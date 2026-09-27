//! Real-activation GEMM numerics: rerun the last layer's prefill GEMMs
//! (QKV, O, up, down) on the activations a real prefill left in scratch,
//! once per caller-supplied executor (e.g. one per coopmat accumulator
//! policy), and compare every result against an f64 CPU reference.

use anyhow::{Result, ensure};
use tensor_ash::{DType, Executor, MatmulOp, Tensor};

use super::{Model, mm};

/// Error of one executor's result against the f64 reference, over the
/// sampled rows.
#[derive(Clone, Debug)]
pub struct ErrStats {
    pub max_abs: f64,
    /// `max_abs / max|ref|` over the sampled outputs.
    pub max_abs_over_max_ref: f64,
    /// Root-mean-square error.
    pub rms: f64,
    /// `rms / rms(ref)`.
    pub rel_rms: f64,
    /// Mean signed error (a rounding-bias detector).
    pub mean: f64,
    /// Outputs that are not finite (f16 overflow).
    pub nonfinite: usize,
}

/// One GEMM's report.
#[derive(Clone, Debug)]
pub struct GemmNumerics {
    pub name: &'static str,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rows_checked: usize,
    /// Largest |A| (activation) and |B| (weight) element.
    pub max_abs_a: f32,
    pub max_abs_b: f32,
    /// Largest |ref| over the sampled outputs, and its RMS.
    pub max_abs_ref: f64,
    pub rms_ref: f64,
    /// The unavoidable floor of any f16-C route: the reference rounded
    /// to f16 (RNE), measured like the executors below.
    pub f16_store_floor: ErrStats,
    /// (label, kernel name, stats) per executor, in the order given.
    pub results: Vec<(String, &'static str, ErrStats)>,
}

fn err_stats(got: &[f64], reference: &[f64]) -> ErrStats {
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    let mut ref_max = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut nonfinite = 0usize;
    let mut finite = 0usize;
    for (&g, &r) in got.iter().zip(reference) {
        ref_max = ref_max.max(r.abs());
        ref_sq += r * r;
        if !g.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = g - r;
        max_abs = max_abs.max(e.abs());
        sum_sq += e * e;
        sum += e;
        finite += 1;
    }
    let n = finite.max(1) as f64;
    let rms = (sum_sq / n).sqrt();
    let rms_ref = (ref_sq / reference.len().max(1) as f64).sqrt();
    ErrStats {
        max_abs,
        max_abs_over_max_ref: max_abs / ref_max.max(f64::MIN_POSITIVE),
        rms,
        rel_rms: rms / rms_ref.max(f64::MIN_POSITIVE),
        mean: sum / n,
        nonfinite,
    }
}

/// f64 reference rows `rows` of `A[m,k] @ B[k,n]`, row-major, threaded
/// over rows.
fn reference_rows(a: &[f32], b: &[f32], k: usize, n: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows.len() * n];
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_rows, chunk_out) in rows.chunks(per).zip(out.chunks_mut(per * n)) {
            scope.spawn(move || {
                for (&row, dst) in chunk_rows.iter().zip(chunk_out.chunks_mut(n)) {
                    let a_row = &a[row * k..(row + 1) * k];
                    for (kk, &av) in a_row.iter().enumerate() {
                        let av = av as f64;
                        let b_row = &b[kk * n..(kk + 1) * n];
                        for (d, &bv) in dst.iter_mut().zip(b_row) {
                            *d += av * bv as f64;
                        }
                    }
                }
            });
        }
    });
    out
}

fn download(exec: &Executor, t: &Tensor) -> Result<Vec<f32>> {
    let mut host = vec![0.0f32; t.shape().iter().product::<u32>() as usize];
    exec.download(t, &mut host)?;
    Ok(host)
}

impl Model {
    /// Prefill `tokens` (the model's own executor/policy), then rerun
    /// the LAST layer's four prefill GEMMs on the activations left in
    /// scratch through each executor in `execs` (which must share this
    /// model's Vulkan context) and report errors vs an f64 reference
    /// over `sample_rows` evenly spaced rows.  Requires the f16-
    /// activation prefill route (T >= 128, T % 64 == 0), so every GEMM
    /// is the a16 coopmat route llama prefill actually takes.
    pub fn gemm_numerics(
        &mut self,
        tokens: &[u32],
        execs: &[(String, &Executor)],
        sample_rows: usize,
    ) -> Result<Vec<GemmNumerics>> {
        self.prefill(tokens)?;
        let s = self
            .prefill_scratch
            .as_ref()
            .expect("prefill built scratch");
        ensure!(
            s.xn.dtype() == DType::F16,
            "gemm numerics needs the f16-activation prefill route (T >= 128, T % 64 == 0, \
             LLAMA_ASH_ACT unset); got T = {}",
            s.t
        );
        let layer = self.layers.last().expect("model has layers");
        let t = s.t as usize;
        let rows: Vec<usize> = (0..sample_rows.min(t))
            .map(|i| i * t / sample_rows.min(t))
            .collect();
        let gemms: [(&'static str, &Tensor, &Tensor); 4] = [
            ("qkv", &s.xn, &layer.w_qkv),
            ("o", &s.attn_flat, &layer.wo),
            ("up", &s.on, &layer.w_up),
            ("down", &s.gate, &layer.w_down),
        ];
        let mut reports = Vec::with_capacity(gemms.len());
        for (name, a, b) in gemms {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let host_a = download(&self.exec, a)?;
            let host_b = download(&self.exec, b)?;
            let reference = reference_rows(&host_a, &host_b, k as usize, n as usize, &rows);
            let rounded: Vec<f64> = reference
                .iter()
                .map(|&r| tensor_ash::dtype::round_f32_via_f16(r as f32) as f64)
                .collect();
            let max_abs_ref = reference.iter().fold(0.0f64, |acc, r| acc.max(r.abs()));
            let rms_ref =
                (reference.iter().map(|r| r * r).sum::<f64>() / reference.len() as f64).sqrt();
            // A fresh output so the model's scratch stays intact.
            let c = Tensor::uninit_device_f16(&self.ctx, &[m, n])?;
            let mut results = Vec::with_capacity(execs.len());
            for (label, exec) in execs {
                let kernel = exec.dispatch_info_a16(1, m, n, k).kernel;
                exec.run_ops(&[MatmulOp::new(mm(a, b, &c))])?;
                let out = download(exec, &c)?;
                let got: Vec<f64> = rows
                    .iter()
                    .flat_map(|&row| out[row * n as usize..(row + 1) * n as usize].iter())
                    .map(|&v| v as f64)
                    .collect();
                results.push((label.clone(), kernel, err_stats(&got, &reference)));
            }
            reports.push(GemmNumerics {
                name,
                m,
                n,
                k,
                rows_checked: rows.len(),
                max_abs_a: host_a.iter().fold(0.0f32, |acc, v| acc.max(v.abs())),
                max_abs_b: host_b.iter().fold(0.0f32, |acc, v| acc.max(v.abs())),
                max_abs_ref,
                rms_ref,
                f16_store_floor: err_stats(&rounded, &reference),
                results,
            });
        }
        Ok(reports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn err_stats_measures_bias_and_overflow() {
        let reference = [1.0, -2.0, 4.0, 0.5];
        let got = [1.5, -2.0, f64::INFINITY, 0.0];
        let stats = err_stats(&got, &reference);
        assert_eq!(stats.nonfinite, 1);
        assert!((stats.max_abs - 0.5).abs() < 1e-12);
        assert!((stats.mean - 0.0).abs() < 1e-12); // +0.5, 0, -0.5
        assert!((stats.max_abs_over_max_ref - 0.125).abs() < 1e-12);
    }

    #[test]
    fn reference_rows_matches_naive() {
        let (m, k, n) = (5usize, 7usize, 3usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.11).cos()).collect();
        let rows = [0usize, 2, 4];
        let got = reference_rows(&a, &b, k, n, &rows);
        for (ri, &row) in rows.iter().enumerate() {
            for col in 0..n {
                let want: f64 = (0..k)
                    .map(|kk| a[row * k + kk] as f64 * b[kk * n + col] as f64)
                    .sum();
                assert!((got[ri * n + col] - want).abs() < 1e-9);
            }
        }
    }
}
