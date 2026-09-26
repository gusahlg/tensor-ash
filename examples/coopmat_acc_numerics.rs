//! Numerics of the coopmat1 accumulator classes (f32 / f16 / f16-promote)
//! against an f64 CPU reference.
//!
//! Part 1 (correctness gate): every coopmat1 kernel — 4 tiles x {a32, a16}
//! x {f32, f16, promote} accumulators — pinned via its own pipeline on a
//! small tile-aligned shape, FULL output compared to the f64 reference
//! within a class-aware bound.  Fails (exit != 0) on any violation.
//!
//! Part 2 (error table): the auto-routed prefill shapes 512x2048x2048 and
//! 512x2048x5632 (and 512x5632x2048), a16 and a32, at each accumulator
//! policy, on two input distributions:
//!   * `unif`: A, B ~ U[-1, 1)                    (the ml_bench fixture)
//!   * `llm`:  A ~ N(0, 1) with 0.5% outlier channels x 20 (post-norm LLM
//!     activations), B ~ N(0, 0.02^2) (projection weights)
//! reporting max abs error, max err / max|ref|, relative RMS error and the
//! mean signed error (rounding bias) over 64 sampled rows x all columns.
//!
//! Run (4060 first):  cargo run --release --example coopmat_acc_numerics
//! `ML_NUMERICS_ROWS` (default 64) sets the sampled rows of part 2.

use std::sync::Arc;

use anyhow::{Result, bail, ensure};
use tensor_ash::{
    CoopmatAcc, Executor, ExecutorConfig, KERNEL_SPECS, KernelSelection, MatmulCall,
    MatmulPipeline, Tensor, VulkanContext, dtype::round_f32_via_f16,
};
use tensor_ash_test_support::{LcgRng, fill_det};

fn normal_fill(out: &mut [f32], seed: u64, sigma: f32) {
    let mut rng = LcgRng::new(seed);
    let mut unit = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    for pair in out.chunks_mut(2) {
        let (u1, u2) = (unit(), unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        pair[0] = (r * t.cos()) as f32 * sigma;
        if pair.len() > 1 {
            pair[1] = (r * t.sin()) as f32 * sigma;
        }
    }
}

#[derive(Clone, Copy)]
enum Dist {
    Unif,
    Llm,
}

impl Dist {
    fn name(self) -> &'static str {
        match self {
            Dist::Unif => "unif",
            Dist::Llm => "llm",
        }
    }

    /// Host A [m, k] and B [k, n], already rounded to f16 (every route
    /// multiplies f16 values; the reference must see the same inputs).
    fn inputs(self, m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        match self {
            Dist::Unif => {
                fill_det(&mut a, seed);
                fill_det(&mut b, seed + 1);
            }
            Dist::Llm => {
                normal_fill(&mut a, seed, 1.0);
                // 0.5% of the K channels are outlier features (x20).
                let mut rng = LcgRng::new(seed + 7);
                let outliers: Vec<usize> = (0..(k / 200).max(1))
                    .map(|_| rng.range_u32(0, k as u32 - 1) as usize)
                    .collect();
                for row in a.chunks_mut(k) {
                    for &c in &outliers {
                        row[c] *= 20.0;
                    }
                }
                normal_fill(&mut b, seed + 1, 0.02);
            }
        }
        for v in a.iter_mut().chain(b.iter_mut()) {
            *v = round_f32_via_f16(*v);
        }
        (a, b)
    }
}

/// f64 reference rows of `A @ B`.
fn reference_rows(a: &[f32], b: &[f32], k: usize, n: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows.len() * n];
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get());
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_rows, chunk_out) in rows.chunks(per).zip(out.chunks_mut(per * n)) {
            scope.spawn(move || {
                for (&row, dst) in chunk_rows.iter().zip(chunk_out.chunks_mut(n)) {
                    for (kk, &av) in a[row * k..(row + 1) * k].iter().enumerate() {
                        let av = av as f64;
                        for (d, &bv) in dst.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                            *d += av * bv as f64;
                        }
                    }
                }
            });
        }
    });
    out
}

/// sqrt(sum_k (a_k b_k)^2) per sampled output: the typical size of a
/// random-sign running sum, the scale the accumulation error follows.
fn l2_rows(a: &[f32], b: &[f32], k: usize, n: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows.len() * n];
    for (ri, &row) in rows.iter().enumerate() {
        let dst = &mut out[ri * n..(ri + 1) * n];
        for (kk, &av) in a[row * k..(row + 1) * k].iter().enumerate() {
            let av = av as f64;
            for (d, &bv) in dst.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                let p = av * bv as f64;
                *d += p * p;
            }
        }
        for d in dst.iter_mut() {
            *d = d.sqrt();
        }
    }
    out
}

struct Stats {
    max_abs: f64,
    max_over_ref: f64,
    rel_rms: f64,
    mean: f64,
    /// max |err| / (2^-11 * l2): error in units of one f16 rounding of
    /// the typical running-sum size.
    max_units: f64,
    nonfinite: usize,
}

fn stats(got: &[f64], reference: &[f64], l2: &[f64]) -> Stats {
    let (mut max_abs, mut sq, mut sum, mut rsq, mut rmax, mut units) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut nonfinite = 0;
    let mut finite = 0usize;
    for ((&g, &r), &l) in got.iter().zip(reference).zip(l2) {
        rmax = rmax.max(r.abs());
        rsq += r * r;
        if !g.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = g - r;
        max_abs = max_abs.max(e.abs());
        units = units.max(e.abs() / (l * 2f64.powi(-11)).max(f64::MIN_POSITIVE));
        sq += e * e;
        sum += e;
        finite += 1;
    }
    let nf = finite.max(1) as f64;
    Stats {
        max_abs,
        max_over_ref: max_abs / rmax.max(f64::MIN_POSITIVE),
        rel_rms: (sq / nf).sqrt() / (rsq / reference.len().max(1) as f64).sqrt(),
        mean: sum / nf,
        max_units: units,
        nonfinite,
    }
}

struct Case<'a> {
    ctx: &'a Arc<VulkanContext>,
    a16: bool,
    m: u32,
    n: u32,
    k: u32,
    host_a: &'a [f32],
    host_b: &'a [f32],
}

/// Run `C = A @ B` through `exec` and download all of C.
fn run(exec: &Executor, case: &Case<'_>) -> Result<Vec<f32>> {
    let (m, n, k) = (case.m, case.n, case.k);
    let act = |shape: &[u32]| {
        if case.a16 {
            Tensor::uninit_device_f16(case.ctx, shape)
        } else {
            Tensor::uninit_device(case.ctx, shape)
        }
    };
    let a = act(&[1, m, k])?;
    let b = Tensor::uninit_device_f16(case.ctx, &[1, k, n])?;
    let c = act(&[1, m, n])?;
    exec.upload(case.host_a, &a)?;
    exec.upload(case.host_b, &b)?;
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])?;
    let mut out = vec![0.0f32; (m * n) as usize];
    exec.download(&c, &mut out)?;
    Ok(out)
}

fn executor(
    ctx: &Arc<VulkanContext>,
    pipe: &Arc<MatmulPipeline>,
    acc: CoopmatAcc,
) -> Result<Executor> {
    Executor::new_with_config(
        ctx.clone(),
        pipe.clone(),
        ExecutorConfig {
            n_slots: 1,
            max_calls_per_submit: 4,
            tune: false,
            coopmat_acc: acc,
        },
    )
}

/// Part 1: every coopmat1 kernel, full output, class-aware bound.
fn correctness_gate(ctx: &Arc<VulkanContext>) -> Result<usize> {
    let (m, n, k) = (256usize, 256usize, 1024usize);
    let (host_a, host_b) = Dist::Unif.inputs(m, n, k, 42);
    let rows: Vec<usize> = (0..m).collect();
    let reference = reference_rows(&host_a, &host_b, k, n, &rows);
    let l2 = l2_rows(&host_a, &host_b, k, n, &rows);
    let mut failures = 0;
    println!("## part 1: every coopmat1 kernel, 256x256x1024 unif, full output vs f64");
    println!("| kernel | acc | max abs err | max err / (2^-11 l2) | rel rms | mean err | verdict |");
    println!("|---|---|---|---|---|---|---|");
    for (index, spec) in KERNEL_SPECS.iter().enumerate() {
        if !spec.name.contains("coopmat") {
            continue;
        }
        let selection = KernelSelection::parse(spec.name)?;
        debug_assert_eq!(selection.index(), Some(index));
        let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(ctx, selection)?);
        let exec = executor(ctx, &pipe, CoopmatAcc::F32)?;
        let case = Case {
            ctx,
            a16: spec.a_f16(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            host_a: &host_a,
            host_b: &host_b,
        };
        let got: Vec<f64> = run(&exec, &case)?.iter().map(|&v| v as f64).collect();
        let s = stats(&got, &reference, &l2);
        // Bound in units of 2^-11 * l2 (see ml_bench's acc16_budget):
        // f32 acc ~ output rounding only; f16 acc K/16 roundings of the
        // running sum; promote 8 roundings of a 128-K chunk partial.
        let c_f16_units = if spec.a_f16() { 8.0 } else { 0.0 };
        let limit = match spec.accumulator() {
            CoopmatAcc::F32 => 1.0 + c_f16_units,
            CoopmatAcc::F16 => (k as f64 / 16.0) + c_f16_units,
            CoopmatAcc::F16Promote => 8.0 * (k as f64 / 128.0).sqrt() + c_f16_units,
        };
        let ok = s.nonfinite == 0 && s.max_units <= limit;
        if !ok {
            failures += 1;
        }
        println!(
            "| {} | {} | {:.3e} | {:.2} (limit {:.1}) | {:.3e} | {:+.2e} | {} |",
            spec.name,
            spec.accumulator().as_str(),
            s.max_abs,
            s.max_units,
            limit,
            s.rel_rms,
            s.mean,
            if ok { "ok" } else { "FAIL" }
        );
    }
    Ok(failures)
}

/// Part 2: error table on the prefill shapes through the auto route.
fn error_table(ctx: &Arc<VulkanContext>) -> Result<()> {
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        ctx,
        KernelSelection::Auto,
    )?);
    let policies = [CoopmatAcc::F32, CoopmatAcc::F16, CoopmatAcc::F16Promote];
    let execs = policies
        .iter()
        .map(|&acc| executor(ctx, &pipe, acc))
        .collect::<Result<Vec<_>>>()?;
    let sample = std::env::var("ML_NUMERICS_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64usize);
    println!();
    println!("## part 2: prefill shapes, auto route, {sample} sampled rows x all N vs f64");
    println!(
        "| dist | A | MxNxK | max|ref| | rms(ref) | acc | kernel | max abs err | max err / max|ref| | rel rms err | mean err | max err / (2^-11 l2) | nonfinite |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for dist in [Dist::Unif, Dist::Llm] {
        for (m, n, k) in [
            (512usize, 2048usize, 2048usize),
            (512, 5632, 2048),
            (512, 2048, 5632),
        ] {
            let (host_a, host_b) = dist.inputs(m, n, k, 7 + k as u64);
            let rows: Vec<usize> = (0..sample.min(m)).map(|i| i * m / sample.min(m)).collect();
            let reference = reference_rows(&host_a, &host_b, k, n, &rows);
            let l2 = l2_rows(&host_a, &host_b, k, n, &rows);
            let rmax = reference.iter().fold(0.0f64, |acc, r| acc.max(r.abs()));
            let rrms = (reference.iter().map(|r| r * r).sum::<f64>() / reference.len() as f64)
                .sqrt();
            for a16 in [true, false] {
                let case = Case {
                    ctx,
                    a16,
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    host_a: &host_a,
                    host_b: &host_b,
                };
                for (exec, acc) in execs.iter().zip(policies) {
                    let kernel = if a16 {
                        exec.dispatch_info_a16(1, m as u32, n as u32, k as u32)
                            .kernel
                    } else {
                        exec.dispatch_info_for(1, m as u32, n as u32, k as u32, true)
                            .kernel
                    };
                    let out = run(exec, &case)?;
                    let got: Vec<f64> = rows
                        .iter()
                        .flat_map(|&r| out[r * n..(r + 1) * n].iter())
                        .map(|&v| v as f64)
                        .collect();
                    let s = stats(&got, &reference, &l2);
                    println!(
                        "| {} | {} | {m}x{n}x{k} | {rmax:.3} | {rrms:.4} | {} | {kernel} | {:.3e} | {:.3e} | {:.3e} | {:+.2e} | {:.2} | {} |",
                        dist.name(),
                        if a16 { "a16" } else { "a32" },
                        acc.as_str(),
                        s.max_abs,
                        s.max_over_ref,
                        s.rel_rms,
                        s.mean,
                        s.max_units,
                        s.nonfinite,
                    );
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let ctx = VulkanContext::new(false)?;
    ensure!(
        ctx.coopmat_enabled && ctx.f16_storage_enabled,
        "device lacks cooperative-matrix / f16-storage support"
    );
    ensure!(
        ctx.coopmat_f16_acc_enabled,
        "device reports no 16x16x16 f16-accumulator coopmat configuration"
    );
    println!("{}", ctx.diagnostics());
    let failures = correctness_gate(&ctx)?;
    if std::env::var("ML_NUMERICS_GATE_ONLY").is_err() {
        error_table(&ctx)?;
    }
    if failures > 0 {
        bail!("{failures} coopmat kernel(s) exceeded their numerics bound");
    }
    Ok(())
}
