//! T2 / T6: per-shape kernel-class throughput theses.
//!
//! T2 holds every decode GEMV class (the TinyLlama-family M=1 f16w row
//! GEMVs) to a fraction of its bandwidth floor, where the floor is
//! bytes moved (f16 weights, f32 activation row in, f32 row out) over
//! the recorded device bandwidth.  Packed-B is used where llama-ash
//! packs (q, k/v, gate/up, lm_head) so T2 sees the sequential K-walk
//! that made lm_head hit 94% of device BW; o and down stay unpacked.
//! The verdict is taken on a DRAM-cold measurement: every timed
//! dispatch reads a different copy of B from a rotation set far larger
//! than any L2, because decode streams ~2 GB of weights per token and
//! never sees a weight L2-hot — while a repeated standalone dispatch
//! of an 8 MB q/o weight is served from AD107's 24 MB L2 (and the
//! 1 MB k/v weight even from GA104's 4 MB), which reported
//! "efficiencies" above 100% of the DRAM floor.  The L2-hot
//! `run_case` number is still reported alongside (it also validates
//! numerics and names the routed kernel).
//! T6 holds the f16w coopmat GEMM to a
//! TFLOPS floor on REAL aligned prefill shapes (the 34.6 TF/s headline
//! was 4096^3; what 512-row shapes sustain is the falsifiable claim),
//! with 4096^3 kept as a regression reference.
//!
//! Both reuse the `cases` machinery (`run_case`): timestamped medians,
//! post-timing sampled validation, auto kernel routing with the routed
//! kernel reported per row.

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    Executor, MatmulCall, MatmulOp, Tensor, VulkanContext, f16w_row_tile_n, pack_f16w_row_tiles,
};
use tensor_ash_test_support::fill_det;

use super::expect::Section;
use super::{Row, Verdict, check_min, key};
use crate::bench::cases::{BenchCase, SampleSummary};
use crate::bench::commands::run_case;
use crate::bench::env::env_u32;

/// Decode GEMV classes: `(label, K, N, packed)` for a `[1,K] @ [K,N]`
/// f16w row GEMV (TinyLlama-1.1B geometry). `packed` follows llama-ash
/// (`Layer::wq_p`..`w_up_p`, `lm_head_p`): q, k/v, gate/up, and lm_head
/// use `[N/tile][K][tile]`; o and down stay row-major (packing them
/// was measured-neutral).
const GEMV_SHAPES: &[(&str, u32, u32, bool)] = &[
    ("q 2048x2048", 2048, 2048, true),
    ("o 2048x2048", 2048, 2048, false),
    ("k/v 2048x256", 2048, 256, true),
    ("gate/up 2048x5632", 2048, 5632, true),
    ("down 5632x2048", 5632, 2048, false),
    ("lm_head 2048x32000", 2048, 32000, true),
];

/// Bytes of distinct B copies the DRAM-cold T2 measurement rotates
/// through: 8x AD107's 24 MB L2 (48x GA104's 4 MB), so each timed
/// dispatch reads weights the previous dispatches evicted.
const T2_ROTATION_BYTES: f64 = 192.0 * 1024.0 * 1024.0;
const T2_MAX_COPIES: usize = 256;

/// Median GPU time (us) of an M=1 f16w row GEMV whose B cycles through
/// enough distinct copies that no dispatch finds its weights in L2 —
/// the decode condition.  A and C stay resident (as in decode, where
/// the activation row is L2-hot).  `None` without GPU timestamps.
fn gemv_dram_cold_us(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    k: u32,
    n: u32,
    packed_b: bool,
    iters: u32,
    warmup: u32,
) -> Result<Option<f64>> {
    let b_bytes = 2.0 * k as f64 * n as f64;
    let copies = ((T2_ROTATION_BYTES / b_bytes).ceil() as usize).clamp(2, T2_MAX_COPIES);
    let a = Tensor::uninit_device(ctx, &[1, 1, k])?;
    let c = Tensor::uninit_device(ctx, &[1, 1, n])?;
    let mut h_a = vec![0.0f32; k as usize];
    fill_det(&mut h_a, 7);
    exec.upload(&h_a, &a)?;
    let mut h_b = vec![0.0f32; k as usize * n as usize];
    fill_det(&mut h_b, 11);
    if packed_b {
        let tile = f16w_row_tile_n(k, n) as usize;
        h_b = pack_f16w_row_tiles(&h_b, k as usize, n as usize, tile);
    }
    let bs = (0..copies)
        .map(|_| {
            let b = Tensor::uninit_device_f16(ctx, &[1, k, n])?;
            exec.upload(&h_b, &b)?;
            Ok(b)
        })
        .collect::<Result<Vec<_>>>()?;
    let run = |b: &Tensor| {
        let call = MatmulCall {
            a: &a,
            b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        };
        if packed_b {
            exec.run_ops(&[MatmulOp::new(call).with_packed_b()])
        } else {
            exec.run_matmuls(std::slice::from_ref(&call))
        }
    };
    for i in 0..warmup as usize {
        run(&bs[i % copies])?;
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for i in 0..iters as usize {
        match run(&bs[(warmup as usize + i) % copies])?.gpu_time_ns {
            Some(ns) => samples.push(ns as f64 / 1e3),
            None => return Ok(None),
        }
    }
    Ok(SampleSummary::new(samples).map(|summary| summary.median))
}

pub(crate) fn t2_gemv_efficiency(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    section: Option<&Section>,
) -> Result<Vec<Row>> {
    if !ctx.f16_storage_enabled {
        return Ok(vec![Row {
            thesis: "T2",
            item: "decode GEMV classes".into(),
            prediction: "-".into(),
            measured: "device lacks f16 storage".into(),
            verdict: Verdict::Skip,
        }]);
    }
    let iters = env_u32("ML_ITERS", 50).max(1);
    let warmup = env_u32("ML_WARMUP", 5);
    let bw_gbps = key(section, "mem_bw_gbps");
    let min_frac = key(section, "t2_min_bw_frac");

    let mut rows = Vec::new();
    for &(label, k, n, packed_b) in GEMV_SHAPES {
        let case = BenchCase {
            label: Cow::Owned(format!("gemv {label}")),
            b: 1,
            m: 1,
            n,
            k,
            b_f16: true,
            packed_b,
        };
        let result = run_case(ctx, exec, case, iters, warmup)?;
        let item = if packed_b {
            format!("GEMV {label} f16w packed")
        } else {
            format!("GEMV {label} f16w")
        };
        let cold_us = gemv_dram_cold_us(ctx, exec, k, n, packed_b, iters, warmup)?;
        let (Some(gpu), Some(cold_us)) = (result.gpu_ms, cold_us) else {
            rows.push(Row {
                thesis: "T2",
                item,
                prediction: "-".into(),
                measured: "device reports no GPU timestamps".into(),
                verdict: Verdict::Skip,
            });
            continue;
        };
        // f16 weights + f32 activation row in + f32 output row.
        let bytes = 2.0 * k as f64 * n as f64 + 4.0 * (k as f64 + n as f64);
        let measured_gbps = bytes / (cold_us * 1e3);
        let hot_us = gpu.median * 1e3;
        let (prediction, measured, verdict) = match bw_gbps {
            Some(bw) => {
                let frac = measured_gbps / bw;
                (
                    format!(
                        ">= {:.0}% of BW floor ({:.1} us at {bw:.0} GB/s)",
                        min_frac.unwrap_or(f64::NAN) * 100.0,
                        bytes / (bw * 1e3),
                    ),
                    format!(
                        "{cold_us:.1} us = {measured_gbps:.0} GB/s = {:.0}% \
                         (L2-hot {hot_us:.1} us) [{}]",
                        frac * 100.0,
                        result.dispatch.kernel,
                    ),
                    check_min(frac, min_frac),
                )
            }
            None => (
                "unrecorded (needs mem_bw_gbps)".into(),
                format!(
                    "{cold_us:.1} us = {measured_gbps:.0} GB/s (L2-hot {hot_us:.1} us) [{}]",
                    result.dispatch.kernel,
                ),
                Verdict::Info,
            ),
        };
        rows.push(Row {
            thesis: "T2",
            item,
            prediction,
            measured,
            verdict,
        });
    }
    Ok(rows)
}

/// Real aligned 2048-class prefill GEMM shapes `(label, M, N, K)`:
/// every GEMM TinyLlama's pp512 prefill runs per layer (the q|k|v
/// projections are one concatenated N=2560 GEMM).
const COOPMAT_SHAPES: &[(&str, u32, u32, u32)] = &[
    ("prefill qkv 512x2560x2048", 512, 2560, 2048),
    ("prefill o 512x2048x2048", 512, 2048, 2048),
    ("prefill gate/up 512x5632x2048", 512, 5632, 2048),
    ("prefill down 512x2048x5632", 512, 2048, 5632),
];

pub(crate) fn t6_coopmat_ceiling(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    section: Option<&Section>,
) -> Result<Vec<Row>> {
    if !ctx.coopmat_enabled {
        return Ok(vec![Row {
            thesis: "T6",
            item: "f16w coopmat GEMM".into(),
            prediction: "-".into(),
            measured: "device lacks KHR cooperative matrix".into(),
            verdict: Verdict::Skip,
        }]);
    }
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let min_tflops = key(section, "t6_min_tflops");
    let ref_tflops = key(section, "t6_ref_tflops");
    let ref_rel_tol = key(section, "t6_ref_rel_tol").unwrap_or(0.15);

    let mut rows = Vec::new();
    let measure = |label: &str, m: u32, n: u32, k: u32| -> Result<(String, f64, &'static str)> {
        let case = BenchCase {
            label: Cow::Owned(format!("coopmat {label}")),
            b: 1,
            m,
            n,
            k,
            b_f16: true,
            packed_b: false,
        };
        let result = run_case(ctx, exec, case, iters, warmup)?;
        let kernel = result.dispatch.kernel;
        let gpu_us = result
            .gpu_ms
            .map_or(f64::NAN, |summary| summary.median * 1e3);
        Ok((
            format!("{:.2} TF/s ({gpu_us:.0} us) [{kernel}]", result.tflops),
            result.tflops,
            kernel,
        ))
    };

    for &(label, m, n, k) in COOPMAT_SHAPES {
        let (measured, tflops, kernel) = measure(label, m, n, k)?;
        // Routing away from the tensor cores on an eligible shape is
        // itself a broken assumption, independent of the rate.
        let verdict = if !kernel.contains("coopmat") && min_tflops.is_some() {
            Verdict::Fail
        } else {
            check_min(tflops, min_tflops)
        };
        rows.push(Row {
            thesis: "T6",
            item: format!("coopmat {label} f16w"),
            prediction: min_tflops
                .map_or_else(|| "unrecorded".into(), |min| format!(">= {min:.0} TF/s")),
            measured,
            verdict,
        });
    }

    let (measured, tflops, _kernel) = measure("reference 4096^3", 4096, 4096, 4096)?;
    rows.push(Row {
        thesis: "T6",
        item: "coopmat reference 4096^3 f16w".into(),
        prediction: ref_tflops.map_or_else(
            || "unrecorded".into(),
            |reference| format!("~{reference:.1} TF/s (-{:.0}% floor)", ref_rel_tol * 100.0),
        ),
        measured,
        verdict: check_min(tflops, ref_tflops.map(|r| r * (1.0 - ref_rel_tol))),
    });
    Ok(rows)
}
