//! T2 / T6: per-shape kernel-class throughput theses.
//!
//! T2 holds every decode GEMV class (the TinyLlama-family M=1 f16w row
//! GEMVs) to a fraction of its bandwidth floor, where the floor is
//! bytes moved (f16 weights, f32 activation row in, f32 row out) over
//! the recorded device bandwidth.  T6 holds the f16w coopmat GEMM to a
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
use tensor_ash::{Executor, VulkanContext};

use super::expect::Section;
use super::{Row, Verdict, check_min, key};
use crate::bench::cases::BenchCase;
use crate::bench::commands::run_case;
use crate::bench::env::env_u32;

/// Decode GEMV classes: `(label, K, N)` for a `[1,K] @ [K,N]` f16w row
/// GEMV (TinyLlama-1.1B geometry, the campaign's reference model).
const GEMV_SHAPES: &[(&str, u32, u32)] = &[
    ("q/o 2048x2048", 2048, 2048),
    ("k/v 2048x256", 2048, 256),
    ("gate/up 2048x5632", 2048, 5632),
    ("down 5632x2048", 5632, 2048),
    ("lm_head 2048x32000", 2048, 32000),
];

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
    for &(label, k, n) in GEMV_SHAPES {
        let case = BenchCase {
            label: Cow::Owned(format!("gemv {label}")),
            b: 1,
            m: 1,
            n,
            k,
            b_f16: true,
        };
        let result = run_case(ctx, exec, case, iters, warmup)?;
        let item = format!("GEMV {label} f16w");
        let Some(gpu) = result.gpu_ms else {
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
        let measured_gbps = bytes / (gpu.median * 1e6);
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
                        "{:.1} us = {measured_gbps:.0} GB/s = {:.0}% [{}]",
                        gpu.median * 1e3,
                        frac * 100.0,
                        result.dispatch.kernel,
                    ),
                    check_min(frac, min_frac),
                )
            }
            None => (
                "unrecorded (needs mem_bw_gbps)".into(),
                format!(
                    "{:.1} us = {measured_gbps:.0} GB/s [{}]",
                    gpu.median * 1e3,
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

/// Real aligned 2048-class prefill GEMM shapes `(label, M, N, K)`.
const COOPMAT_SHAPES: &[(&str, u32, u32, u32)] = &[
    ("prefill q/o 512x2048x2048", 512, 2048, 2048),
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
