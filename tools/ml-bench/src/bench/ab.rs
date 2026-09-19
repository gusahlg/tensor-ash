//! `ml_bench ab` — interleaved multi-kernel A/B measurement.
//!
//! Several kernels on several shapes in one process, one Vulkan context,
//! tensors allocated once per case. Rounds rotate which kernel runs first
//! so GPU clock drift does not systematically favour the same pick.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tensor_ash::{
    Executor, ExecutorConfig, KERNEL_SPECS, KernelSelection, MatmulCall, MatmulPipeline, Tensor,
    VulkanContext,
};

use super::cases::{BenchCase, SampleSummary};
use super::commands::{
    prepare_case_tensors, reference_a_for_kernel, validate_samples, CaseTensors,
};
use super::env::{OutputMode, env_bool, env_string, env_u32, env_usize};
use super::report::csv_escape;

/// Parse `ML_AB_KERNELS` (comma-separated catalog names / aliases / `auto`).
pub(super) fn parse_ab_kernels(raw: &str) -> Result<Vec<(String, KernelSelection)>> {
    let raw = raw.trim();
    if raw.is_empty() {
        bail!(
            "ML_AB_KERNELS must be a comma-separated list of kernel names or auto \
             (e.g. ML_AB_KERNELS=k64_bda_v4_tm8_tn4,m128n64k64_bda_v4,auto)"
        );
    }
    let mut out = Vec::new();
    for token in raw.split(',') {
        let token = token.trim();
        if token.is_empty() {
            bail!("ML_AB_KERNELS has an empty entry; expected a kernel name or auto");
        }
        let selection = KernelSelection::parse(token)
            .with_context(|| format!("ML_AB_KERNELS entry '{token}'"))?;
        out.push((token.to_string(), selection));
    }
    Ok(out)
}

/// Kernel indices for one round: start index rotated by `round` so no
/// kernel is always first.
pub(super) fn rotate_kernel_indices(n_kernels: usize, round: usize) -> Vec<usize> {
    if n_kernels == 0 {
        return Vec::new();
    }
    (0..n_kernels)
        .map(|i| (i + round) % n_kernels)
        .collect()
}

struct AbKernel {
    label: String,
    selection: KernelSelection,
    exec: Executor,
}

pub(super) fn run(
    ctx: &Arc<VulkanContext>,
    n_slots: usize,
    specs: impl IntoIterator<Item = String>,
) -> Result<()> {
    let kernels = parse_ab_kernels(&env_string("ML_AB_KERNELS", ""))?;
    let cases = specs
        .into_iter()
        .map(|spec| BenchCase::parse(&spec))
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        !cases.is_empty(),
        "ab requires at least one label,b,m,n,k argument"
    );

    let rounds = env_usize("ML_ROUNDS", 5).max(1);
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 5);
    let no_burn = env_bool("ML_AB_NO_BURN");
    let mode = OutputMode::from_env();

    let mut entries = Vec::with_capacity(kernels.len());
    for (label, selection) in &kernels {
        let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(ctx, *selection)?);
        let exec = Executor::new_with_config(
            ctx.clone(),
            pipe,
            ExecutorConfig {
                n_slots: n_slots.max(1),
                max_calls_per_submit: 8,
                tune: false,
            },
        )?;
        entries.push(AbKernel {
            label: label.clone(),
            selection: *selection,
            exec,
        });
    }

    if !no_burn {
        burn_in(ctx, &entries[0])?;
    }

    let kernel_list = kernels
        .iter()
        .map(|(label, _)| label.as_str())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "ab protocol: burn-in {}~300ms 4096^3 on first kernel; rounds={rounds} \
         iters={iters} warmup={warmup}; kernels={kernel_list}; per-round kernel \
         start rotated by round index; gpu_ms = median of per-round GPU medians",
        if no_burn { "skipped; " } else { "" },
    );

    let mut prepared = Vec::with_capacity(cases.len());
    for case in &cases {
        prepared.push(prepare_case_tensors(
            ctx,
            &entries[0].exec,
            case,
            /*requantize_a_for_tensor_core=*/ false,
        )?);
    }

    // `None` = this (case, kernel) cannot run; skip remaining rounds.
    let mut round_ms: Vec<Vec<Option<Vec<f64>>>> =
        vec![vec![Some(Vec::with_capacity(rounds)); entries.len()]; cases.len()];

    for round in 0..rounds {
        for (ci, (case, tensors)) in cases.iter().zip(prepared.iter()).enumerate() {
            for ki in rotate_kernel_indices(entries.len(), round) {
                if round_ms[ci][ki].is_none() {
                    continue;
                }
                let entry = &entries[ki];
                match time_round(&entry.exec, tensors, iters, warmup) {
                    Ok(median_ms) => {
                        if round == 0 {
                            validate_round_zero(&entry.exec, case, tensors)?;
                        }
                        if let Some(samples) = &mut round_ms[ci][ki] {
                            samples.push(median_ms);
                        }
                    }
                    Err(err) => {
                        log::info!(
                            "ab: case {} kernel {} n/a ({err:#})",
                            case.label,
                            entry.label
                        );
                        round_ms[ci][ki] = None;
                    }
                }
            }
        }
    }

    print_results(mode, &cases, &entries, &round_ms);
    Ok(())
}

fn burn_in(ctx: &Arc<VulkanContext>, first: &AbKernel) -> Result<()> {
    const DIM: u32 = 4096;
    let f16_b = first
        .selection
        .index()
        .map(|i| KERNEL_SPECS[i].weights_f16())
        .unwrap_or(false);
    let a = Tensor::uninit_device(ctx, &[1, DIM, DIM])?;
    let b = if f16_b {
        Tensor::uninit_device_f16(ctx, &[1, DIM, DIM])?
    } else {
        Tensor::uninit_device(ctx, &[1, DIM, DIM])?
    };
    let c = Tensor::uninit_device(ctx, &[1, DIM, DIM])?;
    let call = MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    };
    let start = Instant::now();
    let mut n = 0u32;
    while start.elapsed() < Duration::from_millis(300) {
        first
            .exec
            .run_matmuls(std::slice::from_ref(&call))
            .context("ab burn-in 4096^3")?;
        n += 1;
    }
    log::info!(
        "ab burn-in: {n} x 4096^3 in {:.0} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

fn time_round(exec: &Executor, tensors: &CaseTensors, iters: u32, warmup: u32) -> Result<f64> {
    let call = MatmulCall {
        a: &tensors.a,
        b: &tensors.b,
        c: &tensors.c,
        alpha: 1.0,
        accumulate: false,
    };
    for _ in 0..warmup {
        exec.run_matmuls(std::slice::from_ref(&call))?;
    }
    let mut gpu_ms = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let stats = exec.run_matmuls(std::slice::from_ref(&call))?;
        let Some(ns) = stats.gpu_time_ns else {
            bail!("GPU timestamps unavailable");
        };
        gpu_ms.push(ns as f64 / 1e6);
    }
    SampleSummary::new(gpu_ms)
        .map(|summary| summary.median)
        .context("benchmark produced no GPU-time samples")
}

fn validate_round_zero(exec: &Executor, case: &BenchCase, tensors: &CaseTensors) -> Result<()> {
    let kernel = exec
        .dispatch_info_for(case.b, case.m, case.n, case.k, case.b_f16)
        .kernel;
    let h_a = reference_a_for_kernel(&tensors.h_a, kernel, case.b_f16);
    validate_samples(
        exec,
        &tensors.c,
        h_a.as_ref(),
        &tensors.h_b,
        case.b,
        case.m,
        case.n,
        case.k,
    )
    .with_context(|| {
        format!(
            "ab sampled correctness failed for case {} kernel {kernel}",
            case.label
        )
    })
}

fn print_results(
    mode: OutputMode,
    cases: &[BenchCase],
    entries: &[AbKernel],
    round_ms: &[Vec<Option<Vec<f64>>>],
) {
    match mode {
        OutputMode::Table => {
            println!(
                "{:<24} {:<28} {:>10} {:>10} {:>8} {:>8}",
                "case", "kernel", "gpu_ms", "gpu_ms_min", "TF/s", "ratio"
            );
            println!("{}", "-".repeat(92));
        }
        OutputMode::Csv => {
            println!("case,kernel,gpu_ms,gpu_ms_min,tflops,ratio,b,m,n,k");
        }
    }

    for (ci, case) in cases.iter().enumerate() {
        let flops = 2.0 * case.b as f64 * case.m as f64 * case.n as f64 * case.k as f64;
        let baseline_tf = cell_stats(&round_ms[ci][0], flops).and_then(|s| s.tflops);
        for (ki, entry) in entries.iter().enumerate() {
            let stats = cell_stats(&round_ms[ci][ki], flops);
            let ratio = match (stats.and_then(|s| s.tflops), baseline_tf) {
                (Some(tf), Some(base)) if base > 0.0 => Some(tf / base),
                _ => None,
            };
            match mode {
                OutputMode::Table => println!(
                    "{:<24} {:<28} {:>10} {:>10} {:>8} {:>8}",
                    case.label,
                    entry.label,
                    fmt_num(stats.map(|s| s.gpu_ms), 3),
                    fmt_num(stats.map(|s| s.gpu_ms_min), 3),
                    fmt_num(stats.and_then(|s| s.tflops), 2),
                    fmt_num(ratio, 3),
                ),
                OutputMode::Csv => println!(
                    "{},{},{},{},{},{},{},{},{},{}",
                    csv_escape(&case.label),
                    csv_escape(&entry.label),
                    fmt_num(stats.map(|s| s.gpu_ms), 6),
                    fmt_num(stats.map(|s| s.gpu_ms_min), 6),
                    fmt_num(stats.and_then(|s| s.tflops), 6),
                    fmt_num(ratio, 6),
                    case.b,
                    case.m,
                    case.n,
                    case.k,
                ),
            }
        }
    }
}

#[derive(Clone, Copy)]
struct CellStats {
    gpu_ms: f64,
    gpu_ms_min: f64,
    tflops: Option<f64>,
}

fn cell_stats(samples: &Option<Vec<f64>>, flops: f64) -> Option<CellStats> {
    let samples = samples.as_ref()?;
    let summary = SampleSummary::new(samples.iter().copied())?;
    let tflops = (summary.median > 0.0).then_some(flops / summary.median * 1e-9);
    Some(CellStats {
        gpu_ms: summary.median,
        gpu_ms_min: summary.min,
        tflops,
    })
}

fn fmt_num(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.precision$}"),
        _ => "n/a".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ab_kernels_accepts_catalog_names_and_auto() {
        let parsed = parse_ab_kernels(
            "k64_bda_v4_tm8_tn4, m128n64k64_bda_v4, large_bda_v4, auto",
        )
        .unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].0, "k64_bda_v4_tm8_tn4");
        assert_eq!(parsed[0].1, KernelSelection::K64BdaV4Tm8Tn4);
        assert_eq!(parsed[1].0, "m128n64k64_bda_v4");
        assert_eq!(parsed[1].1, KernelSelection::M128N64K64BdaV4);
        assert_eq!(parsed[2].0, "large_bda_v4");
        assert_eq!(parsed[2].1, KernelSelection::LargeBdaV4);
        assert_eq!(parsed[3].0, "auto");
        assert_eq!(parsed[3].1, KernelSelection::Auto);
    }

    #[test]
    fn parse_ab_kernels_is_case_insensitive_and_keeps_the_token() {
        let parsed = parse_ab_kernels(" AUTO ").unwrap();
        assert_eq!(parsed[0].0, "AUTO");
        assert_eq!(parsed[0].1, KernelSelection::Auto);
    }

    #[test]
    fn parse_ab_kernels_rejects_empty_and_unknown() {
        for invalid in ["", "   ", ",", "auto,", ",auto", "not_a_kernel", "auto,,k64"] {
            assert!(
                parse_ab_kernels(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn rotate_kernel_indices_cycles_the_start() {
        assert_eq!(rotate_kernel_indices(3, 0), vec![0, 1, 2]);
        assert_eq!(rotate_kernel_indices(3, 1), vec![1, 2, 0]);
        assert_eq!(rotate_kernel_indices(3, 2), vec![2, 0, 1]);
        assert_eq!(rotate_kernel_indices(3, 3), vec![0, 1, 2]);
        assert_eq!(rotate_kernel_indices(3, 4), vec![1, 2, 0]);
        assert_eq!(rotate_kernel_indices(4, 0), vec![0, 1, 2, 3]);
        assert_eq!(rotate_kernel_indices(4, 1), vec![1, 2, 3, 0]);
        assert_eq!(rotate_kernel_indices(1, 7), vec![0]);
        assert_eq!(rotate_kernel_indices(0, 1), Vec::<usize>::new());
    }

    #[test]
    fn fmt_num_renders_na_for_missing_or_non_finite() {
        assert_eq!(fmt_num(None, 3), "n/a");
        assert_eq!(fmt_num(Some(f64::NAN), 3), "n/a");
        assert_eq!(fmt_num(Some(f64::INFINITY), 2), "n/a");
        assert_eq!(fmt_num(Some(1.2346), 3), "1.235");
    }

    #[test]
    fn cell_stats_median_and_min_of_round_medians() {
        let samples = Some(vec![4.0, 1.0, 2.0, 3.0, 5.0]);
        let stats = cell_stats(&samples, 2.0e12).unwrap();
        assert_eq!(stats.gpu_ms, 3.0);
        assert_eq!(stats.gpu_ms_min, 1.0);
        assert!((stats.tflops.unwrap() - 2.0e12 / 3.0 * 1e-9).abs() < 1e-12);
        assert!(cell_stats(&Some(Vec::new()), 1.0).is_none());
        assert!(cell_stats(&None, 1.0).is_none());
    }
}
