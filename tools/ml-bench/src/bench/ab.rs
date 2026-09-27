//! `ml_bench ab` — interleaved multi-kernel A/B measurement.
//!
//! Several kernels on several shapes in one process, one Vulkan context,
//! tensors allocated once per case. Rounds rotate which kernel runs first
//! so GPU clock drift does not systematically favour the same pick.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tensor_ash::{
    BinaryOp, CoopmatAcc, Executor, ExecutorConfig, KERNEL_SPECS, KernelSelection, MatmulCall,
    MatmulPipeline, Tensor, VulkanContext,
};

use super::cases::{BenchCase, SampleSummary};
use super::commands::{
    CaseTensors, prepare_case_tensors, reference_a_for_kernel, validate_samples,
};
use super::env::{OutputMode, env_bool, env_string, env_u32, env_usize};
use super::report::csv_escape;

/// One `ML_AB_KERNELS` entry: the token as typed, its kernel
/// selection, and the executor's coopmat accumulator policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AbEntry {
    pub label: String,
    pub selection: KernelSelection,
    pub acc: CoopmatAcc,
}

/// Parse `ML_AB_KERNELS` (comma-separated catalog names / aliases /
/// `auto`, each optionally `@f32|@f16|@promote` to set that entry's
/// executor accumulator policy — e.g. `auto,auto@f16` A/Bs the routed
/// tile at both precisions; explicit kernels carry their own class).
pub(super) fn parse_ab_kernels(raw: &str) -> Result<Vec<AbEntry>> {
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
        let (name, acc) = match token.split_once('@') {
            Some((name, acc)) => (
                name,
                CoopmatAcc::parse(acc).with_context(|| format!("ML_AB_KERNELS entry '{token}'"))?,
            ),
            None => (token, CoopmatAcc::F32),
        };
        let selection = KernelSelection::parse(name)
            .with_context(|| format!("ML_AB_KERNELS entry '{token}'"))?;
        out.push(AbEntry {
            label: token.to_string(),
            selection,
            acc,
        });
    }
    Ok(out)
}

/// Kernel indices for one round: start index rotated by `round` so no
/// kernel is always first.
pub(super) fn rotate_kernel_indices(n_kernels: usize, round: usize) -> Vec<usize> {
    if n_kernels == 0 {
        return Vec::new();
    }
    (0..n_kernels).map(|i| (i + round) % n_kernels).collect()
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
    // ML_AB_COLD_MB=N: before every timed GEMM, stream an N MiB f32
    // buffer pair through the GPU (evicting L2) and re-read A, so B
    // (the weights) is cold and A (the activations) warm — how a model
    // forward sees each layer's GEMM.  Default off: back-to-back
    // repeats keep a B that fits the L2 resident, which flatters
    // x-fastest small tiles on big-L2 parts.
    let cold_mb = env_u32("ML_AB_COLD_MB", 0);
    let mode = OutputMode::from_env();

    let mut entries = Vec::with_capacity(kernels.len());
    for entry in &kernels {
        let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
            ctx,
            entry.selection,
        )?);
        let exec = Executor::new_with_config(
            ctx.clone(),
            pipe,
            ExecutorConfig {
                n_slots: n_slots.max(1),
                max_calls_per_submit: 8,
                tune: false,
                coopmat_acc: entry.acc,
            },
        )?;
        entries.push(AbKernel {
            label: entry.label.clone(),
            selection: entry.selection,
            exec,
        });
    }

    if !no_burn {
        burn_in(ctx, &entries[0])?;
    }

    let kernel_list = kernels
        .iter()
        .map(|entry| entry.label.as_str())
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "ab protocol: burn-in {}~300ms 4096^3 on first kernel; rounds={rounds} \
         iters={iters} warmup={warmup}; kernels={kernel_list}; per-round kernel \
         start rotated by round index; gpu_ms = median of per-round GPU medians; \
         cold_l2={}",
        if no_burn { "skipped; " } else { "" },
        if cold_mb > 0 {
            format!("{cold_mb} MiB flush + A re-read before each GEMM")
        } else {
            "off".into()
        },
    );
    let flush = if cold_mb > 0 {
        let elems = cold_mb.saturating_mul(1 << 18); // MiB of f32
        Some((
            Tensor::uninit_device(ctx, &[elems])?,
            Tensor::uninit_device(ctx, &[elems])?,
        ))
    } else {
        None
    };

    let mut prepared = Vec::with_capacity(cases.len());
    let mut a_scratch = Vec::with_capacity(cases.len());
    for case in &cases {
        let tensors = prepare_case_tensors(
            ctx,
            &entries[0].exec,
            case,
            /*requantize_a_for_tensor_core=*/ false,
        )?;
        a_scratch.push(if flush.is_some() {
            let shape = [tensors.a.len() as u32];
            Some(if case.a16 {
                Tensor::uninit_device_f16(ctx, &shape)?
            } else {
                Tensor::uninit_device(ctx, &shape)?
            })
        } else {
            None
        });
        prepared.push(tensors);
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
                let cold = flush
                    .as_ref()
                    .zip(a_scratch[ci].as_ref())
                    .map(|((f0, f1), a_copy)| Cold { f0, f1, a_copy });
                match time_round(&entry.exec, tensors, iters, warmup, cold) {
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
    let spec = first.selection.index().map(|i| &KERNEL_SPECS[i]);
    let f16_b = spec.is_some_and(|s| s.weights_f16());
    // An a16 kernel pin only accepts f16 A/C (the f16-activation route).
    let a16 = spec.is_some_and(|s| s.a_f16());
    let act = |shape: &[u32]| {
        if a16 {
            Tensor::uninit_device_f16(ctx, shape)
        } else {
            Tensor::uninit_device(ctx, shape)
        }
    };
    let a = act(&[1, DIM, DIM])?;
    let b = if f16_b {
        Tensor::uninit_device_f16(ctx, &[1, DIM, DIM])?
    } else {
        Tensor::uninit_device(ctx, &[1, DIM, DIM])?
    };
    let c = act(&[1, DIM, DIM])?;
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

/// L2-eviction buffers for `ML_AB_COLD_MB` plus an A-sized scratch the
/// re-read of A writes to.
struct Cold<'t> {
    f0: &'t Tensor,
    f1: &'t Tensor,
    a_copy: &'t Tensor,
}

impl Cold<'_> {
    /// Evict L2 (stream both flush buffers), then pull A back in.
    /// Separate submissions: none of this is inside the GEMM's GPU
    /// timestamps.
    fn prime(&self, exec: &Executor, a: &Tensor) -> Result<()> {
        exec.run_binary(self.f0, self.f1, self.f0, BinaryOp::AddScaled { beta: 0.0 })?;
        exec.run_binary(a, a, self.a_copy, BinaryOp::AddScaled { beta: 0.0 })?;
        Ok(())
    }
}

fn time_round(
    exec: &Executor,
    tensors: &CaseTensors,
    iters: u32,
    warmup: u32,
    cold: Option<Cold<'_>>,
) -> Result<f64> {
    let call = MatmulCall {
        a: &tensors.a,
        b: &tensors.b,
        c: &tensors.c,
        alpha: 1.0,
        accumulate: false,
    };
    for _ in 0..warmup {
        if let Some(cold) = &cold {
            cold.prime(exec, &tensors.a)?;
        }
        exec.run_matmuls(std::slice::from_ref(&call))?;
    }
    let mut gpu_ms = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        if let Some(cold) = &cold {
            cold.prime(exec, &tensors.a)?;
        }
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

/// The kernel `exec` dispatches for `case` (auto resolves per shape).
fn route_kernel(exec: &Executor, case: &BenchCase) -> &'static str {
    if case.a16 {
        exec.dispatch_info_a16(case.b, case.m, case.n, case.k)
            .kernel
    } else {
        exec.dispatch_info_for(case.b, case.m, case.n, case.k, case.b_f16)
            .kernel
    }
}

fn validate_round_zero(exec: &Executor, case: &BenchCase, tensors: &CaseTensors) -> Result<()> {
    let kernel = route_kernel(exec, case);
    // a16 fixtures are already f16-rounded (the device A is half).
    let h_a = reference_a_for_kernel(&tensors.h_a, kernel, case.b_f16 && !case.a16);
    validate_samples(
        exec,
        &tensors.c,
        h_a.as_ref(),
        &tensors.h_b,
        case.b,
        case.m,
        case.n,
        case.k,
        case.a16,
        kernel,
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
                "{:<24} {:<28} {:>10} {:>10} {:>8} {:>8}  route",
                "case", "kernel", "gpu_ms", "gpu_ms_min", "TF/s", "ratio"
            );
            println!("{}", "-".repeat(120));
        }
        OutputMode::Csv => {
            println!("case,kernel,gpu_ms,gpu_ms_min,tflops,ratio,b,m,n,k,route");
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
            let route = route_kernel(&entry.exec, case);
            match mode {
                OutputMode::Table => println!(
                    "{:<24} {:<28} {:>10} {:>10} {:>8} {:>8}  {}",
                    case.label,
                    entry.label,
                    fmt_num(stats.map(|s| s.gpu_ms), 3),
                    fmt_num(stats.map(|s| s.gpu_ms_min), 3),
                    fmt_num(stats.and_then(|s| s.tflops), 2),
                    fmt_num(ratio, 3),
                    route,
                ),
                OutputMode::Csv => println!(
                    "{},{},{},{},{},{},{},{},{},{},{}",
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
                    route,
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
        let parsed =
            parse_ab_kernels("k64_bda_v4_tm8_tn4, m128n64k64_bda_v4, large_bda_v4, auto").unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].label, "k64_bda_v4_tm8_tn4");
        assert_eq!(parsed[0].selection, KernelSelection::K64BdaV4Tm8Tn4);
        assert_eq!(parsed[1].label, "m128n64k64_bda_v4");
        assert_eq!(parsed[1].selection, KernelSelection::M128N64K64BdaV4);
        assert_eq!(parsed[2].label, "large_bda_v4");
        assert_eq!(parsed[2].selection, KernelSelection::LargeBdaV4);
        assert_eq!(parsed[3].label, "auto");
        assert_eq!(parsed[3].selection, KernelSelection::Auto);
        assert!(parsed.iter().all(|e| e.acc == CoopmatAcc::F32));
    }

    #[test]
    fn parse_ab_kernels_is_case_insensitive_and_keeps_the_token() {
        let parsed = parse_ab_kernels(" AUTO ").unwrap();
        assert_eq!(parsed[0].label, "AUTO");
        assert_eq!(parsed[0].selection, KernelSelection::Auto);
    }

    #[test]
    fn parse_ab_kernels_reads_an_accumulator_suffix() {
        let parsed = parse_ab_kernels("auto,auto@f16,a16_coopmat_m64_acc16p").unwrap();
        assert_eq!(parsed[0].acc, CoopmatAcc::F32);
        assert_eq!(parsed[1].label, "auto@f16");
        assert_eq!(parsed[1].selection, KernelSelection::Auto);
        assert_eq!(parsed[1].acc, CoopmatAcc::F16);
        assert_eq!(
            parsed[2].selection,
            KernelSelection::F16wA16CoopmatM64Acc16p
        );
        assert!(parse_ab_kernels("auto@bf16").is_err());
    }

    #[test]
    fn parse_ab_kernels_rejects_empty_and_unknown() {
        for invalid in [
            "",
            "   ",
            ",",
            "auto,",
            ",auto",
            "not_a_kernel",
            "auto,,k64",
        ] {
            assert!(parse_ab_kernels(invalid).is_err(), "accepted {invalid:?}");
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
