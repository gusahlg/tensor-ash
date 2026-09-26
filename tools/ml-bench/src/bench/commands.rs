use std::borrow::Cow;
use std::env;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tensor_ash::{
    CoopmatAcc, DeviceKind, Executor, MatmulCall, MatmulOp, Tensor, VulkanContext, f16w_row_tile_n,
    pack_f16w_row_tiles,
};
use tensor_ash_test_support::{cpu_bmm, fill_det, max_abs_err, tolerance};

use super::cases::{BenchCase, BenchResult, SampleSummary, host_len, sweep_cases};
use super::env::{OutputMode, SweepMode, env_u32, env_usize};
use super::report::{BenchReporter, csv_escape};

/// Shared measurement knobs: `ML_ITERS`/`ML_WARMUP` plus a reporter
/// scaled by `ML_PEAK_TFLOPS`, header already printed.
fn reporter_setup(ctx: &Arc<VulkanContext>) -> (BenchReporter<'_>, u32, u32) {
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);
    let mut reporter = BenchReporter::new(OutputMode::from_env(), peak, ctx);
    reporter.print_header();
    (reporter, iters, warmup)
}

pub(super) fn correctness(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 2);
    let m = env_u32("ML_M", 64);
    let n = env_u32("ML_N", 80);
    let k = env_u32("ML_K", 48);
    correctness_impl(ctx, exec, b, m, n, k)
}

fn correctness_impl(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    #[allow(non_snake_case)] B: u32,
    #[allow(non_snake_case)] M: u32,
    #[allow(non_snake_case)] N: u32,
    #[allow(non_snake_case)] K: u32,
) -> Result<()> {
    let a = Tensor::uninit_device(ctx, &[B, M, K])?;
    let b = Tensor::uninit_device(ctx, &[B, K, N])?;
    let c = Tensor::uninit_device(ctx, &[B, M, N])?;

    let mut ha = vec![0.0f32; host_len(&[B, M, K])?];
    let mut hb = vec![0.0f32; host_len(&[B, K, N])?];
    fill_det(&mut ha, 1);
    fill_det(&mut hb, 2);
    exec.upload(&ha, &a)?;
    exec.upload(&hb, &b)?;

    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])?;

    let mut hc = vec![0.0f32; host_len(&[B, M, N])?];
    exec.download(&c, &mut hc)?;

    let cpu = cpu_bmm(&ha, &hb, None, B, M, N, K, 1.0, false);
    let (e, idx) = max_abs_err(&hc, &cpu);
    let tol = tolerance(K);
    log::info!(
        "correctness: max|err|={e:.3e}  tol={tol:.3e}  \
         at idx {idx}: gpu={:.6}  cpu={:.6}",
        hc[idx],
        cpu[idx],
    );
    anyhow::ensure!(e <= tol, "correctness failed: err {e:.3e} > tol {tol:.3e}");
    println!("CORRECTNESS OK (err={e:.3e}, tol={tol:.3e})");
    Ok(())
}

pub(super) fn sweep(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let default_sweep = if ctx.device_kind() == DeviceKind::Cpu {
        SweepMode::Smoke
    } else {
        SweepMode::Standard
    };
    let sweep_mode = SweepMode::from_env(default_sweep);
    let (mut reporter, iters, warmup) = reporter_setup(ctx);

    for case in sweep_cases(sweep_mode) {
        let result = run_case(ctx, exec, case.clone(), iters, warmup)?;
        reporter.print_case(&result);
    }
    Ok(())
}

pub(super) fn cases(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    specs: impl IntoIterator<Item = String>,
) -> Result<()> {
    let cases = specs
        .into_iter()
        .map(|spec| BenchCase::parse(&spec))
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        !cases.is_empty(),
        "cases requires at least one label,b,m,n,k[,f16w][,packed] argument"
    );
    let (mut reporter, iters, warmup) = reporter_setup(ctx);
    for case in cases {
        reporter.print_case(&run_case(ctx, exec, case, iters, warmup)?);
    }
    Ok(())
}

/// Device tensors plus the host fixtures used to fill them and to
/// sample-validate C. Shared by `cases` and `ab` so both paths upload
/// the same deterministic A/B.
pub(super) struct CaseTensors {
    pub(super) a: Tensor,
    pub(super) b: Tensor,
    pub(super) c: Tensor,
    pub(super) h_a: Vec<f32>,
    pub(super) h_b: Vec<f32>,
}

/// Allocate A/B/C for `case`, fill from the same seeds `cases` has
/// always used, and upload through `exec`.
///
/// When `requantize_a_for_tensor_core` is true and the executor's
/// route for this f16w shape is a tensor-core kernel, A is rounded to
/// f16 and re-uploaded so the CPU reference matches in-shader
/// quantization. `ab` passes false and keeps a single f32 A shared
/// across kernels; it rounds a copy at validation time instead.
pub(super) fn prepare_case_tensors(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    case: &BenchCase,
    requantize_a_for_tensor_core: bool,
) -> Result<CaseTensors> {
    let bsz = case.b;
    let m = case.m;
    let n = case.n;
    let k = case.k;
    // `a16`: A and C are f16 storage (the f16-activations route).
    let act = |shape: &[u32]| {
        if case.a16 {
            Tensor::uninit_device_f16(ctx, shape)
        } else {
            Tensor::uninit_device(ctx, shape)
        }
    };
    let a = act(&[bsz, m, k])?;
    let b = if case.b_f16 {
        Tensor::uninit_device_f16(ctx, &[bsz, k, n])?
    } else {
        Tensor::uninit_device(ctx, &[bsz, k, n])?
    };
    let c = act(&[bsz, m, n])?;
    let mut h_a = vec![0.0f32; host_len(&[bsz, m, k])?];
    let mut h_b = vec![0.0f32; host_len(&[bsz, k, n])?];
    fill_det(&mut h_a, 7);
    fill_det(&mut h_b, 11);
    if case.a16 {
        // The device holds halves; the reference must see the same A.
        for value in &mut h_a {
            *value = tensor_ash::dtype::round_f32_via_f16(*value);
        }
    }

    let split_k2 = env_u32("ML_SPLIT_K2", 0);
    if case.packed_b {
        anyhow::ensure!(
            case.b_f16,
            "packed-B cases require f16 weights (the 'packed' flag implies f16w)"
        );
        anyhow::ensure!(
            bsz == 1 && m == 1,
            "packed-B GEMV requires B=1 M=1 (got B={bsz} M={m})"
        );
        anyhow::ensure!(
            n.is_multiple_of(32),
            "packed-B GEMV requires N a multiple of 32 (got {n})"
        );
        anyhow::ensure!(
            split_k2 < 2,
            "packed-B is incompatible with ML_SPLIT_K2={split_k2}"
        );
    }

    if case.b_f16 {
        // Validation compares against exactly what the GPU stores.
        // Round before pack/upload so the packed layout holds the
        // same f16 values the kernel will multiply.
        for value in &mut h_b {
            *value = tensor_ash::dtype::round_f32_via_f16(*value);
        }
    }
    exec.upload(&h_a, &a)?;
    if case.packed_b {
        let tile = f16w_row_tile_n(k, n) as usize;
        let packed = pack_f16w_row_tiles(&h_b, k as usize, n as usize, tile);
        exec.upload(&packed, &b)?;
    } else {
        exec.upload(&h_b, &b)?;
    }
    if case.b_f16 && !case.a16 && requantize_a_for_tensor_core {
        // The tensor-core routes also quantize A (coopmat1 while
        // staging, cm2 in its tensor-load decode callback).
        let kernel = if case.packed_b {
            exec.dispatch_info_packed(n, k).kernel
        } else {
            exec.dispatch_info_for(bsz, m, n, k, true).kernel
        };
        if kernel_quantizes_a(kernel) {
            for value in &mut h_a {
                *value = tensor_ash::dtype::round_f32_via_f16(*value);
            }
            exec.upload(&h_a, &a)?;
        }
    }
    Ok(CaseTensors { a, b, c, h_a, h_b })
}

pub(super) fn kernel_quantizes_a(kernel: &str) -> bool {
    kernel.contains("coopmat") || kernel.contains("cm2")
}

/// CPU-side A for [`validate_samples`]. Tensor-core routes quantize A
/// in-shader; when the shared device tensor was left as f32, round a
/// copy so the reference matches.
pub(super) fn reference_a_for_kernel<'a>(
    h_a: &'a [f32],
    kernel: &str,
    b_f16: bool,
) -> Cow<'a, [f32]> {
    if b_f16 && kernel_quantizes_a(kernel) {
        Cow::Owned(
            h_a.iter()
                .copied()
                .map(tensor_ash::dtype::round_f32_via_f16)
                .collect(),
        )
    } else {
        Cow::Borrowed(h_a)
    }
}

pub(super) fn run_case(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    case: BenchCase,
    iters: u32,
    warmup: u32,
) -> Result<BenchResult> {
    let CaseTensors { a, b, c, h_a, h_b } = prepare_case_tensors(ctx, exec, &case, true)?;
    let bsz = case.b;
    let m = case.m;
    let n = case.n;
    let k = case.k;

    let flops = 2.0f64 * bsz as f64 * m as f64 * n as f64 * k as f64;
    let split_k2 = env_u32("ML_SPLIT_K2", 0);

    let run_once = || {
        let call = MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        };
        if case.packed_b {
            exec.run_ops(&[MatmulOp::new(call).with_packed_b()])
        } else if split_k2 < 2 {
            exec.run_matmuls(std::slice::from_ref(&call))
        } else {
            exec.run_matmuls_split_k2(call, split_k2)
        }
    };

    for _ in 0..warmup {
        run_once()?;
    }

    // Keep GPU and wall measurements paired so host overhead is derived from
    // the same submission rather than from two unrelated best-case samples.
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = Instant::now();
        let stats = run_once()?;
        samples.push((
            t0.elapsed().as_secs_f64() * 1e3,
            stats.gpu_time_ns.map(|ns| ns as f64 / 1e6),
        ));
    }
    let routed = if case.packed_b {
        exec.dispatch_info_packed(n, k).kernel
    } else if case.a16 {
        exec.dispatch_info_a16(bsz, m, n, k).kernel
    } else {
        exec.dispatch_info_for(bsz, m, n, k, case.b_f16).kernel
    };
    validate_samples(
        exec,
        &c,
        &h_a,
        &h_b,
        bsz,
        m,
        n,
        k,
        case.a16,
        CoopmatAcc::of_kernel_name(routed),
    )?;
    let wall_ms = SampleSummary::new(samples.iter().map(|(wall, _)| *wall))
        .context("benchmark produced no wall-time samples")?;
    let gpu_ms = SampleSummary::new(samples.iter().filter_map(|(_, gpu)| *gpu));
    let host_overhead_ms = SampleSummary::new(
        samples
            .iter()
            .filter_map(|(wall, gpu)| gpu.map(|gpu| (wall - gpu).max(0.0))),
    );
    let tflops = gpu_ms
        .filter(|summary| summary.median > 0.0)
        .map_or(f64::NAN, |summary| flops / summary.median * 1e-9);
    let wall_tflops = if wall_ms.median > 0.0 {
        flops / wall_ms.median * 1e-9
    } else {
        f64::NAN
    };
    // Split factors below 2 execute the normal dispatch path, so report it.
    let dispatch = if case.packed_b {
        exec.dispatch_info_packed(n, k)
    } else if case.a16 {
        exec.dispatch_info_a16(bsz, m, n, k)
    } else if split_k2 < 2 {
        exec.dispatch_info_for(bsz, m, n, k, case.b_f16)
    } else {
        tensor_ash::DispatchInfo::split_k2(m, n, split_k2)
    };
    Ok(BenchResult {
        case,
        dispatch,
        flops,
        wall_ms,
        gpu_ms,
        host_overhead_ms,
        tflops,
        wall_tflops,
    })
}

/// Validate a fixed number of well-spaced outputs after timing. This catches
/// broken kernels without turning large benchmark cases into full CPU GEMMs.
/// `c_f16` widens the budget by the f16 store's rounding (2^-11 relative,
/// doubled like the accumulation term).
#[allow(clippy::too_many_arguments)]
/// Extra per-sample error budget of an f16-accumulator coopmat route:
/// one f16 rounding (2^-11 relative) of a running sum of typical size
/// `l2 = sqrt(sum_i (a_i b_i)^2)` per 16-deep MMA, grown linearly in
/// the MMA count so even round-toward-zero accumulation (a biased
/// walk) stays inside it — a smoke gate against wrong-tile / garbage
/// output (errors of order `l2` itself), not a precision spec; the
/// measured error lives in `benchmarks/campaign/f16acc.md`.
pub(super) fn acc16_budget(acc: CoopmatAcc, k: u32, l2: f32) -> f32 {
    match acc {
        CoopmatAcc::F32 => 0.0,
        // Pure f16: K/16 roundings of the full running sum.
        CoopmatAcc::F16 => (k as f32 / 16.0) * 2.0f32.powi(-11) * l2,
        // Promote: 8 roundings per 128-K chunk partial (|chunk| ~
        // l2*sqrt(128/K)), chunks add in f32.
        CoopmatAcc::F16Promote => 8.0 * 2.0f32.powi(-11) * l2 * (k as f32 / 128.0).sqrt(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_samples(
    exec: &Executor,
    c: &Tensor,
    a: &[f32],
    b: &[f32],
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    c_f16: bool,
    acc: CoopmatAcc,
) -> Result<()> {
    const CHECKS: usize = 8;
    let mut output = vec![0.0; host_len(&[batch, m, n])?];
    exec.download(c, &mut output)?;
    let checks = output.len().min(CHECKS);
    let plane = m as usize * n as usize;
    let mut max_error = 0.0f32;
    let mut max_budget = 2.0 * tolerance(k);
    let mut worst = 0;
    for sample in 0..checks {
        let index = if checks == 1 {
            0
        } else {
            sample * (output.len() - 1) / (checks - 1)
        };
        let batch_index = index / plane;
        let within_plane = index % plane;
        let row = within_plane / n as usize;
        let col = within_plane % n as usize;
        let products = (0..k as usize).map(|inner| {
            a[(batch_index * m as usize + row) * k as usize + inner]
                * b[(batch_index * k as usize + inner) * n as usize + col]
        });
        let expected = products.clone().sum::<f32>();
        let l2 = products.map(|p| p * p).sum::<f32>().sqrt();
        // Excess error over this sample's budget; the worst one fails.
        let budget = 2.0 * tolerance(k)
            + if c_f16 {
                expected.abs() * 2.0f32.powi(-10)
            } else {
                0.0
            }
            + acc16_budget(acc, k, l2);
        let error = (output[index] - expected).abs();
        if !error.is_finite() || error - budget > max_error - max_budget {
            max_error = error;
            max_budget = budget;
            worst = index;
        }
    }
    anyhow::ensure!(
        max_error <= max_budget,
        "sampled correctness failed at output {worst}: error {max_error:.3e} > {max_budget:.3e}"
    );
    Ok(())
}

pub(super) fn single(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 1);
    let m = env_u32("ML_M", 4096);
    let n = env_u32("ML_N", 4096);
    let k = env_u32("ML_K", 4096);
    let case = BenchCase {
        b_f16: false,
        packed_b: false,
        a16: false,
        label: Cow::Owned(format!("B={b} M={m} N={n} K={k}")),
        b,
        m,
        n,
        k,
    };
    let (mut reporter, iters, warmup) = reporter_setup(ctx);
    let result = run_case(ctx, exec, case, iters, warmup)?;
    reporter.print_case(&result);
    Ok(())
}

pub(super) fn concurrent(ctx: Arc<VulkanContext>, exec: Arc<Executor>) -> Result<()> {
    let n_threads = env_usize(
        "ML_THREADS",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
    let iters = env_u32("ML_ITERS", 50).max(1);
    let (m, n, k) = (512u32, 512u32, 512u32);

    let triples: Vec<(Tensor, Tensor, Tensor)> = (0..n_threads)
        .map(|t| {
            let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
            let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
            let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
            let mut ha = vec![0.0f32; (m * k) as usize];
            let mut hb = vec![0.0f32; (k * n) as usize];
            fill_det(&mut ha, 100 + t as u64);
            fill_det(&mut hb, 200 + t as u64);
            exec.upload(&ha, &a).unwrap();
            exec.upload(&hb, &b).unwrap();
            (a, b, c)
        })
        .collect();
    let triples = Arc::new(triples);

    let flops_per_call = 2.0f64 * m as f64 * n as f64 * k as f64;
    let total_calls = iters as u64 * n_threads as u64;
    let total_flops = flops_per_call * total_calls as f64;

    let t0 = Instant::now();
    let mut handles = Vec::new();
    for t in 0..n_threads {
        let exec = exec.clone();
        let triples = triples.clone();
        handles.push(std::thread::spawn(move || -> Result<()> {
            let (a, b, c) = &triples[t];
            for _ in 0..iters {
                exec.run_matmuls(&[MatmulCall {
                    a,
                    b,
                    c,
                    alpha: 1.0,
                    accumulate: false,
                }])
                .context("run_matmuls")?;
            }
            Ok(())
        }));
    }
    for h in handles {
        h.join().unwrap()?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let tflops = total_flops / dt * 1e-12;
    println!(
        "concurrent: {n_threads} threads x {iters} iters of 512^3   \
         total {:.1} GFLOPs  in {dt:.3}s  ->  {tflops:.2} TFLOPS (aggregate)",
        total_flops * 1e-9,
    );
    Ok(())
}

pub(super) fn transfer(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let mb = env_usize("ML_TRANSFER_MB", 16).max(1);
    let iters = env_u32("ML_ITERS", 10).max(1);
    let bytes = mb
        .checked_mul(1024 * 1024)
        .context("ML_TRANSFER_MB overflows usize")?;
    let n_f32 = bytes / std::mem::size_of::<f32>();
    anyhow::ensure!(
        n_f32 <= u32::MAX as usize,
        "transfer tensor is too large for current u32 shape storage"
    );

    let tensor = Tensor::uninit_device(ctx, &[n_f32 as u32])?;
    let src = vec![1.0f32; n_f32];
    let mut dst = vec![0.0f32; n_f32];

    exec.upload(&src, &tensor)?;
    exec.download(&tensor, &mut dst)?;

    let mut upload_ms = Vec::with_capacity(iters as usize);
    let mut download_ms = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = Instant::now();
        exec.upload(&src, &tensor)?;
        upload_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        let t0 = Instant::now();
        exec.download(&tensor, &mut dst)?;
        download_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let upload = SampleSummary::new(upload_ms).context("no upload timing samples")?;
    let download = SampleSummary::new(download_ms).context("no download timing samples")?;
    let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let upload_gibs = gib / (upload.median / 1e3);
    let download_gibs = gib / (download.median / 1e3);
    match OutputMode::from_env() {
        OutputMode::Table => {
            println!();
            println!(
                "transfer: {mb} MiB x {} samples  upload={upload_gibs:.2} GiB/s  download={download_gibs:.2} GiB/s",
                upload.count,
            );
            println!(
                "latency ms (min/median/p95): upload={:.3}/{:.3}/{:.3}  download={:.3}/{:.3}/{:.3}",
                upload.min, upload.median, upload.p95, download.min, download.median, download.p95,
            );
        }
        OutputMode::Csv => {
            println!(
                "device,kind,bytes,iters,upload_gibs,download_gibs,sample_count,upload_min_ms,upload_median_ms,upload_p95_ms,download_min_ms,download_median_ms,download_p95_ms"
            );
            println!(
                "{},{},{},{},{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                csv_escape(ctx.device_name()),
                ctx.device_kind().as_str(),
                bytes,
                iters,
                upload_gibs,
                download_gibs,
                upload.count,
                upload.min,
                upload.median,
                upload.p95,
                download.min,
                download.median,
                download.p95,
            );
        }
    }
    Ok(())
}

/// Compare synchronous dispatch against prepared replay and pipelined
/// prepared submission for one repeated shape.  Host overhead dominates
/// the smallest GEMMs, so this reports per-call wall time.
pub(super) fn prepared(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 1);
    let m = env_u32("ML_M", 256);
    let n = env_u32("ML_N", 256);
    let k = env_u32("ML_K", 256);
    let iters = env_u32("ML_ITERS", 500).max(1) as usize;
    let warmup = env_u32("ML_WARMUP", 20) as usize;

    let shape: &[u32] = &[b, m, k];
    let shape_b: &[u32] = &[b, k, n];
    let shape_c: &[u32] = &[b, m, n];
    let a = Tensor::uninit_device(ctx, shape)?;
    let bt = Tensor::uninit_device(ctx, shape_b)?;
    let c_sync = Tensor::uninit_device(ctx, shape_c)?;
    let c_prep = Tensor::uninit_device(ctx, shape_c)?;
    let c_ping = Tensor::uninit_device(ctx, shape_c)?;
    let c_pong = Tensor::uninit_device(ctx, shape_c)?;
    let mut host_a = vec![0.0f32; host_len(shape)?];
    let mut host_b = vec![0.0f32; host_len(shape_b)?];
    fill_det(&mut host_a, 41);
    fill_det(&mut host_b, 42);
    exec.upload(&host_a, &a)?;
    exec.upload(&host_b, &bt)?;

    fn call<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
        MatmulCall {
            a,
            b,
            c,
            alpha: 1.0,
            accumulate: false,
        }
    }
    let per_call_us =
        |elapsed: std::time::Duration, calls: usize| elapsed.as_secs_f64() * 1e6 / calls as f64;

    // Prepare everything up front, then burn in the GPU clocks with the
    // sync path for a fixed wall budget so mode ordering does not hand
    // the later modes warmer clocks than the first.
    let mut prep = exec.prepare_matmuls(&[call(&a, &bt, &c_prep)])?;
    let mut ping = exec.prepare_matmuls(&[call(&a, &bt, &c_ping)])?;
    let mut pong = exec.prepare_matmuls(&[call(&a, &bt, &c_pong)])?;
    let burn_start = Instant::now();
    while burn_start.elapsed() < std::time::Duration::from_millis(300) {
        exec.run_matmuls(&[call(&a, &bt, &c_sync)])?;
    }

    // Mode 1: the regular synchronous path.
    for _ in 0..warmup {
        exec.run_matmuls(&[call(&a, &bt, &c_sync)])?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        exec.run_matmuls(&[call(&a, &bt, &c_sync)])?;
    }
    let sync_us = per_call_us(start.elapsed(), iters);

    // Mode 2: record once, replay synchronously.
    for _ in 0..warmup {
        prep.run()?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        prep.run()?;
    }
    let prepared_us = per_call_us(start.elapsed(), iters);

    // Mode 3: two prepared objects ping-ponging so submission and GPU
    // execution overlap; sustained throughput, not per-call latency.
    // Same total warmup calls as the other modes.
    for _ in 0..warmup.div_ceil(2) {
        ping.run()?;
        pong.run()?;
    }
    let pairs = iters.div_ceil(2);
    let start = Instant::now();
    // SAFETY: ping/pong are waited before this function returns and are
    // never leaked while in flight.
    unsafe {
        ping.submit()?;
        pong.submit()?;
        for _ in 1..pairs {
            ping.wait()?;
            ping.submit()?;
            pong.wait()?;
            pong.submit()?;
        }
    }
    ping.wait()?;
    pong.wait()?;
    let pipelined_us = per_call_us(start.elapsed(), pairs * 2);

    let expected = cpu_bmm(&host_a, &host_b, None, b, m, n, k, 1.0, false);
    for (label, c) in [
        ("sync", &c_sync),
        ("prepared", &c_prep),
        ("ping", &c_ping),
        ("pong", &c_pong),
    ] {
        let mut got = vec![0.0f32; expected.len()];
        exec.download(c, &mut got)?;
        let (error, index) = max_abs_err(&got, &expected);
        anyhow::ensure!(
            error <= 2.0 * tolerance(k),
            "prepared bench '{label}' output error {error:.3e} at {index}"
        );
    }

    let route = exec.dispatch_info(b, m, n, k);
    println!("prepared: B={b} {m}x{n}x{k}, {iters} iters (all outputs validated)");
    match route.split_k2_splits {
        Some(splits) => println!(
            "  note: sync auto-routes split-K2 x{splits}; prepared replays the DP kernel \
             — the ratios include that routing difference"
        ),
        None => println!("  kernel: {}", route.kernel),
    }
    println!("  sync run_matmuls : {sync_us:>8.2} us/call");
    println!(
        "  prepared replay  : {prepared_us:>8.2} us/call  ({:.2}x)",
        sync_us / prepared_us
    );
    println!(
        "  pipelined x2     : {pipelined_us:>8.2} us/call  ({:.2}x)",
        sync_us / pipelined_us
    );
    Ok(())
}

pub(super) fn self_check(ctx: &VulkanContext, n_slots: usize) -> Result<()> {
    println!("tensor-ash self-check");
    println!("status: OK");
    println!("selected: {}", ctx.diagnostics());
    println!("executor_slots: {n_slots}");
    println!("glslc: {}", command_path("glslc"));
    println!("vulkaninfo: {}", command_path("vulkaninfo"));
    println!(
        "LD_LIBRARY_PATH: {}",
        env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".into()),
    );
    println!(
        "VK_ICD_FILENAMES: {}",
        env::var("VK_ICD_FILENAMES").unwrap_or_else(|_| "<unset>".into()),
    );
    if ctx.device_kind() == DeviceKind::Cpu {
        println!(
            "warning: selected device is CPU/software Vulkan; benchmark results are not useful for GPU tuning"
        );
    }
    Ok(())
}

fn command_path(command: &str) -> String {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {command}"))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "<missing>".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_quantizes_a_detects_tensor_core_routes() {
        assert!(kernel_quantizes_a("f16w_coopmat_aligned"));
        assert!(kernel_quantizes_a("f16w_cm2"));
        assert!(!kernel_quantizes_a("k64_bda_v4_tm8_tn4"));
        assert!(!kernel_quantizes_a("f16w_row_bda_k16"));
    }

    #[test]
    fn reference_a_for_kernel_only_rounds_tensor_core_f16w() {
        let a = [0.1f32, 0.2];
        assert!(matches!(
            reference_a_for_kernel(&a, "large_bda_v4", true),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            reference_a_for_kernel(&a, "f16w_coopmat_aligned", false),
            Cow::Borrowed(_)
        ));
        let rounded = reference_a_for_kernel(&a, "f16w_coopmat_aligned", true);
        assert!(matches!(rounded, Cow::Owned(_)));
        assert_ne!(rounded[0], 0.1);
    }
}
