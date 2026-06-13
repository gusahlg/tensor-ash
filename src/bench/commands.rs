use std::borrow::Cow;
use std::env;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tensor_ash::{
    DeviceKind, Executor, MatmulCall, Tensor, VulkanContext,
    testing::{cpu_bmm, fill_det, max_abs_err},
};

use super::cases::{BenchCase, BenchResult, host_len, sweep_cases};
use super::env::{OutputMode, SweepMode, env_u32, env_usize};
use super::report::{BenchReporter, csv_escape};

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

    let mut ha = vec![0.0f32; (B * M * K) as usize];
    let mut hb = vec![0.0f32; (B * K * N) as usize];
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

    let mut hc = vec![0.0f32; (B * M * N) as usize];
    exec.download(&c, &mut hc)?;

    let cpu = cpu_bmm(&ha, &hb, None, B, M, N, K, 1.0, false);
    let (e, idx) = max_abs_err(&hc, &cpu);
    let tol = 8.0 * (K as f32) * f32::EPSILON;
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
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let default_sweep = if ctx.device_kind() == DeviceKind::Cpu {
        SweepMode::Smoke
    } else {
        SweepMode::Standard
    };
    let sweep_mode = SweepMode::from_env(default_sweep);
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);
    let mut reporter = BenchReporter::new(OutputMode::from_env(), peak, ctx);
    reporter.print_header();

    for case in sweep_cases(sweep_mode) {
        let result = run_case(ctx, exec, case.clone(), iters, warmup)?;
        reporter.print_case(&result);
    }
    Ok(())
}

pub(super) fn run_case(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    case: BenchCase,
    iters: u32,
    warmup: u32,
) -> Result<BenchResult> {
    let bsz = case.b;
    let m = case.m;
    let n = case.n;
    let k = case.k;
    let a = Tensor::uninit_device(ctx, &[bsz, m, k])?;
    let b = Tensor::uninit_device(ctx, &[bsz, k, n])?;
    let c = Tensor::uninit_device(ctx, &[bsz, m, n])?;
    let mut h_a = vec![0.0f32; host_len(&[bsz, m, k])?];
    let mut h_b = vec![0.0f32; host_len(&[bsz, k, n])?];
    fill_det(&mut h_a, 7);
    fill_det(&mut h_b, 11);
    exec.upload(&h_a, &a)?;
    exec.upload(&h_b, &b)?;

    let flops = 2.0f64 * bsz as f64 * m as f64 * n as f64 * k as f64;

    for _ in 0..warmup {
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])?;
    }

    let mut best_gpu_ns = u64::MAX;
    let mut best_wall_ns = u128::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        let stats = exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])?;
        let wall_ns = t0.elapsed().as_nanos();
        best_wall_ns = best_wall_ns.min(wall_ns);
        if let Some(gpu_ns) = stats.gpu_time_ns {
            best_gpu_ns = best_gpu_ns.min(gpu_ns);
        }
    }
    let gpu_ms = if best_gpu_ns != u64::MAX {
        best_gpu_ns as f64 / 1e6
    } else {
        f64::NAN
    };
    let wall_ms = best_wall_ns as f64 / 1e6;
    let tflops = if best_gpu_ns != u64::MAX {
        flops / best_gpu_ns as f64 * 1e-3
    } else {
        f64::NAN
    };
    Ok(BenchResult {
        case,
        flops,
        wall_ms,
        gpu_ms,
        tflops,
    })
}

pub(super) fn single(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 1);
    let m = env_u32("ML_M", 4096);
    let n = env_u32("ML_N", 4096);
    let k = env_u32("ML_K", 4096);
    let iters = env_u32("ML_ITERS", 20).max(1);
    let warmup = env_u32("ML_WARMUP", 3);
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);
    let case = BenchCase {
        label: Cow::Owned(format!("B={b} M={m} N={n} K={k}")),
        b,
        m,
        n,
        k,
    };
    let mut reporter = BenchReporter::new(OutputMode::from_env(), peak, ctx);
    reporter.print_header();
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

    let mut best_upload = f64::INFINITY;
    let mut best_download = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        exec.upload(&src, &tensor)?;
        best_upload = best_upload.min(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        exec.download(&tensor, &mut dst)?;
        best_download = best_download.min(t0.elapsed().as_secs_f64());
    }

    let gib = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
    let upload_gibs = gib / best_upload;
    let download_gibs = gib / best_download;
    match OutputMode::from_env() {
        OutputMode::Table => {
            println!();
            println!(
                "transfer: {mb} MiB x {iters} iters  upload={upload_gibs:.2} GiB/s  download={download_gibs:.2} GiB/s"
            );
        }
        OutputMode::Csv => {
            println!("device,kind,bytes,iters,upload_gibs,download_gibs");
            println!(
                "{},{},{},{},{:.6},{:.6}",
                csv_escape(ctx.device_name()),
                ctx.device_kind().as_str(),
                bytes,
                iters,
                upload_gibs,
                download_gibs,
            );
        }
    }
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
