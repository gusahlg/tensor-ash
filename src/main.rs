//! Benchmark binary for `ml_project`.
//!
//! Reports GPU-time TFLOPS (via vkCmdWriteTimestamp) — separating actual
//! kernel time from CPU-side submission overhead.
//!
//! Subcommands:
//!   correctness   Quick correctness sanity check vs CPU reference.
//!   sweep         Throughput sweep across representative shapes.
//!   single        One configurable run: B,M,N,K from ML_B/M/N/K env vars.
//!   concurrent    Many parallel run_matmuls from N host threads.
//!
//! Without a subcommand, runs `correctness` then `sweep`.
//!
//! Env knobs:
//!   ML_B,ML_M,ML_N,ML_K  shape for `single`
//!   ML_ITERS             timing iterations (default 20)
//!   ML_WARMUP            warm-up iterations (default 3)
//!   ML_SLOTS             executor slot count (default 2)
//!   ML_THREADS           threads for `concurrent` (default = cpus)
//!   ML_VALIDATE          set 1 to enable Vulkan validation
//!
//! Run:
//!   cargo run --release --bin ml_bench -- sweep

use std::env;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use ml_project::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

// ---------- env helpers ----------
fn env_u32(k: &str, d: u32) -> u32 { env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn env_usize(k: &str, d: usize) -> usize { env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn env_bool(k: &str) -> bool { env::var(k).ok().is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) }

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cmd = env::args().nth(1).unwrap_or_else(|| "all".into());
    let validate = env_bool("ML_VALIDATE");
    let n_slots  = env_usize("ML_SLOTS", 2);

    let ctx  = VulkanContext::new(validate)?;
    let pipe = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipe, n_slots, /*max_calls=*/256)?;

    log::info!(
        "device: compute_family={} slots={} timestamps={}",
        ctx.compute_family, n_slots, ctx.timestamps_supported,
    );

    match cmd.as_str() {
        "correctness"        => correctness(&ctx, &exec)?,
        "sweep"              => sweep(&ctx, &exec)?,
        "single"             => single(&ctx, &exec)?,
        "concurrent"         => concurrent(ctx.clone(), Arc::new(exec))?,
        "all" | _            => { correctness(&ctx, &exec)?; sweep(&ctx, &exec)?; }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Correctness — small, fast, prints max abs err.
// ---------------------------------------------------------------------------
fn correctness(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    const B: u32 = 2; const M: u32 = 64; const N: u32 = 80; const K: u32 = 48;

    let a = Tensor::zeros_device(ctx, &[B, M, K])?;
    let b = Tensor::zeros_device(ctx, &[B, K, N])?;
    let c = Tensor::zeros_device(ctx, &[B, M, N])?;

    let mut ha = vec![0.0f32; (B * M * K) as usize];
    let mut hb = vec![0.0f32; (B * K * N) as usize];
    fill_det(&mut ha, 1);
    fill_det(&mut hb, 2);
    exec.upload(&ha, &a)?;
    exec.upload(&hb, &b)?;

    exec.run_matmuls(&[MatmulCall { a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false }])?;

    let mut hc = vec![0.0f32; (B * M * N) as usize];
    exec.download(&c, &mut hc)?;

    let cpu = cpu_bmm(&ha, &hb, B, M, N, K);
    let (e, idx) = max_abs_err(&hc, &cpu);
    let tol = 8.0 * (K as f32) * f32::EPSILON;
    log::info!(
        "correctness: max|err|={e:.3e}  tol={tol:.3e}  \
         at idx {idx}: gpu={:.6}  cpu={:.6}",
        hc[idx], cpu[idx],
    );
    anyhow::ensure!(e <= tol, "correctness failed: err {e:.3e} > tol {tol:.3e}");
    println!("CORRECTNESS OK (err={e:.3e}, tol={tol:.3e})");
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Sweep — representative shapes.  Reports best-of-N GPU TFLOPS and the
//    wall-clock vs GPU-time delta (= CPU submit overhead).
// ---------------------------------------------------------------------------
fn sweep(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let iters  = env_u32("ML_ITERS", 20);
    let warmup = env_u32("ML_WARMUP", 3);

    println!();
    println!(
        "{:<28} {:>14} {:>10} {:>10} {:>9} {:>9}",
        "shape", "FLOPs", "wall(ms)", "gpu(ms)", "TF/s", "%peak",
    );
    println!("{}", "-".repeat(80));

    let cases: &[(&str, u32, u32, u32, u32)] = &[
        ("square 512^3   B=1",   1, 512, 512, 512),
        ("square 1024^3  B=1",   1, 1024, 1024, 1024),
        ("square 2048^3  B=1",   1, 2048, 2048, 2048),
        ("square 4096^3  B=1",   1, 4096, 4096, 4096),
        ("batched 32×1024^2",   32, 1024, 1024, 1024),
        ("batched 8×2048^2",     8, 2048, 2048, 2048),
        ("tall   8192×1024×1024",1, 8192, 1024, 1024),
        ("wide   1024×8192×1024",1, 1024, 8192, 1024),
        ("thin K 4096×4096×512", 1, 4096, 4096,  512),
        ("fat  K 1024×1024×8192",1, 1024, 1024, 8192),
        ("odd   1023×1025×1027", 1, 1023, 1025, 1027),
    ];

    // Theoretical peak — RTX 3070: 5888 CUDA cores * 2 FMA * 1.725 GHz ≈ 20.3 TFLOPS FP32.
    // Allow override via env so other GPUs report something sensible.
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok().and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.3);

    for &(label, bsz, m, n, k) in cases {
        run_case(ctx, exec, label, bsz, m, n, k, iters, warmup, peak)?;
    }
    Ok(())
}

fn run_case(
    ctx: &Arc<VulkanContext>, exec: &Executor,
    label: &str, bsz: u32, m: u32, n: u32, k: u32,
    iters: u32, warmup: u32, peak_tflops: f64,
) -> Result<()> {
    let a = Tensor::zeros_device(ctx, &[bsz, m, k])?;
    let b = Tensor::zeros_device(ctx, &[bsz, k, n])?;
    let c = Tensor::zeros_device(ctx, &[bsz, m, n])?;
    // Upload non-zero (peak FP32 throughput is unaffected by data values,
    // but doing so guards against denormal pathologies).
    let mut h_a = vec![0.0f32; (bsz * m * k) as usize];
    let mut h_b = vec![0.0f32; (bsz * k * n) as usize];
    fill_det(&mut h_a, 7);
    fill_det(&mut h_b, 11);
    exec.upload(&h_a, &a)?;
    exec.upload(&h_b, &b)?;

    let flops = 2.0f64 * bsz as f64 * m as f64 * n as f64 * k as f64;

    for _ in 0..warmup {
        exec.run_matmuls(&[MatmulCall { a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false }])?;
    }

    let mut best_gpu_ns = u64::MAX;
    let mut best_wall_ns = u128::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        let stats = exec.run_matmuls(&[MatmulCall {
            a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false,
        }])?;
        let wall_ns = t0.elapsed().as_nanos();
        best_wall_ns = best_wall_ns.min(wall_ns);
        if let Some(gpu_ns) = stats.gpu_time_ns {
            best_gpu_ns = best_gpu_ns.min(gpu_ns);
        }
    }
    let gpu_ms  = if best_gpu_ns != u64::MAX { best_gpu_ns as f64 / 1e6 } else { f64::NAN };
    let wall_ms = best_wall_ns as f64 / 1e6;
    let tflops  = if best_gpu_ns != u64::MAX { flops / best_gpu_ns as f64 * 1e-3 } else { f64::NAN };
    let pct     = tflops / peak_tflops * 100.0;
    println!(
        "{:<28} {:>14.3} {:>10.3} {:>10.3} {:>9.2} {:>8.1}%",
        label, flops / 1e9, wall_ms, gpu_ms, tflops, pct,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Single — one configurable matmul.
// ---------------------------------------------------------------------------
fn single(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let b = env_u32("ML_B", 1);
    let m = env_u32("ML_M", 4096);
    let n = env_u32("ML_N", 4096);
    let k = env_u32("ML_K", 4096);
    let iters  = env_u32("ML_ITERS", 20);
    let warmup = env_u32("ML_WARMUP", 3);
    let peak = env::var("ML_PEAK_TFLOPS")
        .ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.3);
    let label = format!("B={b} M={m} N={n} K={k}");
    println!();
    println!(
        "{:<28} {:>14} {:>10} {:>10} {:>9} {:>9}",
        "shape", "FLOPs", "wall(ms)", "gpu(ms)", "TF/s", "%peak",
    );
    println!("{}", "-".repeat(80));
    run_case(ctx, exec, &label, b, m, n, k, iters, warmup, peak)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Concurrent — many threads issuing run_matmuls in parallel.
// ---------------------------------------------------------------------------
fn concurrent(ctx: Arc<VulkanContext>, exec: Arc<Executor>) -> Result<()> {
    let n_threads = env_usize("ML_THREADS",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let iters     = env_u32("ML_ITERS", 50);
    let (m, n, k) = (512u32, 512u32, 512u32);

    // Each thread owns its triple of tensors.
    let triples: Vec<(Tensor, Tensor, Tensor)> = (0..n_threads).map(|t| {
        let a = Tensor::zeros_device(&ctx, &[m, k]).unwrap();
        let b = Tensor::zeros_device(&ctx, &[k, n]).unwrap();
        let c = Tensor::zeros_device(&ctx, &[m, n]).unwrap();
        let mut ha = vec![0.0f32; (m*k) as usize];
        let mut hb = vec![0.0f32; (k*n) as usize];
        fill_det(&mut ha, 100 + t as u64);
        fill_det(&mut hb, 200 + t as u64);
        exec.upload(&ha, &a).unwrap();
        exec.upload(&hb, &b).unwrap();
        (a, b, c)
    }).collect();
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
                    a, b, c, alpha: 1.0, accumulate: false,
                }]).context("run_matmuls")?;
            }
            Ok(())
        }));
    }
    for h in handles { h.join().unwrap()?; }
    let dt = t0.elapsed().as_secs_f64();
    let tflops = total_flops / dt * 1e-12;
    println!(
        "concurrent: {n_threads} threads × {iters} iters of 512^3   \
         total {:.1} GFLOPs  in {dt:.3}s  →  {tflops:.2} TFLOPS (aggregate)",
        total_flops * 1e-9,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fill_det(out: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for v in out.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        *v = (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
    }
}

fn cpu_bmm(a: &[f32], b: &[f32], bsz: u32, m: u32, n: u32, k: u32) -> Vec<f32> {
    let m_ = m as usize; let n_ = n as usize; let k_ = k as usize;
    let mk = m_ * k_; let kn = k_ * n_; let mn = m_ * n_;
    let mut out = vec![0.0f32; bsz as usize * mn];
    for bi in 0..bsz as usize {
        let a = &a[bi*mk..(bi+1)*mk];
        let b = &b[bi*kn..(bi+1)*kn];
        let c = &mut out[bi*mn..(bi+1)*mn];
        for i in 0..m_ {
            for j in 0..n_ {
                let mut acc = 0.0f64;
                for kk in 0..k_ {
                    acc += (a[i*k_+kk] as f64) * (b[kk*n_+j] as f64);
                }
                c[i*n_+j] = acc as f32;
            }
        }
    }
    out
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32; let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > m { m = e; idx = i; }
    }
    (m, idx)
}
