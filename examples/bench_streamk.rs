//! Stream-K head-to-head benchmark.
//!
//! Runs each shape through both `Executor::run_matmuls` (the regular
//! auto-selected kernel) and `Executor::run_matmuls_stream_k`, then
//! prints a comparison table.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tensor_ash::{
    DevicePreference, Executor, KernelSelection, MatmulCall, MatmulPipeline, Tensor, VulkanContext,
};

const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("sq_512", 512, 512, 512),
    ("sq_1024", 1024, 1024, 1024),
    ("sq_2048", 2048, 2048, 2048),
    ("sq_4096", 4096, 4096, 4096),
    ("sq_4096_k1024", 4096, 4096, 1024),
    ("attn_2048x4096x512", 2048, 4096, 512),
    ("attn_4096x2048x512", 4096, 2048, 512),
    ("attn_proj_2048x512x512", 2048, 512, 512),
    ("attn_proj_512x2048x512", 512, 2048, 512),
    ("attn_qkv_1024x3072x512", 1024, 3072, 512),
    ("attn_qkv_2048x6144x1024", 2048, 6144, 1024),
    ("ffn_proj_2048x8192x2048", 2048, 8192, 2048),
    ("medium_768", 768, 768, 768),
    ("deep_k_1024x1024x8192", 1024, 1024, 8192),
];

const WARMUP: u32 = 5;
const ITERS: u32 = 30;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let ctx = VulkanContext::new_with_device_preference(false, DevicePreference::Auto)?;
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        KernelSelection::Auto,
    )?);
    let exec = Executor::new(ctx.clone(), pipe, 2, 32)?;

    println!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "case", "base_ms", "base_TF", "sk_ms", "sk_TF", "speedup", "max_err"
    );
    println!("{}", "-".repeat(86));

    let mut wins = 0;
    let mut losses = 0;
    let mut tied = 0;

    for &(name, m, n, k) in SHAPES {
        let nm = (m as usize) * (k as usize);
        let nb = (k as usize) * (n as usize);
        let nc = (m as usize) * (n as usize);
        let mut h_a = vec![0.0f32; nm];
        let mut h_b = vec![0.0f32; nb];
        // Simple deterministic fill.
        for (i, v) in h_a.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32) - 0.5;
        }
        for (i, v) in h_b.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(40503) as f32 / u32::MAX as f32) - 0.5;
        }
        let a = Tensor::uninit_device(&ctx, &[m, k])?;
        let b = Tensor::uninit_device(&ctx, &[k, n])?;
        let c_base = Tensor::uninit_device(&ctx, &[m, n])?;
        let c_sk = Tensor::uninit_device(&ctx, &[m, n])?;
        exec.upload(&h_a, &a)?;
        exec.upload(&h_b, &b)?;

        let call_base = || MatmulCall {
            a: &a,
            b: &b,
            c: &c_base,
            alpha: 1.0,
            accumulate: false,
        };
        let call_sk = || MatmulCall {
            a: &a,
            b: &b,
            c: &c_sk,
            alpha: 1.0,
            accumulate: false,
        };

        // Warmup both paths.
        for _ in 0..WARMUP {
            exec.run_matmuls(&[call_base()])?;
            exec.run_matmuls_stream_k(call_sk())?;
        }

        // Time baseline.
        let t0 = Instant::now();
        let mut base_gpu_total_ns: u64 = 0;
        for _ in 0..ITERS {
            let stats = exec.run_matmuls(&[call_base()])?;
            base_gpu_total_ns += stats.gpu_time_ns.unwrap_or(0);
        }
        let base_wall_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
        let base_gpu_ms = base_gpu_total_ns as f64 / ITERS as f64 / 1_000_000.0;

        // Time Stream-K.
        let t1 = Instant::now();
        let mut sk_gpu_total_ns: u64 = 0;
        for _ in 0..ITERS {
            let stats = exec.run_matmuls_stream_k(call_sk())?;
            sk_gpu_total_ns += stats.gpu_time_ns.unwrap_or(0);
        }
        let sk_wall_ms = t1.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
        let sk_gpu_ms = sk_gpu_total_ns as f64 / ITERS as f64 / 1_000_000.0;

        // FLOPs.
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let base_tf = flops / (base_gpu_ms * 1e6) / 1e3;
        let sk_tf = flops / (sk_gpu_ms * 1e6) / 1e3;
        let speedup = base_gpu_ms / sk_gpu_ms;

        // Correctness diff.
        let mut out_base = vec![0.0f32; nc];
        let mut out_sk = vec![0.0f32; nc];
        exec.download(&c_base, &mut out_base)?;
        exec.download(&c_sk, &mut out_sk)?;
        let max_err = out_base
            .iter()
            .zip(out_sk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let tag = if speedup > 1.02 {
            wins += 1;
            "WIN"
        } else if speedup < 0.98 {
            losses += 1;
            "LOSS"
        } else {
            tied += 1;
            ""
        };
        println!(
            "{:<28} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>7.3}x {:>9.2e} {}",
            name, base_gpu_ms, base_tf, sk_gpu_ms, sk_tf, speedup, max_err, tag
        );
        // Suppress wall-clock unused warnings.
        let _ = (base_wall_ms, sk_wall_ms);
    }

    println!("{}", "-".repeat(86));
    println!("wins={wins}  losses={losses}  unchanged={tied}");
    Ok(())
}
