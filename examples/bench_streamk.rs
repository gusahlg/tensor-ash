//! Stream-K head-to-head benchmark.
//!
//! Runs each shape through:
//!   - `Executor::run_matmuls` (the regular auto-selected kernel,
//!     i.e. the BDA_V4 large kernel for square inputs)
//!   - `Executor::run_matmuls_stream_k_dp_only` (Stream-K's DP-flat
//!     kernel only, no SK-tail, no pre-fill — isolates the DP-flat
//!     kernel's per-tile throughput)
//!   - `Executor::run_matmuls_stream_k` (full hybrid: DP + SK-tail
//!     + pre-fill if needed)
//!
//! The DP-only probe is for diagnosis: it tells us whether the gap
//! between Stream-K and the base path is in kernel quality or in
//! persistent-grid + atomic-add + pre-fill overhead.
//!
//! Samples are interleaved (base, dp, sk repeated per round, 5 rounds
//! of 30 iters each) to absorb GPU noise from another agent that may
//! be running in parallel.

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
const ROUNDS: u32 = 5;
const ITERS_PER_ROUND: u32 = 30;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let ctx = VulkanContext::new_with_device_preference(false, DevicePreference::Auto)?;
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        KernelSelection::Auto,
    )?);
    let exec = Executor::new(ctx.clone(), pipe, 2, 32)?;

    println!(
        "{:<28} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>9}",
        "case", "base_TF", "dp_TF", "sk_TF", "dp/base", "sk/base", "sk/dp", "best_TF", "max_err"
    );
    println!("{}", "-".repeat(100));

    for &(name, m, n, k) in SHAPES {
        let nm = (m as usize) * (k as usize);
        let nb = (k as usize) * (n as usize);
        let nc = (m as usize) * (n as usize);
        let mut h_a = vec![0.0f32; nm];
        let mut h_b = vec![0.0f32; nb];
        for (i, v) in h_a.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32) - 0.5;
        }
        for (i, v) in h_b.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(40503) as f32 / u32::MAX as f32) - 0.5;
        }
        let a = Tensor::uninit_device(&ctx, &[m, k])?;
        let b = Tensor::uninit_device(&ctx, &[k, n])?;
        let c_base = Tensor::uninit_device(&ctx, &[m, n])?;
        let c_dp = Tensor::uninit_device(&ctx, &[m, n])?;
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
        let call_dp = || MatmulCall {
            a: &a,
            b: &b,
            c: &c_dp,
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

        // Probe whether DP-only mode is supported for this shape (it
        // requires 128x128/BK=32 alignment).  If not, just measure
        // base + sk.
        let dp_ok = m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32);

        // Warmup.
        for _ in 0..WARMUP {
            exec.run_matmuls(&[call_base()])?;
            if dp_ok {
                let _ = exec.run_matmuls_stream_k_dp_only(call_dp());
            }
            let _ = exec.run_matmuls_stream_k(call_sk());
        }

        // Interleaved sampling: ROUNDS rounds, each with
        // ITERS_PER_ROUND base, ITERS_PER_ROUND dp, ITERS_PER_ROUND sk,
        // rotating order each round.
        let mut base_ns: u64 = 0;
        let mut dp_ns: u64 = 0;
        let mut sk_ns: u64 = 0;
        let mut total_iters: u32 = 0;
        for _ in 0..ROUNDS {
            for _ in 0..ITERS_PER_ROUND {
                let s = exec.run_matmuls(&[call_base()])?;
                base_ns += s.gpu_time_ns.unwrap_or(0);
                if dp_ok && let Ok(s) = exec.run_matmuls_stream_k_dp_only(call_dp()) {
                    dp_ns += s.gpu_time_ns.unwrap_or(0);
                }
                let s = exec.run_matmuls_stream_k(call_sk())?;
                sk_ns += s.gpu_time_ns.unwrap_or(0);
            }
            total_iters += ITERS_PER_ROUND;
        }

        let base_ms = base_ns as f64 / total_iters as f64 / 1_000_000.0;
        let dp_ms = dp_ns as f64 / total_iters as f64 / 1_000_000.0;
        let sk_ms = sk_ns as f64 / total_iters as f64 / 1_000_000.0;

        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let base_tf = flops / (base_ms * 1e6) / 1e3;
        let dp_tf = if dp_ok && dp_ms > 0.0 {
            flops / (dp_ms * 1e6) / 1e3
        } else {
            0.0
        };
        let sk_tf = flops / (sk_ms * 1e6) / 1e3;

        let dp_speedup = if dp_ok { dp_tf / base_tf } else { 0.0 };
        let sk_speedup = sk_tf / base_tf;
        let sk_vs_dp = if dp_ok && dp_tf > 0.0 {
            sk_tf / dp_tf
        } else {
            0.0
        };

        let best_tf = base_tf.max(dp_tf).max(sk_tf);

        // Correctness check vs base.
        let mut out_base = vec![0.0f32; nc];
        let mut out_sk = vec![0.0f32; nc];
        exec.download(&c_base, &mut out_base)?;
        exec.download(&c_sk, &mut out_sk)?;
        let max_err_sk = out_base
            .iter()
            .zip(out_sk.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_err_dp = if dp_ok {
            let mut out_dp = vec![0.0f32; nc];
            exec.download(&c_dp, &mut out_dp)?;
            out_base
                .iter()
                .zip(out_dp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        let max_err = max_err_sk.max(max_err_dp);

        let dp_disp = if dp_ok {
            format!("{:>7.3}", dp_tf)
        } else {
            "    n/a".to_string()
        };
        let dp_sp = if dp_ok {
            format!("{:>6.3}x", dp_speedup)
        } else {
            "    n/a".to_string()
        };
        let sk_dp_sp = if dp_ok && sk_vs_dp > 0.0 {
            format!("{:>6.3}x", sk_vs_dp)
        } else {
            "    n/a".to_string()
        };
        println!(
            "{:<28} {:>7.3} {} {:>7.3} {} {:>6.3}x {} {:>8.3} {:>9.2e}",
            name, base_tf, dp_disp, sk_tf, dp_sp, sk_speedup, sk_dp_sp, best_tf, max_err
        );
        let _ = Instant::now();
    }

    println!("{}", "-".repeat(100));
    Ok(())
}
