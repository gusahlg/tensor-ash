//! Tutorial 2: batched matmuls, broadcasting, and honest timing.
//!
//! Builds on tutorial 1. Three new ideas:
//!
//!   1. Rank-3 tensors give you batched matmul, exactly like
//!      `torch.matmul` on `[B, M, K] @ [B, K, N]` inputs.
//!   2. A batch dimension of 1 broadcasts — one weight matrix shared
//!      across the whole batch, like PyTorch's broadcasting rules.
//!   3. `RunStats` tells you how long the GPU *actually* worked, which
//!      is not the same thing as how long your function call took.
//!
//! Run it:
//!
//!     cargo run --release --example tutorial_2_batching_and_timing

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

/// Deterministic pseudo-random values in [-0.5, 0.5] — cheap stand-in
/// for `torch.randn`, good enough for exercising a kernel.
fn random_data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn main() -> Result<()> {
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;
    println!("Using: {}\n", ctx.device_name());

    // ------------------------------------------------------------------
    // Part 1: batched matmul with a broadcast weight.
    //
    // PyTorch equivalent:
    //     x = torch.randn(8, 64, 128)   # 8 independent inputs
    //     w = torch.randn(128, 32)      # ONE weight matrix
    //     y = x @ w                     # w broadcasts -> [8, 64, 32]
    //
    // In tensor-ash the broadcast is expressed by shapes alone: A is
    // rank-3 with batch 8, B is rank-2 (batch 1), so B is reused for
    // every batch element. The output C must be rank-3 with batch 8.
    // ------------------------------------------------------------------
    let (batch, m, k, n) = (8u32, 64u32, 128u32, 32u32);
    let x = Tensor::uninit_device(&ctx, &[batch, m, k])?;
    let w = Tensor::uninit_device(&ctx, &[k, n])?;
    let y = Tensor::uninit_device(&ctx, &[batch, m, n])?;

    exec.upload(&random_data((batch * m * k) as usize, 1), &x)?;
    exec.upload(&random_data((k * n) as usize, 2), &w)?;

    let stats = exec.run_matmuls(&[MatmulCall {
        a: &x,
        b: &w,
        c: &y,
        alpha: 1.0,
        accumulate: false,
    }])?;
    println!(
        "batched broadcast matmul [{batch}x{m}x{k}] @ [{k}x{n}]: \
         GPU time {:.3} ms",
        stats.gpu_time_ns.unwrap_or(0) as f64 / 1e6
    );

    // ------------------------------------------------------------------
    // Part 2: `accumulate` — adding into an existing result.
    //
    // PyTorch equivalent: y += 0.5 * (x @ w2)
    //
    // Instead of computing into a temporary and launching a separate
    // add, set `accumulate: true` and the GEMM adds into C in the same
    // kernel. (alpha plays the role of the scalar multiplier.)
    // ------------------------------------------------------------------
    let w2 = Tensor::uninit_device(&ctx, &[k, n])?;
    exec.upload(&random_data((k * n) as usize, 3), &w2)?;

    exec.run_matmuls(&[MatmulCall {
        a: &x,
        b: &w2,
        c: &y, // y already holds x @ w; this adds 0.5 * x @ w2 on top
        alpha: 0.5,
        accumulate: true,
    }])?;
    println!("accumulated 0.5 * (x @ w2) into the previous result\n");

    // ------------------------------------------------------------------
    // Part 3: timing — wall clock vs GPU clock.
    //
    // When people benchmark PyTorch naively they time the Python call,
    // which mixes three very different costs:
    //   * host overhead   (building the command, talking to the driver)
    //   * GPU compute     (the actual matmul)
    //   * synchronization (waiting for the GPU to finish)
    //
    // tensor-ash reports GPU compute separately: `RunStats.gpu_time_ns`
    // comes from timestamps the GPU itself records around your work.
    // The difference between wall time and GPU time is your per-call
    // overhead — and the reason tutorials 3 and 4 batch work together.
    // ------------------------------------------------------------------
    let (m, n, k) = (1024u32, 1024u32, 1024u32);
    let a = Tensor::uninit_device(&ctx, &[m, k])?;
    let b = Tensor::uninit_device(&ctx, &[k, n])?;
    let c = Tensor::uninit_device(&ctx, &[m, n])?;
    exec.upload(&random_data((m * k) as usize, 4), &a)?;
    exec.upload(&random_data((k * n) as usize, 5), &b)?;

    let call = MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    };

    // Warm up first: the very first dispatch of a shape pays one-time
    // costs (kernel selection, GPU clocks ramping up from idle).
    for _ in 0..5 {
        exec.run_matmuls(&[call])?;
    }

    let t0 = Instant::now();
    let stats = exec.run_matmuls(&[call])?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = stats.gpu_time_ns.unwrap_or(0) as f64 / 1e6;

    println!("1024^3 matmul:");
    println!("  wall clock : {wall_ms:.3} ms  (what your program waited)");
    println!("  GPU time   : {gpu_ms:.3} ms  (what the GPU computed)");
    println!(
        "  overhead   : {:.3} ms  (submission + sync)",
        wall_ms - gpu_ms
    );
    if let Some(tflops) = stats.tflops() {
        println!("  throughput : {tflops:.2} TFLOP/s");
    }

    Ok(())
}
