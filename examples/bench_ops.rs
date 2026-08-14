//! GPU timings for the non-GEMM model ops at model-ish sizes.
//!
//!     LD_LIBRARY_PATH=... cargo run --release --example bench_ops

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    CopyDesc, Executor, MatmulPipeline, RopeDesc, SoftmaxMask, Tensor, VulkanContext,
};

fn median_gpu_ms(mut run: impl FnMut() -> Result<Option<u64>>, iters: u32) -> Result<f64> {
    let mut samples = Vec::new();
    for _ in 0..iters {
        if let Some(ns) = run()? {
            samples.push(ns);
        }
    }
    samples.sort_unstable();
    Ok(samples
        .get(samples.len() / 2)
        .map_or(f64::NAN, |&ns| ns as f64 / 1e6))
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;

    let report = |label: &str, bytes: u64, ms: f64| {
        let gbps = bytes as f64 / (ms * 1e6);
        println!("{label:<28} {ms:>8.4} ms   {gbps:>7.1} GB/s");
    };

    // Burn the clocks in with real work first.
    let (rows, cols) = (4096_u32, 4096_u32);
    let x = Tensor::uninit_device(&ctx, &[rows, cols])?;
    let y = Tensor::uninit_device(&ctx, &[rows, cols])?;
    let w = Tensor::uninit_device(&ctx, &[cols])?;
    let b = Tensor::uninit_device(&ctx, &[cols])?;
    exec.upload(&vec![0.5; (rows * cols) as usize], &x)?;
    exec.upload(&vec![1.0; cols as usize], &w)?;
    exec.upload(&vec![0.1; cols as usize], &b)?;
    for _ in 0..40 {
        exec.run_rms_norm(&x, &w, &y, 1e-5)?;
    }

    let rw = (rows as u64 * cols as u64) * 4 * 2; // one read + one write
    let ms = median_gpu_ms(
        || {
            Ok(exec
                .run_softmax_rows(&x, &y, 1.0, SoftmaxMask::Full)?
                .gpu_time_ns)
        },
        30,
    )?;
    report("softmax 4096x4096", rw, ms);
    let ms = median_gpu_ms(|| Ok(exec.run_rms_norm(&x, &w, &y, 1e-5)?.gpu_time_ns), 30)?;
    report("rmsnorm 4096x4096", rw, ms);
    let ms = median_gpu_ms(
        || Ok(exec.run_layer_norm(&x, &w, &b, &y, 1e-5)?.gpu_time_ns),
        30,
    )?;
    report("layernorm 4096x4096", rw, ms);

    // RoPE at llama-8B decode prefill-ish size: T=512, H=32, dh=128.
    let (tokens, heads, dh) = (512_u32, 32_u32, 128_u32);
    let act = Tensor::uninit_device(&ctx, &[tokens, heads * dh])?;
    let table = Tensor::uninit_device(&ctx, &[8192, dh / 2, 2])?;
    exec.upload(&vec![0.5; (tokens * heads * dh) as usize], &act)?;
    exec.upload(&vec![0.7; (8192 * dh) as usize], &table)?;
    let desc = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim: dh,
        pos_base: 0,
        ..Default::default()
    };
    let bytes = (tokens as u64 * heads as u64 * dh as u64) * 4 * 2;
    let ms = median_gpu_ms(
        || Ok(exec.run_rope(&act, &table, &act, desc)?.gpu_time_ns),
        30,
    )?;
    report("rope 512x32x128 in-place", bytes, ms);

    // Strided transpose 4096x4096 (worst-case scatter on one side).
    let ms = median_gpu_ms(
        || {
            Ok(exec
                .run_copy_strided(
                    &x,
                    &y,
                    CopyDesc {
                        extent: [rows, cols, 1],
                        src_offset: 0,
                        src_strides: [1, cols, 0],
                        dst_offset: 0,
                        dst_strides: [rows, 1, 0],
                        ..Default::default()
                    },
                )?
                .gpu_time_ns)
        },
        30,
    )?;
    report("transpose 4096x4096", rw, ms);
    Ok(())
}
