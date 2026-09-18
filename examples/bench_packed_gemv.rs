//! Packed vs unpacked f16w row-GEMV on TinyLlama decode shapes.
//!
//!     cargo run --release --example bench_packed_gemv

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    DType, Executor, MatmulCall, MatmulOp, MatmulPipeline, Tensor, VulkanContext, f16w_row_tile_n,
    pack_f16w_row_tiles,
};
use tensor_ash_test_support::fill_det;

fn median_ns(mut run: impl FnMut() -> Result<Option<u64>>, iters: u32) -> Result<u64> {
    let mut samples = Vec::new();
    for _ in 0..iters {
        if let Some(ns) = run()? {
            samples.push(ns);
        }
    }
    samples.sort_unstable();
    Ok(*samples.get(samples.len() / 2).unwrap_or(&0))
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 8)?;

    // Clock burn-in: ~300 ms of real GEMM.
    let burn_a = Tensor::uninit_device(&ctx, &[512, 2048])?;
    let burn_b = Tensor::uninit_device_f16(&ctx, &[2048, 2048])?;
    let burn_c = Tensor::uninit_device(&ctx, &[512, 2048])?;
    let zeros = vec![0.0f32; 512 * 2048];
    exec.upload(&zeros, &burn_a)?;
    exec.upload(&zeros, &burn_b)?;
    let mut burned = 0u64;
    while burned < 300_000_000 {
        let stats = exec.run_matmuls(&[MatmulCall {
            a: &burn_a,
            b: &burn_b,
            c: &burn_c,
            alpha: 1.0,
            accumulate: false,
        }])?;
        burned += stats.gpu_time_ns.unwrap_or(0);
    }

    let shapes = [
        ("k/v  1x256x2048", 256_u32, 2048_u32),
        ("q/o  1x2048x2048", 2048, 2048),
        ("up   1x5632x2048", 5632, 2048),
        ("down 1x2048x5632", 2048, 5632),
        ("lm   1x32000x2048", 32000, 2048),
    ];
    println!(
        "{:<22} {:>10} {:>10} {:>8}",
        "shape", "unpacked", "packed", "ratio"
    );
    for (label, n, k) in shapes {
        let mut host_a = vec![0.0; k as usize];
        let mut host_b = vec![0.0; (k * n) as usize];
        fill_det(&mut host_a, 1);
        fill_det(&mut host_b, 2);
        let a = Tensor::uninit_device(&ctx, &[1, k])?;
        let b = Tensor::uninit_device_f16(&ctx, &[k, n])?;
        let b_p = Tensor::uninit_device_f16(&ctx, &[k, n])?;
        let c = Tensor::uninit_device(&ctx, &[1, n])?;
        let c_p = Tensor::uninit_device(&ctx, &[1, n])?;
        exec.upload(&host_a, &a)?;
        exec.upload(&host_b, &b)?;
        let tile = f16w_row_tile_n(k, n) as usize;
        let packed = pack_f16w_row_tiles(&host_b, k as usize, n as usize, tile);
        exec.upload(&packed, &b_p)?;
        assert_eq!(b.dtype(), DType::F16);

        let unpack_op = MatmulOp::new(MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        });
        let pack_op = MatmulOp::new(MatmulCall {
            a: &a,
            b: &b_p,
            c: &c_p,
            alpha: 1.0,
            accumulate: false,
        })
        .with_packed_b();
        for _ in 0..4 {
            exec.run_ops(&[unpack_op])?;
            exec.run_ops(&[pack_op])?;
        }
        let unpack = median_ns(|| Ok(exec.run_ops(&[unpack_op])?.gpu_time_ns), 20)?;
        let pack = median_ns(|| Ok(exec.run_ops(&[pack_op])?.gpu_time_ns), 20)?;
        println!(
            "{label:<22} {:>8.1} us {:>8.1} us {:>7.3}",
            unpack as f64 / 1e3,
            pack as f64 / 1e3,
            pack as f64 / unpack as f64
        );
    }
    Ok(())
}
