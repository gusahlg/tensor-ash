//! Probe: DP vs two-stage split-K on deep-K shapes.
//!
//! Prints min GPU ms per strategy so the reduction paths can be
//! compared on the local device.
//!
//!   cargo run --release --example bench_splitk2

use anyhow::Result;
use std::sync::Arc;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

const ITERS: usize = 10;
const WARMUP: usize = 3;

fn min_gpu_ms(mut run: impl FnMut() -> Result<Option<u64>>) -> Result<Option<f64>> {
    for _ in 0..WARMUP {
        run()?;
    }
    let mut best: Option<u64> = None;
    for _ in 0..ITERS {
        if let Some(ns) = run()? {
            best = Some(best.map_or(ns, |b| b.min(ns)));
        }
    }
    Ok(best.map(|ns| ns as f64 / 1e6))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let ctx = VulkanContext::new(false)?;
    let pipe = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipe, 2, 8)?;
    println!("{}", ctx.diagnostics());

    let shapes: &[(u32, u32, u32)] = &[
        (64, 64, 8192),
        (128, 128, 8192),
        (256, 256, 4096),
        (256, 256, 8192),
        (512, 512, 4096),
        (128, 128, 2048),
        (256, 256, 1024),
        (512, 512, 2048),
    ];

    println!(
        "{:<20} {:>9} {:>22} {:>22}",
        "shape (MxNxK)", "dp_ms", "splitk2_ms(best)", "splitk2 splits"
    );
    for &(m, n, k) in shapes {
        let a = Tensor::uninit_device(&ctx, &[m, k])?;
        let b = Tensor::uninit_device(&ctx, &[k, n])?;
        let c = Tensor::uninit_device(&ctx, &[m, n])?;
        let ha = vec![0.5f32; (m * k) as usize];
        let hb = vec![0.25f32; (k * n) as usize];
        exec.upload(&ha, &a)?;
        exec.upload(&hb, &b)?;
        let call = || MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        };

        let dp = min_gpu_ms(|| Ok(exec.run_matmuls(&[call()])?.gpu_time_ns))?;

        let mut best2: Option<(f64, u32)> = None;
        for splits in [4u32, 8, 16, 32, 64] {
            if k / splits < 64 {
                continue;
            }
            if let Some(ms) =
                min_gpu_ms(|| Ok(exec.run_matmuls_split_k2(call(), splits)?.gpu_time_ns))?
                && best2.is_none_or(|(b, _)| ms < b)
            {
                best2 = Some((ms, splits));
            }
        }

        println!(
            "{:<20} {:>9} {:>22} {:>22}",
            format!("{m}x{n}x{k}"),
            dp.map_or("-".into(), |v| format!("{v:.3}")),
            best2.map_or("-".into(), |(v, _)| format!("{v:.3}")),
            best2.map_or("-".into(), |(_, s)| s.to_string()),
        );
    }
    Ok(())
}
