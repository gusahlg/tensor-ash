//! A/B kernel bench: `NV_cooperative_matrix2` flash attention vs the
//! SIMT flash kernels, same session, interleaved iterations.
//!
//!     LD_LIBRARY_PATH=... cargo run --release --example bench_flash_cm2

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{Executor, FlashAttentionDesc, MatmulPipeline, Tensor, VulkanContext};

const ITERS: usize = 50;

fn median_us(mut samples: Vec<u64>) -> f64 {
    samples.sort_unstable();
    samples
        .get(samples.len() / 2)
        .map_or(f64::NAN, |&ns| ns as f64 / 1e3)
}

struct Case {
    heads: u32,
    dh: u32,
    t: u32,
    kv_f16: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;
    if !ctx.coopmat2_enabled {
        anyhow::bail!("NV_cooperative_matrix2 not enabled on this device");
    }

    let cases = [
        Case {
            heads: 32,
            dh: 64,
            t: 512,
            kv_f16: false,
        },
        Case {
            heads: 32,
            dh: 64,
            t: 512,
            kv_f16: true,
        },
        Case {
            heads: 32,
            dh: 64,
            t: 2048,
            kv_f16: false,
        },
        Case {
            heads: 32,
            dh: 64,
            t: 2048,
            kv_f16: true,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 512,
            kv_f16: false,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 512,
            kv_f16: true,
        },
    ];

    println!(
        "{:<24} {:>12} {:>12} {:>8}",
        "case", "simt us", "cm2 us", "speedup"
    );

    for case in &cases {
        let (heads, dh, t) = (case.heads, case.dh, case.t);
        let q = Tensor::uninit_device(&ctx, &[heads, t, dh])?;
        let (kt, v) = if case.kv_f16 {
            (
                Tensor::uninit_device_f16(&ctx, &[heads, dh, t])?,
                Tensor::uninit_device_f16(&ctx, &[heads, t, dh])?,
            )
        } else {
            (
                Tensor::uninit_device(&ctx, &[heads, dh, t])?,
                Tensor::uninit_device(&ctx, &[heads, t, dh])?,
            )
        };
        let out = Tensor::uninit_device(&ctx, &[heads, t, dh])?;
        let n = (heads * t * dh) as usize;
        let mut host = vec![0.0f32; n];
        tensor_ash_test_support::fill_det(&mut host, 42);
        exec.upload(&host, &q)?;
        tensor_ash_test_support::fill_det(&mut host, 43);
        exec.upload(&host, &kt)?;
        tensor_ash_test_support::fill_det(&mut host, 44);
        exec.upload(&host, &v)?;
        let desc = FlashAttentionDesc {
            kv_len: t,
            pos_base: 0,
            scale: 1.0 / (dh as f32).sqrt(),
        };

        // Warm the clocks with >= 300 ms of real GPU work before
        // timing (falls back to a fixed iteration count if the device
        // reports no timestamps).
        exec.run_flash_attention(&q, &kt, &v, &out, desc)?;
        let mut burn = 0u64;
        while burn < 300_000_000 {
            burn += exec
                .run_flash_attention_simt(&q, &kt, &v, &out, desc)?
                .gpu_time_ns
                .unwrap_or(10_000_000);
        }

        // Interleave the two kernels so both see the same clock state.
        let (mut simt, mut cm2) = (Vec::new(), Vec::new());
        for _ in 0..ITERS {
            simt.extend(
                exec.run_flash_attention_simt(&q, &kt, &v, &out, desc)?
                    .gpu_time_ns,
            );
            cm2.extend(
                exec.run_flash_attention(&q, &kt, &v, &out, desc)?
                    .gpu_time_ns,
            );
        }
        let (simt, cm2) = (median_us(simt), median_us(cm2));
        println!(
            "{:<24} {:>12.1} {:>12.1} {:>7.2}x",
            format!(
                "H{heads} dh{dh} T{t} kv_{}",
                if case.kv_f16 { "f16" } else { "f32" }
            ),
            simt,
            cm2,
            simt / cm2,
        );
    }
    Ok(())
}
