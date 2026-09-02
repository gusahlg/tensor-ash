//! Causal prefill attention: composed path (Q@Kt -> masked softmax ->
//! P@V) vs the fused flash kernel, per-stage GPU medians.
//!
//!     LD_LIBRARY_PATH=... cargo run --release --example bench_attention

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    Executor, FlashAttentionDesc, MatmulCall, MatmulPipeline, SoftmaxMask, Tensor, VulkanContext,
};

const ITERS: usize = 20;

fn median(mut samples: Vec<u64>) -> f64 {
    samples.sort_unstable();
    samples
        .get(samples.len() / 2)
        .map_or(f64::NAN, |&ns| ns as f64 / 1e6)
}

struct Case {
    heads: u32,
    dh: u32,
    t: u32,
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;

    let cases = [
        Case {
            heads: 32,
            dh: 64,
            t: 512,
        },
        Case {
            heads: 32,
            dh: 64,
            t: 1024,
        },
        Case {
            heads: 32,
            dh: 64,
            t: 2048,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 512,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 1024,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 2048,
        },
        Case {
            heads: 32,
            dh: 128,
            t: 4096,
        },
    ];

    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "case", "qk ms", "sm ms", "pv ms", "sum ms", "flash ms", "speedup"
    );

    for case in &cases {
        let (heads, dh, t) = (case.heads, case.dh, case.t);
        let q = Tensor::uninit_device(&ctx, &[heads, t, dh])?;
        let kt = Tensor::uninit_device(&ctx, &[heads, dh, t])?;
        let v = Tensor::uninit_device(&ctx, &[heads, t, dh])?;
        let scores = Tensor::uninit_device(&ctx, &[heads, t, t])?;
        let out = Tensor::uninit_device(&ctx, &[heads, t, dh])?;
        let n = (heads * t * dh) as usize;
        let mut host = vec![0.0f32; n];
        tensor_ash_test_support::fill_det(&mut host, 42);
        exec.upload(&host, &q)?;
        tensor_ash_test_support::fill_det(&mut host, 43);
        exec.upload(&host, &kt)?;
        tensor_ash_test_support::fill_det(&mut host, 44);
        exec.upload(&host, &v)?;
        let scale = 1.0 / (dh as f32).sqrt();

        // Warm the clocks with real work before timing anything.
        for _ in 0..5 {
            exec.run_matmuls(&[MatmulCall {
                a: &q,
                b: &kt,
                c: &scores,
                alpha: 1.0,
                accumulate: false,
            }])?;
        }

        let (mut qk, mut sm, mut pv) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..ITERS {
            let stats = exec.run_matmuls(&[MatmulCall {
                a: &q,
                b: &kt,
                c: &scores,
                alpha: 1.0,
                accumulate: false,
            }])?;
            qk.extend(stats.gpu_time_ns);
            let stats = exec.run_softmax_rows(
                &scores,
                &scores,
                scale,
                SoftmaxMask::Causal {
                    prefix: 0,
                    rows_per_group: t,
                },
            )?;
            sm.extend(stats.gpu_time_ns);
            let stats = exec.run_matmuls(&[MatmulCall {
                a: &scores,
                b: &v,
                c: &out,
                alpha: 1.0,
                accumulate: false,
            }])?;
            pv.extend(stats.gpu_time_ns);
        }
        let (qk, sm, pv) = (median(qk), median(sm), median(pv));

        // Fused path, if the kernel is available in this build.
        let flash = bench_flash(&exec, &q, &kt, &v, &out, case, scale)?;

        let sum = qk + sm + pv;
        println!(
            "{:<22} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10} {:>10}",
            format!("H{heads} dh{dh} T{t}"),
            qk,
            sm,
            pv,
            sum,
            flash
                .map(|ms| format!("{ms:.3}"))
                .unwrap_or_else(|| "-".into()),
            flash
                .map(|ms| format!("{:.2}x", sum / ms))
                .unwrap_or_else(|| "-".into()),
        );
    }
    Ok(())
}

fn bench_flash(
    exec: &Executor,
    q: &Tensor,
    kt: &Tensor,
    v: &Tensor,
    out: &Tensor,
    case: &Case,
    scale: f32,
) -> Result<Option<f64>> {
    let desc = FlashAttentionDesc {
        kv_len: case.t,
        pos_base: 0,
        scale,
        token_major_heads: None,
        out_token_major_heads: None,
    };
    let mut samples = Vec::new();
    for _ in 0..ITERS {
        let stats = exec.run_flash_attention(q, kt, v, out, desc)?;
        samples.extend(stats.gpu_time_ns);
    }
    if samples.is_empty() {
        return Ok(None);
    }
    Ok(Some(median(samples)))
}
