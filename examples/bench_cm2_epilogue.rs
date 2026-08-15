//! Fused-epilogue CM2 GEMM vs the composed coopmat1 + separate-Binary
//! baseline, interleaved in one session with a clock burn-in first.
//!
//!     LD_LIBRARY_PATH=... cargo run --release --example bench_cm2_epilogue
//!
//! Runs the two epilogue families the T >= 256 llama MLP un-demote
//! deletes Binary passes for, on the TinyLlama prefill shapes:
//! - Silu+Mul (gate projection, 512x5632x2048): fused epilogue op
//!   vs plain matmul + `BinaryOp::SiluMul` pass.
//! - AddScaled (down projection + residual, 512x2048x5632): fused
//!   epilogue op vs plain matmul + `BinaryOp::AddScaled` pass.
//!
//! Auto routing is left in place: the fused op rides `f16w_cm2` via
//! `epilogue_fallback_index`, the plain matmul keeps the measured
//! coopmat1 winner.  Both variants run as a single `run_exec_ops`
//! submission, so the GPU timestamp covers the whole chain.

use anyhow::Result;
use std::sync::Arc;
use tensor_ash::{
    Activation, BinaryOp, Epilogue, EpilogueBinary, ExecOp, Executor, MatmulCall, MatmulOp,
    MatmulPipeline, Tensor, VulkanContext,
};

fn fill_det(values: &mut [f32], seed: u64) {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for value in values.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *value = ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 64)?;
    if !ctx.coopmat2_enabled {
        // Deliberate under ML_NO_COOPMAT2=1: the fused ops demote to
        // the SIMT epilogue family (the pre-cm2 baseline).
        eprintln!("coopmat2 off: fused ops ride the SIMT demote");
    }

    let iters = 30;
    let upload = |shape: &[u32], seed: u64, f16: bool| -> Result<Tensor> {
        let t = if f16 {
            Tensor::uninit_device_f16(&ctx, shape)?
        } else {
            Tensor::uninit_device(&ctx, shape)?
        };
        let mut host = vec![0.0f32; Tensor::numel(shape) as usize];
        fill_det(&mut host, seed);
        for v in &mut host {
            *v *= 0.05; // keep silu/products tame
        }
        exec.upload(&host, &t)?;
        Ok(t)
    };

    // (label, m, n, k, family) — family: 0 = SiluMul, 1 = AddScaled.
    let cases: &[(&str, u32, u32, u32, u8)] = &[
        ("gate silu_mul 512x5632x2048", 512, 5632, 2048, 0),
        ("down add_scaled 512x2048x5632", 512, 2048, 5632, 1),
    ];

    // Burn the clocks in (~300 ms of real tensor-core work).
    {
        let a = upload(&[512, 2048], 1, false)?;
        let b = upload(&[2048, 5632], 2, true)?;
        let c = Tensor::uninit_device(&ctx, &[512, 5632])?;
        for _ in 0..60 {
            exec.run_matmuls(&[MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            }])?;
        }
    }

    for &(label, m, n, k, family) in cases {
        let a = upload(&[m, k], 10, false)?;
        let b = upload(&[k, n], 11, true)?;
        let d = upload(&[m, n], 12, false)?;
        let c = Tensor::uninit_device(&ctx, &[m, n])?;
        let plain = MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        };
        let epilogue = if family == 0 {
            Epilogue {
                activation: Activation::Silu,
                binary: EpilogueBinary::Mul { d: &d },
                ..Epilogue::NONE
            }
        } else {
            Epilogue {
                binary: EpilogueBinary::AddScaled { d: &d, beta: 1.0 },
                ..Epilogue::NONE
            }
        };
        let binary = if family == 0 {
            BinaryOp::SiluMul
        } else {
            BinaryOp::AddScaled { beta: 1.0 }
        };
        let fused_ops = [ExecOp::Matmul(MatmulOp::with_epilogue(plain, epilogue))];
        // Composed order matches llama's T >= 256 branch: plain GEMM
        // into C, then the standalone binary pass combining C with D.
        let composed_ops = [
            ExecOp::Matmul(MatmulOp::new(plain)),
            ExecOp::Binary {
                a: &c,
                b: &d,
                out: &c,
                op: binary,
            },
        ];

        let mut fused = Vec::new();
        let mut composed = Vec::new();
        for _ in 0..3 {
            exec.run_exec_ops(&fused_ops)?;
            exec.run_exec_ops(&composed_ops)?;
        }
        for _ in 0..iters {
            if let Some(ns) = exec.run_exec_ops(&fused_ops)?.gpu_time_ns {
                fused.push(ns as f64 / 1e6);
            }
            if let Some(ns) = exec.run_exec_ops(&composed_ops)?.gpu_time_ns {
                composed.push(ns as f64 / 1e6);
            }
        }
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let fused_ms = median(fused);
        let composed_ms = median(composed);
        println!(
            "{label:<32} fused {fused_ms:>7.4} ms ({:>6.2} TF/s)   composed {composed_ms:>7.4} ms ({:>6.2} TF/s)   fused/composed {:>5.3}",
            flops / (fused_ms * 1e-3) / 1e12,
            flops / (composed_ms * 1e-3) / 1e12,
            fused_ms / composed_ms,
        );
    }
    Ok(())
}
