//! Synthetic transformer-layer matmul workload.
//!
//! Chains the FP32 matmuls of one Llama-style decoder layer (fused QKV
//! projection, attention output projection, FFN gate/up/down) at several
//! batch sizes M, comparing two execution strategies:
//!
//!   * **split** — two dependency-ordered `run_matmuls` submissions
//!     per layer (the only correct option before v1.3): the second
//!     submission fence-waits on the first, and the SwiGLU
//!     nonlinearity is skipped entirely (the library had no elementwise
//!     ops).
//!   * **graph+fused** — one `run_op_graph` submission covering the
//!     whole layer, with the SwiGLU (`silu(gate) * up`) fused into the
//!     gate GEMM's epilogue.  One fence wait per layer, hazards
//!     barriered automatically, and the activation costs no extra pass.
//!
//! This is not a real model: weights are random, no attention softmax,
//! no norms, no residuals.  It's the *matmul shape* of one decoder
//! layer.
//!
//! Defaults to TinyLlama-1.1B dimensions because FP32 weights for
//! Llama-7B (~810 MB per layer) crowd the 8 GB RTX 3070 once we add
//! scratch.  Override with `MODEL=7b`.  Pick M with `--m=N` or a
//! comma-list `--m=1,16,128`.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tensor_ash::{
    Activation, DevicePreference, Epilogue, EpilogueBinary, Executor, KernelSelection, MatmulCall,
    MatmulOp, MatmulPipeline, Tensor, VulkanContext,
};

#[derive(Clone, Copy, Debug)]
struct LlamaCfg {
    name: &'static str,
    hidden: u32,
    intermediate: u32,
}

const TINY: LlamaCfg = LlamaCfg {
    name: "TinyLlama-1.1B",
    hidden: 2048,
    intermediate: 5632,
};
const L7B: LlamaCfg = LlamaCfg {
    name: "Llama-7B",
    hidden: 4096,
    intermediate: 11008,
};

const WARMUP: u32 = 3;
const ITERS: u32 = 20;

fn rng_buf(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn pick_m_list() -> Vec<u32> {
    for arg in std::env::args().skip(1) {
        if let Some(rest) = arg.strip_prefix("--m=") {
            return rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
    }
    vec![1, 16, 128, 256]
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let cfg = match std::env::var("MODEL").as_deref() {
        Ok("7b") | Ok("7B") => L7B,
        _ => TINY,
    };
    let m_list = pick_m_list();

    let ctx = VulkanContext::new_with_device_preference(false, DevicePreference::Auto)?;
    let pipe = Arc::new(MatmulPipeline::new_with_kernel_selection(
        &ctx,
        KernelSelection::Auto,
    )?);
    let exec = Executor::new(ctx.clone(), pipe, 2, 32)?;

    let h = cfg.hidden;
    let i_ = cfg.intermediate;

    println!("Model: {} (hidden={}, intermediate={})", cfg.name, h, i_);
    println!("{}", ctx.diagnostics());
    println!();

    // Per-layer weights (alloc once, reuse across M sweep).
    let w_qkv = Tensor::uninit_device(&ctx, &[h, 3 * h])?;
    let w_o = Tensor::uninit_device(&ctx, &[h, h])?;
    let w_gate = Tensor::uninit_device(&ctx, &[h, i_])?;
    let w_up = Tensor::uninit_device(&ctx, &[h, i_])?;
    let w_down = Tensor::uninit_device(&ctx, &[i_, h])?;
    exec.upload(&rng_buf((h * 3 * h) as usize, 1), &w_qkv)?;
    exec.upload(&rng_buf((h * h) as usize, 2), &w_o)?;
    exec.upload(&rng_buf((h * i_) as usize, 3), &w_gate)?;
    exec.upload(&rng_buf((h * i_) as usize, 4), &w_up)?;
    exec.upload(&rng_buf((i_ * h) as usize, 5), &w_down)?;

    println!(
        "{:<6} {:>13} {:>13} {:>10} {:>13} {:>13} {:>8}",
        "M", "split_ms", "graph_ms", "speedup", "split_tok/s", "graph_tok/s", "layer_TF"
    );
    println!("{}", "-".repeat(82));

    for &m in &m_list {
        // Per-M activations.
        let x = Tensor::uninit_device(&ctx, &[m, h])?;
        let qkv = Tensor::uninit_device(&ctx, &[m, 3 * h])?;
        let attn_out = Tensor::uninit_device(&ctx, &[m, h])?;
        let up = Tensor::uninit_device(&ctx, &[m, i_])?;
        let gated = Tensor::uninit_device(&ctx, &[m, i_])?;
        let down = Tensor::uninit_device(&ctx, &[m, h])?;
        exec.upload(&rng_buf((m * h) as usize, 6), &x)?;

        fn call<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
            MatmulCall {
                a,
                b,
                c,
                alpha: 1.0,
                accumulate: false,
            }
        }

        // --- Strategy A: dependency-ordered separate submissions. ---
        // qkv/o/gate/up are independent; down depends on the FFN
        // output, so it needs its own submission (fence between).
        // SwiGLU is NOT computed (no elementwise capability pre-v1.3),
        // so `down` consumes the raw `up` — strictly *less* work than
        // strategy B.
        let run_split = || -> Result<()> {
            exec.run_matmuls(&[
                call(&x, &w_qkv, &qkv),
                call(&x, &w_o, &attn_out),
                call(&x, &w_gate, &gated),
                call(&x, &w_up, &up),
            ])?;
            exec.run_matmuls(&[call(&up, &w_down, &down)])?;
            Ok(())
        };

        // --- Strategy B: whole layer in one graph submission with the
        // SwiGLU fused into the gate GEMM's epilogue. ---
        let run_graph = || -> Result<()> {
            exec.run_op_graph(&[
                MatmulOp::new(call(&x, &w_qkv, &qkv)),
                MatmulOp::new(call(&x, &w_o, &attn_out)),
                MatmulOp::new(call(&x, &w_up, &up)),
                MatmulOp::with_epilogue(
                    call(&x, &w_gate, &gated),
                    Epilogue {
                        bias: None,
                        activation: Activation::Silu,
                        binary: EpilogueBinary::Mul { d: &up },
                    },
                ),
                MatmulOp::new(call(&gated, &w_down, &down)),
            ])?;
            Ok(())
        };

        for _ in 0..WARMUP {
            run_split()?;
            run_graph()?;
        }

        let t0 = Instant::now();
        for _ in 0..ITERS {
            run_split()?;
        }
        let split_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let t1 = Instant::now();
        for _ in 0..ITERS {
            run_graph()?;
        }
        let graph_ms = t1.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let flops_layer = 2.0 * m as f64 * h as f64 * (3 * h) as f64          // qkv
            + 2.0 * m as f64 * h as f64 * h as f64                            // o
            + 2.0 * m as f64 * h as f64 * i_ as f64 * 2.0                     // gate + up
            + 2.0 * m as f64 * i_ as f64 * h as f64; // down
        let layer_tf = flops_layer / (graph_ms / 1000.0) / 1e12;

        println!(
            "{:<6} {:>13.3} {:>13.3} {:>9.2}x {:>13.0} {:>13.0} {:>8.2}",
            m,
            split_ms,
            graph_ms,
            split_ms / graph_ms,
            m as f64 / (split_ms / 1000.0),
            m as f64 / (graph_ms / 1000.0),
            layer_tf,
        );
    }

    println!();
    println!(
        "(split = 2 dependency-ordered submissions per layer, no activation. \
         graph = 1 submission, hazards auto-barriered, SwiGLU fused into the \
         gate GEMM. Wall-clock per layer, includes host dispatch.)"
    );

    Ok(())
}
