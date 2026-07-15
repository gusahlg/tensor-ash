//! Tutorial 4: a whole network forward pass in ONE submission.
//!
//! Tutorials 1-3 called the GPU once per operation. Each call blocks
//! until the GPU finishes — fine for learning, wasteful for a real
//! model: a 20-layer network would pay 20+ round-trips of "submit,
//! wait, submit, wait, ...".
//!
//! `run_op_graph` fixes that. You hand it the *whole* list of ops for a
//! forward pass; it records them into a single GPU command stream,
//! figures out which ops depend on which (by watching which tensors
//! each op reads and writes), inserts the minimal synchronization
//! between dependent ops, and submits everything at once. One
//! submission, one wait — conceptually similar to what CUDA Graphs or
//! `torch.compile` do for PyTorch.
//!
//! We'll run a small 2-layer MLP:
//!
//!     hidden = relu(x @ w1 + b1)      # layer 1 (fused, tutorial 3)
//!     logits = hidden @ w2 + b2       # layer 2 (fused bias)
//!
//! Run it:
//!
//!     cargo run --release --example tutorial_4_mlp_in_one_graph

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, Executor, MatmulCall, MatmulOp, MatmulPipeline, Tensor,
    VulkanContext,
};

fn random_data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x as f32 / u32::MAX as f32) - 0.5) * 0.1
        })
        .collect()
}

fn main() -> Result<()> {
    let ctx = VulkanContext::new(false)?;
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;
    println!("Using: {}\n", ctx.device_name());

    // Network dimensions: 64 samples, 512 -> 1024 -> 256.
    let (batch, d_in, d_hidden, d_out) = (64u32, 512u32, 1024u32, 256u32);

    // ------------------------------------------------------------------
    // Weights: upload ONCE, reuse forever.
    //
    // This mirrors real inference: model weights go to the GPU at load
    // time and stay there. Only the input changes between calls. If you
    // find yourself re-uploading weights every forward pass, that's the
    // first thing to fix.
    // ------------------------------------------------------------------
    let w1 = Tensor::uninit_device(&ctx, &[d_in, d_hidden])?;
    let b1 = Tensor::uninit_device(&ctx, &[d_hidden])?;
    let w2 = Tensor::uninit_device(&ctx, &[d_hidden, d_out])?;
    let b2 = Tensor::uninit_device(&ctx, &[d_out])?;
    exec.upload(&random_data((d_in * d_hidden) as usize, 1), &w1)?;
    exec.upload(&random_data(d_hidden as usize, 2), &b1)?;
    exec.upload(&random_data((d_hidden * d_out) as usize, 3), &w2)?;
    exec.upload(&random_data(d_out as usize, 4), &b2)?;

    // Activations: also allocated once and reused across forward passes
    // (like pre-allocated buffers, not like PyTorch's fresh tensor per
    // op — allocation is not free, so reuse pays).
    let x = Tensor::uninit_device(&ctx, &[batch, d_in])?;
    let hidden = Tensor::uninit_device(&ctx, &[batch, d_hidden])?;
    let logits = Tensor::uninit_device(&ctx, &[batch, d_out])?;

    exec.upload(&random_data((batch * d_in) as usize, 5), &x)?;

    // ------------------------------------------------------------------
    // The forward pass as data, not as control flow.
    //
    // Note what we're NOT doing: no manual synchronization, no "wait
    // for layer 1 before layer 2". The graph executor sees that op #2
    // reads `hidden`, which op #1 writes, and inserts the barrier
    // itself. Ops that *don't* depend on each other are left free to
    // overlap on the GPU.
    // ------------------------------------------------------------------
    let forward = |exec: &Executor| -> Result<()> {
        exec.run_op_graph(&[
            // hidden = relu(x @ w1 + b1)
            MatmulOp::with_epilogue(
                MatmulCall {
                    a: &x,
                    b: &w1,
                    c: &hidden,
                    alpha: 1.0,
                    accumulate: false,
                },
                Epilogue {
                    bias: Some(&b1),
                    activation: Activation::Relu,
                    binary: EpilogueBinary::None,
                },
            ),
            // logits = hidden @ w2 + b2   (depends on the op above)
            MatmulOp::with_epilogue(
                MatmulCall {
                    a: &hidden,
                    b: &w2,
                    c: &logits,
                    alpha: 1.0,
                    accumulate: false,
                },
                Epilogue {
                    bias: Some(&b2),
                    activation: Activation::None,
                    binary: EpilogueBinary::None,
                },
            ),
        ])?;
        Ok(())
    };

    // Warm up (first call of each shape does one-time kernel selection).
    for _ in 0..5 {
        forward(&exec)?;
    }

    // Time the whole forward pass, end to end.
    const ITERS: u32 = 100;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        forward(&exec)?;
    }
    let per_pass_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
    println!("2-layer MLP forward ({batch} samples): {per_pass_ms:.3} ms per pass");
    println!("(one submission and one GPU wait per pass, however many layers)\n");

    // ------------------------------------------------------------------
    // Sanity-check the output against a CPU forward pass.
    // ------------------------------------------------------------------
    let mut got = vec![0.0f32; (batch * d_out) as usize];
    exec.download(&logits, &mut got)?;

    // Rebuild the same inputs on the CPU (random_data is deterministic).
    let hx = random_data((batch * d_in) as usize, 5);
    let hw1 = random_data((d_in * d_hidden) as usize, 1);
    let hb1 = random_data(d_hidden as usize, 2);
    let hw2 = random_data((d_hidden * d_out) as usize, 3);
    let hb2 = random_data(d_out as usize, 4);

    let matmul_bias_cpu =
        |a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize, relu: bool| {
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for kk in 0..k {
                        acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
                    }
                    acc += bias[j] as f64;
                    out[i * n + j] = if relu {
                        acc.max(0.0) as f32
                    } else {
                        acc as f32
                    };
                }
            }
            out
        };
    let h_ref = matmul_bias_cpu(
        &hx,
        &hw1,
        &hb1,
        batch as usize,
        d_in as usize,
        d_hidden as usize,
        true,
    );
    let y_ref = matmul_bias_cpu(
        &h_ref,
        &hw2,
        &hb2,
        batch as usize,
        d_hidden as usize,
        d_out as usize,
        false,
    );

    let worst = got
        .iter()
        .zip(&y_ref)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f32, f32::max);
    println!("largest difference vs CPU forward pass: {worst:.2e}");
    assert!(worst < 1e-3, "GPU forward pass diverged from CPU");
    println!("ok!");

    // ------------------------------------------------------------------
    // Where to go from here
    //
    // * Auto-tuning: run your program once with `ML_TUNE=1` set in the
    //   environment. The first time each matmul shape appears,
    //   tensor-ash measures every suitable kernel on your actual GPU
    //   and remembers the fastest one on disk — every later run (with
    //   or without ML_TUNE) uses the measured winner automatically.
    //   You can also pre-warm shapes explicitly with
    //   `exec.tune_shape(batch, m, n, k)`.
    //
    // * Residuals and gating: `EpilogueBinary::AddScaled { d, beta }`
    //   fuses `+ beta * d` (skip connections) and
    //   `EpilogueBinary::Mul { d }` fuses elementwise gating — see
    //   `examples/synth_llama_layer.rs` for a transformer-style layer
    //   using both a graph and a fused SwiGLU.
    //
    // * Everything here is FP32, row-major, and explicit. If a result
    //   looks wrong, print shapes and remember: `uninit_device` memory
    //   is garbage until something writes it.
    // ------------------------------------------------------------------
    Ok(())
}
