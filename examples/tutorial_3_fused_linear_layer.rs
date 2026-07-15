//! Tutorial 3: a neural-network linear layer in ONE GPU dispatch.
//!
//! In PyTorch, `nn.Linear` followed by an activation looks like:
//!
//! ```python
//! y = torch.relu(x @ w + bias)
//! ```
//!
//! Under the hood that is (at least) two GPU kernels: the matmul writes
//! `x @ w + bias` to memory, then the ReLU reads all of it back and
//! writes it again. For layers where the matmul is small, that extra
//! round-trip over memory is a real cost.
//!
//! tensor-ash lets you *fuse* the bias and activation into the matmul's
//! final step ("epilogue"): while each output value is still sitting in
//! a GPU register, the kernel adds the bias, applies the activation,
//! and only then writes to memory. One dispatch, one write.
//!
//! Run it:
//!
//!     cargo run --release --example tutorial_3_fused_linear_layer

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, Executor, MatmulCall, MatmulOp, MatmulPipeline, Tensor,
    VulkanContext,
};

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

    // The layer: 32 input samples, 256 features in, 128 features out.
    //   x: [32, 256]   w: [256, 128]   bias: [128]   y: [32, 128]
    let (m, k, n) = (32u32, 256u32, 128u32);
    let x = Tensor::uninit_device(&ctx, &[m, k])?;
    let w = Tensor::uninit_device(&ctx, &[k, n])?;
    let bias = Tensor::uninit_device(&ctx, &[n])?;
    let y = Tensor::uninit_device(&ctx, &[m, n])?;

    let host_x = random_data((m * k) as usize, 1);
    let host_w = random_data((k * n) as usize, 2);
    let host_bias = random_data(n as usize, 3);
    exec.upload(&host_x, &x)?;
    exec.upload(&host_w, &w)?;
    exec.upload(&host_bias, &bias)?;

    // ------------------------------------------------------------------
    // The fused call.
    //
    // A `MatmulOp` is a `MatmulCall` (what to multiply) plus an
    // `Epilogue` (what to do with each output value before it's
    // written). The epilogue runs in this order:
    //
    //     value = alpha * (A@B)        the matmul itself
    //     value += bias[column]        if `bias` is set
    //     value  = activation(value)   Relu / Silu / Gelu / None
    //     value  = value (op) D        optional second tensor, see below
    //
    // `EpilogueBinary::None` means no second tensor. The other options
    // are `AddScaled { d, beta }` (residual connections: value +=
    // beta * d) and `Mul { d }` (gating, e.g. SwiGLU in transformers).
    //
    // Note: `run_ops` instead of `run_matmuls` — same thing, but takes
    // ops with epilogues.
    // ------------------------------------------------------------------
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &x,
            b: &w,
            c: &y,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: Some(&bias),
            activation: Activation::Relu,
            binary: EpilogueBinary::None,
        },
    )])?;

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&y, &mut got)?;

    // ------------------------------------------------------------------
    // Check against a plain CPU implementation, the way you'd write it
    // on paper: y[i][j] = relu( sum_k x[i][k] * w[k][j] + bias[j] ).
    // ------------------------------------------------------------------
    let mut worst = 0.0f32;
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0.0f64;
            for kk in 0..k as usize {
                acc += host_x[i * k as usize + kk] as f64 * host_w[kk * n as usize + j] as f64;
            }
            acc += host_bias[j] as f64;
            let expected = acc.max(0.0) as f32; // ReLU
            worst = worst.max((got[i * n as usize + j] - expected).abs());
        }
    }
    println!("relu(x @ w + bias) in one dispatch");
    println!("largest difference vs CPU reference: {worst:.2e}");
    assert!(worst < 1e-3, "GPU result diverged from CPU reference");

    // ------------------------------------------------------------------
    // Takeaway
    //
    // The unfused version of this layer would be:
    //   1. matmul kernel      : write 32*128 floats
    //   2. bias+ReLU kernel   : read them all, write them all again
    //
    // The fused version writes each value exactly once. On top of that
    // you saved a whole kernel launch. For big matmuls the arithmetic
    // dominates and fusion matters less; for the small/medium layers
    // that make up most real networks, it's free performance.
    //
    // Available activations: Activation::{None, Relu, Silu, Gelu}
    // (Gelu is the tanh approximation, same as PyTorch's
    // `nn.GELU(approximate="tanh")`).
    // ------------------------------------------------------------------
    println!("ok!");
    Ok(())
}
