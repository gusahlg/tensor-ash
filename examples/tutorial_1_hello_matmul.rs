//! Tutorial 1: your first matmul.
//!
//! If you've used PyTorch, you've written something like:
//!
//! ```python
//! a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
//! b = torch.tensor([[ 7.,  8.], [ 9., 10.], [11., 12.]], device="cuda")
//! c = a @ b
//! print(c.cpu())
//! ```
//!
//! PyTorch hides four separate steps in those lines: picking a GPU,
//! compiling/loading kernels, copying data to the GPU, and copying the
//! result back. tensor-ash makes each step explicit — that's the whole
//! point of the library: nothing happens behind your back, so nothing
//! costs time behind your back either.
//!
//! Run it:
//!
//!     cargo run --release --example tutorial_1_hello_matmul
//!
//! (On NixOS use `nix-shell` / `nix develop` first so the Vulkan loader
//! is on the library path.)

use std::sync::Arc;

use anyhow::Result;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

fn main() -> Result<()> {
    // ------------------------------------------------------------------
    // Step 1: connect to a GPU.
    //
    // `VulkanContext` is roughly `torch.device("cuda")` plus the driver
    // handshake. It finds a Vulkan-capable GPU (preferring a discrete
    // one) and sets up a compute queue. The `false` disables Vulkan's
    // debug validation layers — turn them on while developing if you
    // suspect you're holding the API wrong.
    // ------------------------------------------------------------------
    let ctx = VulkanContext::new(false)?;
    println!("Using: {}", ctx.device_name());

    // ------------------------------------------------------------------
    // Step 2: compile the kernels.
    //
    // `MatmulPipeline` holds every compiled GPU matmul program. Think of
    // it as the cuBLAS handle PyTorch creates behind the scenes. Build
    // it once and share it; the first run on a machine compiles shaders
    // to GPU code (cached on disk, so later runs start fast).
    // ------------------------------------------------------------------
    let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);

    // ------------------------------------------------------------------
    // Step 3: create an executor.
    //
    // The `Executor` submits work to the GPU and waits for it. The two
    // numbers are: how many submissions can be in flight at once (2 is
    // a good default), and the max matmuls per submission.
    // ------------------------------------------------------------------
    let exec = Executor::new(ctx.clone(), pipeline, 2, 16)?;

    // ------------------------------------------------------------------
    // Step 4: allocate GPU tensors.
    //
    // A `Tensor` is a shape plus a block of GPU memory holding f32s in
    // row-major order — the same layout as a NumPy array. Unlike
    // PyTorch there is no `.to("cuda")`: tensors live on the GPU from
    // birth, and `uninit_device` means exactly that — the memory holds
    // garbage until you upload into it (or a matmul writes it).
    //
    // We'll compute C = A @ B with A: 2x3 and B: 3x2, so C is 2x2.
    // ------------------------------------------------------------------
    let a = Tensor::uninit_device(&ctx, &[2, 3])?;
    let b = Tensor::uninit_device(&ctx, &[3, 2])?;
    let c = Tensor::uninit_device(&ctx, &[2, 2])?;

    // ------------------------------------------------------------------
    // Step 5: copy input data to the GPU.
    //
    // Data is passed as a flat row-major slice. This 2x3 matrix:
    //
    //     [[1, 2, 3],
    //      [4, 5, 6]]
    //
    // is uploaded as [1, 2, 3, 4, 5, 6] — first row, then second row.
    // ------------------------------------------------------------------
    exec.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &a)?;
    exec.upload(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &b)?;

    // ------------------------------------------------------------------
    // Step 6: run the matmul.
    //
    // A `MatmulCall` describes one GEMM: C = alpha * A @ B, and if
    // `accumulate` is true, the old contents of C are added in
    // (C += alpha * A @ B). With alpha = 1.0 and accumulate = false
    // this is plain `c = a @ b`.
    //
    // `run_matmuls` blocks until the GPU is done — like calling
    // `torch.cuda.synchronize()` after every op. (Later tutorials show
    // how to batch many matmuls into one submission so you don't pay
    // that round-trip per op.)
    // ------------------------------------------------------------------
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])?;

    // ------------------------------------------------------------------
    // Step 7: copy the result back and look at it.
    // ------------------------------------------------------------------
    let mut result = vec![0.0f32; 4];
    exec.download(&c, &mut result)?;

    println!("C = [[{}, {}],", result[0], result[1]);
    println!("     [{}, {}]]", result[2], result[3]);

    // The same numbers PyTorch would give you:
    //   [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]   [ 58,  64]
    //   [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12] = [139, 154]
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    println!("matches the hand-computed answer — success!");

    Ok(())
}
