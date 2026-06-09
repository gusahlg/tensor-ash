//! `tensor-ash` — a minimal Vulkan compute library focused on
//! high-throughput batched matrix multiplication for AI workloads.
//!
//! The public surface is intentionally narrow:
//!
//!   * [`VulkanContext`] — instance + device + single compute queue.
//!   * [`MatmulPipeline`] — the compiled GEMM kernel (one shader module
//!     + descriptor layout + pipeline).
//!   * [`Tensor`] — shape + GPU buffer (rank 2 or 3, contiguous, f32).
//!   * [`Executor`] — submits work to the GPU; thread-safe; reports
//!     per-run GPU time via timestamp queries.
//!
//! ```no_run
//! use std::sync::Arc;
//! use tensor_ash::{VulkanContext, MatmulPipeline, Executor, Tensor, MatmulCall};
//!
//! # fn try_main() -> anyhow::Result<()> {
//! let ctx       = VulkanContext::new(false)?;
//! let pipeline  = Arc::new(MatmulPipeline::new(&ctx)?);
//! let exec      = Executor::new(ctx.clone(), pipeline, /*n_slots=*/2, /*max_calls=*/64)?;
//!
//! let a = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
//! let b = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
//! let c = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
//!
//! exec.upload(&vec![1.0f32; 8*256*256], &a)?;
//! exec.upload(&vec![1.0f32; 8*256*256], &b)?;
//! let stats = exec.run_matmuls(&[MatmulCall {
//!     a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false,
//! }])?;
//! println!("{:?}", stats.tflops());
//! # Ok(()) }
//! ```

pub mod buffer;
pub mod context;
pub mod executor;
pub mod matmul;
pub mod pipeline;
pub mod tensor;
pub mod testing;

pub use buffer::{Buffer, BufferLocation};
pub use context::{DeviceKind, DevicePreference, DeviceSummary, VulkanContext};
pub use executor::Executor;
pub use matmul::{MatmulCall, MatrixShape, RunStats};
pub use pipeline::{
    KernelSelection, KernelVariant, MatmulKernel, MatmulPipeline, MatmulPushConstants,
    SMALL_TILE_M, SMALL_TILE_N, TILE_M, TILE_N,
};
pub use tensor::Tensor;
