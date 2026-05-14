# ml_project

`ml_project` is a lightweight Vulkan compute layer for high-throughput FP32
matrix multiplication. The immediate goal is to make GPU-backed model
experiments feel simple from Rust while keeping low-level control over memory,
submission, timing, and batching.

## Core Features

- Vulkan 1.2 compute backend through `ash`.
- Device-local `Tensor` buffers with explicit upload/download.
- Rank-2 and rank-3 row-major FP32 GEMM: `[M, K] @ [K, N]` and batched
  `[B, M, K] @ [B, K, N]`.
- Batch broadcasting from `A` and/or `B` when their batch dimension is `1`.
- Batched submission of many independent `MatmulCall`s in one GPU submit.
- Thread-safe `Executor` with a small submission-slot pool.
- GPU timestamp-based runtime stats and TFLOPS reporting when supported.

## Requirements

Install a Vulkan runtime/driver, the Vulkan loader (`libvulkan.so.1` on Linux),
Rust, and `glslc` from the Vulkan SDK or shaderc. The build script compiles
`shaders/*.comp` into SPIR-V and embeds the result in the binary.

## Benchmark Tool

```bash
cargo run --release --bin ml_bench -- correctness
cargo run --release --bin ml_bench -- sweep
cargo run --release --bin ml_bench -- single
cargo run --release --bin ml_bench -- concurrent
```

Useful knobs:

```bash
ML_B=4 ML_M=1024 ML_N=1024 ML_K=1024 cargo run --release --bin ml_bench -- single
ML_ITERS=100 ML_WARMUP=10 cargo run --release --bin ml_bench -- sweep
ML_THREADS=8 cargo run --release --bin ml_bench -- concurrent
ML_VALIDATE=1 cargo run --release --bin ml_bench -- correctness
```

## Library Sketch

```rust
use std::sync::Arc;
use ml_project::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

let ctx = VulkanContext::new(false)?;
let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
let exec = Executor::new(ctx.clone(), pipeline, 2, 64)?;

let a = Tensor::zeros_device(&ctx, &[8, 256, 256])?;
let b = Tensor::zeros_device(&ctx, &[8, 256, 256])?;
let c = Tensor::zeros_device(&ctx, &[8, 256, 256])?;

exec.upload(&host_a, &a)?;
exec.upload(&host_b, &b)?;

let stats = exec.run_matmuls(&[MatmulCall {
    a: &a,
    b: &b,
    c: &c,
    alpha: 1.0,
    accumulate: false,
}])?;

exec.download(&c, &mut host_c)?;
println!("{:?}", stats.tflops());
```

Run CPU-only checks with:

```bash
cargo test
```

Run the GPU correctness suite on a Vulkan-capable machine with:

```bash
cargo test --release --test correctness -- --ignored --test-threads=1
```

## Vision

The pitch is a durable Rust-first GPU building block for machine learning:
small enough to inspect, fast enough to sit near hardware limits, and explicit
enough that future kernels, schedulers, and model code can be layered on top
without hiding the important performance controls.
