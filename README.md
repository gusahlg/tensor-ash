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
- Automatic small/large GEMM shader selection, with a manual override for
  kernel A/B testing.
- GPU timestamp-based runtime stats and TFLOPS reporting when supported.

## Requirements

Install a Vulkan runtime/driver, the Vulkan loader (`libvulkan.so.1` on Linux),
Rust, and `glslc` from the Vulkan SDK or shaderc. The build script compiles
`shaders/*.comp` into SPIR-V and embeds the result in the binary.

On NixOS or in a Nix-based environment, use the included shell:

```bash
nix-shell
cargo run --release --bin ml_bench -- self-check
```

## Benchmark Tool

```bash
cargo run --release --bin ml_bench -- self-check
cargo run --release --bin ml_bench -- correctness
cargo run --release --bin ml_bench -- sweep
cargo run --release --bin ml_bench -- single
cargo run --release --bin ml_bench -- concurrent
cargo run --release --bin ml_bench -- transfer
```

Useful knobs:

```bash
ML_B=4 ML_M=1024 ML_N=1024 ML_K=1024 cargo run --release --bin ml_bench -- single
ML_ITERS=100 ML_WARMUP=10 cargo run --release --bin ml_bench -- sweep
ML_THREADS=8 cargo run --release --bin ml_bench -- concurrent
ML_VALIDATE=1 cargo run --release --bin ml_bench -- correctness
ML_DEVICE=discrete cargo run --release --bin ml_bench -- self-check
ML_OUTPUT=csv ML_SWEEP=smoke cargo run --release --bin ml_bench -- sweep
ML_TRANSFER_MB=64 ML_OUTPUT=csv cargo run --release --bin ml_bench -- transfer
ML_KERNEL=small ML_B=1 ML_M=512 ML_N=512 ML_K=512 cargo run --release --bin ml_bench -- single
```

`ML_DEVICE` accepts `auto`, `discrete`, `integrated`, `virtual`, `cpu`,
`index:N`, `name:TEXT`, or a bare name substring. If `ML_DEVICE=discrete`
cannot find a discrete GPU, the tool fails clearly instead of silently
benchmarking software Vulkan.

`ML_SWEEP=smoke` runs tiny sanity benchmarks, `standard` runs the default
discrete-GPU sweep, and `full` restores the largest shape set. CPU/software
Vulkan defaults to the smoke sweep.

`ML_KERNEL` accepts `auto`, `large`, or `small`. The default `auto` selector
uses the lower-register 64x64 kernel for small, small-K, or edge-heavy shapes
and the 128x128 kernel for larger aligned shapes. The override is mainly for
benchmark tuning.

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

## Troubleshooting

If the benchmark fails with `failed to load Vulkan loader: libvulkan.so.1`,
make the loader visible to the dynamic linker. In the Nix shell this is handled
for you; manually, it looks like:

```bash
export LD_LIBRARY_PATH=/path/to/vulkan-loader/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
```

If `self-check` selects `llvmpipe`, benchmark results are software-renderer
results and are not useful for GPU tuning. Use `ML_DEVICE=discrete` to require a
real GPU, or set `VK_ICD_FILENAMES` to the desired ICD JSON when the Vulkan
loader is seeing the wrong driver set.

## Cross-Library Benchmarks

The comparison harness benchmarks `ml_project`, NumPy, PyTorch CPU, optional
PyTorch CUDA, and transfer bandwidth. It writes raw JSON plus a Markdown report:

```bash
nix-shell --run 'python3 scripts/bench_compare.py --case-set extended --iters 5 --warmup 2 --torch-threads 1'
```

Results are written to `benchmarks/latest.json` and `benchmarks/latest.md`.
These reports are most useful after `ml_bench self-check` sees a real GPU; if it
selects `llvmpipe`, treat the `ml_project` rows as software-Vulkan correctness
and overhead data.

Latest local RTX 3070 run: `ml_project` was fastest on 11/11 shared GEMM cases,
with a 30.2x geometric-mean throughput ratio versus single-threaded NumPy and
30.2x versus single-threaded PyTorch CPU. PyTorch CUDA was not available in the
Nix Python environment, so CUDA/cuBLAS remains a separate run when those
bindings are present.

## Vision

The pitch is a durable Rust-first GPU building block for machine learning:
small enough to inspect, fast enough to sit near hardware limits, and explicit
enough that future kernels, schedulers, and model code can be layered on top
without hiding the important performance controls.
