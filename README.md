# tensor-ash

`tensor-ash` is a small Vulkan compute library for high-throughput FP32 matrix
multiplication, written in Rust on top of [`ash`](https://crates.io/crates/ash).
It is designed to sit close to the hardware: explicit GPU buffers, explicit
submission, GPU-timestamp measurements, and a narrow public surface that's easy
to inspect and tune.

The current scope is batched GEMM. The compute backend, kernel selector, and
executor are deliberately built so other kernels can be added without churning
the public API.

## Features

- Vulkan 1.2 compute backend through `ash`, with optional validation layers.
- Device-local `Tensor` buffers with explicit upload / download through a
  staging path.
- Rank-2 and rank-3 row-major FP32 GEMM: `[M, K] @ [K, N]` and batched
  `[B, M, K] @ [B, K, N]`, with batch broadcasting when either side has `B == 1`.
- Batched submission of many independent `MatmulCall`s in a single GPU submit.
- Two GEMM kernel variants (64x64 and 128x128 tiles) with automatic shape-based
  selection, plus a manual override for A/B tuning.
- Thread-safe `Executor` backed by a small pool of submission slots.
- GPU timestamp queries → per-run kernel time and TFLOPS when the device
  supports them.
- Persistent, device-qualified Vulkan pipeline cache under
  `$XDG_CACHE_HOME/tensor-ash/` to amortize pipeline creation across runs.
- Device selection by kind, index, or name substring, with a clear failure when
  a discrete GPU is required but unavailable.

## Requirements

- A Vulkan 1.2 runtime/driver and the Vulkan loader (`libvulkan.so.1` on Linux).
- Rust (2024 edition).
- `glslc` from the Vulkan SDK or `shaderc`. `build.rs` compiles
  `shaders/*.comp` to SPIR-V and embeds the result in the binary.

On NixOS or any nix-based environment, use the included shell, which puts the
Vulkan loader and shader tools on `PATH`:

```bash
nix-shell
cargo run --release --bin ml_bench -- self-check
```

## Library sketch

```rust
use std::sync::Arc;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

let ctx = VulkanContext::new(false)?;
let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
let exec = Executor::new(ctx.clone(), pipeline, /*slots=*/ 2, /*max_calls=*/ 64)?;

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

## Benchmark binary (`ml_bench`)

```bash
cargo run --release --bin ml_bench -- self-check     # device + toolchain report
cargo run --release --bin ml_bench -- correctness    # GPU vs CPU reference
cargo run --release --bin ml_bench -- sweep          # shape sweep, best-of-N
cargo run --release --bin ml_bench -- single         # one configurable GEMM
cargo run --release --bin ml_bench -- concurrent     # N threads × matmuls
cargo run --release --bin ml_bench -- transfer       # upload/download GiB/s
```

Common environment knobs:

```bash
ML_B=4 ML_M=1024 ML_N=1024 ML_K=1024 cargo run --release --bin ml_bench -- single
ML_ITERS=100 ML_WARMUP=10 cargo run --release --bin ml_bench -- sweep
ML_THREADS=8 cargo run --release --bin ml_bench -- concurrent
ML_DEVICE=discrete cargo run --release --bin ml_bench -- self-check
ML_KERNEL=small ML_B=1 ML_M=512 ML_N=512 ML_K=512 cargo run --release --bin ml_bench -- single
ML_OUTPUT=csv ML_SWEEP=smoke cargo run --release --bin ml_bench -- sweep
ML_TRANSFER_MB=64 ML_OUTPUT=csv cargo run --release --bin ml_bench -- transfer
```

- `ML_DEVICE` accepts `auto`, `discrete`, `integrated`, `virtual`, `cpu`,
  `index:N`, `name:TEXT`, or a bare name substring. If `ML_DEVICE=discrete`
  cannot find a discrete GPU, the tool fails clearly instead of silently
  benchmarking software Vulkan.
- `ML_SWEEP` accepts `smoke` (tiny sanity shapes), `standard` (default
  discrete-GPU sweep), and `full` (largest shape set). CPU/software Vulkan
  defaults to `smoke`.
- `ML_KERNEL` accepts `auto`, `large`, or `small`. The default `auto` selector
  picks the 64x64 kernel for small, small-K, or edge-heavy shapes and the
  128x128 kernel for larger aligned shapes. The override is for benchmark
  tuning.
- `ML_OUTPUT` selects `table` (default) or `csv` output.

## Tests

CPU-only unit tests (no GPU required):

```bash
cargo test
```

End-to-end GPU correctness suite (compares every result against an
f64-accumulated CPU reference):

```bash
cargo test --release --test correctness -- --ignored --test-threads=1
```

`--test-threads=1` keeps a single Vulkan instance live at a time, which avoids
flaky driver behavior on some platforms.

## Cross-library benchmarks

A Python harness in `scripts/bench_compare.py` benchmarks `tensor-ash`,
NumPy, PyTorch CPU, optional PyTorch CUDA, and transfer bandwidth. It writes
raw JSON plus a Markdown report:

```bash
nix-shell --run 'python3 scripts/bench_compare.py --case-set extended --iters 5 --warmup 2 --torch-threads 1'
```

Results land in `benchmarks/latest.json` and `benchmarks/latest.md`. These are
most meaningful after `ml_bench self-check` confirms a real GPU was selected;
if `llvmpipe` is picked, treat the `tensor-ash` rows as software-Vulkan
correctness and overhead data, not GPU performance.

A recent run on an RTX 3070 (`benchmarks/latest.md`) had `tensor-ash` fastest
on all 11 shared GEMM cases, with a ~30x geometric-mean throughput ratio
versus single-threaded NumPy and single-threaded PyTorch CPU. PyTorch CUDA was
not present in that environment, so cuBLAS comparison is a separate run.

## Troubleshooting

If the binary fails with `failed to load Vulkan loader: libvulkan.so.1`, the
loader is not visible to the dynamic linker. The Nix shell handles this; doing
it manually looks like:

```bash
export LD_LIBRARY_PATH=/path/to/vulkan-loader/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
```

If `self-check` reports `llvmpipe` as the selected device, results are
software-renderer numbers and are not useful for GPU tuning. Use
`ML_DEVICE=discrete` to require a real GPU, or set `VK_ICD_FILENAMES` to the
desired ICD JSON when the loader is seeing the wrong driver set.

## Project layout

```
src/
  context.rs   Vulkan instance, device, queue, persistent pipeline cache
  buffer.rs    Device/staging buffer wrappers
  tensor.rs    Shape + GPU buffer
  pipeline.rs  Shader modules, descriptor layout, kernel variants
  matmul.rs    MatmulCall, shape validation, RunStats
  executor.rs  Thread-safe executor: upload/download/run_matmuls
  main.rs      `ml_bench` CLI
shaders/
  matmul_f32.comp        128x128-tile GEMM (large kernel)
  matmul_f32_small.comp  64x64-tile GEMM  (small kernel)
scripts/
  bench_compare.py       Cross-library comparison harness
benchmarks/
  latest.{json,md}       Last cross-library run
```

## Vision

A durable Rust-first GPU building block for ML inference: small enough to
inspect, fast enough to sit near hardware limits, and explicit enough that
future kernels, schedulers, and model code can be layered on top without
hiding the important performance controls.
