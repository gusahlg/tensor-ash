# tensor-ash

Version: `1.0.0`

`tensor-ash` is a small Vulkan compute library for high-throughput FP32 matrix
multiplication, written in Rust on top of [`ash`](https://crates.io/crates/ash).
It is designed to sit close to the hardware: explicit GPU buffers, explicit
submission, GPU-timestamp measurements, and a narrow public surface that's easy
to inspect and tune.

The current scope is batched GEMM. The compute backend, kernel selector, and
executor are deliberately built so other kernels can be added without churning
the public API.

## Scope and API status

`v1.0.0` is the stable baseline for the Rust API and benchmark tooling. The
current workspace builds on that baseline with:

- FP32 rank-2/rank-3 GEMM on Vulkan compute.
- Explicit device-local tensors and explicit upload/download operations.
- GPU timestamp-based kernel timing.
- Thread-safe submission through `Executor`.
- C ABI wrapper crate for C callers.
- Reproducible local benchmark and correctness workflows.

This project is not yet a drop-in backend for external inference runtimes such
as Ollama, ggml, PyTorch, or TensorFlow. The C ABI exposes this library as a
callable GEMM component, but those runtimes still need their own adapter layer
or backend implementation to route model graph matmuls into `tensor-ash`.

## Features

- Vulkan 1.2 compute backend through `ash`, with optional validation layers.
- Device-local `Tensor` buffers with explicit upload / download through a
  staging path.
- Rank-2 and rank-3 row-major FP32 GEMM: `[M, K] @ [K, N]` and batched
  `[B, M, K] @ [B, K, N]`, with batch broadcasting when either side has `B == 1`.
- Batched submission of many independent `MatmulCall`s in a single GPU submit.
- Multiple GEMM kernel variants with automatic shape-based selection, plus a
  manual override for A/B tuning.
- Thread-safe `Executor` backed by a small pool of submission slots.
- GPU timestamp queries → per-run kernel time and TFLOPS when the device
  supports them.
- Persistent, device-qualified Vulkan pipeline cache under
  `$XDG_CACHE_HOME/tensor-ash/` to amortize pipeline creation across runs.
- Device selection by kind, index, or name substring, with a clear failure when
  a discrete GPU is required but unavailable.
- `tensor-ash-capi` workspace crate that builds `libtensor_ash.so` and
  `libtensor_ash.a` for C callers.

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

The flake shell adds the benchmark tooling, CUDA compiler/runtime packages,
Python, and `uv`:

```bash
nix develop .#benchmark
uv venv .venv-bench
source .venv-bench/bin/activate
uv pip install -r requirements-benchmark.txt
```

## Library sketch

```rust
use std::sync::Arc;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

let ctx = VulkanContext::new(false)?;
let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
let exec = Executor::new(ctx.clone(), pipeline, /*slots=*/ 2, /*max_calls=*/ 64)?;

let a = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
let b = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
let c = Tensor::uninit_device(&ctx, &[8, 256, 256])?;

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

## C ABI

The C ABI lives in the `tensor-ash-capi` workspace crate and is declared in
`include/tensor_ash.h`. It exposes opaque context, executor, and tensor handles
plus upload, download, single-GEMM, and batched-GEMM entry points. Errors return
`-1` or null and can be inspected with `ta_last_error()`.

Build the C library:

```bash
cargo build --release -p tensor-ash-capi
```

Compile and run the C smoke example:

```bash
cc -Iinclude examples/c_smoke.c -Ltarget/release -ltensor_ash \
  -Wl,-rpath,"$PWD/target/release" -o /tmp/tensor_ash_c_smoke
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
```

The example computes a 2x3 by 3x2 GEMM through the C API and verifies the
expected `[58, 64, 139, 154]` result.

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
ML_KERNEL=k64 ML_B=1 ML_M=256 ML_N=256 ML_K=256 cargo run --release --bin ml_bench -- single
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
- `ML_KERNEL` accepts `auto`, `large`, `small`, `m64n128`, `m128n64`,
  `m128n64k64`, `m64n32`, or `k64`. The default `auto` selector picks among
  the tuned variants by shape and batch count. The override is for benchmark
  tuning.
- `ML_OUTPUT` selects `table` (default) or `csv` output.

## Tests

CPU-only unit tests (no GPU required):

```bash
cargo test
cargo clippy --all-targets -- -D warnings
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
NumPy, PyTorch CPU, PyTorch CUDA/cuBLAS, CuPy CUDA/cuBLAS, optional JAX,
optional TensorFlow, and transfer bandwidth. Missing Python frameworks are
reported as skipped instead of failing the whole run. The harness writes raw
JSON plus a Markdown report:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set extended --iters 20 --warmup 5 --torch-threads 1 --transfer-mb 64'
```

Results land in `benchmarks/latest.json` and `benchmarks/latest.md`. These are
most meaningful after `ml_bench self-check` confirms a real GPU was selected;
if `llvmpipe` is picked, treat the `tensor-ash` rows as software-Vulkan
correctness and overhead data, not GPU performance.

For a GPU-focused run without CPU framework baselines:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set extended --iters 20 --warmup 5 --skip-cpu-frameworks'
```

The broader refactor, verification, benchmark procedure, and Ollama backend
attempt are documented in `benchmarks/process.md`.

A recent run on an RTX 3070 (`benchmarks/latest.md`) includes real PyTorch
CUDA/cuBLAS and CuPy CUDA/cuBLAS rows. In that run, `tensor-ash` was fastest on
7/11 cases, reached 8.987 TFLOPS on `square_1024`, had a 1.10x geometric-mean
throughput ratio versus PyTorch CUDA, and had a 2.51x geometric-mean throughput
ratio versus CuPy CUDA. The single-threaded CPU baselines remain useful for
context: `tensor-ash` measured about 32x geometric-mean throughput versus both
NumPy and PyTorch CPU.

Kernel selector tuning can be reproduced without overwriting the benchmark
report:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/tune_kernels.py --case-set extended --iters 50 --warmup 10 --skip-build'
```

## Release notes

See `CHANGELOG.md` for the `v1.0.0` release summary.

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
  context/     Vulkan instance/device setup, selection, debug, cache paths
  pipeline/    Pipeline layout, kernel variants, selector, creation helpers
  executor/    Thread-safe executor, submission slots, command recording
  bench/       `ml_bench` subcommands, output formatting, case definitions
  buffer.rs    Device/staging buffer wrappers
  tensor.rs    Shape + GPU buffer
  matmul.rs    MatmulCall, shape validation, RunStats
  testing.rs   Deterministic CPU reference/test helpers
  main.rs      Thin `ml_bench` entry point
shaders/
  matmul_kernel.glsl     Shared GEMM kernel body (parameterized by tile size)
  matmul_f32.comp        128x128-tile wrapper (large kernel)
  matmul_f32_small.comp  64x64-tile wrapper   (small kernel)
  matmul_f32_m64n128.comp    64x128 wrapper
  matmul_f32_m128n64.comp    128x64 wrapper
  matmul_f32_m128n64k64.comp 128x64 wrapper with BK=64
  matmul_f32_m64n32.comp     64x32 wrapper
  matmul_f32_k64.comp        64x64 wrapper with BK=64
capi/
  src/          C ABI wrapper crate over the Rust API
include/
  tensor_ash.h  Public C header
examples/
  c_smoke.c     C ABI smoke test
scripts/
  bench_compare.py       Cross-library comparison harness
  tune_kernels.py        Manual kernel-variant tuning helper
benchmarks/
  latest.{json,md}       Last cross-library run
  process.md             Refactor, verification, and benchmark process notes
CHANGELOG.md             Release notes
tests/
  correctness.rs         Ignored GPU suite entry point
  correctness/           Topical GPU correctness modules
```

## Vision

A durable Rust-first GPU building block for ML inference: small enough to
inspect, fast enough to sit near hardware limits, and explicit enough that
future kernels, schedulers, and model code can be layered on top without
hiding the important performance controls.
