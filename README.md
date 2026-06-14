# tensor-ash

Version: `1.1.0-dev` (post-`v1.0.0` optimization pass)

`tensor-ash` is a small Vulkan compute library for high-throughput FP32 matrix
multiplication, written in Rust on top of [`ash`](https://crates.io/crates/ash).
It is designed to sit close to the hardware: explicit GPU buffers, explicit
submission, GPU-timestamp measurements, and a narrow public surface that's easy
to inspect and tune.

The current scope is batched GEMM. The compute backend, kernel selector, and
executor are built around a data-driven `KERNEL_SPECS` registry, so adding a
new tile shape is a two-line change plus a `.comp` wrapper — the rest of the
pipeline picks it up automatically.

## Scope and API status

`v1.0.0` is the stable baseline. The current workspace builds on it with FP32
rank-2/rank-3 GEMM on Vulkan compute (batch-broadcasting), explicit
device-local tensors with explicit upload/download, GPU timestamp timing,
thread-safe submission through `Executor`, a C ABI wrapper crate, and Vulkan
1.2 `bufferDeviceAddress` end-to-end (context, buffer, push constants,
shaders) enabling the `GL_EXT_buffer_reference` kernel families.

This project is not yet a drop-in backend for external inference runtimes
(Ollama, ggml, PyTorch, TensorFlow). The C ABI exposes this library as a
callable GEMM component; those runtimes still need their own adapter layer.

## Features

- Vulkan 1.2 compute backend through `ash`, with optional validation layers.
- Vulkan 1.2 `bufferDeviceAddress` (BDA) plumbed through `VulkanContext`,
  `Buffer`, and the push-constant layout so kernels can dereference A/B/C as
  raw 64-bit pointers.
- Data-driven `KERNEL_SPECS` table of compiled shader variants + shape-aware
  auto-selector. The selector promotes every eligible pick to the `BDA_V4`
  family when the device supports it (with a `BDA` fallback for the TN=2
  `m64n32` tile, which has no V4 sibling).
- Manual override (`ML_KERNEL=...`) over the entire registry.
- Persistent device-qualified Vulkan pipeline cache under
  `$XDG_CACHE_HOME/tensor-ash/`.
- Device selection by kind, index, or name substring.
- `tensor-ash-capi` workspace crate that builds `libtensor_ash.so` /
  `libtensor_ash.a` for C callers.

### Kernel families: BDA and BDA_V4

The post-`v1.0.0` perf work added two parallel families on top of the original
descriptor-bound tiles:

- **BDA** — `GL_EXT_buffer_reference` global loads → `LDG.E.128` (one 128-bit
  transaction instead of four 32-bit ones).
- **BDA_V4** — `shared uvec4` Bs staging → `LDS.E.128` shared-memory reads.
  Layers cleanly on top of BDA.
- **BDA_V4 aligned** — strict no-bounds-check siblings of the 128x128 and
  m128n64k64 tiles for shapes where `M % BM == N % BN == K % BK == 0`.
  Sources only emit the LDG.E.128 / LDS.E.128 / FFMA hot path and the
  STG.E.128 epilogue; the dispatcher promotes through `maybe_to_aligned`
  when the shape qualifies.

On an RTX 3070, every tile we measured gains roughly +5-15% from BDA and
another +5-15% from V4. The auto-selector promotes its picks to BDA_V4 at
runtime when the device exposes `bufferDeviceAddress`, with a BDA fallback for
`m64n32`. Explicit `ML_KERNEL=...` picks are honored verbatim.

### Stream-K (experimental)

The `Executor` exposes two opt-in entry points for shapes where the regular
data-parallel dispatch leaves a small partial wave at the end:

- `run_matmuls_stream_k(call)` always routes through the hybrid Stream-K
  pipeline (CUTLASS-style DP-flat bulk dispatch + persistent SK-tail with
  hardware `atomicAdd` from `VK_EXT_shader_atomic_float`).
- `run_matmuls_auto_stream_k(call, tail_fraction_max)` consults a heuristic
  gate and falls back to `run_matmuls` when Stream-K wouldn't help.

Restrictions: single matmul, batch == 1, aligned shapes
(`M%128 == N%128 == K%32 == 0`), `accumulate == false`, and the device must
expose `shaderBufferFloat32AtomicAdd`. The gate is intentionally
conservative — most showcase shapes still win on plain DP, so callers that
don't know their workload should stick to `run_matmuls`.

## Requirements

- Vulkan 1.2 runtime/driver and loader (`libvulkan.so.1` on Linux).
  `bufferDeviceAddress` is required for the BDA kernel families; the selector
  falls back to descriptor-bound variants when it's unavailable.
- Rust (2024 edition).
- `glslc` / `shaderc`. `build.rs` compiles `shaders/*.comp` to SPIR-V and
  embeds the result.

```bash
nix-shell
cargo run --release --bin ml_bench -- self-check
```

The flake adds the benchmark tooling, CUDA compiler/runtime, Python, and `uv`:

```bash
nix develop .#benchmark
uv venv .venv-bench && source .venv-bench/bin/activate
uv pip install -r requirements-benchmark.txt
```

## Library sketch

```rust
use std::sync::Arc;
use tensor_ash::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

let ctx = VulkanContext::new(false)?;
let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
let exec = Executor::new(ctx.clone(), pipeline, 2, 64)?;

let a = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
let b = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
let c = Tensor::uninit_device(&ctx, &[8, 256, 256])?;
exec.upload(&host_a, &a)?;
exec.upload(&host_b, &b)?;

let stats = exec.run_matmuls(&[MatmulCall {
    a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false,
}])?;
exec.download(&c, &mut host_c)?;
println!("{:?}", stats.tflops());
```

## C ABI

The C ABI lives in `tensor-ash-capi` and is declared in
`include/tensor_ash.h`. Errors return `-1` or null and can be inspected with
`ta_last_error()`.

```bash
cargo build --release -p tensor-ash-capi
cc -Iinclude examples/c_smoke.c -Ltarget/release -ltensor_ash \
  -Wl,-rpath,"$PWD/target/release" -o /tmp/tensor_ash_c_smoke
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
```

The smoke test computes a 2x3 by 3x2 GEMM and verifies the expected
`[58, 64, 139, 154]`.

## Benchmark binary (`ml_bench`)

Subcommands: `self-check`, `correctness`, `sweep`, `single`, `concurrent`,
`transfer`. Useful env knobs:

```bash
ML_B=4 ML_M=1024 ML_N=1024 ML_K=1024 cargo run --release --bin ml_bench -- single
ML_KERNEL=k64_bda_v4 cargo run --release --bin ml_bench -- single
ML_DEVICE=discrete cargo run --release --bin ml_bench -- self-check
ML_OUTPUT=csv ML_SWEEP=smoke cargo run --release --bin ml_bench -- sweep
```

`ML_KERNEL=auto` is the default. Concrete names span every entry in
`KERNEL_SPECS`: descriptor-bound tiles (`large`, `small`, `m64n128`,
`m128n64`, `m128n64k64`, `m64n32`, `k64`, `bk16`, `v2`, `m64n128k64`,
`m128n128_t4`, `m256n64`, `v3`), their `*_bda` / `*_bda_v4` siblings,
and the strict-aligned `large_bda_v4_aligned` /
`m128n64k64_bda_v4_aligned` variants for shapes divisible by their
tile dims.
`ML_DEVICE` accepts `auto`, `discrete`, `integrated`, `virtual`, `cpu`,
`index:N`, `name:TEXT`, or a bare name substring.

## Tests

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release --test correctness -- --ignored --test-threads=1
```

## Cross-library benchmarks

`scripts/bench_compare.py` runs `tensor-ash`, NumPy, PyTorch CPU, PyTorch
CUDA/cuBLAS, CuPy CUDA/cuBLAS, optional JAX, optional TensorFlow, a **pure
cuBLAS C++ binary** (`benchmarks/cublas_bench/`), and transfer bandwidth.

The pure cuBLAS row is the apples-to-apples GPU baseline: it calls cuBLAS
through its C API directly with `CUBLAS_PEDANTIC_MATH` to force real FP32 (no
silent TF32 fallback) and times with CUDA events. That isolates the kernel
from PyTorch wrapper overhead.

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set showcase --iters 30 --warmup 10 --torch-threads 1 --transfer-mb 64'
```

Results land in `benchmarks/latest.{json,md}`. If `self-check` reports
`llvmpipe`, treat the `tensor-ash` rows as software-Vulkan correctness data,
not GPU performance. The broader process, including the BDA work and Ollama
backend attempt, lives in `benchmarks/process.md`.

### Results

Latest run: `showcase` set on an RTX 3070 (`benchmarks/latest.md`), 30 iters /
10 warmups, FP32 throughout. Numbers are quoted as `% peak` against the RTX
3070's 20.32 TFLOPS FP32 ceiling (`ML_PEAK_TFLOPS`, overridable for other
GPUs).

- `tensor-ash` is the fastest measured backend on **14/26** showcase cases.
- vs the apples-to-apples **pure cuBLAS** baseline: geomean **1.146x** across
  26 shared cases (range 0.83x-2.31x).
- vs **PyTorch CUDA/cuBLAS**: geomean **~1.32x** when PyTorch is installed
  (range 0.82x-3.62x in earlier runs) — the wrapper overhead alone shows up
  as a meaningful gap.
- vs **CuPy CUDA/cuBLAS** when installed: geomean **~2.6x**.
- vs single-threaded **NumPy / PyTorch CPU**: ~45-48x.
- Headline silicon-limit points: `attn_qkv_1024x3072x512` hits **51.2% peak**
  (10.41 TFLOPS) vs pure cuBLAS at **56.1%**; `square_1024` hits **50.1%
  peak** (10.17 TFLOPS) vs pure cuBLAS at **54.0%**.
- New cuBLAS-beating shapes in v1.1.0: `medium_384` (1.045x) and
  `tall_512x256x256` (essentially tied, 0.998x); `skinny_1024x128x512` and
  `wide_128x1024x512` close from ~0.88x to ~0.93-0.98x.
- Median synchronous host/submission overhead: ~0.022 ms per GEMM call;
  reported TFLOPS uses GPU timestamps and excludes it.

So: ahead of pure cuBLAS on geomean, decisively ahead of PyTorch CUDA, still
trailing cuBLAS on a handful of big square / non-pow2 cases where its
hand-tuned kernels show their edge. The BDA / BDA_V4 path is where the bulk
of recent gains came from — every tile we measured gains ~5-15% from
`LDG.E.128` and another ~5-15% from `LDS.E.128` — plus a +4-7% win on the
K64-routed shape band from the `k64_bda_v4_tm8_tn4` register-tile variant
landed in v1.1.0.

Selector tuning without overwriting the benchmark report:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/tune_kernels.py --case-set showcase --iters 50 --warmup 10 --skip-build'
```

## Release notes & troubleshooting

See `CHANGELOG.md` for `v1.0.0` and the unreleased post-`v1.0.0` changes.

If the binary fails with `failed to load Vulkan loader: libvulkan.so.1`, the
loader isn't visible to the dynamic linker — the Nix shell handles it,
otherwise set `LD_LIBRARY_PATH` to the loader directory. If `self-check`
reports `llvmpipe`, results are software-renderer numbers; use
`ML_DEVICE=discrete` or `VK_ICD_FILENAMES`.

## Project layout

```
src/
  context/         Vulkan instance/device setup, BDA + atomic-float feature
                   wiring, cache paths
  pipeline/        Data-driven KERNEL_SPECS registry, auto-selector, BDA and
                   aligned promotion helpers
  executor/        Thread-safe executor, submission slots, command recording
    splitk.rs      Experimental split-K pipeline
    streamk.rs     Experimental Stream-K (hybrid DP-flat + SK-tail) pipeline
  bench/           ml_bench subcommands, output formatting, case definitions
  buffer.rs        Device/staging buffer wrappers (BDA-aware)
shaders/
  matmul_kernel.glsl              Original descriptor-bound GEMM body
  matmul_bda_kernel.glsl          buffer_reference body (LDG.E.128)
  matmul_bda_v4_kernel.glsl       BDA + shared uvec4 Bs body (LDS.E.128)
  matmul_bda_v4_aligned_kernel.glsl
                                  Strict no-bounds-check BDA_V4 hot path
  matmul_streamk_kernel.glsl      Stream-K SK-tail (persistent + atomicAdd)
  matmul_streamk_dp_kernel.glsl   Stream-K DP-flat bulk dispatch
  matmul_f32*.comp                Tile wrappers; *_bda / *_bda_v4 siblings
capi/, include/, examples/   C ABI crate, public header, smoke test
scripts/
  bench_compare.py   Cross-library harness (cuBLAS pure row, % peak, showcase)
  tune_kernels.py    Manual kernel-variant tuning helper
benchmarks/
  cublas_bench/      Pure cuBLAS C++ benchmark binary + Makefile
  latest.{json,md}   Last cross-library run
  process.md         Refactor, verification, and benchmark process notes
tests/correctness/   Ignored GPU correctness modules
CHANGELOG.md         Release notes
```

## Vision

A durable Rust-first GPU building block for ML inference: small enough to
inspect, fast enough to sit near hardware limits, and explicit enough that
future kernels and schedulers can be layered on top without hiding the
important performance controls.
