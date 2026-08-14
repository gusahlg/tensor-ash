# tensor-ash

Version: `1.4.1`

`tensor-ash` is a small Vulkan compute library for high-throughput FP32 matrix
multiplication, written in Rust on top of [`ash`](https://crates.io/crates/ash).
It is designed to sit close to the hardware: explicit GPU buffers, explicit
submission, GPU-timestamp measurements, and a narrow public surface that's easy
to inspect and tune.

The current scope is batched GEMM plus fused GEMM epilogues (bias,
activations, residual/gating), executed either as independent batches or as
dependent graphs in a single submission. The compute backend, kernel
selector, and executor are built around a data-driven `KERNEL_SPECS`
registry, so adding a new tile shape takes one catalog declaration plus a
`.comp` wrapper — the generated selection enum, parser, stable registry order,
pipeline builder, and measured auto-tuner all pick it up automatically.

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
- **Measured auto-tuning** (`ML_TUNE=1`): first sight of a new shape
  benchmarks every eligible kernel — and the two-stage split-K on deep-K
  shapes — on the caller's real problem, then persists the winner per
  device/driver/shader-build under `$XDG_CACHE_HOME/tensor-ash/`.
  Persisted winners apply on every later run automatically.
- **Fused epilogues** (`MatmulOp` / `Epilogue`): shared `[N]` or per-batch
  `[B, N]` bias + ReLU/SiLU/GELU + residual-add or SwiGLU-style gating applied
  in-register at store time — `silu(x @ W_gate) * up` is one dispatch.
- **Graph submission** (`run_matmul_graph` / `run_op_graph`): dependent
  chains recorded into one command buffer with automatic hazard barriers;
  one fence wait per chain.
- **Prepared submission** (`prepare_matmuls` / `prepare_ops` /
  `prepare_op_graph`): validate, route, and record a fixed batch once, then
  replay with one `vkQueueSubmit` per call; the split `submit`/`wait` lets two
  prepared objects ping-pong so submission overlaps GPU execution (1.26x over
  the synchronous path on the smallest GEMMs, 1.9x versus v1.4.1).
- **Model ops** (`run_softmax_rows` / `run_rms_norm` / `run_layer_norm` /
  `run_rope` / `run_copy_strided`): masked row softmax (exact-zero masked
  tail — zero-padded KV caches compose exactly), bandwidth-rate RMSNorm /
  LayerNorm, partial-rotary RoPE, and a generic strided copy for
  transpose / KV-append / head reshaping. Together with fused-epilogue
  GEMM these cover a full transformer decoder block.
- **Fused flash-attention prefill** (`run_flash_attention`): causal
  online-softmax attention in one dispatch — no materialized score
  tensor, causal tile skipping, GQA, warm-cache offsets; interoperates
  with the composed softmax/matmul path on the same KV-cache layouts.
- **FP16 weights** (`Tensor::uninit_device_f16`): store B as IEEE half with
  f32 accumulation — half the weight memory and ~1.9x on bandwidth-bound
  decode GEMV shapes, neutral on compute-bound ones. The `&[f32]` host API
  is unchanged (CPU round-to-nearest-even conversion); the auto-router and
  tuner pick `f16w_*` kernels per storage type.
- **Two-stage split-K** (`run_matmuls_split_k2`): deterministic scratch-plane
  partials + reduce, no atomics; up to 16x over the data-parallel path on
  deep-K skinny shapes and auto-routed by a conservative heuristic or tuner.
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
  STG.E.128 epilogue; they remain explicit experimental selections because
  the specialization-constant V4 path is faster on the measured device.
- **Row BDA** — eight K-slice warps cooperate on one output-row tile
  without materializing the mostly-empty 64x64 tiles used by a general GEMM.
  It is selected automatically for `M=1`, saturates memory bandwidth even for
  a lone row, scales to large batches of private neural-network layers, and
  supports the same broadcasting, accumulation, and fused epilogues as the
  other bounds-checked BDA kernels.
- **Column BDA** — one warp cooperatively reduces K for two output rows. It is
  selected automatically for `N=1`, reuses B across the two rows, and supports
  broadcasting, accumulation, and fused epilogues.
- **Outer BDA** — register-only rank-1 update for `K=1`, where a tiled GEMM
  wastes ~97% of its staging and math. Each thread owns a 4-row x vec4 tile
  with no shared memory or barriers; selected automatically for `K=1` and
  general-shape correct under explicit selection.

On an RTX 3070, every tile we measured gains roughly +5-15% from BDA and
another +5-15% from V4. The auto-selector promotes its picks to BDA_V4 at
runtime when the device exposes `bufferDeviceAddress`, with a BDA fallback for
`m64n32`. Explicit `ML_KERNEL=...` picks are honored verbatim.

## Requirements

- Vulkan 1.2 runtime/driver and loader (`libvulkan.so.1` on Linux).
  `bufferDeviceAddress` is required for the BDA kernel families; the selector
  falls back to descriptor-bound variants when it's unavailable.
- Rust (2024 edition).
- `glslc` / `shaderc`. `build.rs` compiles `shaders/*.comp` to SPIR-V and
  embeds the result.

```bash
nix-shell
cargo run --release -p ml-bench -- self-check
```

The flake adds the benchmark tooling, CUDA compiler/runtime, Python, and `uv`:

```bash
nix develop .#benchmark
uv venv .venv-bench && source .venv-bench/bin/activate
uv pip install -r requirements-benchmark.txt
```

## Learning the API (start here)

Four short, heavily-commented tutorials live in `examples/`, written for
people who have used PyTorch a little but never touched explicit GPU
programming. Each builds on the previous one:

```bash
cargo run --release --example tutorial_1_hello_matmul       # context/pipeline/executor, upload, matmul, download
cargo run --release --example tutorial_2_batching_and_timing # rank-3 batching, broadcasting, accumulate, GPU vs wall timing
cargo run --release --example tutorial_3_fused_linear_layer  # relu(x@W + b) as ONE dispatch via fused epilogues
cargo run --release --example tutorial_4_mlp_in_one_graph    # a whole MLP forward in one submission with run_op_graph
```

After those, `examples/synth_llama_layer.rs` shows a transformer-style
layer (graph submission + fused SwiGLU), and `examples/bench_splitk2.rs`
compares the reduction strategies.

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

Library integrations that should not read process-wide tuning state can use
`Executor::new_with_config` and `ExecutorConfig`; the compatibility
`Executor::new` constructor continues to honor `ML_TUNE` for CLI workflows.

Fused epilogues and dependent graphs:

```rust
use tensor_ash::{Activation, Epilogue, EpilogueBinary, MatmulOp};

// One submission for a dependent chain; `gated = silu(x@Wg) * up` is a
// single dispatch, and the hazard barriers are inserted automatically.
exec.run_op_graph(&[
    MatmulOp::new(MatmulCall { a: &x, b: &w_up, c: &up, alpha: 1.0, accumulate: false }),
    MatmulOp::with_epilogue(
        MatmulCall { a: &x, b: &w_gate, c: &gated, alpha: 1.0, accumulate: false },
        Epilogue { bias: None, activation: Activation::Silu,
                   binary: EpilogueBinary::Mul { d: &up } },
    ),
    MatmulOp::new(MatmulCall { a: &gated, b: &w_down, c: &out, alpha: 1.0, accumulate: false }),
])?;
```

## C ABI

The C ABI lives in `tensor-ash-capi` and is declared in
`include/tensor_ash.h`. Errors return `-1` or null and can be inspected with
`ta_last_error()`. The surface covers real inference workloads:

- Lifecycle: `ta_context_create`, `ta_executor_create` /
  `ta_executor_create_v2` (explicit tuning policy instead of `ML_TUNE`),
  plus capability queries `ta_context_supports_f16` /
  `ta_context_supports_bda`.
- Tensors: f32 (`ta_tensor_create`) and f16-storage
  (`ta_tensor_create_f16`, `ta_tensor_is_f16`) device tensors.
  `ta_upload` / `ta_download` always take `float` on the host and convert
  to/from half automatically for f16 tensors.
- Ops: `ta_matmul` / `ta_matmul_batch` for plain GEMMs; `ta_run_ops` /
  `ta_run_op_graph` for `ta_matmul_op` batches with fused epilogues
  (bias, relu/silu/gelu, residual add-scaled or gating mul), the graph
  variant inserting automatic barriers between dependent ops.
- Model ops: `ta_softmax_rows` (full / prefix / causal masks via
  `TA_SOFTMAX_MASK_*`), `ta_rms_norm`, `ta_layer_norm`, `ta_rope`
  (`ta_rope_desc`), and `ta_copy_strided` (`ta_copy_desc`). In-place
  (input == output) is allowed for softmax/norm/rope; `ta_copy_strided`
  requires `src != dst`. All require `ta_context_supports_bda`.
- Prepared replay: `ta_prepared_create` records a fixed op batch once;
  `ta_prepared_run` or the pipelined `ta_prepared_submit` /
  `ta_prepared_wait` replay it with one queue submit per call;
  `ta_prepared_destroy` fence-waits. The executor and all referenced
  tensors must outlive the `ta_prepared` handle (documented in the
  header).
- Diagnostics: `ta_dispatch_info_for` reports the selected kernel, tile,
  and split-K route for a shape; `ta_tune_shape` pre-warms the measured
  tuner.

```bash
cargo build --release -p tensor-ash-capi
cc -Iinclude examples/c_smoke.c -Ltarget/release -ltensor_ash -lm \
  -Wl,-rpath,"$PWD/target/release" -o /tmp/tensor_ash_c_smoke
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
```

The smoke test computes a 2x3 by 3x2 GEMM and verifies the expected
`[58, 64, 139, 154]`, then exercises a fused bias+SiLU op, prepared
replay (including submit/wait), an f16-weights matmul, and the model
ops (RMSNorm, prefix-masked softmax, strided-copy transpose) when the
device supports them.

## Benchmark binary (`ml_bench`)

Subcommands: `self-check`, `correctness`, `sweep`, `single`, `cases`,
`concurrent`, `transfer`. Useful env knobs:

```bash
ML_B=4 ML_M=1024 ML_N=1024 ML_K=1024 cargo run --release -p ml-bench -- single
ML_KERNEL=k64_bda_v4 cargo run --release -p ml-bench -- single
ML_TUNE=1 ML_M=768 ML_N=768 ML_K=768 cargo run --release -p ml-bench -- single
ML_DEVICE=discrete cargo run --release -p ml-bench -- self-check
ML_OUTPUT=csv ML_SWEEP=smoke cargo run --release -p ml-bench -- sweep
ML_OUTPUT=csv cargo run --release -p ml-bench -- cases \
  square,1,512,512,512 edge,1,511,513,515
```

Timed runs validate sampled outputs outside the measurement window and report
paired wall/GPU minimum, median, and p95 values. TFLOPS uses the median; CSV
also identifies the selected kernel, tile, and split-K route.

`ML_TUNE=1` measures every eligible kernel (and the two-stage split-K on
deep-K shapes) the first time a shape is seen and persists the winner under
`$XDG_CACHE_HOME/tensor-ash/tuned_kernels_*.txt`, keyed to the driver
version + shader build. Persisted winners are applied on **every** run —
tuning enabled or not — whenever `ML_KERNEL` is `auto`. Delete the store (or
update the driver / rebuild shaders) to reset it.

`ML_KERNEL=auto` is the default. `KERNEL_SPECS` currently contains 37 concrete
choices. They include the
descriptor-bound tiles (`large`, `small`, `m64n128`, `m128n64`,
`m128n64k64`, `m64n32`, `k64`, `bk16`, `v2`, `m64n128k64`,
`m128n128_t4`, `m256n64`, `v3`), BDA / BDA_V4 and register-tile variants,
the strict-aligned kernels, and the shape-specialized `row_bda` (`M=1`),
`col_bda` (`N=1`), and `outer_bda` (`K=1`) kernels. The authoritative names
and aliases live together in `src/pipeline/catalog.rs`.
`ML_DEVICE` accepts `auto`, `discrete`, `integrated`, `virtual`, `cpu`,
`index:N`, `name:TEXT`, or a bare name substring.

## Tests

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1
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

Latest run: `showcase` set on an RTX 3070, driver 595.80
(`benchmarks/latest.md`), 30 iters / 10 warmups, FP32 throughout, measured
selection pre-tuned with `ML_TUNE=1`. Numbers are quoted as `% peak` against
the RTX 3070's 20.32 TFLOPS FP32 ceiling (`ML_PEAK_TFLOPS`, overridable for
other GPUs).

- `tensor-ash` is the fastest measured backend on **16/26** showcase cases
  (was 14/26 in v1.2.x).
- vs the apples-to-apples **pure cuBLAS** baseline: geomean **1.18x** across
  26 shared cases (range 0.83x-2.55x; was 1.146x).
- vs **PyTorch CUDA/cuBLAS**: geomean **1.40x** (range 0.87x-3.64x).
- vs **CuPy CUDA/cuBLAS**: geomean **2.74x**.
- vs single-threaded **NumPy / PyTorch CPU**: ~47-48x.
- Headline silicon-limit points: `attn_qkv_1024x3072x512` hits **53.4% peak**
  (10.84 TFLOPS); `square_1024` **50.1%** (10.19 TFLOPS).
- Biggest v1.3.0 movers, all from the measured tuner discovering that the
  `k64_bda_v4_tm8_tn4` register tile generalizes far beyond its hand-written
  routing rules: `batched_2x512` 7.86→9.43 TFLOPS (+20%), `batched_8x256`
  7.35→8.61 (+17%), `tiny_b16_192` 6.16→7.09 (+15%), `small_k_1024x1024x64`
  6.69→7.37 (+10%), `square_512` 6.46→6.87 (+6%), `batched_64x128`
  6.93→7.81 (+13%).
- Off-showcase, the two-stage split-K transforms the deep-K band:
  `64x64x8192` runs 16.5x faster than the data-parallel path
  (0.413→0.025 ms), `128x128x8192` 4.6x, `256x256x4096` 1.8x — and is
  deterministic, unlike the atomic split-K.
- Median synchronous host/submission overhead: ~0.010 ms per GEMM call (a
  bounded spin before the blocking fence wait removed the scheduler wakeup);
  reported TFLOPS uses GPU timestamps and excludes it. Dependent chains
  amortize it via `run_matmul_graph`, and repeated fixed batches drop to
  ~0.013 ms/call end-to-end with two ping-ponged `PreparedOps`.

So: ahead of pure cuBLAS on geomean, decisively ahead of PyTorch CUDA, still
trailing cuBLAS on a handful of big square / non-pow2 cases
(`medium_768` 0.83x, `non_pow2_1023x1025x1027` 0.84x) where its hand-tuned
kernels keep an in-kernel edge — the tuner confirms our best registry kernel
is already selected there, so closing that gap needs new kernel work, not
better selection.

Selector tuning without overwriting the benchmark report:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/tune_kernels.py --case-set showcase --iters 50 --warmup 10 --skip-build'
```

## Release notes & troubleshooting

See `CHANGELOG.md` for the current unreleased work and the versioned release
history.

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
  matmul.rs        Public facade over operation types and shape resolution
  matmul/          API types, checked shape/batch resolution, unit tests
  pipeline/        Generated KERNEL_SPECS catalog, shader ABI, pipeline
                   creation/runtime, auto-selection, measured tuning
    tuning.rs      Measured-selection store (persistent per-device winners)
  executor/        Thread-safe dispatch facade over validation, transfers,
                   timed submission, recording, tuning, and reductions
    recording/     Descriptor updates and graph hazard/barrier recording
    splitk2.rs     Two-stage split-K (scratch partials + reduce, no atomics)
  buffer.rs        Device/staging buffer wrappers (BDA-aware)
  tensor.rs        Context-owned tensor abstraction
tools/ml-bench/    Independent benchmark CLI package, reports, and cases
tools/test-support/ Dependency-free CPU references and deterministic fixtures
                    shared by integration tests and `ml-bench`
shaders/
  matmul_kernel.glsl              Original descriptor-bound GEMM body
  matmul_bda_kernel.glsl          buffer_reference body (LDG.E.128)
  matmul_bda_v4_kernel.glsl       BDA + shared uvec4 Bs body (LDS.E.128)
  matmul_epilogue_common.glsl     Fused-epilogue helpers (spec consts 4..6)
  matmul_bda_v4_aligned_kernel.glsl
                                  Strict no-bounds-check BDA_V4 hot path
  matmul_splitk2_kernel.glsl      Two-stage split-K stage 1 (partial planes)
  matmul_f32_splitk2_reduce.comp  Two-stage split-K stage 2 (plane sum)
  matmul_f32*.comp                Tile wrappers; *_bda / *_bda_v4 siblings
capi/                      C ABI workspace crate; lifecycle, tensor, and
                           matmul exports are split under capi/src/api/
include/, examples/        Public C header, smoke test, Rust examples
scripts/
  bench_compare.py   Cross-library CLI and compatibility facade
  bench_compare_backends.py  Framework and native benchmark adapters
  bench_compare_models.py    Cases and result data models
  bench_compare_report.py    JSON/Markdown aggregation and reporting
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
