# tensor-ash Refactor, Verification, and Benchmark Process

Generated: 2026-06-10. Structure and verification guidance updated
2026-08-06 after the workspace-level architecture cleanup. Historical
benchmark measurements remain labeled with the date and environment in which
they were collected.

## Scope

This report records the maintainability work, post-`v1.0.0` optimization, C
ABI port, local verification, cross-framework GEMM benchmark, and Ollama
backend attempt. The structure, commands, and file-size audit describe the
current workspace; older performance numbers below are retained as historical
measurements rather than implied results of the refactor.

The workspace now contains four deliberately separate packages:

- `tensor-ash`: the Rust/Vulkan FP32 GEMM library.
- `tensor-ash-capi`: a C ABI wrapper that builds `libtensor_ash.so` and
  `libtensor_ash.a`.
- `ml-bench`: a non-published benchmark/correctness CLI. Its logging,
  environment parsing, cases, and reporting no longer add dependencies or a
  binary target to the core library package.
- `tensor-ash-test-support`: a dependency-free, non-published home for
  deterministic fixtures and CPU reference math shared by integration tests
  and `ml-bench`; those helpers are no longer part of the runtime API.

The C ABI is a callable GEMM layer. It is not yet an Ollama or ggml backend
implementation, so Ollama cannot use it as a model runtime backend without
additional adapter work.

## Refactor Summary

The large files were decomposed along runtime ownership and change boundaries:

- `src/executor/mod.rs` is now the slot-owning facade. Normal/graph dispatch,
  validation, upload/download, timed queue submission, online tuning,
  split-K execution, and Stream-K execution/scheduling live in separate
  modules. Descriptor updates and graph barrier construction are further
  isolated under `src/executor/recording/`.
- `src/matmul.rs` is a compatibility facade. Public operations/statistics live
  in `src/matmul/api.rs`, while checked shape, broadcasting, batch-stride, and
  FLOP resolution live in `src/matmul/resolution.rs`.
- `src/pipeline/catalog.rs` is the single source of kernel identity. One macro
  declaration generates `KernelSelection`, parsing aliases, stable indices,
  and `KERNEL_SPECS`; ABI specialization, Vulkan creation/runtime, selection,
  and persistence are separate modules.
- `src/context/` separates device discovery, debug plumbing, pipeline-cache
  policy, and context construction/ownership.
- `tools/ml-bench/` is its own workspace package rather than `src/bench/` plus
  a binary in the library package.
- `tools/test-support/` owns deterministic input generation, reference BMM,
  error measurement, and tolerances shared by tests and the benchmark CLI.
- `scripts/bench_compare.py` is a small CLI/compatibility facade over
  `bench_compare_backends.py`, `bench_compare_models.py`, and
  `bench_compare_report.py`.
- `capi/src/api/` splits exported lifecycle, tensor-transfer, and matmul calls.
  Shared handle validation remains in `handles.rs`; panic/error barriers remain
  in `error.rs`; C-compatible records remain in `types.rs`.
- `tests/correctness/` remains organized by feature, with Vulkan-dependent
  integration tests explicitly ignored during ordinary CPU-only test runs.

The split also tightened ownership invariants: an `Executor` rejects a
pipeline created from another `VulkanContext`; Rust tensors and C tensor
handles retain their originating context; C operations reject cross-context
tensor handles; and release builds unwind at the C boundary so its
`catch_unwind` safety barrier remains effective.

## Performance Changes

Layered post-`v1.0.0`:

- Descriptor updates use a stack-allocated fast path for the common case of
  one descriptor set and one GEMM call. The previous path always allocated
  vectors sized for batched submissions.
- The shader has a `K_MULTIPLE` specialization constant. When the host knows
  `K` is a multiple of the kernel K tile, the selected pipeline can fold out
  the K-tail branch and modulo test.
- Manual/tuned tile family expanded to seven primary shapes after the initial
  CUDA comparison: `large` (128x128), `small` (64x64), `m64n128`, `m128n64`,
  `m128n64k64`, `m64n32`, `k64`. The auto-selector uses batch-aware measured
  rules so the new variants improve batch-1 medium/skinny shapes without
  regressing batched cases.
- The pipeline is now data-driven through a single `KERNEL_SPECS` table:
  pipeline construction, kernel selection, and the `ML_KERNEL=...` parser all
  derive from one `kernel_catalog!` declaration, so adding a tile takes one
  catalog entry plus a `.comp` wrapper.
- Vulkan 1.2 `bufferDeviceAddress` is enabled end-to-end (`VulkanContext`,
  `Buffer`, `MatmulPushConstants.{a,b,c}_ptr`), enabling the
  `GL_EXT_buffer_reference` BDA kernel family (LDG.E.128 global loads) and
  the BDA_V4 family on top of that (LDS.E.128 shared reads via `shared uvec4`
  Bs).
- The auto-selector promotes its picks to the BDA_V4 sibling when the device
  exposes `bufferDeviceAddress`, with a plain BDA fallback for the TN=2
  `m64n32` tile. Explicit `ML_KERNEL=...` selections are honored verbatim.

`KERNEL_SPECS` currently lists 35 SPIR-V variants, including descriptor-bound,
BDA, BDA_V4, register-tile, strict-aligned, exploratory, and row/GEMV entries.
The catalog macro prevents the selection enum, parser aliases, registry order,
and index mapping from drifting apart. Multiplied by the `K_MULTIPLE` /
`ACCUMULATE` / `ALPHA_IS_ONE` / `INTERIOR_ONLY` specialization bits, startup
builds 16 zero-epilogue pipelines per kernel; non-zero epilogues are created
lazily and everything is amortized by the persistent Vulkan pipeline cache
under `$XDG_CACHE_HOME/tensor-ash/`.

## Added Tests

- The generated kernel catalog tests every selection/index mapping, every
  case-insensitive alias, and unique registry names.
- Matmul resolution tests require bias tensors to have actual `[N]` or
  `[B, N]` shape rather than merely the same element count.
- Test-support checks now treat non-finite output as an infinite error and
  reject mismatched slice lengths.
- Stream-K schedule tests are CPU-only; all nine Vulkan execution cases are
  ignored during ordinary workspace tests, matching the rest of the GPU suite.
- Python adapter tests now exercise per-case failure containment in addition to
  model/report helpers.
- C ABI version string is NUL-terminated.
- C ABI null upload reports a per-thread error instead of panicking.
- C ABI destroy functions accept null handles as no-ops.
- Existing kernel-variant indexing tests now cover the `k_multiple` bit.
- Ignored GPU correctness now forces `m64n128`, `m128n64`, and `k64` kernels
  on aligned and partial-tile shapes.
- Python benchmark helper tests cover GPU-row classification, skipped-row FLOP
  accounting, and `nvidia-smi` formatting.

Previously added tests are still present:

- Device preference parser rejects an empty `name:` filter.
- Matmul shape resolution rejects batch-stride overflow.
- Matmul FLOP accounting rejects `u64` overflow.
- CPU reference GEMM explicitly tests B-side broadcasting.
- Ignored GPU suite includes `manual_large_kernel_handles_partial_tiles`.

## File Size Check

Largest source files after the split:

| file | lines | note |
| --- | ---: | --- |
| `scripts/bench_compare_backends.py` | 467 | isolated framework/native adapters |
| `src/pipeline/mod.rs` | 410 | pipeline ownership and caches |
| `src/executor/splitk2.rs` | 398 | two-stage split-K pipeline and planning |
| `src/executor/streamk_exec.rs` | 367 | Stream-K validation and GPU recording |
| `scripts/bench_compare_report.py` | 357 | JSON/Markdown analysis |
| `tools/ml-bench/src/bench/commands.rs` | 338 | standalone benchmark subcommands |
| `src/context/device.rs` | 320 | device selection and tests |
| `src/executor/streamk_schedule.rs` | 309 | pure host scheduling policy and tests |

No Rust or Python source file remains above 500 lines. The remaining larger
files each represent one cohesive subsystem; pipeline ownership and split-K2
creation are the next candidates if either grows materially.

## Verification Commands

Run the complete current-workspace check from the repository root:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p ml-bench
cargo build --release -p tensor-ash-capi
python3 scripts/bench_compare_test.py
cc -Iinclude examples/c_smoke.c -Ltarget/release -ltensor_ash \
  -Wl,-rpath,"$PWD/target/release" \
  -o /tmp/tensor_ash_c_smoke
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
nix-shell --run 'cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1'
```

Important runtime note:

```text
target/release/ml_bench self-check
```

fails outside the Nix shell with `failed to load Vulkan loader:
libvulkan.so.1`. Inside `nix-shell`, the same binary selects the NVIDIA RTX
3070 correctly.

The following GPU and C outputs are retained from the earlier recorded run;
they are not a substitute for rerunning the current verification sequence.

GPU correctness result:

```text
23 ignored release integration tests passed on NVIDIA GeForce RTX 3070.
```

C smoke result:

```text
tensor-ash C smoke OK: 58.000000 64.000000 139.000000 154.000000
```

Vulkan device inventory:

| index | device | kind |
| ---: | --- | --- |
| 0 | NVIDIA GeForce RTX 3070 | discrete GPU |
| 1 | llvmpipe (LLVM 21.1.8, 256 bits) | CPU/software Vulkan |

## GEMM Benchmark Command

Benchmark shell and CUDA Python setup:

```bash
nix develop .#benchmark
uv venv .venv-bench
source .venv-bench/bin/activate
uv pip install -r requirements-benchmark.txt
```

Showcase benchmark command (current default for cross-library runs):

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set showcase --iters 30 --warmup 10 --torch-threads 1 --transfer-mb 64'
```

Full raw data and table report:

- `benchmarks/latest.json`
- `benchmarks/latest.md`

Environment (latest recorded run):

- GPU: NVIDIA GeForce RTX 3070 (discrete, Vulkan 1.4.329, vendor=0x10de)
- Driver: 595.71.05
- Tensor timestamps: enabled
- CUDA Python rows: PyTorch 2.11.0+cu128 and CuPy 13.6.0
- Pure cuBLAS C++ row: `benchmarks/cublas_bench/cublas_bench.cu`,
  `CUBLAS_PEDANTIC_MATH`, CUDA events
- CPU framework threads: 1
- Cases: 26 `showcase` FP32 GEMM shapes (squares, batched, attention-style,
  non-power-of-two, tall/skinny/wide, small-K, tiny batches)
- Iterations: 30
- Warmups: 10
- Peak FP32 throughput used for `% peak`: 20.32 TFLOPS
  (RTX 3070; overridable via `ML_PEAK_TFLOPS`)

Summary:

| comparison | result |
| --- | ---: |
| `tensor-ash` fastest measured backend | 13 / 26 cases |
| vs pure cuBLAS (apples-to-apples) geomean | 1.11x |
| vs PyTorch CUDA/cuBLAS geomean | 1.32x |
| vs CuPy CUDA/cuBLAS geomean | 2.59x |
| vs NumPy single-thread geomean | 47.56x |
| vs PyTorch CPU single-thread geomean | 45.33x |
| best `tensor-ash` throughput | 10.18 TFLOPS / 50.1% peak (`square_1024`) |
| best `tensor-ash` `attn_qkv_1024x3072x512` | 10.41 TFLOPS / 51.2% peak |
| pure cuBLAS on `square_1024` | 10.92 TFLOPS / 53.8% peak |
| pure cuBLAS on `attn_qkv_1024x3072x512` | 11.44 TFLOPS / 56.3% peak |
| largest single-case gap | `non_pow2_1023x1025x1027`: 1.2x cublas_pure lead |
| median synchronous host overhead | 0.020 ms |
| transfer upload | 9.04 GiB/s |
| transfer download | 9.48 GiB/s |

Headline cases (TFLOPS, with `% peak` in brackets):

| case | `tensor-ash` | pure cuBLAS | PyTorch CUDA | CuPy CUDA |
| --- | ---: | ---: | ---: | ---: |
| `square_512` | 6.47 (31.8%) | 7.49 (36.9%) | 6.61 (32.5%) | 3.49 (17.2%) |
| `square_1024` | 10.18 (50.1%) | 10.92 (53.8%) | 10.65 (52.4%) | 8.98 (44.2%) |
| `medium_768` | 8.36 (41.2%) | 9.99 (49.1%) | 9.41 (46.3%) | 6.80 (33.4%) |
| `attn_proj_2048x512x512` | 9.75 (48.0%) | 9.81 (48.3%) | 9.34 (45.9%) | 7.04 (34.6%) |
| `attn_qkv_1024x3072x512` | 10.41 (51.2%) | 11.44 (56.3%) | 11.18 (55.0%) | 9.90 (48.7%) |
| `batched_2x512` | 7.65 (37.6%) | 7.83 (38.5%) | 7.09 (34.9%) | 4.63 (22.8%) |
| `batched_4x256` | 5.66 (27.9%) | 3.58 (17.6%) | 3.09 (15.2%) | 1.61 (7.9%) |
| `batched_8x256` | 7.24 (35.6%) | 6.72 (33.1%) | 5.73 (28.2%) | 3.08 (15.2%) |
| `non_pow2_1023x1025x1027` | 8.25 (40.6%) | 10.26 (50.5%) | 10.03 (49.3%) | 8.52 (41.9%) |
| `tiny_b32_128` | 6.32 (31.1%) | 4.73 (23.3%) | 4.11 (20.2%) | 1.83 (9.0%) |

Performance notes:

- Pure cuBLAS is still strongest on the largest GEMMs (`non_pow2_1023^3`,
  `attn_qkv`, `square_1024`, `medium_768`), where its hand-tuned kernels
  show their edge. The biggest single-case gap is `non_pow2_1023x1025x1027`
  at 8.25 vs 10.26 TFLOPS.
- `tensor-ash` wins decisively on batched and tiny-batch shapes where the
  per-call submission overhead amortizes over many matmuls (e.g.
  `batched_4x256`: 5.66 vs cuBLAS 3.58 TFLOPS, `tiny_b32_128`: 6.32 vs
  cuBLAS 4.73 TFLOPS).
- `attn_proj_2048x512x512` is now effectively a tie with pure cuBLAS
  (9.75 vs 9.81 TFLOPS) after the BDA_V4 pass.
- Median host/submission overhead is ~20 us per synchronous call. Reported
  TFLOPS uses GPU timestamps, so any remaining gap is shader efficiency,
  not upload/download or host timing.

Skipped framework rows:

- JAX: module not installed.
- TensorFlow: module not installed.

## Local AI / Ollama Attempt

Ollama is installed. Local model inventory from the existing Ollama service:

| model | size | notes |
| --- | ---: | --- |
| `qwen2.5:7b` | 4.7 GB | architecture `qwen2`, 7.6B parameters, Q4_K_M |
| `gemma4:latest` | 9.6 GB | installed, not benchmarked in this pass |

No Ollama models were running before the standard baseline test.

Standard Ollama baseline command:

```bash
ollama run --verbose qwen2.5:7b 'Return the numbers 1 through 64, one per line. No explanation.' --keepalive 1m
```

Cold standard baseline result:

| metric | value |
| --- | ---: |
| total duration | 24.590 s |
| load duration | 4.636 s |
| prompt eval | 46 tokens in 1.363 s |
| prompt eval rate | 33.76 tokens/s |
| generation eval | 183 tokens in 18.450 s |
| generation eval rate | 9.92 tokens/s |

Warm standard baseline result:

| metric | value |
| --- | ---: |
| total duration | 18.484 s |
| load duration | 99.500 ms |
| prompt eval | 46 tokens in 111.216 ms |
| prompt eval rate | 413.61 tokens/s |
| generation eval | 183 tokens in 18.136 s |
| generation eval rate | 10.09 tokens/s |

Custom backend attempt:

```bash
env OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_DEBUG=1 \
  OLLAMA_LLM_LIBRARY="$PWD/target/release/libtensor_ash.so" \
  ollama serve
```

The first sandboxed attempt failed while creating `~/.ollama/id_ed25519`
because the home directory was read-only in the sandbox. The escalated attempt
started a temporary server, but debug logs still showed Ollama discovering its
own CUDA runner:

```text
inference compute ... library=CUDA ... name=CUDA0 description="NVIDIA GeForce RTX 3070"
```

The temporary server did not see the existing models because its default model
directory was `/home/gusahlg/.ollama/models`, while the installed model blob
reported by `ollama show qwen2.5:7b --modelfile` is under
`/var/lib/ollama/models`.

Second custom-server attempt:

```bash
env OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_DEBUG=1 \
  OLLAMA_MODELS=/var/lib/ollama/models \
  OLLAMA_LLM_LIBRARY="$PWD/target/release/libtensor_ash.so" \
  ollama serve
```

This failed before model load:

```text
Error: mkdir /var/lib/ollama: file exists: ensure path elements are traversable
```

`/var/lib/ollama` is a symlink to `private/ollama`, and the launched user
process could not traverse `/var/lib/ollama/models`.

Backend integration status:

- `libtensor_ash.so` now exposes a C GEMM API.
- Ollama's `OLLAMA_LLM_LIBRARY` is not a generic GEMM callback interface.
- Ollama expects its own runner/backend ABI, and the debug run still selected
  Ollama's CUDA runner rather than loading `libtensor_ash.so` as a GEMM
  provider.
- Therefore the measured Ollama numbers above are standard Ollama only, not a
  `tensor-ash` backend run.
- A real comparison requires a ggml/Ollama adapter that maps model graph matmul
  calls to `ta_matmul` or `ta_matmul_batch`.

## Apples-to-Apples cuBLAS Methodology

The Python framework rows (PyTorch CUDA, CuPy CUDA) bundle the kernel with
wrapper overhead: tensor view bookkeeping, Python-side event allocation,
dispatch through the framework's autograd/dispatch layer, and (worst of all
for FP32 comparisons) a silent TF32 fallback for SGEMM on Ampere unless the
caller explicitly opts out.

The `cublas_pure` row in `benchmarks/latest.md` is driven by
`benchmarks/cublas_bench/cublas_bench.cu`, a small CUDA C++ binary that:

1. **Calls cuBLAS directly through its C API** (`cublasSgemm` /
   `cublasSgemmStridedBatched`). No PyTorch, no CuPy, no Python event loop.
2. **Forces real FP32** via `cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)`.
   `CUBLAS_DEFAULT_MATH` on Ampere silently routes SGEMM through TF32, which
   would slash precision and inflate throughput by roughly 8x — not the
   comparison we want for an FP32 library.
3. **Times with CUDA events** (`cudaEventRecord` / `cudaEventElapsedTime`),
   which is the closest CUDA equivalent to the Vulkan timestamp queries
   `ml_bench` uses. The two numbers are directly comparable: both exclude
   host overhead, both measure pure GPU time on the device clock.
4. **Reads cases from stdin as CSV** and emits CSV on stdout, so
   `scripts/bench_compare.py` can call it for every case in the showcase set
   with the same iteration / warmup counts as the Vulkan path.

Row-major / column-major: `tensor-ash` uses row-major tensors and cuBLAS is
column-major, so the cuBLAS call swaps `A` and `B` and computes `C^T = B^T A^T`.
That's a transpose-free reinterpretation, not a copy — the SGEMM kernel that
runs is the same one PyTorch / CuPy dispatch to. See the comment at the top
of `cublas_bench.cu` for the exact argument mapping.

## Optimization Gameplan

Status legend: DONE, in progress, NEXT.

1. **DONE** — Data-driven `KERNEL_SPECS` pipeline. Adding a tile shape is now
   one `kernel_catalog!` declaration plus a `.comp` wrapper.
2. **DONE** — Vulkan 1.2 `bufferDeviceAddress` end-to-end.
3. **DONE** — BDA kernel family (`GL_EXT_buffer_reference` → `LDG.E.128`).
   +5-15% on every tile we measured. Variants exist for all seven primary
   tiles: `large`, `small`, `m64n128`, `m128n64`, `m128n64k64`, `m64n32`,
   `k64`.
4. **DONE** — BDA_V4 kernel family (`shared uvec4` Bs → `LDS.E.128`). Another
   +5-15% over plain BDA on every TN ≥ 4 tile. No V4 path for the TN=2
   `m64n32` tile (LDS.128 over a 2-column stride is non-sensical); the
   auto-selector falls back to plain BDA for that one.
5. **DONE** — Auto-selector promotes its picks to BDA_V4 (with BDA fallback
   for `m64n32`) when `bufferDeviceAddress` is available. Explicit
   `ML_KERNEL=...` selections are honored verbatim.
6. **DONE** — Auto-selector rules retuned: medium rectangular shapes with
   `min_mn ∈ [512, 2048)` and aspect ≤ 5x go to `m128n64k64`; mid-square
   `min_mn ∈ [192, 480]` goes to `k64`; the `m64n32` rule is skipped at
   `batch ≥ 8`.
7. **DONE** — Pure cuBLAS C++ benchmark, `% peak` column, showcase case set
   (see "Apples-to-Apples cuBLAS Methodology" above).
8. **NEXT** — Close the remaining gap on the largest GEMMs. `square_1024`
   currently hits 50.1% peak (10.18 TFLOPS) vs pure cuBLAS at 53.8%
   (10.92 TFLOPS); `non_pow2_1023x1025x1027` is the worst single-case gap
   (8.25 vs 10.26 TFLOPS); `attn_qkv_1024x3072x512` is 51.2% vs 56.3%.
   Promising directions: register-blocking sweep, double the K-strip
   prefetch depth on the BDA_V4 path, evaluate `wmma`-style 16x16 tiles
   once Vulkan cooperative-matrix becomes broadly available.
9. **NEXT** — Implement a ggml/Ollama backend adapter over the C ABI if
   Ollama integration remains the priority.
10. **NEXT** — Add a benchmark focused on C ABI call overhead and
    batched-call throughput.
11. **DONE** — Split `scripts/bench_compare.py` into backend adapters, data
    models/case sets, and report generation, retaining the original module as
    the CLI and compatibility facade.

### Dead Ends — Do Not Retry

- **v3 / double-buffered SMEM with static stage indices**
  (`shaders/matmul_v3_kernel.glsl`, kernel `v3_128x128_bk8_static`). The
  premise was that the v2 dynamic `(kt & 1u)` stage index was preventing the
  NVIDIA driver from interleaving the next-strip global load with the current-
  strip FFMAs. The v3 kernel manually peels the K loop into stage-0/stage-1
  pairs so the indices are literal constants. In practice the driver already
  schedules v2 well, the v3 kernel paid an extra register cost for the
  unrolled control flow, and it lost on every shape we measured against the
  BDA / BDA_V4 path. It's kept in the registry as a manual `ML_KERNEL=v3`
  variant for reproducibility, but the auto-selector ignores it. Don't waste
  another round on a double-buffer rewrite; the win is in global-load width
  (BDA) and shared-load width (BDA_V4), not in stage indexing.
