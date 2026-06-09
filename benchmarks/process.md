# tensor-ash Refactor, Verification, and Benchmark Process

Generated: 2026-06-09

## Scope

This pass focused on maintainability, correctness coverage, and local benchmark
evidence for the current FP32 Vulkan GEMM backend.

The project does not yet expose a C ABI, ggml backend, or Ollama integration
layer. Because of that, Ollama cannot currently be switched to `tensor-ash` as
its GEMM backend. I still measured a standard local Ollama run and recorded the
integration blocker below.

## Refactor Summary

- Split the benchmark binary from one large `src/main.rs` into `src/bench/`.
  - `env.rs`: environment parsing and mode enums.
  - `cases.rs`: benchmark case definitions and host shape sizing.
  - `report.rs`: table/CSV reporting.
  - `commands.rs`: benchmark subcommands.
- Split `src/executor.rs` into `src/executor/`.
  - `slot.rs`: per-submission Vulkan resources.
  - `recording.rs`: descriptor updates and matmul command recording.
  - `mod.rs`: public executor API and submit flow.
- Split `src/pipeline.rs` into `src/pipeline/`.
  - `types.rs`: public pipeline/kernel types.
  - `selection.rs`: auto-selection thresholds and tests.
  - `create.rs`: shader module and compute pipeline creation.
- Split `src/context.rs` into `src/context/`.
  - `device.rs`: device preference parsing and physical-device selection.
  - `cache.rs`: persistent pipeline-cache path construction.
  - `debug.rs`: Vulkan debug callback.
  - `mod.rs`: Vulkan instance/device/queue context.
- Split the large ignored GPU integration test into topical modules under
  `tests/correctness/`.
- Removed an unsafe raw descriptor-set pointer in executor recording by passing
  an immutable slot view into the submit-recording closure.
- Fixed `build.rs` shader tracking so changes to shared GLSL include files
  trigger shader recompilation.
- Corrected the stale pipeline selector comment: the large kernel can handle
  partial tiles manually, but auto-selection still prefers the small kernel for
  edge-heavy shapes.

## Added Tests

- Device preference parser rejects an empty `name:` filter.
- Matmul shape resolution rejects batch-stride overflow.
- Matmul FLOP accounting rejects `u64` overflow.
- CPU reference GEMM now explicitly tests B-side broadcasting.
- Ignored GPU suite now includes `manual_large_kernel_handles_partial_tiles`.

## File Size Check

Largest Rust files after the split:

| file | lines | note |
| --- | ---: | --- |
| `src/executor/mod.rs` | 410 | cohesive executor API and submit flow |
| `src/matmul.rs` | 375 | cohesive shape/stat API |
| `src/bench/commands.rs` | 333 | benchmark subcommands |
| `src/context/device.rs` | 320 | device selection and tests |
| `src/context/mod.rs` | 300 | Vulkan context setup/drop |

`scripts/bench_compare.py` is still a single 658-line Python CLI harness. It is
acceptable for now because it is one script-style component, but it should be
split into a small Python package if more framework backends or report formats
are added.

## Verification Commands

All commands below passed unless explicitly noted.

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
python3 -m py_compile scripts/bench_compare.py
cargo build --release --bin ml_bench
nix-shell --run 'target/release/ml_bench self-check'
nix-shell --run 'ML_DEVICE=discrete target/release/ml_bench self-check'
nix-shell --run 'cargo test --release --test correctness -- --ignored --test-threads=1'
```

Important runtime note:

```text
target/release/ml_bench self-check
```

fails outside the Nix shell with `failed to load Vulkan loader:
libvulkan.so.1`. Inside `nix-shell`, the same binary selects the NVIDIA RTX
3070 correctly.

GPU correctness result:

```text
23 ignored release integration tests passed on NVIDIA GeForce RTX 3070.
```

Vulkan device inventory:

| index | device | kind |
| ---: | --- | --- |
| 0 | NVIDIA GeForce RTX 3070 | discrete GPU |
| 1 | llvmpipe (LLVM 21.1.8, 256 bits) | CPU/software Vulkan |

## GEMM Benchmark Command

```bash
nix-shell --run 'python3 scripts/bench_compare.py --case-set extended --iters 5 --warmup 2 --torch-threads 1 --transfer-mb 64'
```

Full raw data and table report:

- `benchmarks/latest.json`
- `benchmarks/latest.md`

Environment:

- GPU: NVIDIA GeForce RTX 3070
- Vulkan: 1.4.329 on the selected GPU
- Tensor timestamps: enabled
- CPU framework threads: 1
- Cases: 11 extended FP32 GEMM shapes

Summary:

| comparison | result |
| --- | ---: |
| `tensor-ash` fastest measured backend | 11 / 11 cases |
| vs NumPy single-thread geomean | 29.6x |
| vs PyTorch CPU single-thread geomean | 30.2x |
| best `tensor-ash` throughput in this run | 7.858 TFLOPS (`square_1024`) |
| transfer upload | 9.608 GiB/s |
| transfer download | 10.180 GiB/s |

Skipped framework rows:

- PyTorch CUDA/cuBLAS: CUDA unavailable in this Python environment.
- JAX: module not installed.
- TensorFlow: module not installed.

## Local AI / Ollama Attempt

Ollama is installed. Local model inventory:

| model | size | notes |
| --- | ---: | --- |
| `qwen2.5:7b` | 4.7 GB | architecture `qwen2`, 7.6B parameters, Q4_K_M |
| `gemma4:latest` | 9.6 GB | installed, not benchmarked in this pass |

No Ollama models were running before the test.

Standard Ollama baseline command:

```bash
ollama run --verbose qwen2.5:7b "Return a numbered list from 1 to 50. No explanation." --keepalive 1m
```

Warm standard baseline result:

| metric | value |
| --- | ---: |
| total duration | 14.701 s |
| load duration | 93.866 ms |
| prompt eval | 44 tokens in 610.147 ms |
| prompt eval rate | 72.11 tokens/s |
| generation eval | 141 tokens in 13.890 s |
| generation eval rate | 10.15 tokens/s |

Backend integration status:

- `tensor-ash` is currently an explicit Rust/Vulkan tensor API.
- Ollama uses its own model/runtime backend path and cannot load this crate as a
  GEMM provider from configuration alone.
- A real standard-vs-`tensor-ash` Ollama comparison needs a new integration
  layer, most likely a C ABI plus ggml/Ollama backend implementation that maps
  model graph matmul calls to `tensor-ash`.
- Therefore the measured Ollama numbers above are standard Ollama only, not a
  `tensor-ash` backend run.

## Next Engineering Steps

1. Add a proper benchmarking crate or Criterion-style harness for kernel
   selector tuning.
2. Add optional CUDA-enabled PyTorch/cuBLAS benchmark coverage in an environment
   where CUDA Python packages are installed.
3. Split `scripts/bench_compare.py` into a package if framework coverage grows.
4. Design a C ABI / ggml backend boundary before attempting Ollama integration.
5. Continue shader work on skinny, wide, and small-K specialized kernels.
