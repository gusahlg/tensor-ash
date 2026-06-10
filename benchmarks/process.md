# tensor-ash Refactor, Verification, and Benchmark Process

Generated: 2026-06-10

## Scope

This report records the maintainability audit, post-`v1.0.0` optimization
work, C ABI port, local verification, cross-framework GEMM benchmark, and
Ollama backend attempt.

The project now exposes two related surfaces:

- `tensor-ash`: the Rust/Vulkan FP32 GEMM library.
- `tensor-ash-capi`: a C ABI wrapper that builds `libtensor_ash.so` and
  `libtensor_ash.a`.

The C ABI is a callable GEMM layer. It is not yet an Ollama or ggml backend
implementation, so Ollama cannot use it as a model runtime backend without
additional adapter work.

## Refactor Summary

Earlier maintainability work split the largest Rust files into cohesive
modules:

- `src/bench/`: benchmark command parsing, cases, environment, and reporting.
- `src/executor/`: public executor API, per-submit slots, and command
  recording.
- `src/pipeline/`: pipeline types, kernel selector, and Vulkan pipeline
  creation.
- `src/context/`: device selection, debug callback, pipeline cache paths, and
  Vulkan context setup.
- `tests/correctness/`: topical ignored GPU integration tests.

This pass added and split the C ABI:

- `capi/src/api.rs`: exported `ta_*` functions and C ABI tests.
- `capi/src/error.rs`: per-thread error string and panic/error barriers.
- `capi/src/handles.rs`: opaque C handle wrappers.
- `capi/src/types.rs`: C-compatible public structs.
- `include/tensor_ash.h`: public C header.
- `examples/c_smoke.c`: C smoke test.

The Cargo package is now a workspace with the Rust crate as the core package
and `tensor-ash-capi` as the C ABI wrapper crate.

## Performance Changes

Two low-risk hot-path changes were made:

- Descriptor updates now use a stack-allocated fast path for the common case of
  one descriptor set and one GEMM call. The previous path always allocated
  vectors sized for batched submissions.
- The shader now has a `K_MULTIPLE` specialization constant. When the host
  knows `K` is a multiple of the kernel K tile, the selected pipeline can fold
  out the K-tail branch and modulo test.
- Three additional manual/tuned kernels were added after the CUDA comparison:
  `m64n128`, `m128n64`, and `k64`. The auto selector now uses batch-aware
  measured rules so the new variants improve batch-1 medium/skinny shapes
  without regressing the batched benchmark cases.

The `K_MULTIPLE` specialization increased the number of precompiled variants
per kernel from 8 to 16. The tradeoff is more pipeline variants for less branch
work in aligned GEMM calls. With five kernel shapes, each run now creates up to
80 specialized compute pipelines, amortized by the persistent Vulkan pipeline
cache.

## Added Tests

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
| `scripts/bench_compare.py` | 866 | cross-framework benchmark/report CLI; next split target |
| `src/executor/mod.rs` | 410 | cohesive executor API and submit flow |
| `src/matmul.rs` | 375 | cohesive shape/stat API |
| `capi/src/api.rs` | 353 | exported C ABI functions |
| `src/bench/commands.rs` | 330 | benchmark subcommands |
| `src/context/device.rs` | 320 | device selection and tests |
| `src/context/mod.rs` | 300 | Vulkan context setup/drop |

`scripts/bench_compare.py` is now the largest file. It is still acceptable as a
single script-style component for this pass, but it should be split into a
small Python package before adding more framework backends or report formats.

## Verification Commands

All commands below passed unless explicitly noted.

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p tensor-ash-capi
cc -Iinclude examples/c_smoke.c -Ltarget/release -ltensor_ash \
  -Wl,-rpath,/home/gusahlg/repos/ml_project/target/release \
  -o /tmp/tensor_ash_c_smoke
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
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

Extended benchmark command:

```bash
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set extended --iters 20 --warmup 5 --torch-threads 1 --transfer-mb 64'
```

Full raw data and table report:

- `benchmarks/latest.json`
- `benchmarks/latest.md`

Environment:

- GPU: NVIDIA GeForce RTX 3070
- Driver: 595.71.05
- Vulkan: 1.4.329 on the selected GPU
- Tensor timestamps: enabled
- CUDA Python rows: PyTorch 2.11.0+cu128 and CuPy 13.6.0
- CPU framework threads: 1
- Cases: 11 extended FP32 GEMM shapes
- Iterations: 40
- Warmups: 10

Summary:

| comparison | result |
| --- | ---: |
| `tensor-ash` fastest measured backend | 7 / 11 cases |
| vs PyTorch CUDA/cuBLAS geomean | 1.10x |
| vs CuPy CUDA/cuBLAS geomean | 2.51x |
| vs NumPy single-thread geomean | 31.6x |
| vs PyTorch CPU single-thread geomean | 31.9x |
| best `tensor-ash` throughput in this run | 8.987 TFLOPS (`square_1024`) |
| median synchronous host overhead | 0.019 ms |
| transfer upload | 9.771 GiB/s |
| transfer download | 10.335 GiB/s |

Selected GPU results:

| case | `tensor-ash` TFLOPS | PyTorch CUDA TFLOPS | CuPy CUDA TFLOPS |
| --- | ---: | ---: | ---: |
| `square_256` | 2.731 | 2.320 | 0.670 |
| `square_512` | 5.891 | 6.732 | 3.551 |
| `batched_4x256` | 5.185 | 3.154 | 1.669 |
| `tall_512x256x256` | 4.088 | 3.647 | 1.241 |
| `odd_255x257x263` | 2.352 | 2.263 | 0.697 |
| `square_1024` | 8.987 | 10.760 | 9.152 |
| `batched_2x512` | 7.365 | 7.167 | 4.772 |
| `skinny_1024x128x512` | 4.660 | 4.816 | 2.143 |
| `wide_128x1024x512` | 4.697 | 4.860 | 2.174 |
| `small_k_1024x1024x64` | 5.191 | 5.029 | 2.260 |

Performance note:

- PyTorch CUDA/cuBLAS is still strongest on large square GEMMs; it leads
  `square_1024` by about 1.2x and leads the 512 square case.
- The tuned selector now beats PyTorch CUDA on geometric mean for the extended
  set by improving small, medium, odd, tall, batched, and small-K cases.
- Median CPU/submission overhead is about 19 us per synchronous call. The
  reported TFLOPS uses GPU timestamps, so the remaining large-square gap is
  shader efficiency rather than upload/download or host timing.

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
  OLLAMA_LLM_LIBRARY=/home/gusahlg/repos/ml_project/target/release/libtensor_ash.so \
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
  OLLAMA_LLM_LIBRARY=/home/gusahlg/repos/ml_project/target/release/libtensor_ash.so \
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

## Next Engineering Steps

1. Implement a ggml/Ollama backend adapter over the C ABI if Ollama integration
   remains the priority.
2. Add a benchmark focused on C ABI call overhead and batched-call throughput.
3. Continue shader work on large square GEMMs and the remaining skinny/wide
   CUDA deltas.
4. Keep PyTorch CUDA and CuPy CUDA rows in the standard local benchmark before
   accepting performance-sensitive shader changes.
5. Split `scripts/bench_compare.py` into a package if framework coverage grows.
