# Changelog

## Unreleased

Post-`v1.0.0` development changes.

### Added

- Added `tensor-ash-capi`, a dedicated workspace crate that builds
  `libtensor_ash.so` and `libtensor_ash.a`.
- Added `include/tensor_ash.h` with opaque C handles for context, executor, and
  tensors.
- Added C entry points for context/executor creation, tensor allocation,
  upload/download, single GEMM, batched GEMM, version reporting, and per-thread
  error reporting.
- Added `examples/c_smoke.c`, which verifies a 2x3 by 3x2 GEMM through the C
  API.
- Added C ABI unit tests for version strings, null-handle error reporting, and
  null destroy no-ops.
- Added `flake.nix`/`flake.lock` with a benchmark shell that provides Rust,
  Vulkan, CUDA tooling, Python, and `uv`.
- Added `requirements-benchmark.txt` for CUDA-backed PyTorch and CuPy
  benchmark wheels.
- Added five additional Vulkan GEMM kernel variants: `m64n128`, `m128n64`,
  `m128n64k64`, `m64n32`, and `k64`.
- Added `scripts/tune_kernels.py` for report-free manual kernel selector
  tuning.

### Changed

- Split the C ABI wrapper into `api`, `error`, `handles`, and `types` modules
  so each file has one clear responsibility.
- Extended `scripts/bench_compare.py` with CuPy CUDA/cuBLAS timing,
  `nvidia-smi` metadata, GPU/CPU framework skip flags, and report analysis
  that separates actual GPU framework rows from CPU baselines.
- Extended `scripts/bench_compare.py` with `tensor-ash` wall-time and
  host-overhead reporting, and explicitly pins PyTorch CUDA matmul precision
  to FP32/highest for fair comparison.
- Updated the automatic kernel selector to use batch-aware measured rules for
  the new variants while preserving the existing small/large paths for batched
  cases.
- Added a one-call descriptor update fast path in the executor to avoid heap
  allocation for the common single-GEMM submit path.
- Added a `K_MULTIPLE` shader specialization constant so tile-aligned K
  dimensions can skip the K-tail branch.
- Moved the repository to a Cargo workspace while keeping the Rust crate as the
  core library and the C ABI as a wrapper crate.

### Verification

Validated locally on 2026-06-09 with:

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p tensor-ash-capi
nix-shell --run 'cargo test --release --test correctness -- --ignored --test-threads=1'
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
```

The release GPU correctness suite passed 23/23 ignored integration tests on an
NVIDIA GeForce RTX 3070. The C smoke test returned the expected
`58, 64, 139, 154` GEMM result.

### Benchmark Snapshot

The latest recorded extended benchmark uses 40 iterations and 10 warmups on an
RTX 3070. PyTorch CUDA/cuBLAS and CuPy CUDA/cuBLAS rows now run successfully.
`tensor-ash` was fastest on 7/11 cases, reached 8.987 TFLOPS on
`square_1024`, measured 1.10x geometric-mean throughput versus PyTorch CUDA,
2.51x versus CuPy CUDA, and about 32x versus the single-threaded NumPy and
PyTorch CPU baselines. Median synchronous host/submission overhead was
0.019 ms per GEMM call.

### Known Limitations

- The C ABI is a callable GEMM interface, not an Ollama or ggml backend ABI.
- A real Ollama acceleration comparison still requires a ggml/Ollama adapter
  that maps model graph matmul calls into the C API.

## v1.0.0 - 2026-06-09

Initial stable release of `tensor-ash` as a Rust/Vulkan FP32 GEMM component.

### Highlights

- FP32 rank-2 and rank-3 GEMM with batch broadcasting.
- Two shader tile variants with automatic or manual kernel selection.
- Thread-safe executor with reusable submission slots.
- GPU timestamp timing and TFLOPS reporting.
- Device selection by kind, index, or name substring.
- Persistent device-qualified Vulkan pipeline cache.
- Shared GLSL kernel body for large and small tile wrappers.
- CPU and GPU correctness coverage, including ignored release GPU integration
  tests.
- Cross-framework benchmark harness with NumPy, PyTorch CPU, optional PyTorch
  CUDA, optional JAX, optional TensorFlow, and transfer-bandwidth reporting.

### Verification

Validated on 2026-06-09 with:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
python3 -m py_compile scripts/bench_compare.py
cargo build --release --bin ml_bench
nix-shell --run 'ML_DEVICE=discrete target/release/ml_bench self-check'
nix-shell --run 'cargo test --release --test correctness -- --ignored --test-threads=1'
```

The release GPU correctness suite passed 23/23 ignored integration tests on an
NVIDIA GeForce RTX 3070.

### Benchmark Snapshot

The latest local benchmark report is in `benchmarks/latest.md`; raw data is in
`benchmarks/latest.json`. On the RTX 3070 benchmark run, `tensor-ash` was the
fastest measured backend on 11/11 shared GEMM cases versus single-threaded
NumPy and PyTorch CPU baselines.

### Known Limitations

- Current API is Rust/Vulkan-focused and does not expose a C ABI.
- Not yet integrated as a backend for Ollama, ggml, PyTorch, TensorFlow, or
  other inference runtimes.
- Current math scope is FP32 GEMM only.
- CUDA/cuBLAS comparison depends on a Python environment with CUDA-enabled
  PyTorch, which was not available in the recorded local benchmark.
