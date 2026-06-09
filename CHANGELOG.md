# Changelog

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
