# Changelog

## [Unreleased]

Post-`v1.0.0` optimization pass focused on closing the gap to pure cuBLAS on
the RTX 3070 baseline. The two big structural changes are a data-driven
kernel registry and end-to-end Vulkan 1.2 `bufferDeviceAddress`.

### Added

- **Data-driven kernel pipeline.** All compiled tile variants now live in a
  single `KERNEL_SPECS` table in `src/pipeline/types.rs`. The pipeline
  builder, selector, and `ML_KERNEL=...` parser all read from it, so adding a
  new tile is a two-line registry entry plus a `.comp` wrapper.
- **Vulkan 1.2 `bufferDeviceAddress` support.** Plumbed through
  `VulkanContext` (feature enable + buffer-usage flag), `Buffer` (device
  address capture), and the `MatmulPushConstants` layout (`a_ptr`, `b_ptr`,
  `c_ptr` u64s).
- **BDA kernel family** (`GL_EXT_buffer_reference`). Compiles A/B/C global
  loads to `LDG.E.128` instead of four 32-bit transactions. Variants for the
  seven primary tile shapes: `large_bda`, `small_bda`, `m64n128_bda`,
  `m128n64_bda`, `m128n64k64_bda`, `m64n32_bda`, `k64_bda`.
- **BDA_V4 kernel family**. Adds `shared uvec4` Bs staging on top of BDA so
  shared reads compile to `LDS.E.128`. Variants: `large_bda_v4`,
  `small_bda_v4`, `m64n128_bda_v4`, `m128n64_bda_v4`, `m128n64k64_bda_v4`,
  `k64_bda_v4`. (The TN=2 `m64n32` tile has no V4 path — LDS.128 over a
  2-column stride is non-sensical.)
- **Pure cuBLAS C++ benchmark binary** (`benchmarks/cublas_bench/`,
  `cublas_bench.cu` + `Makefile`). Calls cuBLAS through its C API directly
  with `CUBLAS_PEDANTIC_MATH` to force real FP32 (no silent TF32 fallback)
  and times via CUDA events. This is the apples-to-apples GPU baseline for
  `tensor-ash`, isolating the kernel from PyTorch wrapper overhead.
- **`scripts/bench_compare.py`:** `cublas_pure` backend row driven by the C++
  binary; `% peak` column derived from `ML_PEAK_TFLOPS` (default 20.32 for
  RTX 3070 FP32); a new `showcase` case set covering tiny-batch (`tiny_b32_128`,
  `tiny_b16_192`, `tiny_b8_192`, `batched_16x128`, `batched_32x64`,
  `batched_64x128`, `batched_128x64`), medium squares (`medium_384`,
  `medium_768`), attention-style projections (`attn_proj_2048x512x512`,
  `attn_proj_512x2048x512`, `attn_qkv_1024x3072x512`), and non-power-of-two
  shapes (`non_pow2_513x515x517`, `non_pow2_1023x1025x1027`).

### Changed

- **Auto-selector promotes BDA picks to BDA_V4** when the device supports
  `bufferDeviceAddress`, with a plain BDA fallback for the TN=2 `m64n32`
  tile. Explicit `ML_KERNEL=...` selections are honored verbatim so per-tile
  A/B tuning still works.
- **Auto-selector shape rules retuned** on top of the BDA promotion:
  - medium rectangular shapes with `min_mn ∈ [512, 2048)` and aspect ≤ 5x
    now route to `m128n64k64` (a clean win on `medium_768`, `non_pow2_1023*`,
    `attn_proj_2048x512x512`);
  - mid-square shapes with `min_mn ∈ [192, 480]` and `k ≥ 256` now route to
    `k64` (~15% win on `medium_384`);
  - the `m64n32` rule for tiny near-square GEMMs is now skipped when
    `batch ≥ 8`, since the extra workgroups make its lower per-WG arithmetic
    intensity dominate the edge-waste savings (the `64x64` small kernel wins
    on `batched_8x256`).
- Pipeline construction reads tile dimensions out of `KERNEL_SPECS` instead
  of duplicating constants in the selector.

### Performance

Latest `showcase` run on an RTX 3070 (`benchmarks/latest.md`), 30 iters /
10 warmups, FP32 throughout, `ML_PEAK_TFLOPS=20.32`:

| comparison | result |
| --- | ---: |
| `tensor-ash` fastest measured backend | 13 / 26 cases |
| vs pure cuBLAS (apples-to-apples) geomean | 1.11x |
| vs PyTorch CUDA / cuBLAS geomean | 1.32x |
| vs CuPy CUDA / cuBLAS geomean | 2.59x |
| vs NumPy single-thread geomean | 47.56x |
| vs PyTorch CPU single-thread geomean | 45.33x |
| best `tensor-ash` throughput | 10.18 TFLOPS / 50.1% peak (`square_1024`) |
| best `tensor-ash` `attn_qkv_1024x3072x512` | 10.41 TFLOPS / 51.2% peak |
| pure cuBLAS on `square_1024` | 10.92 TFLOPS / 53.8% peak |
| pure cuBLAS on `attn_qkv_1024x3072x512` | 11.44 TFLOPS / 56.3% peak |
| median synchronous host overhead | 0.020 ms / GEMM call |
| transfer upload / download | 9.04 / 9.48 GiB/s |

Per-tile, the BDA path gains ~5-15% over the descriptor-bound original on
every tile we measured, and the BDA_V4 path gains another ~5-15% over plain
BDA. The auto-selector's promotion captures roughly +10-30% across the
showcase set "for free" relative to v1.0.0.

The largest remaining gap to pure cuBLAS is on `attn_qkv_1024x3072x512`
(51.2% vs 56.3% peak) and `square_1024` (50.1% vs 53.8% peak), where its
hand-tuned kernels still show their edge.

### Verification

Validated locally on 2026-06-11 with:

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p tensor-ash-capi
nix-shell --run 'cargo test --release --test correctness -- --ignored --test-threads=1'
nix-shell --run 'env LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH /tmp/tensor_ash_c_smoke'
nix develop .#benchmark --command bash -lc \
  '.venv-bench/bin/python scripts/bench_compare.py --case-set showcase --iters 30 --warmup 10 --torch-threads 1 --transfer-mb 64'
```

GPU correctness suite passes on NVIDIA GeForce RTX 3070; the C smoke test
returns the expected `58, 64, 139, 154` GEMM result.

### Known Limitations

- The C ABI is a callable GEMM interface, not an Ollama or ggml backend ABI.
- A real Ollama acceleration comparison still requires a ggml/Ollama adapter
  that maps model graph matmul calls into the C API.
- BDA / BDA_V4 kernels require Vulkan 1.2 `bufferDeviceAddress`. On devices
  without that feature, the selector falls back to the descriptor-bound
  variants.

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
