# Changelog

## [Unreleased]

## v1.2.0 - 2026-06-14

Follow-up release on top of `v1.1.0` focused on closing more cuBLAS-loss
cases and cleaning up the experimental Stream-K / Split-K paths. Headline
wins: a register-tile (TM/TN) variant of the K64 kernel that unlocks +6%
on the K64-routed shape band, descriptor-set elision on the BDA hot path
(~2-4% wall-time on small/batched shapes), Stream-K DP-flat kernel quality
fix that closes a 12% glslang `OpSwitch` overhead by reusing the regular
`large_bda_v4` SPIR-V, and a hardware float32 atomicAdd in the (still
opt-in) Split-K kernel. Final geomean vs pure cuBLAS: **1.146x** across
26 showcase cases (up from 1.12x in v1.1.0), fastest on **14/26**.

### Added

- **Stream-K experimental pipeline.** Hybrid CUTLASS-style DP-flat bulk
  dispatch + persistent SK-tail with hardware `atomicAdd` from
  `VK_EXT_shader_atomic_float`. Two new `Executor` entry points:
  - `run_matmuls_stream_k(call)` — always route through Stream-K.
  - `run_matmuls_auto_stream_k(call, tail_fraction_max)` — gate by shape
    and fall back to `run_matmuls` when Stream-K wouldn't help.
  Restrictions: single matmul, batch == 1, aligned shapes
  (`M%128 == N%128 == K%32 == 0`), `accumulate == false`, device must
  expose `shaderBufferFloat32AtomicAdd`. Shipped behind an opt-in entry
  point — the standard `run_matmuls` path is unchanged.
- **Strict-aligned BDA_V4 kernels** (opt-in only, see below).
  `large_bda_v4_aligned` and `m128n64k64_bda_v4_aligned` strip the
  bounds-checked scalar load helpers and edge epilogue paths at source
  level; the SPIR-V contains only the LDG.E.128 / LDS.E.128 / FFMA hot
  path and the STG.E.128 epilogue. Still selectable via
  `ML_KERNEL=*_bda_v4_aligned` for shape-specific A/B work; **no longer
  auto-promoted** — see the Changed section below.
- **Descriptor-set elision on the BDA hot path.** BDA kernels carry A/B/C
  device addresses in push constants, so they don't need descriptor-bound
  SSBOs. A second push-constant-only `pipeline_layout_bda` short-circuits
  `update_descriptor_sets` + `cmd_bind_descriptor_sets` on every BDA
  dispatch. Wall-time wins of ~2-4% on small/batched shapes; TF/s metric
  is flat (the savings are host-side).
- **Hardware float32 atomicAdd in the Split-K kernel** via
  `VK_EXT_shader_atomic_float` (Ampere `RED.E.ADD.F32`). Replaces the
  emulated `atomicCompSwap` CAS loop that prior versions shipped.
  Split-K remains opt-in via `Executor::run_matmuls_split_k`; the
  hardware atomic makes the API correct + competitive on deep-K shapes
  if ever wired into auto.
- **`k64_bda_v4_tm8_tn4` register-tile variant.** Same shader as
  `k64_bda_v4` (BM=BN=64, BK=64, all v4 features) but with TM=8, TN=4
  and a 128-thread workgroup instead of 256. Halving the active thread
  count + doubling per-thread arithmetic hits a clean register-blocking
  sweet spot for K64-routed shapes (+6% on `medium_384`, +7.5% on
  `skinny_1024x128x512`, +6.4% on `wide_128x1024x512`, +4.0% on
  `tall_512x256x256`). Auto-selector promotes the K64 path to this
  variant. Companion experimental variants (`k64_bda_v4_tm4_tn8`,
  `k64_bda_v4_tm8_tn8`, `m128n64k64_bda_v4_tm8_tn8`,
  `m128n64k64_bda_v4_tm16_tn4`) ship as ML_KERNEL-selectable repro
  artifacts but lose on this device.
- **`VK_EXT_shader_atomic_float` plumbing** in `VulkanContext`: probe
  the device extension list + `PhysicalDeviceShaderAtomicFloatFeaturesEXT`
  for `shaderBufferFloat32AtomicAdd`, enable when present, expose
  through `ctx.shader_buffer_float32_atomic_add_enabled`.

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
- **Auto-selector no longer promotes the aligned variants.** Interleaved A/B
  sampling showed the source-stripped `*_bda_v4_aligned` kernels consistently
  measure 2-5% slower than their bounds-checked siblings on this device, so
  `maybe_to_aligned` is gone. The kernels stay built and ML_KERNEL-selectable
  for cross-device validation.
- **Auto-selector promotes the `K64` path to `k64_bda_v4_tm8_tn4`** (rather
  than `k64_bda_v4`) on devices that support `bufferDeviceAddress`. See the
  Added section for the magnitude on each routed shape.
- **Stream-K DP-flat kernel reuses unmodified `large_bda_v4` SPIR-V.** The
  prior standalone `matmul_streamk_dp_kernel.glsl` contained an early-out
  `return` that forced glslang to wrap `main()` in an OpSwitch structured
  control-flow merge — costing ~12% on Ampere. The hybrid schedule now
  rounds `dp_tiles_total` down to a multiple of `n_tiles` so DP-flat needs
  no return statement and runs the regular BDA_V4 kernel byte-for-byte.
  Deletes 219 lines of redundant shader; full Stream-K is now within 1-3%
  of `large_bda_v4` on aligned big shapes (was 7-25% behind).

### Performance

Latest `showcase` run on an RTX 3070 (`benchmarks/latest.md`), 30 iters /
10 warmups, FP32 throughout, `ML_PEAK_TFLOPS=20.32`:

| comparison | result |
| --- | ---: |
| `tensor-ash` fastest measured backend | 14 / 26 cases |
| vs pure cuBLAS (apples-to-apples) geomean | 1.146x |
| best `tensor-ash` throughput | 10.41 TFLOPS / 51.2% peak (`attn_qkv_1024x3072x512`) |
| best `tensor-ash` `square_1024` | 10.17 TFLOPS / 50.1% peak |
| pure cuBLAS on `attn_qkv_1024x3072x512` | 11.41 TFLOPS / 56.1% peak |
| pure cuBLAS on `square_1024` | 10.98 TFLOPS / 54.0% peak |
| median synchronous host overhead | ~0.022 ms / GEMM call |

Per-tile, the BDA path gains ~5-15% over the descriptor-bound original on
every tile we measured, and the BDA_V4 path gains another ~5-15% over plain
BDA. The descriptor-set elision shaves another ~2-4% wall time on small/
batched shapes (host-side savings invisible to TF/s). The K64 register-tile
change captures another +4-7% on the K64-routed shape band. The
auto-selector's overall promotion captures roughly +10-40% across the
showcase set relative to v1.0.0.

Two shapes that lost to cuBLAS in earlier v1.x snapshots now beat it:
`medium_384` (1.045x) and `tall_512x256x256` (essentially tied at 0.998x);
plus `skinny_1024x128x512` and `wide_128x1024x512` close from ~0.88x to
~0.93-0.98x. The largest remaining gaps to pure cuBLAS are on `medium_768`
(0.83x), `non_pow2_1023x1025x1027` (0.84x), `odd_255x257x263` (0.92x), and
`square_1024` (0.93x), where its hand-tuned kernels still show their edge —
the FP32 SIMT compiler's FMA throughput ceiling on GA104 appears to be the
structural limit, not selector quality. See `feedback_*_dead_end` notes in
session memory for the ruled-out levers (Stream-K, Split-K w/ hardware
atomic, CTA swizzle, subgroupShuffle).

### Verification

Validated locally on 2026-06-14 with:

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
