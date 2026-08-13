# Changelog

## [Unreleased]

## v1.4.1 - 2026-08-13

Measured-performance and correctness patch release.

### Added

- `ml_bench cases` benchmarks many labeled shapes in one Vulkan process,
  avoiding repeated pipeline startup and GPU-clock perturbation. Timed results
  retain paired wall/GPU samples and report sample counts, minimum, median,
  nearest-rank p95, host overhead, wall throughput, and the selected
  kernel/tile/reduction route.
- Timed GEMMs now validate a fixed set of output elements after measurement so
  a fast but incorrect kernel cannot be recorded as a benchmark win.
- Cross-backend JSON schema v2 records timing scope/statistic, git state,
  device/driver information, route metadata, and matching median distributions
  for tensor-ash and cuBLAS. A compact regression set covers the four measured
  worst shapes plus tile boundaries, batching, deep K, and both GEMV axes.
- Addressing, alignment, aliasing, split-K, timestamp-wrap, and dispatch-limit
  regression coverage, including supported strict-aligned happy paths.

### Changed

- Automatic kernel selection accounts for the whole batched grid and compares
  actual padded work at tile boundaries. On an RTX 3070 this raises median
  throughput by about 20% for batch-8 1024-cubed GEMM, 18% for batch-32
  512-cubed GEMM, and 27% for the 4095-cubed edge case.
- Runtime tuning filters irrelevant/invalid candidates, balances measurement
  order, uses median samples consistently, releases its slot before probing
  split-K2, and reports the actual selected reduction route.
- Pure-BDA submissions skip descriptor-update allocation and selection work.

### Fixed

- Strict aligned kernels, including the static V3 kernel's special K=16
  requirement, reject unsupported dimensions instead of permitting out-of-
  bounds shader access or loop underflow.
- Matmul resolution rejects rank-2 and batched layouts whose shader-visible
  element offsets exceed 32-bit addressing, while accepting the exact valid
  boundary.
- Output tensors may no longer alias either GEMM input; split-K2 scratch and
  split counts are range checked; one-split calls fall back before optional
  feature/pipeline setup; Stream-K validates the actual dispatch axes.
- Shader full-tile and K-strip predicates avoid 32-bit arithmetic overflow,
  and timestamp deltas honor the device's valid counter width.

## v1.4.0 - 2026-08-06

### Added

- `row_bda`, a warp-sized row/GEMV kernel for large batches of `M=1`
  products. The automatic selector uses it on BDA-capable devices instead of
  dispatching mostly-empty 64x64 GEMM tiles. It retains arbitrary-M
  correctness, batch broadcasting, alpha/accumulate behavior, and every fused
  epilogue. Dedicated Vulkan tests cover tails, broadcasting, multiple rows,
  batched bias, activation, and residual addition.

### Changed

- Batched private-network layers now approach memory-bandwidth limits on an
  RTX 3070. Across the six shapes used by town-hall at batch 10,000, measured
  GPU time improved by 2.4x to 5.8x.
- Large implementation files were split by responsibility: executor dispatch,
  validation, transfer, submission, tuning, reductions, Stream-K scheduling,
  and recording are separate modules; matmul API types are separate from shape
  resolution; pipeline identity is generated from one catalog declaration;
  persistent execution/resource setup and C lifecycle/tensor/matmul exports are
  likewise isolated.
- `ml_bench` moved out of the core package into the non-published `ml-bench`
  workspace package. Run it with `cargo run --release -p ml-bench -- ...`;
  release builds still produce `target/release/ml_bench`.
- Deterministic fixtures and CPU reference math moved from the public
  `tensor_ash::testing` module to the non-published
  `tensor-ash-test-support` workspace package shared by integration tests and
  `ml-bench`.
- The cross-library Python CLI is now a small compatibility facade over
  backend adapters, data models/case sets, and report generation modules.
- `ExecutorConfig` and `Executor::new_with_config` provide explicit slot,
  submission-limit, and tuning policy for library callers; the existing
  constructor remains compatible with `ML_TUNE`.
- Raw `Buffer`/`Tensor` fields are private. Read-only Vulkan interop is exposed
  through accessors while ownership stays with the wrapper.
- All workspace packages now share version `1.4.0` and Rust edition `2024`.

### Fixed

- `Executor` construction and tensor operations reject resources owned by a
  different `VulkanContext`, including through the C ABI, before invalid
  cross-device handles can reach Vulkan.
- `Buffer::new` reports a normal error for zero-sized allocations instead of
  panicking.
- `PersistentMatmul::bench` derives FLOP counts from the validated matmul
  shape instead of accepting a caller-supplied value that could misreport
  throughput.
- Release builds use unwinding so the C ABI's `catch_unwind` boundary can
  convert Rust panics into `ta_last_error()` instead of aborting the process.
- All Vulkan-dependent Stream-K correctness cases are ignored during ordinary
  CPU-only `cargo test --workspace` runs.

### Removed

- Unused core dependencies on `rayon` and `thiserror`; `env_logger` is now a
  development/tool dependency rather than part of the runtime library graph.

## v1.3.0 - 2026-07-15

VkSplat-inspired systems release: the library stops being "a pile of
GEMM shaders + a hand-tuned selector" and gains measured runtime
adaptation, fused epilogues, dependent-graph submission, and a
deterministic two-stage split-K.  Design notes: the guiding ideas are
work-elimination before work-optimization, accumulate-locally-then-
commit, fuse-so-intermediates-never-touch-VRAM, and measure-instead-of-
model (per the VkSplat Eurographics'26 write-up + `IDEAS.md` items
14/65/79/95/109/120).

### Added

- **Measured kernel auto-tuner with a persistent per-device store**
  (`src/pipeline/tuning.rs`).  With `ML_TUNE=1`, the first plain
  single-call submission of a new shape measures every BDA-family
  kernel on the *caller's real problem* (interleaved rounds, min GPU
  timestamp per candidate, 1 discarded warmup round, >2% margin
  required to unseat the heuristic prior) and records the winner in
  `$XDG_CACHE_HOME/tensor-ash/tuned_kernels_v<vendor>_<device>.txt`.
  The store header pins driver version + an FNV hash of every embedded
  SPIR-V, so driver updates or shader edits invalidate it wholesale.
  Persisted winners are loaded and applied on every run (tuning off or
  on); `ML_KERNEL=` overrides everything.  `Executor::tune_shape()`
  pre-warms shapes explicitly.  Measured wins on the RTX 3070 showcase
  set vs the hand heuristic: batched_2x512 +19%, batched_8x256 +19%,
  batched_4x256 +14%, small_k_1024x1024x64 +11%, square_512 +8%,
  non_pow2_513 +8%, attn_qkv/attn_proj +3-4% — mostly by discovering
  that `k64_bda_v4_tm8_tn4` generalizes far beyond the shapes the
  hand-written rules gave it.
- **Fused GEMM epilogues** (`MatmulOp` + `Epilogue`): per-column bias,
  ReLU / SiLU / tanh-GELU activations, and a binary second operand —
  residual `+ beta*D` or SwiGLU-style gating `* D` — applied while the
  output tile is still in registers.  Implemented as three new
  specialization constants (IDs 4..6) in the BDA and BDA_V4 kernel
  bodies; the zero-epilogue pipelines are bit-identical to before and
  epilogue pipeline variants compile lazily on first use through the
  persistent pipeline cache.  Push constants grew to 80 bytes
  (`d_ptr`, `bias_ptr`, `beta`); descriptor-bound kernels reject
  epilogues.  New entry points `Executor::run_ops` /
  `Executor::run_op_graph`.
- **Dependency-aware graph executor** (`Executor::run_matmul_graph` /
  `run_op_graph`): records a dependent chain of matmuls in one command
  buffer, tracking per-buffer read/write hazards (RAW/WAW/WAR,
  epilogue operands included) and inserting compute→compute barriers
  only where needed — one submit, one fence wait per chain instead of
  one per dependency level.  Independent ops between barriers still
  overlap exactly as in `run_matmuls`.
- **Two-stage split-K** (`Executor::run_matmuls_split_k2` +
  `shaders/matmul_splitk2_kernel.glsl` / `matmul_f32_splitk2_reduce.comp`):
  stage 1 plain-stores per-split partial planes into slot-local scratch
  (no atomics, no `VK_EXT_shader_atomic_float` requirement), stage 2
  sums the planes into C.  Deterministic — bit-stable across runs,
  unlike the atomic split-K.  Measured on RTX 3070: 64x64x8192 16.5x
  over DP (0.413 → 0.025 ms), 128x128x8192 4.6x, 256x256x4096 1.8x,
  128x128x2048 2.1x (and 30% over *atomic* split-K).  The tuner probes
  split counts {4..64} on deep-K low-tile shapes (K >= 1024, <= 48
  128x128-tiles) and records `splitk2=N` in the store; auto dispatch
  routes single plain calls — and graph ops, inline with their own
  scratch regions — through it.
- Correctness suites for all of the above: `tests/correctness/graph.rs`
  (chains, diamonds, accumulate-into-prior-output, bit-identical vs
  sequential), `epilogue.rs` (10 cases across interior-vec4 and edge
  store paths + hazard tracking of epilogue operands + shape
  validation), `splitk2.rs` (deep-K, uneven splits, scalar reduce path,
  batching, determinism), `tuning.rs`.
- `examples/bench_splitk2.rs`: DP vs atomic split-K vs two-stage
  split-K probe.

### Changed

- **`examples/synth_llama_layer.rs`** now compares dependency-ordered
  split submissions against a single `run_op_graph` submission with the
  SwiGLU (`silu(x@W_gate) * up`) fused into the gate GEMM epilogue.
  This also fixes a latent hazard in the old example (`down` consumed
  `up`'s output inside one barrier-less `run_matmuls`).  End-to-end at
  M=1 on the RTX 3070: 907 → ~1235 tok/s (+36%) from tuned kernels +
  split-K2 routing + single-submit graphs, now *including* the SwiGLU
  that the old example never computed.
- `MatmulPushConstants` grew from 56 to 80 bytes (three epilogue
  fields).  The GLSL PC blocks of the BDA/BDA_V4 bodies grew to match;
  descriptor-bound and split-K/stream-K shaders are unchanged (their
  blocks remain prefixes of the pushed range).
- `KERNEL_SPECS` kernels report `supports_epilogue()` (true for
  bounds-checked BDA-family bodies).

### Notes

- Tuning measurements run on the caller's tensors (every candidate
  fully overwrites C with `accumulate=false`), so the first tuned call
  still returns correct results — it just takes a few extra
  milliseconds.  One-shot shapes should leave `ML_TUNE` off and rely on
  the heuristic or a pre-populated store.
- The C ABI is unchanged; epilogues/graphs are not yet exposed through
  `tensor-ash-capi`.

## v1.2.1 - 2026-06-14

Hygiene patch on top of v1.2.0. No perf changes, no public-API changes.

### Changed
- **Removed dead `persistent_v4` branch in `src/bench/mod.rs`.** The
  bench was building a `PersistentMatmul` whenever `ML_KERNEL=persistent_v4`
  was set, then discarding it via `let _ = &persistent;` because
  `commands::single_persistent` was never wired up. The branch now just
  routes through the regular `KernelSelection::from_env()` path. Users
  who had `ML_KERNEL=persistent_v4` will now get a clear parse error
  instead of silently running the default kernel; the public
  `PersistentMatmul` re-export in `lib.rs` is unchanged (its removal is
  reserved for a future major bump).
- **Remote URL**: tracked origin updated from `gusahlg/ml-project.git`
  to `gusahlg/tensor-ash.git` (GitHub had been serving a redirect).

### Removed
- Two orphan experimental shader files with no Rust references:
  `shaders/matmul_f32_strassen.comp` and
  `shaders/matmul_strassen_kernel.glsl`.

### Verified
- `cargo build --release`, `--examples`, `-p tensor-ash-capi` clean.
- `cargo test --release`, `--doc`, and `--test correctness -- --ignored`
  (29 GPU tests) all pass.
- `cargo clippy --release --all-targets` clean.
- `cargo fmt --check` clean.

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
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p tensor-ash-capi
nix-shell --run 'cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1'
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
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
python3 scripts/bench_compare_test.py
cargo build --release -p ml-bench
nix-shell --run 'ML_DEVICE=discrete target/release/ml_bench self-check'
nix-shell --run 'cargo test --release -p tensor-ash --test correctness -- --ignored --test-threads=1'
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
