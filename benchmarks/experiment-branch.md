# Experiment branch log — `experiment/model-inference`

Branched from `main@18d0f88` (v1.5.0-dev) on 2026-08-14. This file tracks every
major shift on this branch: what was tried, why, how it was verified, and
whether it stays. Read top-to-bottom for the narrative; each entry ends with a
**Direction check** stating where the evidence points next.

## Mandate

1. Experimental major shifts, benchmark-verified, on this branch (not main).
2. FP16 support "in the nicest way possible".
3. C ABI capable of driving major local AI models (llama-class decoders).
4. Continuous dedup: every redesign should slightly reduce LOC.
5. Architecture stays clean; research continues on the side.

## Hardware facts (queried 2026-08-14)

The RTX 3070 driver exposes everything the FP16 plan needs:

- `shaderFloat16 = true`, `VK_KHR_shader_float16_int8` — f16 arithmetic in shaders.
- `storageBuffer16BitAccess = true`, `VK_KHR_16bit_storage` — f16 SSBO loads/stores.
- `VK_KHR_cooperative_matrix` rev 2 (plus `VK_NV_cooperative_matrix2`) — tensor
  cores reachable from compute shaders.
- `shaderBFloat16Type/CooperativeMatrix = true` — bf16 is on the table later.

## Strategy

FP16 lands in two phases, each independently benchmarked:

- **Phase A — f16 storage, f32 accumulate.** Half the bytes moved. Decode-shape
  GEMV and outer-product ops are bandwidth-bound (row/col/outer already sit at
  ~90% of 448 GB/s), so f16 weights should approach 2x on exactly the shapes
  local-model decode lives on. Numerically safe: accumulation stays f32.
- **Phase B — cooperative-matrix compute.** Tensor-core f16×f16+f32 for the
  compute-bound prompt-processing shapes. Bigger ceiling (GA104 FP16 TC peak is
  ~2x FP32 FFMA), bigger risk; gated on Phase A's dtype plumbing.

Model ops (softmax, RMSNorm, SwiGLU elementwise, residual add) and the C ABI v2
ride on the same dtype-aware foundation; op-gap survey in flight to size the
minimal set for a llama-class decoder block.

## Entries

### 2026-08-14 — branch created; groundwork surveys launched

Committed the verified overnight v1.5.0-dev work to local `main` (18d0f88) so
the branch has a clean baseline to diff and bench against. Launched four
parallel survey agents: capi surface + conventions, kernel-extension seams,
dedup hotspots, model-op gap list. FP16 hardware capability confirmed (above).

**Direction check:** waiting on surveys before touching code; FP16 Phase A
design proceeds meanwhile since hardware support is confirmed.

### 2026-08-14 — step 1: dead-end feature deletion (−3,382 LOC)

**Why:** three feature stacks were measured dead ends in earlier sessions yet
remained fully wired and publicly exported: Stream-K (−6% to −75% on every
probed shape), the persistent-grid module (same structural atomic cost), and
atomic split-K (17–53% slower than the split-K2 replacement that superseded
it). Deleting them first shrinks the surface every later change (FP16 dtype
plumbing, new ops) has to keep consistent, and matches the standing rule that
redesigns reduce LOC.

**What:** deleted 16 files (streamk{,_exec,_schedule}.rs, persistent/,
splitk.rs, 7 shaders, streamk tests/example paths); `default_num_k_splits`
moved into splitk2.rs (split-K2's auto path uses it); dead executor re-exports
demoted. Split-K2 untouched. `ResolvedMatmul::split_k_push_constants` kept —
split-K2 stage 1 reuses it.

**Verified:** 68/68 GPU correctness tests (was 74; −6 streamk tests),
41 unit tests + doctest, clippy `-D warnings` clean. Bench spot-check
(sq2048/sq4096/tall/batched/tiny/row/deepk) — GPU medians identical to the
pre-deletion session-end artifact; routes unchanged. Diff: 27 files,
+104/−3494.

**Direction check:** codebase is 14.6k LOC (was 18.0k). Next: FP16 Phase A on
the now-smaller kernel surface.

### 2026-08-14 — FP16 research findings (recorded before implementation)

Key facts from the toolchain/driver research pass (full brief in session
notes; sources: Khronos GLSL_KHR_cooperative_matrix spec, NVIDIA
vk_cooperative_matrix_perf, llama.cpp Vulkan backend, Vulkanised 2025):

- Repo glslc (shaderc 2026.1, glslang 16.2) compiles both
  `GL_EXT_shader_explicit_arithmetic_types_float16` buffer_reference blocks
  and `GL_KHR_cooperative_matrix` at `--target-env=vulkan1.2` — no build
  changes needed. Verified by compiling test shaders locally.
- f16 through BDA still requires `storageBuffer16BitAccess` (SPIR-V emits
  `OpCapability StorageBuffer16BitAccess` for PhysicalStorageBuffer too) plus
  `shaderFloat16` (Vulkan11/12Features). Driver has both.
- Ampere FFMA has no f16-source form: H2F conversions are separate, half-rate
  instructions. **Convert at shared-memory staging time, store f32 tiles** —
  inner loop stays the existing proven FFMA pipeline; the 2x global-bandwidth
  win is preserved. Load f16 globals as `uvec4` (8 halves, LDG.128), not
  `f16vec4` (only LDG.64).
- f16 accumulate is rejected permanently: 65504 max-normal overflows K≈1000
  dot products (llama.cpp shipped this bug, ggml #18969); f32 accumulate
  keeps the determinism/accuracy story.
- Coopmat Phase B: KHR coopmat1 first (llama.cpp reports coopmat1 ≈ coopmat2
  perf on NVIDIA for plain GEMM; on a 3090 Vulkan+coopmat beat CUDA on
  pp512). Needs `vulkanMemoryModel` enabled (glslang emits `OpMemoryModel
  Vulkan` for coopmat shaders), subgroup-uniform control flow (no divergent
  edge tricks), shared-staging stores for edges/epilogues (no (row,col)
  info in fragments; `cooperativeMatrixRobustBufferAccess=false` on this
  driver so OOB is UB). Query (M,N,K) configs via ash
  `khr::cooperative_matrix::Instance` at runtime; expect 16x8x8/16x8x16/
  16x16x16 f16·f16+f32 subgroup-scope. Env-var kill-switch pattern
  (llama.cpp `GGML_VK_DISABLE_COOPMAT`) worth copying.
