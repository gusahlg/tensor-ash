# Coopmat tile selection, SM-count-aware (campaign checklist item 3)

Branch `campaign/coopmat-sm` (base `exp/grokmerge` = 7cfca58).  GPUs:
RTX 4060 (AD107, 24 SM, 24 MiB L2, 272 GB/s, driver 595.80, louise-pc)
and RTX 3070 (GA104, 46 SM, 4 MiB L2, 448 GB/s, driver 595.99, desktop).
Raw per-batch CSVs (`ml_bench ab`, `ML_OUTPUT=csv`) and `nvidia-smi`
clock logs are in [`coopmat-sm-data/`](coopmat-sm-data/).

## Decisions

1. **128x64 replaces 128x128 as the large-grid tile** (NEXT.md open
   question 1).  The 3070 4096³ "veto" is a tie: 128x128 34.38 vs
   128x64 34.38 TF/s (same interleaved run).  On the 3070 128x64 is
   never more than 0.6% behind 128x128 on any of 77 shapes and up to
   +56% ahead; on the 4060 it trails only on 4096³ (−3.6%, reproduced)
   and N=32000 at M=768/1024 (−1.9/−1.2%).
2. **Tile choice is SM-count-aware** (question 2).  The old
   `tiles_128 >= 96` threshold is 2 waves on 46 SMs but not on 24; e.g.
   f16w 768x2048x2048: 3070 64x64 31.5 / 128x64 26.7 / 128x128 21.1
   TF/s (old auto = 128x128 = 0.67x of best), 4060 128x64 25.6 best.
   The shipped rule works from `n = ceil(tiles / SMs)` per tile (below).
3. **The BK/WM retune of 128x64 is falsified** (question 3).  On
   512x5632x2048 and 7 more shapes per GPU, BK=64 loses 2-15% (one
   +1.3% cell) and 256 threads (8 x 32x32 subgroups) loses 5.5-19%;
   64x128 with 64x32 slices still compiles to 168 registers and never
   becomes the best tile.
   The lever on that shape was **raster order**: a 64x64 tile launched
   M-fastest (new `f16w[_a16]_coopmat_m64n64_mfast`) lifts a16 gate/up
   512x5632x2048 on the 3070 from 29.1 (64x64) / 33.8 (128x64) to
   **35.2 TF/s**.
4. **L2 capacity matters, but not the way the warm sweep said** (§12).
   The SM-count-only model transfers the 128-row tiles' per-clock
   efficiency between GA104 and AD107 within 3%, but 64x64 runs 16-17%
   less efficiently per clock on the 4 MiB-L2 GA104, so the rule keeps an
   L2 lower bound from an NVIDIA device-ID family table (`ML_L2_MB`
   overrides; unknown = small).  The 100df0d rule also used it to keep
   x-fast order whenever B fits a large L2; that was a **warm-L2
   artifact** of `ml_bench ab` (B stays resident between repeats) and
   made 4060 pp512 1.5% slower than exp/grokmerge.  With weights evicted
   before every GEMM (`ML_AB_COLD_MB=64`) 64x64 M-fast wins all four
   pp512 GEMMs on both GPUs, so 0fc29f9 drops that exception; on a large
   L2 the L2 now only makes M-fast's *A* re-reads cheap.
5. Result (0fc29f9, §12): 3070 — replayed on the clean §4 sweep, 0 of
   77 shapes slower than exp/grokmerge by >1%, geomean ×1.081; clean
   cold M=512 set 11/11 at the measured best (×1.056).  4060 — cold
   sweep geomean ×1.083, 1 shape slower by >1% (384x2560x2048, +2.4%),
   mean regret vs the best of five tiles 0.71%.  TinyLlama pp512 on the
   4060: 0fc29f9 38.46 ms vs grokmerge 39.48 ms GPU prefill (−2.6%,
   6 rotated rounds, faster in every round; 100df0d was +1.0%), §9.
   **3070 embargo from 07:44 (2026-09-27)**: the 3070 pp512 A/B and the
   dedicated 128x64-vs-128x128 A/B are PENDING (§12.5).

## 1. SM count (and L2) on the host

`VulkanContext::shader_cores` / `sm_count()` (src/context/device.rs,
init.rs): `VkPhysicalDeviceShaderSMBuiltinsPropertiesNV::shaderSMCount`
(+ `shaderWarpsPerSM`), else `VK_AMD_shader_core_properties2` active
CUs, `VK_AMD_shader_core_properties` physical CUs, or
`VK_ARM_shader_core_builtins`; the structs chain into
`vkGetPhysicalDeviceProperties2` without enabling the extensions.
`ML_SM_COUNT=N` pins the count, `ML_SM_COUNT=0` forces the SM-agnostic
fallback.  Both drivers report it:

```
device #0: NVIDIA GeForce RTX 4060 (... shader_cores=24 (VK_NV_shader_sm_builtins))
device #0: NVIDIA GeForce RTX 3070 (... shader_cores=46 (VK_NV_shader_sm_builtins))
```

`VulkanContext::l2_bytes_estimate`: NVIDIA device-ID family lower bound
(GA100 40 MiB, GH100 50 MiB, 0x2680-0x2FFF Ada/Blackwell 24 MiB,
Turing/Ampere GA10x 1 MiB; anything else `None`), `ML_L2_MB=N` override.

## 2. Occupancy from driver register counts

`examples/coopmat_occupancy.rs` builds each kernel on a second logical
device with `VK_KHR_pipeline_executable_properties` +
`CAPTURE_STATISTICS` and derives CTAs/SM (64K regs, 48 warps, 100 KiB
shared + 1 KiB/CTA).  Identical on sm_86 and sm_89:

| kernel | tile | threads | shared B | regs | CTAs/SM (limiter) |
|---|---|---:|---:|---:|---:|
| f16w(_a16)_coopmat_aligned | 128x128 | 256 | 18944 | 128 | 2 (regs) |
| f16w(_a16)_coopmat_m64n64 | 64x64 | 128 | 9728 | 72 | **7** (regs) |
| f16w_coopmat_m64n128 / a16 | 64x128 | 128 | 13824 | 168 / 155 | 3 (regs) |
| f16w(_a16)_coopmat_m128n64 | 128x64 | 128 | 14848 | 128 | 4 (regs) |
| candidate m128n64 BK=64 | 128x64 | 128 | 27648 | 168 | 3 |
| candidate m128n64 256 thr | 128x64 | 256 | 14848 | 77 | 3 |
| candidate m64n128 64x32 slices | 64x128 | 128 | 13824 | 168 | 3 |
| candidate m64n64 BK=64 | 64x64 | 128 | 18432 | 96 | 5 |
| f16w(_a16)_coopmat_m64n64_mfast | 64x64 | 128 | 9728 | 72 | 7 |

`ML_STATS_FILTER=coopmat target/release/examples/coopmat_occupancy`
(output: `coopmat-sm-data/occupancy_{4060,3070}.txt`).  Note 64x128's
168 registers are not caused by its 32x64 subgroup slice: the 64x32
slice variant compiles to the same 168.

## 3. Measurement protocol

`ml_bench ab` (merged from `exp/ab` into this branch): one process, one
context, 300 ms 4096³ burn-in, tensors allocated once per case, rounds
rotate which kernel runs first, TF/s from the median of per-round GPU
medians (20 iters, 5 warmup per round), sampled correctness validated on
round 0 of every (case, kernel).  New in this branch: `,a16` cases (f16
A/B/C, llama-ash's prefill route — pinned f16w tiles map to their a16
siblings) and a `route` column recording what `auto` dispatched.

```
ML_AB_KERNELS=f16w_coopmat_aligned,f16w_coopmat_m64n64,f16w_coopmat_m64n128,f16w_coopmat_m128n64,auto \
ML_ROUNDS=7 ML_OUTPUT=csv target/release/ml_bench ab <label,1,M,N,K,f16w|a16> ...
```

Shapes: squares 1024-4096; M ∈ {256,384,512,768,1024,2048} x N ∈
{2048,2560,5632} x K ∈ {2048,5632} plus N=32000/K=2048; TinyLlama
prefill T=128/1920 (f16w) and a16 T ∈ {128,256,512,1024,1920} x
{2048x2048, 2560x2048 (concat QKV), 5632x2048 (gate/up), 2048x5632
(down)}.  Binaries: 4060 = full `glslc -O` build of e1670b8 (fast Rust
profile); 3070 = desktop LOWMEM build of the same commit (the coopmat
shaders compile at `-O` there too; the SPIR-V is byte-identical).
Median SM clock under load (`nvidia-smi`, util >= 90%): 4060 2535 MHz
(2250-2745), 3070 1815 MHz (1770-1890, power-limited at ~218 W).
Noise floor: `auto` and its pinned twin (same kernel, same run) differ
by up to 1-4% on multi-millisecond shapes.

## 4. Raw sweep tables

Bold = best of the four shipped tiles; "new rule" = the shipped
SM-aware pick on that GPU (64x64 M-fast was not in this sweep; see §5).

#### 4060 — sweep (commit e1670b8, `ml_bench ab`, ML_ROUNDS=7, 20 iters/round, TF/s from the median of per-round GPU medians)

| shape (MxNxK) | 128x128 | 64x64 | 64x128 | 128x64 | exp/grokmerge auto | new rule |
|---|---:|---:|---:|---:|---|---|
| 128x2048x2048 | 11.28 | **21.28** | 14.49 | 14.00 | 64x64 | 64x64 |
| 128x2048x5632 | 11.28 | **20.22** | 16.16 | 14.59 | 64x64 | 64x64 |
| 128x2560x2048 | 14.09 | **18.78** | 17.71 | 17.42 | 64x64 | 64x64 |
| 128x5632x2048 | 21.58 | 20.98 | 19.58 | **22.96** | 64x64 | 128x64 |
| 128x32000x2048 | 22.25 | 15.22 | 15.18 | **25.19** | 64x64 | 128x64 |
| 256x2048x2048 | 16.35 | **21.06** | 19.88 | 20.91 | 64x64 | 128x64 |
| 256x2048x5632 | 17.78 | 22.61 | 21.46 | **23.07** | 64x64 | 128x64 |
| 256x2560x2048 | 20.35 | **22.79** | 19.42 | 21.17 | 64x64 | 64x64 |
| 256x2560x5632 | 21.59 | **23.50** | 19.57 | 22.66 | 64x64 | 64x64 |
| 256x5632x2048 | 22.43 | 23.72 | 23.12 | **24.03** | 64x64 | 128x64 |
| 256x5632x5632 | 21.82 | 19.46 | 21.46 | **24.16** | 64x64 | 128x64 |
| 256x32000x2048 | 22.70 | 15.24 | 15.32 | **25.39** | 64x64 | 128x64 |
| 384x2048x2048 | 23.77 | 21.88 | 19.81 | **24.85** | 64x64 | 128x64 |
| 384x2048x5632 | 25.55 | 22.52 | 20.60 | **26.83** | 64x64 | 128x64 |
| 384x2560x2048 | 18.75 | 23.89 | **24.19** | 23.36 | 64x64 | 128x64 |
| 384x2560x5632 | 18.87 | 23.49 | **25.02** | 23.87 | 64x64 | 128x64 |
| 384x5632x2048 | 21.35 | 22.70 | 23.90 | **25.03** | 64x64 | 128x64 |
| 384x5632x5632 | 21.73 | 18.49 | 21.84 | **25.31** | 64x64 | 128x64 |
| 384x32000x2048 | 23.31 | 15.31 | 15.39 | **25.53** | 64x64 | 128x64 |
| 512x2048x2048 | 20.02 | **23.99** | 22.27 | 22.68 | 64x64 | 64x64 |
| 512x2048x5632 | 20.09 | **24.66** | 22.43 | 22.69 | 64x64 | 64x64 |
| 512x2560x2048 | 20.75 | 22.94 | 23.13 | **24.47** | 128x64 | 128x64 |
| 512x2560x5632 | 20.39 | 23.30 | 22.88 | **25.34** | 128x64 | 128x64 |
| 512x5632x2048 | 21.10 | 22.84 | 23.60 | **24.37** | 128x64 | 128x64 |
| 512x5632x5632 | 21.69 | 17.29 | 23.22 | **25.23** | 128x64 | 128x64 |
| 512x32000x2048 | 22.46 | 15.31 | 15.42 | **24.49** | 128x64 | 128x64 |
| 768x2048x2048 | 24.53 | 24.19 | 24.57 | **25.64** | 128x128 | 128x64 |
| 768x2048x5632 | 25.01 | 24.39 | 25.68 | **26.88** | 128x128 | 128x64 |
| 768x2560x2048 | 22.81 | 23.36 | 24.34 | **24.64** | 128x128 | 128x64 |
| 768x2560x5632 | 23.16 | 23.62 | 21.91 | **25.84** | 128x128 | 128x64 |
| 768x5632x2048 | 23.17 | 24.14 | 24.77 | **24.86** | 128x128 | 128x64 |
| 768x5632x5632 | 23.91 | 17.14 | 22.14 | **25.79** | 128x128 | 128x64 |
| 768x32000x2048 | **23.84** | 15.33 | 15.44 | 23.40 | 128x128 | 128x64 |
| 1024x1024x1024 | 16.97 | **21.48** | 19.38 | 19.66 | 64x64 | 64x64 |
| 1024x2048x2048 | 22.54 | 24.47 | 24.88 | **25.42** | 128x128 | 128x64 |
| 1024x2048x5632 | 22.53 | 24.57 | 24.88 | **26.30** | 128x128 | 128x64 |
| 1024x2560x2048 | 22.33 | 23.43 | 24.22 | **24.76** | 128x128 | 128x64 |
| 1024x2560x5632 | 23.09 | 23.39 | 23.12 | **24.68** | 128x128 | 128x64 |
| 1024x5632x2048 | 23.30 | 24.49 | 23.65 | **24.94** | 128x128 | 128x64 |
| 1024x5632x5632 | 24.32 | 16.75 | 22.50 | **24.85** | 128x128 | 128x64 |
| 1024x32000x2048 | **24.04** | 15.34 | 15.43 | 23.76 | 128x128 | 128x64 |
| 1536x1536x1536 | 23.96 | 23.44 | 24.90 | **25.16** | 128x128 | 128x64 |
| 1920x2048x2048 | 23.61 | 24.11 | 25.41 | **26.01** | 128x128 | 128x64 |
| 1920x2048x5632 | 25.27 | 24.79 | 25.49 | **26.45** | 128x128 | 128x64 |
| 1920x2560x2048 | 23.92 | 24.29 | 25.71 | **25.97** | 128x128 | 128x64 |
| 1920x5632x2048 | 24.11 | 24.13 | 24.36 | **24.91** | 128x128 | 128x64 |
| 1920x32000x2048 | 23.72 | 14.71 | 15.01 | **23.78** | 128x128 | 128x64 |
| 2048x2048x2048 | 23.43 | 24.78 | 25.00 | **25.62** | 128x128 | 128x64 |
| 2048x2048x2048 | 24.13 | 25.10 | 25.20 | **26.05** | 128x128 | 128x64 |
| 2048x2048x5632 | 24.57 | 24.32 | 23.52 | **25.48** | 128x128 | 128x64 |
| 2048x2560x2048 | 24.20 | 24.62 | 25.77 | **26.60** | 128x128 | 128x64 |
| 2048x2560x5632 | 24.69 | 21.79 | 22.93 | **25.10** | 128x128 | 128x64 |
| 2048x5632x2048 | 24.26 | 25.22 | 24.45 | **25.57** | 128x128 | 128x64 |
| 2048x5632x5632 | 24.13 | 16.23 | 23.01 | **24.98** | 128x128 | 128x64 |
| 2048x32000x2048 | 23.88 | 15.40 | 15.47 | **24.59** | 128x128 | 128x64 |
| 3072x3072x3072 | 25.98 | 23.90 | 25.05 | **26.06** | 128x128 | 128x64 |
| 4096x4096x4096 | **25.64** | 17.08 | 23.10 | 24.75 | 128x128 | 128x64 |
| a16 128x2048x2048 | 12.89 | **23.87** | 13.93 | 16.99 | 64x64 | 64x64 |
| a16 128x2048x5632 | 12.97 | **25.04** | 14.63 | 16.37 | 64x64 | 64x64 |
| a16 128x2560x2048 | 15.97 | **20.22** | 16.94 | 20.22 | 64x64 | 64x64 |
| a16 128x5632x2048 | 23.63 | **25.23** | 21.79 | 24.31 | 64x64 | 64x64 |
| a16 256x2048x2048 | 17.92 | **24.65** | 21.35 | 23.20 | 64x64 | 64x64 |
| a16 256x2048x5632 | 18.80 | **25.81** | 22.68 | 24.85 | 64x64 | 64x64 |
| a16 256x2560x2048 | 21.56 | **25.77** | 19.80 | 22.05 | 64x64 | 64x64 |
| a16 256x5632x2048 | 24.00 | **28.35** | 24.00 | 25.92 | 64x64 | 64x64 |
| a16 512x2048x2048 | 21.51 | **27.08** | 22.30 | 24.53 | 64x64 | 64x64 |
| a16 512x2048x5632 | 21.07 | **28.18** | 23.02 | 24.66 | 64x64 | 64x64 |
| a16 512x2560x2048 | 21.83 | **27.08** | 23.93 | 26.21 | 128x64 | 64x64 |
| a16 512x5632x2048 | 24.34 | **28.92** | 26.10 | 27.95 | 128x64 | 64x64 |
| a16 1024x2048x2048 | 23.95 | **28.45** | 25.05 | 27.49 | 128x128 | 64x64 |
| a16 1024x2048x5632 | 23.96 | **28.96** | 25.09 | 27.14 | 128x128 | 64x64 |
| a16 1024x2560x2048 | 24.94 | **28.54** | 25.34 | 27.31 | 128x128 | 64x64 |
| a16 1024x5632x2048 | 25.94 | **29.59** | 25.21 | 28.64 | 128x128 | 64x64 |
| a16 1920x2048x2048 | 26.38 | **29.51** | 25.95 | 28.83 | 128x128 | 64x64 |
| a16 1920x2048x5632 | 27.16 | **30.64** | 25.83 | 28.94 | 128x128 | 64x64 |
| a16 1920x2560x2048 | 25.05 | **29.05** | 25.97 | 27.41 | 128x128 | 64x64 |
| a16 1920x5632x2048 | 26.41 | **29.50** | 23.92 | 27.02 | 128x128 | 64x64 |

#### 3070 — sweep (commit e1670b8, `ml_bench ab`, ML_ROUNDS=7, 20 iters/round, TF/s from the median of per-round GPU medians)

| shape (MxNxK) | 128x128 | 64x64 | 64x128 | 128x64 | exp/grokmerge auto | new rule |
|---|---:|---:|---:|---:|---|---|
| 128x2048x2048 | 7.64 | **16.50** | 12.00 | 11.28 | 64x64 | 64x64 |
| 128x2048x5632 | 8.31 | **18.36** | 13.16 | 12.37 | 64x64 | 64x64 |
| 128x2560x2048 | 9.64 | **19.98** | 14.89 | 14.50 | 64x64 | 64x64 |
| 128x5632x2048 | 21.05 | **30.08** | 28.49 | 26.08 | 64x64 | 64x64 |
| 128x32000x2048 | 28.70 | 24.42 | 24.58 | **33.90** | 64x64 | 128x64 |
| 256x2048x2048 | 15.07 | **28.19** | 19.42 | 19.14 | 64x64 | 64x64 |
| 256x2048x5632 | 16.18 | **31.34** | 20.69 | 20.76 | 64x64 | 64x64 |
| 256x2560x2048 | 19.46 | **26.96** | 25.00 | 24.51 | 64x64 | 64x64 |
| 256x2560x5632 | 20.52 | **28.46** | 26.32 | 26.18 | 64x64 | 64x64 |
| 256x5632x2048 | 30.72 | 26.59 | 27.83 | **32.48** | 64x64 | 128x64 |
| 256x5632x5632 | 31.33 | 24.12 | 28.65 | **33.45** | 64x64 | 128x64 |
| 256x32000x2048 | 30.58 | 24.79 | 24.55 | **34.49** | 64x64 | 128x64 |
| 384x2048x2048 | 17.48 | **26.55** | 21.53 | 22.64 | 64x64 | 64x64 |
| 384x2048x5632 | 18.36 | **27.61** | 23.41 | 24.69 | 64x64 | 64x64 |
| 384x2560x2048 | 22.36 | 28.57 | 27.55 | **28.94** | 64x64 | 64x64 |
| 384x2560x5632 | 22.69 | 27.44 | 28.76 | **30.27** | 64x64 | 128x64 |
| 384x5632x2048 | 27.85 | 27.05 | 27.58 | **31.65** | 64x64 | 128x64 |
| 384x5632x5632 | 28.29 | 25.26 | 27.73 | **32.47** | 64x64 | 128x64 |
| 384x32000x2048 | 30.49 | 24.75 | 24.57 | **34.77** | 64x64 | 128x64 |
| 512x2048x2048 | 23.09 | **29.83** | 28.40 | 29.42 | 64x64 | 64x64 |
| 512x2048x5632 | 23.66 | 28.12 | 30.29 | **31.38** | 64x64 | 128x64 |
| 512x2560x2048 | 27.94 | **31.20** | 26.12 | 29.11 | 128x64 | 64x64 |
| 512x2560x5632 | 29.59 | 29.77 | 27.21 | **30.84** | 128x64 | 128x64 |
| 512x5632x2048 | 30.56 | 25.78 | 27.74 | **32.73** | 128x64 | 128x64 |
| 512x5632x5632 | 31.07 | 23.70 | 27.43 | **33.53** | 128x64 | 128x64 |
| 512x32000x2048 | 31.77 | 24.89 | 24.65 | **33.10** | 128x64 | 128x64 |
| 768x2048x2048 | 21.10 | **31.47** | 27.29 | 26.72 | 128x128 | 64x64 |
| 768x2048x5632 | 22.21 | 28.40 | **28.48** | 27.06 | 128x128 | 64x64 |
| 768x2560x2048 | 25.85 | **31.13** | 29.48 | 29.80 | 128x128 | 128x64 |
| 768x2560x5632 | 27.34 | 26.81 | 29.32 | **30.03** | 128x128 | 128x64 |
| 768x5632x2048 | 30.87 | 25.92 | 26.25 | **32.49** | 128x128 | 128x64 |
| 768x5632x5632 | 32.94 | 23.21 | 26.10 | **34.57** | 128x128 | 128x64 |
| 768x32000x2048 | 32.95 | 24.87 | 24.44 | **33.10** | 128x128 | 128x64 |
| 1024x1024x1024 | 21.90 | **29.02** | 26.76 | 27.41 | 64x64 | 64x64 |
| 1024x2048x2048 | 27.62 | **32.27** | 30.13 | 30.92 | 128x128 | 128x64 |
| 1024x2048x5632 | 28.75 | 26.59 | 31.07 | **31.33** | 128x128 | 128x64 |
| 1024x2560x2048 | 28.50 | 29.65 | 32.59 | **33.44** | 128x128 | 128x64 |
| 1024x2560x5632 | 30.19 | 23.88 | 32.40 | **34.50** | 128x128 | 128x64 |
| 1024x5632x2048 | 30.94 | 25.55 | 25.45 | **33.03** | 128x128 | 128x64 |
| 1024x5632x5632 | 32.99 | 22.72 | 25.45 | **33.39** | 128x128 | 128x64 |
| 1024x32000x2048 | 32.93 | 24.53 | 24.29 | **32.96** | 128x128 | 128x64 |
| 1536x1536x1536 | 26.80 | **31.12** | 29.09 | 30.88 | 128x128 | 128x64 |
| 1920x2048x2048 | 28.97 | 31.74 | 32.16 | **32.39** | 128x128 | 128x64 |
| 1920x2048x5632 | 30.90 | 21.02 | 32.49 | **33.64** | 128x128 | 128x64 |
| 1920x2560x2048 | 29.72 | 27.00 | 31.82 | **31.84** | 128x128 | 128x64 |
| 1920x5632x2048 | 31.67 | 25.12 | 25.44 | **34.69** | 128x128 | 128x64 |
| 1920x32000x2048 | **33.27** | 24.98 | 24.60 | 33.23 | 128x128 | 128x64 |
| 2048x2048x2048 | 29.80 | 31.23 | 31.29 | **31.76** | 128x128 | 128x64 |
| 2048x2048x2048 | 30.40 | 31.52 | **32.50** | 32.39 | 128x128 | 128x64 |
| 2048x2048x5632 | 32.34 | 21.21 | 31.34 | **32.79** | 128x128 | 128x64 |
| 2048x2560x2048 | 31.34 | 27.03 | 32.76 | **33.15** | 128x128 | 128x64 |
| 2048x2560x5632 | 33.70 | 21.28 | 32.75 | **34.08** | 128x128 | 128x64 |
| 2048x5632x2048 | 31.55 | 25.05 | 25.26 | **34.43** | 128x128 | 128x64 |
| 2048x5632x5632 | **33.10** | 22.24 | 25.12 | 32.90 | 128x128 | 128x64 |
| 2048x32000x2048 | **33.05** | 24.88 | 24.56 | 32.98 | 128x128 | 128x64 |
| 3072x3072x3072 | 33.42 | 23.30 | 30.61 | **34.29** | 128x128 | 128x64 |
| 4096x4096x4096 | 34.38 | 23.27 | 26.66 | **34.38** | 128x128 | 128x64 |
| a16 128x2048x2048 | 8.84 | **19.33** | 13.63 | 13.51 | 64x64 | 64x64 |
| a16 128x2048x5632 | 8.90 | **20.46** | 14.36 | 13.93 | 64x64 | 64x64 |
| a16 128x2560x2048 | 10.73 | **22.34** | 16.87 | 16.74 | 64x64 | 64x64 |
| a16 128x5632x2048 | 23.13 | **32.39** | 29.97 | 28.98 | 64x64 | 64x64 |
| a16 256x2048x2048 | 17.11 | **32.71** | 19.88 | 21.98 | 64x64 | 64x64 |
| a16 256x2048x5632 | 17.65 | **35.50** | 20.20 | 22.06 | 64x64 | 64x64 |
| a16 256x2560x2048 | 21.20 | **28.47** | 23.75 | 26.63 | 64x64 | 64x64 |
| a16 256x5632x2048 | 33.83 | 30.35 | 29.23 | **34.14** | 64x64 | 64x64 M-fast |
| a16 512x2048x2048 | 24.30 | **32.86** | 29.79 | 32.24 | 64x64 | 64x64 |
| a16 512x2048x5632 | 24.89 | 32.90 | 31.03 | **33.25** | 64x64 | 64x64 |
| a16 512x2560x2048 | 29.65 | **35.41** | 26.73 | 30.20 | 128x64 | 64x64 |
| a16 512x5632x2048 | 32.86 | 30.60 | 28.83 | **35.84** | 128x64 | 64x64 M-fast |
| a16 1024x2048x2048 | 30.31 | **35.26** | 31.19 | 34.05 | 128x128 | 128x64 |
| a16 1024x2048x5632 | 31.71 | 33.69 | 32.15 | **34.53** | 128x128 | 128x64 |
| a16 1024x2560x2048 | 29.80 | 33.63 | 33.47 | **36.44** | 128x128 | 128x64 |
| a16 1024x5632x2048 | 33.59 | 29.11 | 26.60 | **36.31** | 128x128 | 128x64 |
| a16 1920x2048x2048 | 30.42 | **36.38** | 33.98 | 35.39 | 128x128 | 128x64 |
| a16 1920x2048x5632 | 32.80 | 28.78 | 32.99 | **36.97** | 128x128 | 128x64 |
| a16 1920x2560x2048 | 31.77 | 29.95 | 32.73 | **34.60** | 128x128 | 128x64 |
| a16 1920x5632x2048 | 33.95 | 27.24 | 26.72 | **37.77** | 128x128 | 128x64 |

## 5. Candidate kernels (raster order, BK, warp layout)

#### Raster (M-fastest) candidates, f16w — RTX 4060 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 128x64 M-fast | 64x64 | 64x64 M-fast | 128x128 | 128x128 M-fast |
|---|---:|---:|---:|---:|---:|---:|
| g256x32000x2048 | **24.72** | 23.41 | 14.79 | 24.13 | 22.18 | 23.27 |
| g512x32000x2048 | 24.83 | 24.25 | 14.83 | 23.57 | 22.55 | **25.15** |
| g256x5632x5632 | **24.24** | 23.36 | 18.57 | 22.34 | 21.69 | 22.33 |
| g512x5632x2048 | **25.05** | 24.58 | 23.61 | 23.49 | 21.75 | 22.26 |
| g512x2560x2048 | **23.16** | 22.25 | 22.42 | 22.47 | 20.32 | 20.56 |
| g1024x5632x5632 | **25.54** | 24.74 | 16.31 | 23.65 | 22.26 | 25.08 |
| g2048x2048x5632 | **27.22** | 21.52 | 22.62 | 16.58 | 24.05 | 23.64 |
| g768x2048x2048 | 24.70 | **24.91** | 24.13 | 23.76 | 23.63 | 23.36 |
| sq4096x4096x4096 | 24.49 | 21.93 | 21.49 | 17.81 | **25.37** | 21.67 |

#### Raster (M-fastest) candidates, f16w — RTX 3070 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 128x64 M-fast | 64x64 | 64x64 M-fast | 128x128 | 128x128 M-fast |
|---|---:|---:|---:|---:|---:|---:|
| g256x32000x2048 | **34.38** | 31.35 | 24.78 | 32.17 | 29.17 | 31.37 |
| g512x32000x2048 | **34.22** | 31.27 | 24.83 | 30.72 | 31.05 | 32.87 |
| g256x5632x5632 | **33.02** | 31.28 | 23.57 | 26.17 | 30.40 | 31.38 |
| g512x5632x2048 | **32.05** | 31.04 | 25.77 | 31.62 | 29.73 | 30.06 |
| g512x2560x2048 | 27.99 | 26.67 | 30.63 | **31.48** | 27.16 | 27.30 |
| g1024x5632x5632 | **34.05** | 20.30 | 22.95 | 26.49 | 31.94 | 33.36 |
| g2048x2048x5632 | **33.93** | 18.44 | 21.04 | 20.08 | 31.82 | 32.08 |
| g768x2048x2048 | 26.77 | 24.85 | **31.46** | 29.57 | 20.77 | 20.58 |
| sq4096x4096x4096 | 33.38 | 13.87 | 23.28 | 14.17 | **33.83** | 25.58 |

#### Raster candidates, a16 (f16 A/C) — the llama prefill route — RTX 4060 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 64x64 | 64x64 M-fast |
|---|---:|---:|---:|
| a512x2048x2048 | 22.94 | **25.51** | 25.45 |
| a512x2560x2048 | 24.54 | **25.37** | 25.06 |
| a512x5632x2048 | 26.89 | **27.62** | 27.49 |
| a512x2048x5632 | 24.16 | **27.12** | 27.02 |
| a256x5632x2048 | 24.13 | **26.29** | 25.85 |
| a1024x5632x2048 | 26.44 | 27.76 | **27.78** |
| a1920x2048x5632 | 27.15 | **28.18** | 27.16 |
| a1920x5632x2048 | 27.12 | **27.68** | 27.07 |

#### Raster candidates, a16 (f16 A/C) — the llama prefill route — RTX 3070 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 64x64 | 64x64 M-fast |
|---|---:|---:|---:|
| a512x2048x2048 | 29.42 | 30.50 | **31.18** |
| a512x2560x2048 | 28.01 | 33.05 | **33.54** |
| a512x5632x2048 | 33.75 | 29.05 | **35.21** |
| a512x2048x5632 | 31.93 | 32.06 | **33.55** |
| a256x5632x2048 | 32.03 | 28.84 | **33.26** |
| a1024x5632x2048 | 34.66 | 27.77 | **37.05** |
| a1920x2048x5632 | **36.04** | 28.42 | 25.81 |
| a1920x5632x2048 | **35.62** | 26.85 | 30.04 |

#### BK / warp-layout retune candidates, f16w — RTX 4060 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 128x64 BK64 | 128x64 256thr | 64x128 | 64x128 64x32-slices | 64x64 | 64x64 BK64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| g512x5632x2048 | **24.84** | 23.19 | 21.13 | 24.45 | 24.68 | 23.85 | 21.72 |
| g512x2560x2048 | **23.95** | 21.01 | 20.43 | 23.10 | 22.51 | 22.62 | 20.32 |
| g512x2048x2048 | 22.64 | 21.33 | 19.69 | 21.86 | 21.85 | **23.84** | 20.83 |
| g512x2048x5632 | 22.80 | 21.49 | 20.07 | 22.52 | 22.96 | **24.53** | 21.06 |
| g1024x2048x2048 | **24.77** | 22.60 | 20.48 | 23.13 | 23.22 | 23.45 | 20.97 |
| sq2048x2048x2048 | **24.99** | 22.89 | 21.25 | 24.47 | 24.50 | 24.23 | 21.39 |
| sq4096x4096x4096 | 23.59 | 23.11 | 21.41 | 22.56 | **24.10** | 19.07 | 15.84 |
| g256x2048x2048 | **19.99** | 18.17 | 17.28 | 18.93 | 19.07 | 19.90 | 16.69 |

#### BK / warp-layout retune candidates, f16w — RTX 3070 (ML_ROUNDS=9; TF/s)

| shape | 128x64 | 128x64 BK64 | 128x64 256thr | 64x128 | 64x128 64x32-slices | 64x64 | 64x64 BK64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| g512x5632x2048 | **31.60** | 29.45 | 27.35 | 27.34 | 28.57 | 25.86 | 24.64 |
| g512x2560x2048 | 28.02 | 23.91 | 22.76 | 25.09 | 24.63 | **30.29** | 27.98 |
| g512x2048x2048 | **29.75** | 27.65 | 25.93 | 28.50 | 28.45 | 29.74 | 25.31 |
| g512x2048x5632 | **30.45** | 28.12 | 26.80 | 29.75 | 29.92 | 27.94 | 24.42 |
| g1024x2048x2048 | 30.76 | 28.47 | 26.16 | 29.34 | 29.64 | **31.55** | 29.17 |
| sq2048x2048x2048 | 30.55 | 28.92 | 26.52 | 30.84 | **30.98** | 30.82 | 28.84 |
| sq4096x4096x4096 | **33.77** | 31.90 | 29.24 | 26.33 | 25.99 | 23.25 | 23.82 |
| g256x2048x2048 | 18.24 | 18.48 | 17.24 | 19.17 | 19.20 | **27.90** | 24.89 |

M-fastest order fixes 64x64's collapse on wide N (4060 256x32000:
14.8 → 24.1; 3070: 24.8 → 32.2) and helps the 3070 a16 prefill shapes,
but is a loss once A is big (M=2048 / 4096³: −5% to −39%), and the
128-row tiles never beat plain 128x64 with it by more than noise (+1.3%
at best, −58% at 4096³ on the 3070).  Hence the shipped rule chooses the
64x64 raster per shape from the re-read traffic of each order (§7).

## 6. The model

`time ≈ waves(tiles, SMs x CTAs/SM) x per-tile time`, made explicit at
SM granularity (the busiest SM receives `n = ceil(tiles / S)` CTAs and
runs them in batches of `c` co-resident CTAs):

```
T_comp = 2·BM·BN·(K + K0) · [ floor(n/c)·c/P(c) + r/P(r) ] / f_clk,  r = n mod c
P(j)   = p_inf · min(1, j·e1)            (per-SM flop/clk with j resident CTAs)
T_mem  = (|A| + |C| + |B|·rB) / (η·BW),  rB = 1 if |B| <= L2 else ceil(rows / max(1, S·c/cols))
T      = max(T_comp, T_mem)
```

`c` from §2; p_inf, e1, K0 fitted per (tile, A dtype) by log-least
squares on one GPU; S, L2, BW (η = 0.85) and the measured median clock
are that GPU's hardware.  Reproduce with `python3 analysis.py model`
(and `policies`, `rule`) in `coopmat-sm-data/`.

Fitted per-clock efficiency (p_inf in flop/clk/SM; the f16×f16→f32
tensor peak is 512):

| tile | A | 4060 fit | 3070 fit | ratio |
|---|---|---:|---:|---:|
| 128x128 | f32 | 426 | 428 | 1.00 |
| 128x64 | f32 | 436 | 424 | 0.97 |
| 64x128 | f32 | 411 | 382 | 0.93 |
| **64x64** | f32 | **363** | **304** | **0.84** |
| 128x128 | f16 | 452 | 456 | 1.01 |
| 128x64 | f16 | 451 | 435 | 0.96 |
| 64x128 | f16 | 413 | 401 | 0.97 |
| **64x64** | f16 | **493** | **409** | **0.83** |

Cross-GPU prediction (fit on one GPU, predict every cell of the other):

| direction | median abs err | p90 abs err | tile-choice regret (mean / max) | zero-regret shapes |
|---|---:|---:|---:|---:|
| 4060 → 3070 | 4.6% | 21.8% | 2.47% / 38.6% | 75% |
| 3070 → 4060 | 7.3% | 18.8% | 2.12% / 12.8% | 61% |
| 4060 in-sample | 3.2% | 10.4% | 0.38% / 8.7% | 90% |
| 3070 in-sample | 5.0% | 14.1% | 1.53% / 15.3% | 70% |

What the SM count buys (mean tile-choice regret vs the measured best of
the four tiles; `analysis.py policies`):

| policy | 4060 | 3070 |
|---|---:|---:|
| exp/grokmerge rule | 8.19% | 8.63% |
| always 128x64 | 4.62% | 8.05% |
| model fit on the other GPU, **correct** SM count | 2.12% | 2.47% |
| model fit on the other GPU, **other GPU's** SM count | 3.10% | 5.35% |
| shipped rule (§7, in-sample; six fitted ratios) | 0.25% | 0.32% |

Reading: the SM-level wave model transfers (the 128-row tiles to ±3%
per clock; swapping in the wrong SM count doubles the 3070 regret), but
the residual is dominated by one non-SM effect: 64x64's efficiency on
the 4 MiB L2.  The max-regret cells are all a16/f16w 64x64 on large
operands (e.g. a16 1920x5632x2048: 4060 64x64 29.5 vs 128x64 27.0;
3070 64x64 26.9 vs 128x64 35.6) — the two GPUs genuinely disagree there
and no SM-count-only rule can serve both.

## 7. The 100df0d rule (superseded in part by §12)

No SM count → the exp/grokmerge rule, unchanged (`coopmat_tile_fallback`).
With `S` SMs, compare 128x64 and 64x64 (128x128 / 64x128 stay
selectable but are not auto-routed; M not 128-aligned → 64x64):

1. **64x64 raster.**  Window `W = 7·S` CTAs; x-fastest re-streams B
   `xp = ceil(rows / max(1, W/cols))` times, M-fastest re-streams A
   `mp = ceil(cols / max(1, W/rows))` times.  M-fast iff
   `|A|·mp + |B| < 0.75·(|B|·xp + |A|)`, `|A|·(mp−1) <= 8 MiB`, and B
   does not fit a large L2.
2. **Score** `= tiles/(S·n) · min(1, n/3) · ρ` with `n = ceil(tiles/S)`:
   SM load balance times an occupancy floor (under 3 CTAs per SM
   neither tile hides its load/MMA latency).  ρ(128x64) = 1.
3. **ρ(64x64)** (fitted, rounded): large L2 — a16 1.02, f16w 0.92;
   small/unknown L2 — a16 1.02, but 0.8 for a multi-pass x-fast grid;
   f16w 0.9, except single-pass x-fast 1.0 (|A| <= 4 MiB) / 0.85.
4. Ties → 64x64.

The a16 route (`MatmulPipeline::a16_coopmat_index`) uses the same rule;
an explicit coopmat pin (`ML_KERNEL`, `ml_bench ab`) pins the same tile
on both routes.  Unit tests (`pipeline::routing::bda_tests`): 24-SM /
large-L2 (4060) picks, 46-SM / small-L2 (3070) picks, SM count alone
moving the boundary (768x2048x2048: 128x64 on 24 SMs, 64x64 on 46 for
either L2 class), no-SM-count == fallback for every L2 value,
alignment, selection round trips; `context::device::tests` cover the
`ML_SM_COUNT` / `ML_L2_MB` parsers and the L2 table.  GPU tests:
`f16_routes_pick_f16w_kernels_and_skip_splitk2` expects 128x64 at
2048³ when the SM count is known; `coopmat_asymmetric_tiles_match_dual_rounded_reference`
covers both M-fast kernels including a batched ragged grid.

Picks for TinyLlama pp512 (a16, M=512); superseded by the cold refit
(§12), which picks 64x64 M-fast for all four on both GPUs:

| GEMM | 4060 grokmerge → 100df0d → 0fc29f9 | 3070 grokmerge → 100df0d → 0fc29f9 |
|---|---|---|
| q/o 512x2048x2048 | 64x64 → 64x64 → 64x64 M-fast | 64x64 → 64x64 → 64x64 M-fast |
| concat QKV 512x2560x2048 | 128x64 → 64x64 → 64x64 M-fast | 128x64 → 64x64 → 64x64 M-fast |
| gate, up 512x5632x2048 | 128x64 → 64x64 → 64x64 M-fast | 128x64 → 64x64 M-fast → 64x64 M-fast |
| down 512x2048x5632 | 64x64 → 64x64 → 64x64 M-fast | 64x64 → 64x64 → 64x64 M-fast |

## 8. Verification

Fresh interleaved sweep at the branch head: the shipped `auto` plus
every tile it or exp/grokmerge can pick, in one process per batch
(`coopmat-sm-data/verify.sh`, ML_ROUNDS=7), so "old pick" and "new
pick" are timed side by side.  `auto`'s route column is the Rust rule's
actual choice; `analysis.py verify` compares it with the grokmerge pick
using the pinned cells of the same run.

```
ML_AB_KERNELS=f16w_coopmat_m128n64,f16w_coopmat_m64n64,f16w_coopmat_m64n64_mfast,f16w_coopmat_aligned,f16w_coopmat_m64n128,auto \
ML_ROUNDS=7 ML_OUTPUT=csv target/release/ml_bench ab <batch cases from batches.txt>
python3 analysis.py verify verify-4060 4060 --md
```

4060 (verify-4060/, warm, 100df0d binary): 77 shapes, 47 faster than
grokmerge, 1 slower by >1% (384x2560x2048, +2.3%), geomean ×1.083,
mean regret 0.42% (`python3 analysis.py verify verify-4060 4060`).
The cold re-run and the 0fc29f9 replay are in §12.

RTX 3070: sq/m256/m384 ran clean on 2026-09-26 21:00 (`verify-3070/`);
the rest ran on 2026-09-27 07:25-07:45 next to a foreign 3-5 GB GPU
allocation (the user's game starting; embargo at 07:44) and is
quarantined in `verify-3070-contended/` (absolute TF/s −4%, tile-ratio
disagreement with the clean §4 sweep p90 14% vs 5% for the clean
batches).  Command used, one batch per `gpu3070` hold:

```
for b in sq m256 m384 m512 m768 m1024 m2048 t128 t1920 a16_128 a16_256 a16_512 a16_1024 a16_1920; do
  BIN=<worktree>/target/release/ml_bench $SP/bin/gpu3070 coopmat-sm-data/verify.sh verify-3070 $b; sleep 15
done
python3 analysis.py verify verify-3070 3070 --md
```

Until then the 3070 evidence is the pre-embargo sweep of §4 (same
kernels, e1670b8 binary): replaying the shipped rule on it gives 0 of
77 shapes slower than grokmerge by >1% and 45 faster, geomean ×1.079
(`analysis.py rule`; the one M-fast pick uses its in-run ratio from §5).

## 9. TinyLlama pp512

`llama_ash bench -m tinyllama-1.1b-chat-v1.0.f16.gguf --pp 512 --tg 32`,
grokmerge (7cfca58) and branch builds with the same fast profile,
alternating A,B,A,B... (`coopmat-sm-data/llama_ab.sh`), prefill GPU time
from the `prefill timing` log line (graph GPU timestamps).

**RTX 4060, 2026-09-26 (100df0d)**: `pp512_ab_4060.log` — grokmerge
39.42 vs 100df0d 40.00 ms GPU prefill (medians, 5 alternating pairs):
100df0d **1.5% slower**.  `pp512_variants_4060.log` (4 rotated rounds):
base 39.71, `ML_SM_COUNT=0` 39.65, 100df0d 40.08, 100df0d with
`ML_L2_MB=1` (small-L2 branch = 64x64 M-fast on all four GEMMs) 38.36 ms.
This is what triggered the cold-L2 refit (§12).

**RTX 4060, 2026-09-27 (0fc29f9)**: three binaries rotated per round,
6 rounds (`coopmat-sm-data/llama_env_ab.sh`, louise
`~/ta-wt/cm-results/pp4060.sh`; raw `coopmat-sm-data/pp512_final_4060.log`):

```
llama_env_ab.sh 6 pp512_final_4060.log base=ML_X=0:llama_ash-7cfca58 old=ML_X=0:llama_ash-100df0d new=ML_X=0:llama_ash-0fc29f9
```

| build | prefill GPU ms, median (min-max) | vs grokmerge |
|---|---:|---:|
| grokmerge 7cfca58 | 39.48 (39.26-39.98) | — |
| 100df0d | 39.87 (39.67-40.16) | +1.0% |
| **0fc29f9** | **38.46 (38.31-38.72)** | **−2.6%** (every round: −1.9 to −4.1%) |

pp512 throughput 13.2-13.3k vs 12.7-12.9k tokens/s; argmax 29872 in
every run.  Consistent with the rule's per-GEMM cold gains over
grokmerge's picks (`cold-4060-0fc29f9/`, `auto` cells: q/o +1.9%, QKV
+2.0%, gate/up +10.2%, down +0.5%).

**RTX 3070: PENDING** (embargo from 2026-09-27 07:44; an attempt on
2026-09-26 21:01 got 2 rows before the reboot: new 30.27 ms vs
`ML_SM_COUNT=0` 31.52 ms, one round, not a result).  Validated binaries
are ready: `$SP/work/coopmat/bin/llama_ash-{7cfca58,0fc29f9}` (desktop
builds; 0fc29f9 kernels are byte-identical to 34ad9f1, GPU tests pass
on the 4060).  Run when the embargo file is gone:

```
cd $SP/work/coopmat && GPU_WAIT=3600 $SP/bin/gpu3070 timeout 580 \
  <worktree>/benchmarks/campaign/coopmat-sm-data/llama_env_ab.sh 5 results/pp512_final_3070.log \
  base=ML_X=0:$PWD/bin/llama_ash-7cfca58 new=ML_X=0:$PWD/bin/llama_ash-0fc29f9
```

Expected from the clean per-GEMM 3070 data (§12.3/§5): q/o +3%, QKV
+19%, gate/up +6%, down +5% vs grokmerge's picks.

## 10. Negative results (do not retry without a new mechanism)

| experiment | result | why (best guess) |
|---|---|---|
| 128x64 BK=64 | −2 to −15% on 15 of 16 cells (+1.3% on 3070 256x2048x2048) | 168 regs → 3 CTAs/SM; fewer barriers do not pay for occupancy |
| 128x64 on 256 threads (8 x 32x32) | −5.5 to −19% (all 16 cells) | 77 regs but 8 warps/CTA → 3 CTAs/SM, half the MMAs per fragment load |
| 64x128 with 64x32 slices | −2.6 to +6.8% vs 64x128, never the best tile | same 168 registers; the slice shape is not the register driver |
| 64x64 BK=64 | −5 to −17% on 15 of 16 cells | 96 regs, 18 KiB shared → 5 CTAs/SM |
| 128x64 / 128x128 M-fastest | ≤ +1.3% over x-fast 128x64 anywhere; −58% at 4096³ (3070) | 128-row tiles already stream B only M/128 times; A re-reads dominate |
| 64x64 M-fastest on large A | −5 to −39% at M=2048 / 4096³ | A re-streamed per column block; handled by the traffic test |
| SM-count-only joint model | 2-2.5% mean regret, 30-40% max | cannot express the L2-dependent 64x64 efficiency |

## 11. Follow-ups

- **3070, PENDING (embargo)**: pp512 A/B (§9 command); the 15-round
  128x128-vs-128x64 A/B (`$SP/work/coopmat/large3070.sh` (copy in `coopmat-sm-data/jobs/`), §12.5); the
  remaining cold batches (`ML_AB_COLD_MB=64 $SP/work/coopmat/run3070.sh
  cold-3070 a16_128 a16_256 a16_1024 a16_1920 t128 m256 m384 sq m768
  m1024`) and a clean re-run of the quarantined warm batches
  (`run3070.sh verify-3070 m512 a16_512 m768 m1024 m2048 t128 t1920
  a16_128 a16_256 a16_1024 a16_1920`).  Then `python3 coldfit.py`.
- 4060 a16 128x5632x2048 cold (128x64 +17% over the pick): M=128 is a
  single 128-row block; needs more M=128 data before a special case.
- 3070 single-pass M-fast at M=256/384 (rows 4-6) wins warm on 5 of 8
  shapes (+1-6%) but loses 7% at 256x2048x2048; the rule requires 8
  row blocks.  Cold 3070 m256/m384 would settle it.
- A real L2 query (or one-shot probe) instead of the device-ID table.
- 64x128's 168 registers: find what the compiler keeps live (SASS agent).

## 12. Cold-L2 refit (2026-09-27, commit 0fc29f9)

### 12.1 Why

4060 pp512 (§9): the 100df0d rule was 1.5% *slower* than exp/grokmerge,
and forcing the small-L2 branch (`ML_L2_MB=1`, i.e. 64x64 M-fast on all
four GEMMs) was 3.4% *faster* than grokmerge.  The large-L2 branch had
been fitted on warm `ml_bench ab` runs, where B stays in the 24 MiB L2
between the 20 back-to-back repeats, so x-fast order never paid for its
B re-streams.  In a forward pass every weight matrix is cold.

### 12.2 Protocol

`ML_AB_COLD_MB=64` (34ad9f1): before every timed GEMM a 64 MiB buffer
is written (evicting L2) and A is re-read (activations are warm in a
real forward: the previous op produced them).  Same interleaved `ab`
protocol otherwise (7 rounds, rotation, GPU medians), kernels
`f16w_coopmat_{m128n64,m64n64,m64n64_mfast,aligned,m64n128},auto`:

```
ML_AB_COLD_MB=64 BIN=<ml_bench 34ad9f1> coopmat-sm-data/verify.sh <outdir> <batch>...   # one GPU-lock hold per chunk
```

4060: louise `~/ta-wt/cm-var` build (100df0d + cold mode), all 77 shapes
(`cold-4060/`; 2026-09-26 21:02 and 2026-09-27 07:08-07:20, launcher
`louise:~/ta-wt/cm-results/cold4060_rest.sh`).  3070: desktop binary
`$SP/work/coopmat/bin/ml_bench-34ad9f1`, M=512 f16w + a16 (`cold-3070/`,
07:15 / 07:19, launcher `$SP/work/coopmat/run3070.sh`); the other 3070
cold batches were blocked by the embargo.

### 12.3 What the cold data say

64x64 M-fast gain over 64x64 x-fast (cold, same run): on the 4060 it is
**+6% to +68% wherever M-fast moves fewer bytes** (traffic ratio
< 0.75), whether or not B fits the L2 — e.g. a16 512x5632x2048 +10.5%
cold vs +0.9% warm, 256x5632x5632 +13.8%; +0-8% at ratio 0.75-1.0; and
−17% to −32% only once A is large *and* re-streamed (M >= 1920 with
K = 5632, 3072³, 4096³).  On the 3070, single-pass (x-fast `xp` = 1)
M=512 grids gain +2.7% to +14% although the traffic model calls the two
orders equal.  a16 pp512 GEMMs, cold (TF/s):

| GEMM | 4060 64x64 | 4060 M-fast | 4060 128x64 | 3070 64x64 | 3070 M-fast | 3070 128x64 |
|---|---:|---:|---:|---:|---:|---:|
| q/o 512x2048x2048 | 27.14 | **27.27** | 24.54 | 32.24 | **33.34** | 31.33 |
| QKV 512x2560x2048 | 26.22 | **26.61** | 26.08 | 34.41 | **35.34** | 29.64 |
| gate/up 512x5632x2048 | 26.07 | **28.81** | 26.36 | 29.93 | **36.72** | 34.69 |
| down 512x2048x5632 | 28.65 | **28.71** | 25.57 | 32.86 | **34.43** | 33.19 |

The 3070 M-fast wins also hold warm (§5 raster candidates: 31.18 /
33.54 / 35.21 / 33.55 vs 64x64 30.50 / 33.05 / 29.05 / 32.06), so they
are not a cold-only effect.

### 12.4 The refitted rule (`routing.rs`, `CoopmatGrid::m_fast_64`)

64x64 is launched M-fastest iff

* M-fast moves clearly fewer bytes (`|A| mp + |B| < 0.75 (|B| xp + |A|)`,
  unchanged) **or** the x-fast grid is a single pass at least 8 row
  blocks tall and 32 column blocks wide (M >= 512, N >= 2048; new);
* **and** its A re-reads are cheap: `|A| (mp - 1) <= 8 MiB` (unchanged)
  **or** A fits half of a large L2 (new; replaces "never if B fits a
  large L2").

rho is unchanged except that the small-L2 f16w value is keyed on
`xp > 1` only (identical to 100df0d wherever 100df0d could pick M-fast;
a single-pass M-fast grid keeps the single-pass rho).  A grid search of
rho on the cold 4060 set (6x6x5x6 values) improves mean regret only
0.71% -> 0.65%, so rho stays.

**Composition with the f16-accumulate rule** (branch `campaign/f16acc`,
`coopmat_tile_acc16`): the accumulator-independent pieces are now
`CoopmatGrid` — `new` (SM count, L2, operand bytes), `m_fast_64`
(raster) and `balance` (wave balance x occupancy floor).
`coopmat_tile` is the f32-accumulate rule = grid + its rho table.  An
accumulator-aware dispatcher matches on `CoopmatAcc`: `F32` ->
`coopmat_tile`; `F16`/`F16Promote` -> an acc16 rule that builds the same
grid and scores its own candidates (64x128 / 128x128) with `balance`.
f16acc's tiles have no M-fast sibling yet, so that rule would not call
`m_fast_64` until one exists (f16acc's `COOPMAT1` table would also need
a `T64Mfast` row per accumulator, or `selection_acc` must map
`T64Mfast` to the x-fast kernel for the f16 classes).

Candidate rules replayed on every clean same-run set
(`python3 coldfit.py`; "slower" = vs the grokmerge pick's cell in the
same run; verify-3070 = the clean sq/m256/m384 batches only):

```
--- shipped (100df0d)
  cold-4060    n=77 regret mean  1.24% max  17.2%  vs grokmerge x1.077  slower>1%: 2  a512x5632x2048 64x64<-128x64 +1.1%, g384x2560x2048 128x64<-64x64 +2.4%
  cold-3070    n=11 regret mean  1.67% max   4.8%  vs grokmerge x1.039  slower>1%: 0  
  verify-4060  n=77 regret mean  0.42% max   7.6%  vs grokmerge x1.083  slower>1%: 1  g384x2560x2048 128x64<-64x64 +2.3%
  verify-3070  n=19 regret mean  1.34% max   6.5%  vs grokmerge x1.099  slower>1%: 2  sq3072 128x64<-128x128 +1.6%, sq4096 128x64<-128x128 +1.3%
--- drop large-L2 fit exemption
  cold-4060    n=77 regret mean  0.87% max  17.5%  vs grokmerge x1.081  slower>1%: 1  g384x2560x2048 128x64<-64x64 +2.4%
  cold-3070    n=11 regret mean  1.67% max   4.8%  vs grokmerge x1.039  slower>1%: 0  
  verify-4060  n=77 regret mean  0.54% max   7.6%  vs grokmerge x1.081  slower>1%: 5
  verify-3070  n=19 regret mean  1.34% max   6.5%  vs grokmerge x1.099  slower>1%: 2  sq3072 128x64<-128x128 +1.6%, sq4096 128x64<-128x128 +1.3%
--- small-L2 rule everywhere
  cold-4060    n=77 regret mean  1.29% max  17.5%  vs grokmerge x1.077  slower>1%: 5
  cold-3070    n=11 regret mean  1.67% max   4.8%  vs grokmerge x1.039  slower>1%: 0  
  verify-4060  n=77 regret mean  1.09% max  10.2%  vs grokmerge x1.076  slower>1%: 9
  verify-3070  n=19 regret mean  1.34% max   6.5%  vs grokmerge x1.099  slower>1%: 2  sq3072 128x64<-128x128 +1.6%, sq4096 128x64<-128x128 +1.3%
--- NEW: cold-fitted raster (routing.rs)
  cold-4060    n=77 regret mean  0.71% max  17.5%  vs grokmerge x1.083  slower>1%: 1  g384x2560x2048 128x64<-64x64 +2.4%
  cold-3070    n=11 regret mean  0.00% max   0.0%  vs grokmerge x1.056  slower>1%: 0  
  verify-4060  n=77 regret mean  0.66% max   7.6%  vs grokmerge x1.080  slower>1%: 5  a128x5632x2048 64x64m<-64x64 +1.9%, a256x5632x2048 64x64m<-64x64 +1.7%, a512x2048x2048 64x64m<-64x64 +2.5%, a512x2048x5632 64x64m<-64x64 +3.8%, g384x2560x2048 128x64<-64x64 +2.3%
  verify-3070  n=19 regret mean  1.34% max   6.5%  vs grokmerge x1.099  slower>1%: 2  sq3072 128x64<-128x128 +1.6%, sq4096 128x64<-128x128 +1.3%
```

`verify-4060` (warm) is the only set where 0fc29f9 loses to 100df0d:
the four a16 cells where warm x-fast beats M-fast by 1.7-3.8% are the
warm-L2 artifact itself (cold, M-fast wins or ties them).

Every changed 3070 pick has clean evidence (new/grokmerge, same run):
512x2048x2048 cold 1.037; 512x2560x2048 candidates 1.123 / cold 1.117;
a16 256x5632x2048 candidates 1.153; a16 512x2048x2048 candidates 1.023 /
cold 1.034; a16 512x2048x5632 candidates 1.046 / cold 1.048; a16
512x2560x2048 candidates 1.197 / cold 1.192; a16 512x5632x2048
candidates 1.043 / cold 1.058.  Every other §4 shape keeps the 100df0d
pick; replaying 0fc29f9 on the clean §4 sweep gives **0 of 77 3070
shapes slower than grokmerge by >1%, geomean ×1.081** (M-fast cells
scaled by their in-run ratio to 64x64 from the candidate / cold runs).
The one caveat is 3072³ / 4096³ (§12.5).

Known residuals: 4060 a16 128x5632x2048 cold — 128x64 20.53 vs the
pick (64x64 M-fast) 17.47 TF/s (−15%; warm, 64x64 wins, and grokmerge
picks 64x64 too); 4060 384x2560x2048 — 128x64 picked, 64x64 x-fast
+2.4% (both regimes).

#### 4060 — cold sweep (`cold-4060/`, `python3 coldfit.py table cold-4060 4060`; TF/s)

| shape | 128x64 | 64x64 | 64x64 M-fast | 128x128 | 64x128 | grokmerge | 100df0d | new | new/grokmerge |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| 128x2048x2048 | 12.91 | **15.87** | 15.30 | 10.99 | 14.19 | 64x64 | 64x64 | 64x64 | 1.000 |
| 128x2048x5632 | 13.70 | **17.27** | 16.33 | 11.59 | 15.47 | 64x64 | 64x64 | 64x64 | 1.000 |
| 128x2560x2048 | 14.82 | **15.89** | 15.70 | 12.88 | 15.44 | 64x64 | 64x64 | 64x64 | 1.000 |
| 128x5632x2048 | **19.95** | 16.74 | 16.22 | 18.77 | 15.26 | 64x64 | 128x64 | 128x64 | 1.192 |
| 128x32000x2048 | **23.98** | 14.27 | 23.28 | 22.05 | 14.21 | 64x64 | 128x64 | 128x64 | 1.680 |
| 256x2048x2048 | 21.29 | 21.39 | **21.94** | 17.03 | 20.64 | 64x64 | 128x64 | 128x64 | 0.996 |
| 256x2048x5632 | 23.11 | 22.58 | **23.40** | 17.99 | 22.10 | 64x64 | 128x64 | 128x64 | 1.024 |
| 256x2560x2048 | 21.33 | 21.96 | **22.57** | 20.25 | 18.10 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x2560x5632 | 22.71 | 23.33 | **24.13** | 21.44 | 18.89 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x5632x2048 | 22.29 | 20.64 | **23.30** | 20.90 | 18.88 | 64x64 | 128x64 | 128x64 | 1.080 |
| 256x5632x5632 | 24.19 | 21.30 | **24.23** | 22.68 | 19.31 | 64x64 | 128x64 | 128x64 | 1.136 |
| 256x32000x2048 | **25.25** | 14.93 | 25.16 | 23.60 | 15.05 | 64x64 | 128x64 | 128x64 | 1.691 |
| 384x2048x2048 | **25.21** | 22.06 | 21.47 | 23.67 | 19.56 | 64x64 | 128x64 | 128x64 | 1.143 |
| 384x2048x5632 | **27.13** | 22.26 | 22.32 | 25.78 | 20.85 | 64x64 | 128x64 | 128x64 | 1.219 |
| 384x2560x2048 | 21.78 | 22.30 | **23.91** | 18.56 | 23.11 | 64x64 | 128x64 | 128x64 | 0.977 |
| 384x2560x5632 | 23.93 | 23.00 | **24.97** | 19.01 | 24.63 | 64x64 | 128x64 | 128x64 | 1.040 |
| 384x5632x2048 | 24.25 | 22.31 | **24.57** | 21.58 | 21.67 | 64x64 | 128x64 | 128x64 | 1.087 |
| 384x5632x5632 | **25.98** | 20.78 | 25.24 | 22.99 | 21.42 | 64x64 | 128x64 | 128x64 | 1.250 |
| 384x32000x2048 | **24.97** | 15.05 | 24.84 | 23.59 | 15.19 | 64x64 | 128x64 | 128x64 | 1.659 |
| 512x2048x2048 | 22.60 | 23.91 | **23.99** | 19.79 | 21.72 | 64x64 | 64x64 | 64x64 | 1.000 |
| 512x2048x5632 | 23.36 | 24.71 | **24.95** | 20.01 | 22.72 | 64x64 | 64x64 | 64x64 | 1.000 |
| 512x2560x2048 | **24.72** | 23.08 | 23.66 | 20.98 | 22.70 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x2560x5632 | **26.10** | 23.58 | 24.40 | 21.68 | 21.80 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x5632x2048 | 24.62 | 22.99 | **24.98** | 21.90 | 22.29 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x5632x5632 | **25.90** | 19.62 | 25.33 | 23.10 | 23.11 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x32000x2048 | 24.60 | 15.16 | **25.05** | 23.91 | 15.26 | 128x64 | 128x64 | 128x64 | 1.000 |
| 768x2048x2048 | **26.18** | 24.48 | 24.08 | 24.51 | 24.53 | 128x128 | 128x64 | 128x64 | 1.068 |
| 768x2048x5632 | **27.55** | 24.90 | 25.01 | 25.82 | 26.05 | 128x128 | 128x64 | 128x64 | 1.067 |
| 768x2560x2048 | **25.92** | 24.12 | 24.70 | 23.54 | 24.70 | 128x128 | 128x64 | 128x64 | 1.101 |
| 768x2560x5632 | **27.01** | 24.06 | 25.35 | 23.97 | 23.64 | 128x128 | 128x64 | 128x64 | 1.127 |
| 768x5632x2048 | **25.72** | 23.79 | 25.36 | 23.79 | 23.59 | 128x128 | 128x64 | 128x64 | 1.081 |
| 768x5632x5632 | **26.10** | 18.05 | 25.22 | 24.57 | 22.94 | 128x128 | 128x64 | 128x64 | 1.062 |
| 768x32000x2048 | **24.66** | 15.25 | 24.02 | 23.84 | 15.35 | 128x128 | 128x64 | 128x64 | 1.034 |
| 1024x1024x1024 | 21.06 | **22.95** | 22.25 | 17.86 | 20.29 | 64x64 | 64x64 | 64x64 | 1.000 |
| 1024x2048x2048 | **25.94** | 24.21 | 24.37 | 22.47 | 24.43 | 128x128 | 128x64 | 128x64 | 1.154 |
| 1024x2048x5632 | **27.21** | 24.58 | 24.57 | 23.27 | 25.64 | 128x128 | 128x64 | 128x64 | 1.169 |
| 1024x2560x2048 | **25.52** | 24.09 | 24.74 | 23.19 | 24.19 | 128x128 | 128x64 | 128x64 | 1.100 |
| 1024x2560x5632 | **26.55** | 23.50 | 25.17 | 23.64 | 24.43 | 128x128 | 128x64 | 128x64 | 1.123 |
| 1024x5632x2048 | **25.65** | 24.01 | 25.44 | 23.66 | 23.86 | 128x128 | 128x64 | 128x64 | 1.084 |
| 1024x5632x5632 | **25.24** | 17.22 | 24.77 | 24.26 | 23.08 | 128x128 | 128x64 | 128x64 | 1.041 |
| 1024x32000x2048 | **24.65** | 14.70 | 23.44 | 23.84 | 14.94 | 128x128 | 128x64 | 128x64 | 1.034 |
| 1536x1536x1536 | **25.73** | 23.65 | 24.28 | 23.88 | 24.54 | 128x128 | 128x64 | 128x64 | 1.077 |
| 1920x2048x2048 | **27.09** | 24.91 | 25.08 | 25.10 | 25.75 | 128x128 | 128x64 | 128x64 | 1.079 |
| 1920x2048x5632 | **28.09** | 24.92 | 18.87 | 26.28 | 26.69 | 128x128 | 128x64 | 128x64 | 1.069 |
| 1920x2560x2048 | **26.87** | 24.50 | 25.05 | 24.44 | 25.73 | 128x128 | 128x64 | 128x64 | 1.099 |
| 1920x5632x2048 | **26.16** | 24.22 | 25.22 | 24.39 | 25.15 | 128x128 | 128x64 | 128x64 | 1.072 |
| 1920x32000x2048 | **24.18** | 14.77 | 23.37 | 23.92 | 14.99 | 128x128 | 128x64 | 128x64 | 1.011 |
| 2048x2048x2048 | **26.63** | 24.92 | 25.02 | 24.30 | 25.35 | 128x128 | 128x64 | 128x64 | 1.096 |
| 2048x2048x2048 | **26.64** | 24.98 | 25.19 | 24.38 | 25.39 | 128x128 | 128x64 | 128x64 | 1.093 |
| 2048x2048x5632 | **27.52** | 24.89 | 18.76 | 25.08 | 26.20 | 128x128 | 128x64 | 128x64 | 1.098 |
| 2048x2560x2048 | **26.85** | 24.52 | 25.11 | 24.53 | 25.73 | 128x128 | 128x64 | 128x64 | 1.095 |
| 2048x2560x5632 | **26.99** | 22.97 | 18.51 | 25.18 | 25.16 | 128x128 | 128x64 | 128x64 | 1.072 |
| 2048x5632x2048 | **25.75** | 23.82 | 24.98 | 24.25 | 25.06 | 128x128 | 128x64 | 128x64 | 1.062 |
| 2048x5632x5632 | **25.63** | 16.23 | 18.57 | 24.67 | 23.28 | 128x128 | 128x64 | 128x64 | 1.039 |
| 2048x32000x2048 | **24.11** | 14.81 | 23.54 | 23.95 | 14.99 | 128x128 | 128x64 | 128x64 | 1.007 |
| 3072x3072x3072 | **27.41** | 24.18 | 16.38 | 26.17 | 26.96 | 128x128 | 128x64 | 128x64 | 1.047 |
| 4096x4096x4096 | **26.23** | 21.58 | 17.80 | 25.49 | 23.58 | 128x128 | 128x64 | 128x64 | 1.029 |
| a16 128x2048x2048 | 13.96 | **15.83** | 15.58 | 11.11 | 13.75 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 128x2048x5632 | 14.95 | **17.93** | 17.28 | 12.01 | 15.03 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 128x2560x2048 | 15.23 | **15.76** | 15.68 | 12.90 | 15.36 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 128x5632x2048 | **20.53** | 17.51 | 17.47 | 18.87 | 15.58 | 64x64 | 64x64 | 64x64 M-fast | 0.998 |
| a16 256x2048x2048 | 23.05 | **23.51** | 23.09 | 17.85 | 21.19 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 256x2048x5632 | 24.86 | **25.27** | 24.58 | 19.05 | 22.65 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 256x2560x2048 | 22.17 | **24.83** | 24.60 | 20.98 | 18.49 | 64x64 | 64x64 | 64x64 | 1.000 |
| a16 256x5632x2048 | 22.92 | 22.73 | **26.40** | 21.60 | 19.51 | 64x64 | 64x64 | 64x64 M-fast | 1.161 |
| a16 512x2048x2048 | 24.54 | 27.14 | **27.27** | 21.45 | 22.39 | 64x64 | 64x64 | 64x64 M-fast | 1.005 |
| a16 512x2048x5632 | 25.57 | 28.65 | **28.71** | 21.67 | 23.53 | 64x64 | 64x64 | 64x64 M-fast | 1.002 |
| a16 512x2560x2048 | 26.08 | 26.22 | **26.61** | 21.66 | 23.01 | 128x64 | 64x64 | 64x64 M-fast | 1.020 |
| a16 512x5632x2048 | 26.36 | 26.07 | **28.81** | 23.30 | 23.49 | 128x64 | 64x64 | 64x64 M-fast | 1.093 |
| a16 1024x2048x2048 | 28.11 | 28.61 | **28.74** | 24.00 | 25.33 | 128x128 | 64x64 | 64x64 M-fast | 1.197 |
| a16 1024x2048x5632 | 29.36 | 29.38 | **29.91** | 25.28 | 26.39 | 128x128 | 64x64 | 64x64 M-fast | 1.183 |
| a16 1024x2560x2048 | 27.37 | 28.39 | **28.66** | 24.78 | 24.87 | 128x128 | 64x64 | 64x64 M-fast | 1.157 |
| a16 1024x5632x2048 | 27.97 | 28.18 | **29.94** | 25.68 | 25.38 | 128x128 | 64x64 | 64x64 M-fast | 1.166 |
| a16 1920x2048x2048 | 29.68 | 29.85 | **30.00** | 27.09 | 26.72 | 128x128 | 64x64 | 64x64 | 1.102 |
| a16 1920x2048x5632 | 30.07 | **30.52** | 29.93 | 28.02 | 27.77 | 128x128 | 64x64 | 64x64 | 1.089 |
| a16 1920x2560x2048 | 29.18 | 29.13 | **29.50** | 25.83 | 26.54 | 128x128 | 64x64 | 64x64 | 1.128 |
| a16 1920x5632x2048 | 28.77 | 29.02 | **29.86** | 26.40 | 26.83 | 128x128 | 64x64 | 64x64 M-fast | 1.131 |
#### 3070 — cold, M=512 (`cold-3070/`)

| shape | 128x64 | 64x64 | 64x64 M-fast | 128x128 | 64x128 | grokmerge | 100df0d | new | new/grokmerge |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| 512x2048x2048 | 30.75 | 30.78 | **31.93** | 24.03 | 29.53 | 64x64 | 64x64 | 64x64 M-fast | 1.037 |
| 512x2048x5632 | **32.58** | 28.37 | 32.33 | 24.65 | 31.18 | 64x64 | 128x64 | 128x64 | 1.148 |
| 512x2560x2048 | 29.98 | 32.27 | **33.47** | 28.90 | 26.73 | 128x64 | 64x64 | 64x64 M-fast | 1.117 |
| 512x2560x5632 | **31.91** | 30.44 | 31.68 | 30.85 | 28.06 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x5632x2048 | **34.07** | 26.45 | 34.07 | 32.11 | 28.48 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x5632x5632 | **34.86** | 23.89 | 31.08 | 33.20 | 28.24 | 128x64 | 128x64 | 128x64 | 1.000 |
| 512x32000x2048 | **34.73** | 24.51 | 32.63 | 32.67 | 24.56 | 128x64 | 128x64 | 128x64 | 1.000 |
| a16 512x2048x2048 | 31.33 | 32.24 | **33.34** | 24.03 | 29.01 | 64x64 | 64x64 | 64x64 M-fast | 1.034 |
| a16 512x2048x5632 | 33.19 | 32.86 | **34.43** | 24.85 | 31.06 | 64x64 | 64x64 | 64x64 M-fast | 1.048 |
| a16 512x2560x2048 | 29.64 | 34.41 | **35.34** | 29.01 | 26.10 | 128x64 | 64x64 | 64x64 M-fast | 1.192 |
| a16 512x5632x2048 | 34.69 | 29.93 | **36.72** | 32.67 | 28.31 | 128x64 | 64x64 M-fast | 64x64 M-fast | 1.058 |
#### 3070 — warm verify, clean batches (`verify-3070/`, 2026-09-26 21:00, 34ad9f1)

| shape | 128x64 | 64x64 | 64x64 M-fast | 128x128 | 64x128 | grokmerge | 100df0d | new | new/grokmerge |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| 256x2048x2048 | 18.34 | **27.83** | 25.78 | 14.29 | 18.57 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x2048x5632 | 20.91 | **29.94** | 27.81 | 15.93 | 20.98 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x2560x2048 | 24.22 | 26.73 | **28.46** | 19.37 | 24.79 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x2560x5632 | 26.11 | 28.29 | **29.38** | 20.15 | 25.62 | 64x64 | 64x64 | 64x64 | 1.000 |
| 256x5632x2048 | **31.77** | 26.42 | 29.61 | 30.25 | 27.27 | 64x64 | 128x64 | 128x64 | 1.202 |
| 256x5632x5632 | **33.02** | 23.62 | 26.16 | 30.84 | 28.46 | 64x64 | 128x64 | 128x64 | 1.398 |
| 256x32000x2048 | **32.67** | 24.70 | 32.50 | 29.87 | 24.47 | 64x64 | 128x64 | 128x64 | 1.323 |
| 384x2048x2048 | 22.51 | 26.21 | **26.81** | 17.49 | 21.91 | 64x64 | 64x64 | 64x64 | 1.000 |
| 384x2048x5632 | 24.72 | 27.92 | **28.80** | 18.24 | 23.21 | 64x64 | 64x64 | 64x64 | 1.000 |
| 384x2560x2048 | 28.53 | 28.38 | **29.18** | 22.25 | 27.15 | 64x64 | 64x64 | 64x64 | 1.000 |
| 384x2560x5632 | **30.13** | 27.33 | 28.62 | 22.07 | 28.32 | 64x64 | 128x64 | 128x64 | 1.102 |
| 384x5632x2048 | 31.44 | 27.12 | **32.53** | 27.58 | 27.44 | 64x64 | 128x64 | 128x64 | 1.159 |
| 384x5632x5632 | **32.20** | 25.31 | 30.51 | 27.59 | 27.86 | 64x64 | 128x64 | 128x64 | 1.272 |
| 384x32000x2048 | **33.05** | 24.67 | 32.36 | 29.93 | 24.58 | 64x64 | 128x64 | 128x64 | 1.340 |
| 1024x1024x1024 | 27.97 | **29.00** | 28.17 | 21.95 | 26.72 | 64x64 | 64x64 | 64x64 | 1.000 |
| 1536x1536x1536 | 30.45 | **30.63** | 28.26 | 26.59 | 28.73 | 128x128 | 128x64 | 128x64 | 1.145 |
| 2048x2048x2048 | **33.09** | 30.63 | 22.76 | 29.57 | 31.00 | 128x128 | 128x64 | 128x64 | 1.119 |
| 3072x3072x3072 | 33.12 | 23.33 | 14.41 | **33.64** | 30.91 | 128x128 | 128x64 | 128x64 | 0.985 |
| 4096x4096x4096 | 33.69 | 23.61 | 13.77 | **34.12** | 26.65 | 128x128 | 128x64 | 128x64 | 0.987 |
### 12.5 128x64 vs 128x128 on large 3070 shapes (NEXT.md question 1)

Answer: **no measurable reason to keep 128x128; 128x64 stays.**  All
clean 3070 interleaved runs at the large end (TF/s, 128x128 / 128x64):

| shape | §4 sweep (e1670b8) | §5 raster run | verify (34ad9f1) |
|---|---|---|---|
| 4096³ | 34.38 / 34.38 | 33.83 / 33.38 | 34.12 / 33.69 |
| 3072³ | 33.42 / 34.29 | — | 33.64 / 33.12 |
| 2048³ | 29.80 / 31.76 and 30.40 / 32.39 | — | 29.57 / 33.09 |
| 2048x5632x5632 | 33.10 / 32.90 | — | — |
| 2048x32000x2048 | 33.05 / 32.98 | — | — |
| 1920x32000x2048 | 33.27 / 33.23 | — | — |

128x128 leads by at most 1.6% (3072³ verify, while the §4 run has
128x64 ahead by 2.6% on the same shape) and by 0.0-1.3% at 4096³ — inside
the 1-4% same-kernel noise of multi-millisecond shapes (§3) — while
128x64 leads by 6-12% at 2048³ and by up to 56% on mid-size grids.  On
the 4060, cold, 128x64 leads at 4096³ (26.23 vs 25.49) and 3072³
(27.41 vs 26.17).  Strictly, the 4096³ and 3072³ verify cells are the
only clean 3070 measurements where the new pick trails grokmerge's by
more than 1% (1.3%, 1.6%).  A dedicated 15-round, two-kernel warm +
cold A/B on eight large shapes (`$SP/work/coopmat/large3070.sh` (copy in `coopmat-sm-data/jobs/`),
`large.txt`) was queued but refused by the embargo: **PENDING**.  If it
confirms 128x128 ahead by >1% at 4096³, the fix is a "128-tile grid
>= 8 waves on a small L2" branch; nothing in the data supports it yet
(an SM-count-only version would lose 8.5% at 2048³ on the 4060, cold).

### 12.6 Model prediction error across GPUs and regimes

The §6 model (fit per tile and A dtype on one set, predict every cell of
another; regret = the model's choice among the four x-fast tiles vs the
measured best; `python3 coldfit.py model`):

| fit on | predict | shapes | median abs err | p90 abs err | regret mean / max |
|---|---|---:|---:|---:|---:|
| 4060 warm | 4060 warm | 77 | 3.2% | 10.4% | 0.38% / 8.7% |
| 4060 warm | 3070 warm | 77 | 4.6% | 21.8% | 2.47% / 38.6% |
| 4060 warm | 4060 cold | 77 | 5.2% | 18.5% | 0.56% / 17.2% |
| 4060 warm | 3070 cold | 11 | 4.8% | 13.9% | 1.55% / 15.9% |
| 3070 warm | 4060 warm | 77 | 7.3% | 18.8% | 2.12% / 12.8% |
| 3070 warm | 3070 warm | 77 | 5.0% | 14.1% | 1.53% / 15.3% |
| 3070 warm | 4060 cold | 77 | 5.7% | 16.8% | 1.25% / 10.6% |
| 3070 warm | 3070 cold | 11 | 4.7% | 11.7% | 1.73% / 7.6% |
| 4060 cold | 4060 warm | 77 | 9.1% | 18.0% | 0.95% / 12.8% |
| 4060 cold | 3070 warm | 77 | 6.9% | 15.8% | 3.18% / 46.8% |
| 4060 cold | 4060 cold | 77 | 4.4% | 14.1% | 0.35% / 8.9% |
| 4060 cold | 3070 cold | 11 | 4.9% | 10.5% | 0.10% / 1.0% |

Reading: across GPUs the model predicts time with a 4.6-7.3% median
error (p90 11-22%); switching regime (warm <-> cold) costs about as much
as switching GPU.  Its tile choice is good in-regime (<= 0.4% mean
regret on the 4060) but not across GPUs (2-3% mean, 38-47% max),
because of the L2-dependent 64x64 efficiency (§6) — which is why the
shipped rule is a fitted rule with an L2 class, not the model itself.
The refitted raster constants are in-sample (fitted on both GPUs' cold
data); there is no held-out GPU.
