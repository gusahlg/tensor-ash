# campaign/merge-1: integrate + sass + coopmat-sm + f16acc + ptx + v4t(large) + oddk

Status 2026-09-27 ~11:20: **merged, CPU gates green, GPU gates green on both
GPUs; stopped early (usage limit) with three small items pending** (end of
this file).  Branch `campaign/merge-1` (worktree `~/repos/tensor-ash-wt/c-merge1`),
base `campaign/integrate` 6b3ab32.  Raw logs: desktop
`~/repos/tensor-ash-wt/campaign-tools/work/merge1/`, louise
`~/ta-logs/merge1/` (mirrored to `work/merge1/louise/`).

## What merged (in order)

| step | source | commit(s) on merge-1 |
|---|---|---|
| 1 | campaign/sass 3342b90 (`ml_bench isa`, probe API 8608d07) | dfc465f |
| 2 | campaign/coopmat-sm dda5634 (SM count, M-fast 64x64, cold-L2 refit, `ML_AB_COLD_MB`) | cbebe85 |
| 3 | campaign/f16acc bf829ba (CoopmatAcc, acc16/acc16p twins, acc16 tile rule, gen M1) | 42309be |
| 4 | campaign/ptx aafb260 (PTX backend via VK_NV_cuda_kernel_launch) | fc89a46 |
| - | clippy 1.98 `as_chunks` (examples/coopmat_occupancy.rs, ml-bench isa; catalog.rs/create.rs already fixed by c8cd50b) | 024e70f |
| - | drop dead `CoopmatTile::of_selection` | 1f5f8e9 |
| 5 | exp/v4t: `large_bda_v4t` only; f32 Large => LargeBdaV4t; demote LargeBdaV4t => SmallBdaV4 | 04ac4d7 |
| 6 | exp/oddk tail truncation, except the 128x128 tiles (`ODDK_FULL_TAIL`) + `odd_k_tails_match_reference` | bbce3e7 |
| - | packed LM-head tail comment: time-neutral, not 7.4x (f6492f5 kept; fusion's 7a5bb8f is the same change; `LLAMA_ASH_TAIL` not merged) | 1b2a5cf |
| - | tests/ptx policy test for the two-implementation rule | 02fc984 |
| - | cherry-pick fusion d05ea55 (explicit a16 pin on the a16 route) | 7b926ef |
| - | CHANGELOG/README; 3070 thesis section re-recorded | b768ba4 |

## Conflict resolutions

- **routing.rs (coopmat-sm x f16acc)**: coopmat-sm's `CoopmatDevice`/`CoopmatGrid`
  SM-aware rule is the F32 rule; f16acc's measured `coopmat_tile_acc16` serves
  F16/F16Promote; new dispatcher `coopmat_tile_acc(acc, batch, m, n, k, a16, device)`.
  The acc16 rule is NOT re-scored with `CoopmatGrid::balance` (that would be an
  unmeasured rule); `COOPMAT1` gains the f32 M-fast rows and `selection_acc` maps
  `T64Mfast` to the x-fast 64x64 twin for the f16 classes (no M-fast acc16 kernels).
  Unit tests: dispatcher, bijection incl. M-fast.
- **pipeline/mod.rs `a16_coopmat_index`**: f16acc's pin handling + `coopmat_tile_acc`
  with the device (SM + L2) inputs.
- **matmul_coopmat_kernel.glsl**: RASTER_M_FAST + ACC_F16/ACC_F16_PROMOTE; BK stays
  `#ifndef`-overridable in the literal branch.
- **Executor::new_with_config (f16acc x ptx)**: class kept iff
  `config.coopmat_acc == F32 || !ctx.coopmat_enabled || ctx.coopmat_f16_acc_enabled || pipeline.ptx_serves(config.coopmat_acc)`,
  else warn + F32; deprecated `ML_PTX_GEMM=f32|f16|f16p` mapping kept.  This made
  `ptx_policy_is_explicit_and_degrades_to_f32` fail on the 4060 (asserted the PTX-only
  rule; 148/149) -> test fixed in 02fc984.
- **ml-bench** commands.rs (`run_case_warm` on top of `prepare_case_tensors`), mod.rs,
  ab.rs imports, model_theses.rs (`packed_b` + `a16: false`), READMEs/CHANGELOG: unions.
- **d05ea55 cherry-pick**: on this lineage `plan_matmul`'s a16 branch already used
  `a16_coopmat_index`, which honors a16 pins (coopmat-sm 6e72043, 33affa9; f16acc
  3eaebeb/cc9bb05/4016328).  Checked at every measured commit: table pins, M-fast a16
  (33affa9 added it together with the "explicit a16 kernel runs as-is" branch),
  generator configs (cc9bb05) and pk2/pk8 (4016328, out-of-table branch) were all
  honored, and their distinct error levels in f16acc.md §8a corroborate that different
  kernels ran.  **No f16acc / coopmat-sm conclusion relied on the broken path; no
  re-measurement needed.**  The early return is kept for the M/N alignment error.

## oddk and v4t decisions

- oddk applied to every BDA/BDA_V4 tile except 128x128 (large_bda_v4, f16w_large_bda_v4,
  bk16_bda_v4, large_bda): their SPIR-V is byte-identical to before (glslc -Os, compared).
  3070 evidence (night-oddk, 2 runs x 4 pairs, `work/night/r3070/d3oddk{2,3}`, `nk.py cases`):
  +2.9/+2.9% 1023x1025x1027, +3.9/+4.4% 512^2x513, +12.3/+12.0% 128x1024x520,
  +2.6/+5.3% 1024^2x1031; -2.4/-2.5% 2048^2x2049 (large_bda_v4).
- **v4t body keeps its full tail**: an oddk twin (`large_bda_v4t_oddk`, measurement tree
  only) passed `v4t_kernels_match_reference` + `odd_k_tails_match_reference` on the 4060,
  but `ml_bench ab` (ML_ROUNDS=15, both orders, `louise/v4toddk.log`) gave
  **-7.0% / -6.4% at 1023x1025x1027**, +3/+3% 512^2x513, +4/+5% 256^2x263, and
  -0.6..+1.1% elsewhere -> not neutral-or-better, not applied.

## Gates

| gate | host | result | log |
|---|---|---|---|
| fmt --check, clippy --workspace --all-targets -D warnings | desktop | clean (7b926ef) | work/merge1/fmt.log, clippy3.log |
| unit tests | desktop | tensor-ash 76, ml-bench 34, test-support 14, capi 10, llama-ash 4, doc 1: all pass | unit-desktop2.log |
| GPU suite monolithic | 4060 | **149/149** (7b926ef; bbce3e7 had 148/149, fixed test) | louise/suite-mono-4060-2.log |
| `ML_PTX=1 ptx:: coopmat_acc::` + `ML_PTX_GEMM=all f16:: graph:: prepared:: replay::` | 4060 | 9/9, 25/25 | louise/ptx-4060.log |
| GPU suite chunked (25 modules, 3 batches) | 3070 | **149/149** | suite-3070-b{1,2,3}.log |
| PTX tests as above | 3070 | 5/5, 25/25 | ptx-3070.log |
| token gate 24/24 vs llama.cpp CUDA reference | both | **MATCH in all 6 modes** (f32, f16p, f16, ML_PTX=1 x {f32,f16p,f16}) | tokens-4060-2.log, tokens-3070.log |
| `thesis --all` (armed) | 4060 | **PASS, 0 FAIL** x2 (bbce3e7, 7b926ef) | louise/thesis-4060-{1,2}.log |
| `thesis --all` | 3070 | 4 FAIL x2 on the old section (T2 o 80%, T2 k/v 16%, T4 11.7/11.4 us, T5 +31.6/+30.3%), rest PASS -> section re-recorded (b768ba4) | thesis-3070-{1,2}.log |

Commands: `gate.sh {suite-mono,tokens,thesis}` / `job3070.sh {suite,ptx,tokens,thesis,ab}`
(in work/merge1), each under `lgpu`-equivalent locks / `gpu3070`.

## pp512 / tg128 (llama_ash bench, TinyLlama f16, `abv.sh`, rotated rounds, medians)

| GPU | build / mode | pp512 t/s | prefill GPU ms | tg128 t/s |
|---|---|---:|---:|---:|
| 4060 | integrate (4 runs) | 12,912 | 39.33 | 104.1 |
| 4060 | merge-1 f32 (4 runs) | **13,301 (+3.0%)** | 38.23 | 105.2 (noise, 103.8-108.3 both) |
| 4060 | merge-1 `ML_COOPMAT_ACC=f16p` (4) | 14,818 (1.12x) | 34.27 | 104 |
| 4060 | merge-1 `f16` (4) | 16,129 (1.22x) | 31.53 | 104 |
| 4060 | merge-1 `ML_PTX=1` f16p (4) | 16,580 (1.25x) | 30.59 | 104 |
| 4060 | merge-1 `ML_PTX=1` f16 (4) | 17,706 (1.34x) | 28.52 | 105 |
| 3070 | integrate (4 runs) | 14,647 | 34.82 | 172.4 |
| 3070 | merge-1 f32 (4 runs) | **15,538 (+6.1%)** | 32.79 | 172.3 |
| 3070 | merge-1 f16p (2) | 19,157 (1.23x) | 26.62 | 172 |
| 3070 | merge-1 f16 (2) | 18,682 (1.20x) | 27.26 | 172 |
| 3070 | merge-1 `ML_PTX=1` f16p (2) | 20,435 (1.32x) | 24.90 | 172 |
| 3070 | merge-1 `ML_PTX=1` f16 (2) | 21,944 (1.41x) | 23.20 | 172 |

f32 tokens identical everywhere.  coopmat-sm alone on the 3070 (its final3070 A/B,
`work/coopmat/results/pp512_final_3070.log`, 5 rounds): 29.38 vs 31.53 ms prefill GPU
(-6.8%) vs grokmerge — **no regression** (the >1% flag condition is not met).

## Pending / flags (nothing blocks correctness)

1. 3070: one armed `thesis --all` with the re-recorded section (expect 21 PASS), then
   update THESES.md's regression-target table (3070 column still shows 2026-08-15 values)
   and the T4/T5 prose bands.  Command:
   `$SP/bin/gpu3070 bash $SP/work/merge1/job3070.sh thesis`.
2. **3070 GLSL f16 (18.7k) is slower than GLSL f16p (19.2k)**, unlike f16acc's report
   (3070 f16 20.6k).  Likely the acc16 tile rule (4060-fitted, SM-agnostic) vs the new
   SM-aware f32 base on 46 SMs, or a merge interaction in `with_coopmat_acc`; the PTX
   modes are unaffected.  Needs a `ml_bench ab auto@f16,auto@f16p` look on the 3070
   before recommending GLSL f16 on GA104.  f16p (the recommended mode) is fine.
3. 4060 T5 is now +21-22% vs its 25% gate (integrate +11..+18%): margin shrinking because
   the whole got faster while the f32-A pieces did not.
4. Not done: a second 3070 round of the mode A/B (2 runs each), THESES.md edits.
