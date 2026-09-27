# coopmat-sm progress log (resumable state)

Report: `coopmat-sm.md`.  Data + scripts: `coopmat-sm-data/`.
Louise trees: `~/ta-wt/cm-base` (7cfca58 grokmerge build), `cm-sm`
(100df0d), `cm-var` (100df0d + ML_AB_COLD_MB); results `~/ta-wt/cm-results/`.

## 2026-09-26 (successor #2, from 20:37)

- Predecessor stopped ~20:00 (usage limit).  Its last completed
  measurements: 4060 warm verify sweep at 100df0d (`verify-4060/`,
  geomean x1.083 vs grokmerge, 1 shape slower by 2.3%), and the 4060
  pp512 A/Bs:
  - `pp512_ab_4060.log`: grokmerge 39.42 ms vs 100df0d 40.00 ms GPU
    prefill (medians of 5 alternating pairs): **branch 1.5% slower**.
  - `pp512_variants_4060.log` (4 rotated rounds): base 39.71,
    ML_SM_COUNT=0 fallback 39.65, new (large-L2 rule) 40.08,
    ML_L2_MB=1 (small-L2 rule) 38.36 ms.
  - Reading: the large-L2 branch of the rule was fitted on warm-L2 `ab`
    (B stays resident across back-to-back repeats); in a model forward
    the weights are cold and the small-L2 rule (64x64 M-fast on all
    four pp512 GEMMs) wins by 3.4%.
- 34ad9f1: `ML_AB_COLD_MB` (L2 flush + A re-read before each timed GEMM).
- Desktop embargo lifted (no embargo file at 20:37).
- NEXT: cold-L2 sweeps on the 4060 (then 3070), refit/drop the large-L2
  branch, 3070 verify sweep + pp512 A/B, 4060 pp512 re-check.

## 2026-09-26 (successor #4, from 20:53)

- Successor #3 (20:44-20:50) only queued a 4060 cold sweep that never
  started (died waiting for the lock).  Nothing else lost.
- Queued: 4060 cold sweep (`ML_AB_COLD_MB=64`, all 77 shapes, cm-var
  binary = 34ad9f1 code) -> louise `~/ta-wt/cm-results/4060cold`;
  3070 warm verify sweep (desktop binary copy
  `$SP/cm4/bin/ml_bench-34ad9f1`) -> `$SP/cm4/results/verify-3070`,
  then a 3070 cold subset (a16 + m512 + t128) -> `cold-3070`.
- Desktop grokmerge build for the 3070 pp512 A/B: `git archive 7cfca58`
  into `target/cm-base-src/` (ignored), `tabuild build --release -p llama-ash`.

## 2026-09-27 (successor #5, from 07:05)

- Overnight: the 4060 cold sweep (louise `~/ta-wt/cm-results/4060cold`)
  finished sq/m256/m384/m512/m768 before the desktop reboot cut the ssh;
  the 3070 warm verify got sq/m256/m384 (copied to
  `$SP/work/coopmat/results/verify-3070`); the 3070 pp512 variant A/B
  got 2 rows only (unusable).
- 07:08 relaunched on louise (detached, `cm-results/cold4060_rest.sh`):
  m1024 m2048 t128 t1920 done, a16_* queued.  Data -> `coopmat-sm-data/cold-4060/`.
- 3070 queue (`$SP/work/coopmat/run3070.sh`, binary
  `$SP/work/coopmat/bin/ml_bench-34ad9f1`): cold a16_512 m512, warm
  verify remainder, cold a16_* t128 m256 m384.
- Desktop note: user app `project_watt_cubed` runs on the 3070 (~30% util
  at idle clocks); no embargo file.  Check the `.smi.csv` logs.
- 07:20-07:45: 4060 cold sweep complete (77 shapes, `cold-4060/`); 3070
  cold a16_512 + m512 (`cold-3070/`), warm verify remainder.
- **Refit (0fc29f9)**: cold data kill the large-L2 "B fits L2 -> x-fast"
  exception (M-fast wins every a16 pp512 GEMM cold on both GPUs).  New
  raster: fewer bytes (0.75) or tall single pass (M>=512, N>=2048), and
  cheap A re-reads (<=8 MiB or A <= L2/2 on a large L2).  rho unchanged
  except f16w small-L2 keyed on xp.  `coldfit.py` evaluates it.
  CoopmatGrid factors out raster + balance for an acc16 rule.
- 3070 EMBARGO from 07:44 (TABG).  A foreign 3-5 GB GPU allocation was
  present from ~07:25, so the 3070 batches from 07:25 on are quarantined
  in `verify-3070-contended/` (abs TF/s -4%, tile-ratio p90 disagreement
  14% vs 5% clean).  Clean 3070 data: §4 sweep, §5 candidates, verify
  sq/m256/m384, cold m512 (07:19) and a16_512 (07:15, a light GUI app ran).
- 3070 PENDING (embargo): large 128x128-vs-128x64 15-round A/B
  (`$SP/work/coopmat/large3070.sh`), remaining cold batches, pp512 A/B
  with `$SP/work/coopmat/bin/llama_ash-{7cfca58,0fc29f9}`.
- louise: unit tests 14/14 ok at 0fc29f9; GPU tests + pp512 A/B queued
  (`cm-results/pp4060.sh`).
- 08:00-08:20 louise: GPU tests `f16_routes_pick_f16w_kernels_and_skip_splitk2`
  + `coopmat_asymmetric_tiles_match_dual_rounded_reference` pass at 0fc29f9;
  clippy -D warnings clean (tensor-ash, ml-bench, all targets); `auto`
  routes of the 0fc29f9 ml_bench match `coldfit.new_rule` 19/19
  (`cold-4060-0fc29f9/`); pp512 A/B: 0fc29f9 38.46 ms vs grokmerge 39.48
  (−2.6%, 6/6 rounds), 100df0d 39.87 (`pp512_final_4060.log`).
- Report §9, §11, §12 written.  Remaining = 3070 items only (embargo);
  commands in §9 / §11, scripts in `coopmat-sm-data/jobs/`.
