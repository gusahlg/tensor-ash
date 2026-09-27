# Campaign track integrate: land exp/grokmerge, fix the performance model

Branch `campaign/integrate` (worktree `~/repos/tensor-ash-wt/c-integrate`),
based on `exp/grokmerge` (= main 50efb68 + 10 commits: packed decode GEMVs,
64x64/64x128/128x64 f16w coopmat wave-fill tiles, fused prefill QKV pack,
module splits, packed-B in `cases`/C ABI, LOWMEM build.rs).  Worked
2026-09-26 (two agents, cut off) and 2026-09-27 (this continuation).

Hosts: RTX 4060 (AD107, 24 SM, 24 MB L2, 272 GB/s, driver 595.80) on
louise-pc — every new binary ran there first; RTX 3070 (GA104, 46 SM,
4 MB L2, 448 GB/s, driver 595.99.02) on the desktop, only validated
binaries in short batches.  Builds: louise `lbuild` / systemd job scripts
(fast profile: LTO off, 16 CGUs); desktop `tabuild` (same profile).  GPU
timestamps are medians; A/B across processes alternates the order.

Raw outputs: louise `~/ta-logs/integrate/` (scripts + logs), mirrored to
`~/repos/tensor-ash-wt/campaign-tools/work/integrate/`;
2026-09-26 desktop outputs in
`~/repos/tensor-ash-wt/scratch-backup/scratchpad/integ/`.

Token reference (pinned greedy continuation of TinyLlama f16 for
`--ids 1,450,7483,310,3444,338 -n 24`, the leg gate's llama.cpp CUDA
reference):
`3681,29889,13,13,29906,29889,350,29889,450,7483,310,9556,338,5115,29889,13,13,29941,29889,315,29889,450,7483,310`.

## Summary

| item | result | evidence |
|---|---|---|
| CPU gates | fmt clean; clippy `--workspace --all-targets -D warnings` clean (after c8cd50b: rustc/clippy 1.98's `chunks_exact_to_as_chunks` also fails main 50efb68); CPU unit tests all pass | §1 |
| 4060 GPU suite, monolithic | **138/138 pass**, one process, 173 s (all-`-Os` build fc6a92f); also 138/138 in 89 s on 2026-09-26 (-O build) | §2 |
| 3070 GPU suite, chunked per module | **138/138 pass**, 25 modules in 3 batches (all-`-Os` build fc6a92f) | §2 |
| llama-ash token gate, 4060 | **24/24 MATCH** on fc6a92f (and on 48a03a3, 7679d19; main50 also MATCH) | §2 |
| `ml_bench thesis --all`, 4060, armed section | **PASS, 0 FAIL**, 3 of 3 runs on the final harness | §3 |
| `ml_bench thesis --all`, 3070 | **PENDING** (user gaming, embargo 07:44 2026-09-27) | §6 |
| pp512 / tg128 vs main 50efb68, 4060 | pp512 **+27.3%** (12864 vs 10108 t/s), prefill GPU 39.46 vs 47.17 ms (−16.3%), tg128 **+1.5%** (103.80 vs 102.25 t/s), 4 rotated rounds | §4 |
| pp512 / tg128 vs main, 3070 | **PENDING** (embargo) | §6 |
| glslc `-Os` for every shader | UCODE **identical** to `-O` for 3/3 spot-checked routes on the 4060 (incl. tm8_tn8, whose `-O` needs >6 GB); end-to-end prefill ratio to main unchanged (0.8366 vs 0.8371 with `-O`, noise) | §5 |
| T3 performance model | ~0.9 µs/barrier (3070 0.92, 4060 0.78-0.81); the 7.7 µs claim retired | THESES.md T3 |
| packed LM-head prefill tail (f6492f5) | tokens exact; time delta **0** (prefill GPU 39.31 vs 39.35 ms, noise); saves 131 MB VRAM; its commit message's "~3 ms" premise is **false** | §4.1 |

## 1. CPU gates

Desktop, `SP=~/repos/tensor-ash-wt/campaign-tools`, in the worktree:

```console
$SP/bin/taenv cargo fmt --all --check                                    # rc 0
$SP/bin/tabuild clippy --release --workspace --all-targets -- -D warnings  # clean (2026-09-27 07:09, HEAD with -Os build.rs)
$SP/bin/tabuild test --release --workspace                               # all pass: tensor-ash 51, capi 10, test-support 14,
                                                                          # ml-bench 17, llama-ash 2, doc 1; correctness 138 ignored (GPU)
```

The only breakage found on exp/grokmerge was the new clippy 1.98 lint
(c8cd50b, `as_chunks` for fixed-width byte decodes); it also fails on
main 50efb68.

## 2. GPU test suite and token gate

GPU tests are `#[ignore]`d and run as `<correctness binary> --ignored
--test-threads=1` (script `suite-mono.sh`; chunked: `suite-chunk.sh
<module>...`, one process per module).

**4060, monolithic** (the Ada monolithic-suite SIGSEGV fixed by the
loader pin in PR #23 did not recur):

- 2026-09-27 07:27, build fc6a92f (all shaders `-Os`), job
  `louise:~/ta-logs/integrate/jobD.sh` phase P2:
  `test result: ok. 138 passed; 0 failed ... finished in 172.92s`.
  (Slower than yesterday because every pipeline recompiled once: the
  new SPIR-V misses the driver's pipeline cache.)
- 2026-09-26 16:51, build 7679d19 (`-O`): 138 passed in 88.82 s
  (`suite-mono.log`).  Library/test code did not change between
  7679d19 and 48a03a3.

**3070, chunked per module** (desktop, `gpu3070 bash suite-chunk.sh
<modules>` in three batches, 2026-09-27 07:36-07:45, build fc6a92f):
25 modules, **138 passed, 0 failed** (`work/integrate/suite-3070-b{1,2,3}.log`).
Note: a game (TABG) was running on the 3070 during these batches —
nobody had noticed yet; the embargo was set at 07:44, after which no
3070 work ran.  Pass/fail is unaffected; times are not meaningful.
The same 25 modules also passed 138/138 on 2026-09-26 17:04-17:06
(build 7679d19, LOWMEM glslc).

**Token gate** (TinyLlama f16, `llama_ash generate -m
~/models/tinyllama-1.1b-chat-v1.0.f16.gguf --ids 1,450,7483,310,3444,338
-n 24`, compared to the pinned reference above):

| build | host | result | log |
|---|---|---|---|
| main 50efb68 | 4060 | MATCH | `gate-ab-4060.log` |
| 7679d19 | 4060 | MATCH | `gate-ab-4060.log`, `gate-thesis-4060-1.log` |
| 48a03a3 | 4060 | MATCH | `jobC.log` |
| fc6a92f (-Os) | 4060 | MATCH | `jobD.log` |

T7 (inside every `thesis --all` below) additionally checks 17 greedy
tokens identical across 6 configs (prepared/graph/perop × KV f16/f32)
at prefill widths 1/64/300/512: PASS in every run.

## 3. `ml_bench thesis --all` on the 4060 (checklist item 2)

Command (in `~/ta-wt/integrate` on louise, under the 4060 lock):

```console
ML_MODEL=~/models/tinyllama-1.1b-chat-v1.0.f16.gguf ./target/release/ml_bench thesis --all
```

The `["NVIDIA GeForce RTX 4060"]` section was recorded from two
unarmed runs (2026-09-26 17:43 and 19:13, `gate-thesis-4060-{1,2}.log`)
plus a T2/T6 re-run after the harness fixes (`jobA.log`,
`probe1.log`), as floors just under the worse run.  Then armed:

| run | build | verdicts | log |
|---|---|---|---|
| 2026-09-26 20:54 | 48a03a3 | 21 PASS, 0 FAIL, rc 0 | `jobC.log` run 1 |
| 2026-09-26 20:57 | 48a03a3 | 21 PASS, 0 FAIL, rc 0 | `jobC.log` run 2 |
| 2026-09-27 07:31 | fc6a92f (-Os) | 21 PASS, 0 FAIL, rc 0 | `jobD.log` P2 |

Measured ranges over the three armed runs (per-GPU table in
THESES.md): T1 221-227 GB/s = 81-84% of 272 (floor 78%); T2 DRAM-cold
q 89-91%, o 87-90%, k/v 32% (own floor 25%), gate/up 92-95%, down
91-94%, packed lm_head 94-97% (floor 82%); T3 0.78-0.79 µs/barrier;
T4 14.3-14.7 µs/submit (band 10-20); T5 +16.6..+18.5% (gap 25%); T6
qkv 24.4-24.8, o 23.1-23.5, gate/up 23.4-24.4, down 22.3-22.8 TF/s
(floor 20), 4096³ 24.7-25.3 (25 −15%).

What the 4060 exposed in the harness (fixed on this branch, each in
its own commit):

- **T1 SKIPped on every device** since the packed-GEMV leg made
  `decode_many` fence-only (66415ee restores per-token timestamps via
  `Model::time_decode_steps`).
- **T2 measured L2, not DRAM, on a 24 MB-L2 GPU**: q/o 12.5 µs =
  ~670 GB/s on a 272 GB/s part.  Now DRAM-cold (7679d19 → 8171960 →
  48a03a3): one weight allocation, a read-mostly 192 MiB GEMV scrub
  before every timed dispatch.  Two intermediate forms were wrong and
  are documented in `gemv.rs`: rotating two 131 MB lm_head copies read
  842 µs (vs 497 µs in a real decode step), and an elementwise scrub
  left dirty lines whose write-back slowed every class (q 49.3 µs,
  gate/up 129 µs vs 34 / 91 µs in decode).  Final T2 agrees with the
  perop decode breakdown within a few % (o 34.2, up/gate 91-93, down
  89.8, lm_head 497 µs per call; `probe1.log`, `jobA.log`).
- **k/v is launch-bound**: 1.06 MB, floor 3.9 µs, measures 12 µs
  DRAM-cold (6.4 µs L2-hot) = 32% of floor; guarded by its own key
  `t2_kv_min_bw_frac` (7fbd644).
- **T6 clock-state hazard**: after T2 the driver dropped the P-state
  and T6 read 10.7-15.7 TF/s instead of 22.8-26.4 (`probe1.log`);
  every series now warms for 200 ms (8171960).
- `llama_ash bench` tg had silently become a one-submission
  `decode_unrolled` replay (not comparable to llama-bench or older
  scoreboards): per-token again by default, `--tg-unrolled` opt-in
  (4baba3b).

## 4. pp512 / tg128 vs main 50efb68 (4060)

main 50efb68 = a clean `git archive` export built on louise as
`~/ta-wt/main50` (same fast profile).  Script `abn-bench.sh <rounds>
<dirs...>`: one discarded burn-in, then each round runs every build
once in rotated order; `llama_ash bench -m <tinyllama f16> --pp 512
--tg 128` (tg per-token on both builds).

2026-09-27 07:47, integrate fc6a92f (-Os), 4 rounds (`jobD.log` P3):

| build | pp512 t/s | tg128 t/s | prefill GPU ms |
|---|---|---|---|
| main50 | 10086.6 / 10104.0 / 10188.7 / 10112.3 → **10108** | 102.24 / 102.26 / 102.48 / 102.01 → **102.25** | 47.17 |
| integrate | 12889.8 / 12874.1 / 12853.9 / 12849.4 → **12864** | 103.68 / 103.80 / 103.80 / 103.95 → **103.80** | 39.46 |

pp512 +27.3%, tg128 +1.5%.  2026-09-26 19:46 (48a03a3, `-O`,
`jobA.log`, 3-way with the pre-4baba3b tree): main50 pp512 median
10175, integrate 12941; tg128 103.6 vs 105.1.  The earlier 6-pair
A/B (`gate-ab-4060.log`, 17:05) showed tg128 113 vs 104, but that
integrate build still had the unrolled tg (fixed by 4baba3b) — not a
valid comparison, superseded.

### 4.1 The packed LM-head prefill tail (f6492f5)

f6492f5 moved the t > 1 prefill tail (last-row copy, rmsnorm, lm_head)
to the packed M=1 GEMV and dropped the row-major lm_head upload.  Its
commit message cites 3759 µs unpacked vs 511 µs packed (leg 18) and
"~3 ms of a ~39 ms pp512 prefill".  **That premise is false on this
routing**: the auto-routed unpacked head is `f16w_row_bda_k16_v2`,
512.9 µs vs 510.7 µs packed (4060, L2-hot `ml_bench cases
lm_unpacked,1,1,32000,2048,f16w lm_packed,1,1,32000,2048,packed`,
ML_ITERS=100, interleaved twice: 512.9/512.4 vs 510.7/510.5 µs;
`jobC.log`).  The 3759 µs was `examples/bench_packed_gemv`'s unpacked
arm.  campaign/fusion's E0 found the same independently.

Measured end-to-end (4060, `jobA.log`, 3-way interleaved, 4 rounds):
prefill GPU median 39.35 ms without f6492f5 (tree before 4baba3b) vs
39.31 ms with it; pp512 12918 vs 12941 t/s — **no measurable time
delta**.  Tokens exact (gate MATCH; T7 at T=64/300/512 PASS).  The one
real effect is −131 MB VRAM (the unpacked [2048, 32000] f16 copy is no
longer uploaded).  Kept on this branch on those terms; campaign/fusion
carries an equivalent `LLAMA_ASH_TAIL` switch, so at merge exactly one
of the two should land (§7).

## 5. glslc `-Os` for every shader (build.rs, 5431a31)

build.rs now compiles every shader with `-Os`
(`TENSOR_ASH_GLSLC_OPT=-O` restores `-O`); the LOWMEM vmem cap and
its `-Os` fallback are gone.  Reason: campaign/sass
(`c-sass/benchmarks/SASS.md` Q(a)) found byte-identical UCODE for
`-O` and `-Os` on 14/14 shaders on both GPUs, while `-O` needs >3 GB
(>6 GB for tm8_tn8) and was the reason for the desktop/louise build
asymmetry.  The desktop's `tabuild` still exports
`TENSOR_ASH_GLSLC_LOWMEM=1`; it is now ignored.

**Spot check on this branch** (4060, `jobD-p3.sh`): campaign/sass's
`ml_bench isa` (pipeline creation only, no dispatch) compiled each
route from yesterday's louise `-O` SPIR-V (saved to
`~/ta-logs/integrate/spv-O/` before the rebuild) and from today's
`-Os` build output, `ML_KERNEL=<k> ML_ISA_SPV=<spv> ML_ISA_TAG=O|Os
ml_bench isa <case>`:

| route | SPIR-V -O / -Os (bytes) | UCODE hash -O | UCODE hash -Os | regs |
|---|---|---|---|---|
| lmhead,1,1,32000,2048,packed → f16w_row_bda_k16_v2_packed | 13276 / 12972 | eed2d8b6a5e26569 | eed2d8b6a5e26569 | 30 / 30 |
| q512,1,512,2048,2048,f16w → f16w_coopmat_m64n64 | 7720 / 7480 | 018f3c211f1d19eb | 018f3c211f1d19eb | 72 / 72 |
| sq1024,1,1024,1024,1024 → k64_bda_v4_tm8_tn8 | 587228 / 587076 | cd8ac9e5b7fbc99b | cd8ac9e5b7fbc99b | 236 / 236 |

Identical in all three; the tm8_tn8 SPIR-V hashes (451a3481… /
5ee67766…) and UCODE hash equal SASS.md's.  Metadata:
`louise-logs/isa-jobD/*.meta`.  End to end the -Os build's prefill
ratio to main is unchanged: 39.46/47.17 = 0.8366 today vs
39.31/46.97 = 0.8371 with `-O` yesterday (both builds moved +0.4%
together between days).  The night track also measured `-O` ==
`-Os` in timing (CAMPAIGN.md 07:44).

## 6. Pending on the 3070 (embargo)

The user started a game on the desktop; `$SP/GPU3070_EMBARGO` was set
at 2026-09-27 07:44 and `gpu3070` refuses.  Still to run, in three
short batches of validated binaries (desktop build of fc6a92f is in
`c-integrate/target/release`, main50 in
`~/ta-scratch/integrate-main50/target/release`):

```console
cd ~/repos/tensor-ash-wt/c-integrate
$SP/bin/gpu3070 bash $SP/work/integrate/job3070.sh thesis   # token gates + thesis --all (~3 min)
$SP/bin/gpu3070 bash $SP/work/integrate/job3070.sh ab 1     # 2 pairs main50,integrate
$SP/bin/gpu3070 bash $SP/work/integrate/job3070.sh ab 2     # 2 pairs integrate,main50
```

Risk: the 3070 section's `t2_min_bw_frac = 0.80` was recorded L2-hot
(legs 13/15); under DRAM-cold T2 the q/o classes may land lower on
GA104 (4 MB L2 already forced most of them to DRAM, so probably not
much).  A T2 FAIL there means re-recording from measurement, not a
code regression.

## 7. Commits on campaign/integrate (over exp/grokmerge)

c8cd50b clippy 1.98 lint · 9bb5880 T3 ~0.9 µs · 7679d19 T2 DRAM-cold ·
69c6f51 thesis follows llama-ash layout · 66415ee T1 per-token timing ·
4baba3b bench tg per-token · f6492f5 packed prefill LM-head tail ·
8171960 L2 scrub + 200 ms warm · 7fbd644 t2_kv_min_bw_frac · 48a03a3
read-mostly scrub · 5431a31 build.rs `-Os` · fc6a92f `--list` labels ·
8c82021 THESES + 4060 section · 411f674 CHANGELOG.

Merge recommendation: land all of it with the defaults as committed
(`-Os`, per-token tg, armed 4060 section).  f6492f5 is time-neutral
(−131 MB VRAM only): if campaign/fusion's `LLAMA_ASH_TAIL` lands,
drop f6492f5 or make packed its default; do not land both.
