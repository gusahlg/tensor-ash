# Performance theses

The optimization campaign's implicit performance model, made explicit
and falsifiable.  Each thesis below is a quantitative claim about
where the time goes on this stack, paired with an automated
measurement in `ml_bench thesis` that reports predicted vs measured
and a PASS/FAIL verdict.  After every optimization leg, one command
re-validates the whole model and flags which assumptions broke:

```console
# everything (T1-T7; the model theses need a GGUF):
ML_MODEL=models/tinyllama-1.1b-chat-v1.0.f16.gguf \
  target/release/ml_bench thesis --all

# kernel-level subset, no model needed:
target/release/ml_bench thesis t2 t3 t4 t6

target/release/ml_bench thesis --list
```

Output is one table row per measurement: `thesis | item | prediction |
measured | verdict`.  Any FAIL makes the process exit nonzero, so the
command doubles as a CI-style gate.

Predictions are recorded per GPU in
[`thesis-expectations.toml`](thesis-expectations.toml), keyed by the
exact Vulkan device name (override the path with `ML_THESIS_EXPECT`).
A device without a section (NVK, RADV, ...) still measures everything
but reports `info` instead of verdicts; record a section from a clean
run to arm the gate for that GPU.

## Regression targets

Both campaign GPUs are armed.  A leg is not done until
`ml_bench thesis --all` is green on both (the 4060 first: it is the
test host; the 3070 drives a live desktop).

| | RTX 3070 | RTX 4060 |
|---|---|---|
| chip | GA104, sm_86, 46 SM | AD107, sm_89, 24 SM |
| L2 | 4 MB | 24 MB |
| DRAM (spec) | 448 GB/s | 272 GB/s |
| driver (2026-09-26) | 595.99.02 | 595.80 |
| host | desktop (live session) | louise-pc (Ryzen 5 5500) |
| section | `["NVIDIA GeForce RTX 3070"]` | `["NVIDIA GeForce RTX 4060"]` |

Measured state of each (every number a median; commands and raw
tables in `campaign/integrate.md`).  RTX 4060: range over the three
gated `thesis --all` runs on the final harness (campaign/integrate,
2026-09-26 20:54 and 20:57, 2026-09-27 07:31 on the all-`-Os` build).
RTX 3070: the 2026-08-15 baseline — the re-run under the current
harness (DRAM-cold T2, per-token T1, wave-fill T6 routing) is still
pending (desktop GPU embargo, 2026-09-27).

| thesis | measured | RTX 3070 (2026-08-15) | RTX 4060 (2026-09-26/27) |
|---|---|---|---|
| T1 | decode weight BW, tg128 per-token | 356 GB/s = 79.4% | 221-227 GB/s = 81-84% |
| T2 | % of DRAM floor: q / o / k-v / gate-up / down / lm_head | 81-89% (legs 13/15, L2-hot); k/v ~30% | 89-91 / 87-90 / 32 / 92-95 / 91-94 / 94-97% |
| T3 | compute-barrier drain | 0.92 µs | 0.78-0.79 µs |
| T4 | submit + fence round-trip | band 30-60 µs ±20% (baseline value not recorded) | 14.3-14.7 µs |
| T5 | piecewise sum vs whole pp512 | +15.2% | +16.6 to +18.5% |
| T6 | coopmat TF/s: qkv / o / gate-up / down; 4096³ | q/o 24.2 · gate-up 32.2 · down 24.8 (128×128 tile only); 34.6 | 24.4-24.8 / 23.1-23.5 / 23.4-24.4 / 22.3-22.8; 24.7-25.3 |
| T7 | greedy tokens identical, 4 widths × 6 configs | PASS | PASS |

Conventions: the harness burns ~300 ms of large-GEMM work before the
first measurement, and every T2/T5/T6 series additionally warms for
200 ms of its own dispatches before timing (clock-state hazard: the
host-side setup between series idles the GPU long enough for the
driver to drop the P-state — on the 4060 T6 read 10.7-15.7 TF/s
instead of 22.8-26.4 when it followed T2 without that warm-up).  It
interleaves A/B-style series where slopes are differenced, and uses
GPU timestamps (median of the sample set) wherever a submission
boundary would otherwise pollute the number.

## T1 — decode is bandwidth-bound

**Claim.** Decode GPU time/token is priced by weight traffic:
`t ≈ weight-bytes/token / achieved-BW`, with achieved bandwidth at
least 80% of the device's spec bandwidth (≥ 358 GB/s on GA104's 448,
≥ 218 GB/s on AD107's 272).  For TinyLlama-1.1B f16 the weight bytes/token are
computed from the config at run time:
`2 · (layers·(2·embd² + 2·embd·kv_dim + 3·embd·ffn) + embd·vocab)`
≈ 2.07 GB (norm weights, the embedding row, and KV reads are < 1% and
omitted).

**Measurement.** Load the model in its default configuration
(prepared decode, f16 KV), warm up (prefill + 16 decode steps), then
prefill pp512 and decode tg128 with ONE submission per token
(`decode_many`, the path `generate` and `llama_ash bench` use —
not the `--tg-unrolled` one-submit replay); `Model::time_decode_steps`
makes each step read its GPU timestamps (`prepared_total`).  Achieved
GB/s = bytes/token over GPU ns/token.  An info row reports the decode
graph census (201 dispatches, 156 barriers/token since the packed
q/k/v leg; 203/158 before) and host overhead (wall − GPU, including
the per-token timestamp read).

**Provenance.** Legs 13-17: decode settled at ~170 t/s wall with
~5.2-5.4 ms GPU/token on the 3070.  The first thesis baseline
(2026-08-15) measured 356 GB/s = 79.4% of 448 — the recorded floor
sits just under it.  The packed-GEMV leg's switch to a fence-only
replay made T1 SKIP ("no single-submission GPU total") until
`time_decode_steps` restored the timestamps (campaign/integrate).
Expectation keys: `mem_bw_gbps`, `t1_min_bw_frac`.

## T2 — decode GEMV efficiency per shape class

**Claim.** Every bandwidth-sized decode GEMV class (≥ 8 MB of
weights) runs at ≥ 80% of its DRAM floor, where floor =
bytes / `mem_bw_gbps` and bytes = f16 weights + f32 activation row +
f32 output row.  The 1 MB k/v class is NOT bandwidth-bound — one
dispatch costs more than its floor — and is guarded at its own
measured level (`t2_kv_min_bw_frac`); in decode it runs concurrently
with q (no barrier between the three projections), which hides most
of it.

| class | K×N | layout | bytes | floor @448 (3070) | floor @272 (4060) |
|---|---|---|---|---|---|
| q | 2048×2048 | packed | 8.40 MB | 18.8 µs | 30.9 µs |
| o | 2048×2048 | row-major | 8.40 MB | 18.8 µs | 30.9 µs |
| k/v | 2048×256 | packed | 1.06 MB | 2.4 µs | 3.9 µs |
| gate/up | 2048×5632 | packed | 23.1 MB | 51.6 µs | 84.9 µs |
| down | 5632×2048 | row-major | 23.1 MB | 51.6 µs | 84.9 µs |
| lm_head | 2048×32000 | packed | 131.2 MB | 292.9 µs | 482.4 µs |

**Measurement.** Per class (B=1, M=1, f16w, auto-routed — the routed
kernel is reported per row; packed-B exactly where llama-ash packs):
first `run_case` (numerics validation, routed kernel, and an L2-HOT
median of back-to-back dispatches, printed for reference), then the
verdict series: ONE weight allocation, an L2 scrub submitted before
every timed dispatch, ML_ITERS (default 50) GPU timestamps, median.
The scrub is itself a decode-style GEMV — packed f16w
`[1,2048] @ [2048,49152]`, 192 MiB of weights (8× AD107's L2, 48×
GA104's) writing a 192 KiB row — because it must be READ-mostly: a
first version (an elementwise add writing 64 MiB) left L2 full of
dirty lines whose write-back competed with the timed GEMV (4060: q
49 µs, gate/up 129 µs vs 34 / 91 µs in a real decode step).  Decode streams ~2 GB of weights per token, so
no weight is ever L2-hot there; the old back-to-back measurement let
AD107's 24 MB L2 serve q/o (12.5 µs = 2.5× the DRAM floor) and GA104's
4 MB serve k/v.  Cross-check on the 4060 (perop decode breakdown,
one real token): o 34.2 µs, up/gate ~91-93 µs, down 89.8 µs,
lm_head 497 µs per call — within a few % of T2's cold numbers.

**Provenance.** Leg 13/15 forensics: row GEMVs at 81-89% of
bandwidth.  Leg 18 packed lm_head (4060: 3759 → 511 µs; 257 GB/s =
94% of 272) — but the 3759 µs was `examples/bench_packed_gemv`'s
unpacked arm: the auto-routed unpacked head (`f16w_row_bda_k16_v2`)
measures 512.9 µs vs 510.7 µs packed on the same 4060 (L2-hot
`cases`, 2026-09-26, `campaign/integrate.md`; campaign/fusion E0 found
the same), so on today's routing the 94% belongs to the k16_v2 row
kernel, and packing lm_head is worth <1% of its time.  campaign/integrate: DRAM-cold T2 and the k/v
exception.  Keys: `mem_bw_gbps`, `t2_min_bw_frac`,
`t2_kv_min_bw_frac`.

## T3 — a full compute barrier drains ~0.9 µs

**Claim.** One compute-to-compute barrier inside a recorded graph
costs ~0.9 µs of fixed drain + refill (GA104: 0.92 µs, AD107:
0.81 µs).  Decode records 156 hazard barriers per token (7 per layer
× 22 layers + 2 in the tail; 158 before the packed q/k/v leg; T1's
census row reports the live count), so barriers cost ~0.13-0.15 ms of
a ~5.5-9 ms token (1.5-2.5%) — **barrier elimination is not a decode
lever**.  A fusion pays for the dispatches, bytes and kernel time it
removes, not for the barriers.

**Falsification history.** The campaign carried ~7.7 µs/barrier for
months: leg 5 divided a 2.7 ms gap (summed per-op GPU times 7.26 ms vs
the whole command buffer 9.97 ms) by the ~350 barriers of that era,
and leg 15 credited −22 barriers/token with ≈ −170 µs.  The figure
was an attribution, never a measurement.  The first `ml_bench thesis`
baseline (RTX 3070, driver 595.80, 2026-08-15) measured it directly —
serialized chain 34 → 266 µs vs overlapped chain 6 → 31 µs over
32 → 256 dispatches = **0.92 µs/barrier**, 8× below the claim — and
the expectations file was re-recorded (commit c52cd16).  What the
2.7 ms leg-5 gap actually was was never re-attributed; at 0.92 µs the
barriers explain ~0.3 ms of it.

**Measurement.** Two GPU-timestamped `run_exec_ops` chains of the
same trivial kernel at N=32 and N=256, interleaved:

- *serialized*: every op is a 1-add into ONE tensor
  (`ExecOp::Binary`, 256 elements, one workgroup) — the hazard
  tracker emits a barrier before every op (verified N−1 barriers via
  the new plan-only census `Executor::exec_ops_barrier_count`);
- *overlapped*: every op writes its OWN tensor — verified 0 barriers.

Per-barrier cost = difference of the two chains' slopes over N.  No
new shader was needed: an empty kernel cannot force hazard barriers
(no writes), so the minimal barrier-forcing dispatch IS a 1-add, and
`op_binary_f32` at one workgroup already is that kernel wired through
the normal pipeline machinery.

**Scope.** With a trivial predecessor this is the barrier's *fixed*
cost.  Behind a real kernel a barrier also exposes the predecessor's
tail (the last partial wave idles SMs until it drains); that loss is
kernel-dependent and already sits inside every kernel's standalone
single-dispatch timing (T2, T5 pieces), so T5's accounting charges
barriers at this fixed price only.

**Provenance.** Direct measurement, 2026-08-15 baseline (above);
re-measured per GPU in the regression-target table.  Keys:
`t3_barrier_us`, `t3_rel_tol` (±100% — a slope difference at the
sub-µs level is noisy; the gate exists to catch a return to
multi-µs drains, not to resolve 0.1 µs).

## T4 — a submission round-trip costs tens of µs

**Claim.** One `vkQueueSubmit` + fence-wait round-trip costs tens of
µs — the number that made single-submission graphs (legs 15-17) and
the v1.5 PreparedOps + spin-wait work worth building.  It is a HOST
property (CPU, kernel, driver), so each section records its own band:
the desktop (3070) was recorded at 30-60 µs, louise-pc (4060) at
10-20 µs (measured 14.3-14.7 µs in every 2026-09-26/27 run).

**Measurement.** Wall-clock N=64 single-dispatch `run_exec_ops`
submissions vs ONE 64-dispatch submission of the same op list
(independent trivial 1-adds), repeated 9×; per-submit cost =
(median_many − median_one)/(N−1).  Host-side per-op planning appears
in both terms and cancels.

**Provenance.** Campaign ledger (~10 µs pure submit visible in leg 16
forensics as ~10-12 ms over ~300 per-op submissions; synchronous
round-trips measured 30-60 µs in the v1.5 sync-overhead work).  Keys:
`t4_submit_us_lo`, `t4_submit_us_hi`, `t4_edge_tol`.

## T5 — prefill time is accountable piecewise

**Claim.** The pp512 prefill graph's GPU time decomposes:
`whole ≈ Σ GEMM + flash-attn + elementwise + barriers·t3`, with the
independently measured sum within the recorded gap (`t5_rel_gap`:
20% on the 3070, 25% on the 4060) of the measured whole.  The sum
should slightly overestimate (standalone runs cannot overlap
independent neighbours the way the graph does); a large positive gap
means lost overlap or an unaccounted op class, a negative gap means
the graph is paying something the pieces don't see.

**Measurement.** Whole: median of 5 from-scratch pp512 prefills'
`prefill_total` GPU timestamps.  Pieces, at the model's own config
shapes on synthetic tensors, mirroring `prefill_ops`' wide (T ≥ 128,
f16-activation) branch — 12 ops per layer: attn rmsnorm, ONE
concatenated QKV GEMM (512×2560×2048), fused QKV pack (RoPE + KV
append + head-major Q), flash (token-major O), o GEMM, residual add,
ffn rmsnorm, up and gate GEMMs, silu-mul, down GEMM, residual add;
plus the LM-head tail (last-row copy, rmsnorm, packed M=1 lm_head
GEMV): each class measured standalone (GPU-timestamp median; GEMMs
after a 200 ms warm-up), multiplied by its count and layer count.
Barrier term = graph census barriers × T3's measured price (falls
back to the recorded `t3_barrier_us`).  Known biases, all toward
OVERestimation: the GEMM pieces run the f32-A f16w coopmat route
while prefill runs the a16 (f16-activation) twins; the elementwise
pieces run on f32 tensors (prefill's are f16); the fused QKV pack is
priced as the six composed rope/copy dispatches it replaced.  If
`prefill_ops` drifts structurally, this thesis failing is the alarm —
update the inventory in `model_theses.rs` alongside it.

**Provenance.** Leg 16: graph prefill 42.3 ms GPU vs 43.9 ms wall at
pp512.  Baseline 2026-08-15 (3070): +15.2%, GEMM 96% of the whole.
Key: `t5_rel_gap`.

## T6 — coopmat sustains ≥ 30 TF/s on real prefill shapes

**Claim (aspirational).** The f16w KHR-coopmat GEMM sustains
≥ 30 TF/s on the 3070 on the REAL aligned prefill shapes — every
per-layer GEMM of TinyLlama's pp512: concat QKV 512×2560×2048, o
512×2048×2048, gate/up 512×5632×2048, down 512×2048×5632.  The
34.6 TF/s headline was 4096³, and the delta on 512-row shapes is
itself a finding (~33 TF/s average would put pp512 at CUDA parity).
The GATE is a regression floor at the worst measured shape minus
noise (`t6_min_tflops`), plus the 4096³ reference (−15% floor).  A
shape routing AWAY from coopmat is a broken assumption regardless of
rate and fails the row.  Since the wave-fill leg, M=512 shapes route
to the 64×64 / 128×64 tiles (128×64 when N > 4·M), 4096³ to 128×128.

**Measurement.** `run_case_warm` per shape (B=1, f16w with f32 A —
the a16 twins prefill uses are not reachable from `cases` yet —
auto-routed, 200 ms warm-up), timestamped medians, routed kernel
reported.

**Provenance.** Model-inference leg: 34.6 TF/s at 4096³ on the 3070
(3× the f32 ceiling).  Baseline 2026-08-15 (3070, 128×128 only):
24.2 / 32.2 / 24.8 TF/s on q/o, gate/up, down.  Keys:
`t6_min_tflops`, `t6_ref_tflops`, `t6_ref_rel_tol`.

## T7 — token exactness is invariant across execution strategies

**Claim.** Temp-0 generation is bit-identical across every decode
mode (prepared / graph / perop), both KV dtypes (f16 / f32), and
prefill widths T=1/64/300/512 (the single-token prefill; the
fused-epilogue MLP at T<128; the unfused T≥128 branch with f32
activations and the composed rope/append path at T=300 (not a
multiple of 64); and the f16-activation branch with the fused QKV
pack at T=512).  This is the
correctness thesis every performance leg is gated on; scattered
assertions existed in tests and the token gate — this is the unified
sweep.

**Measurement.** For each of the 6 (mode × KV) configs the model is
reloaded (`Model::load_with` + the new `LoadOverrides`, no env
mutation); for each prompt width, prefill + 16 greedy decode steps;
the 17-token sequences must be identical across all configs per
width.  Any divergence FAILs with the config and first divergent
token index.  No recorded expectation needed — correctness is
device-independent.

**Provenance.** Token gate 24/24 byte-identical across legs 13-17;
prepared/graph/perop and KV f32 equality asserted piecemeal in leg
gates.

## Maintaining the expectations

- After an optimization leg *changes* the truth (e.g. barrier count
  drops, GEMV efficiency rises), re-record the affected keys in
  `thesis-expectations.toml` in the same commit, citing the leg.
- New GPU: run `ml_bench thesis --all` twice, copy the closest
  section, replace the numbers with measured-reality guards (floors
  just under the worse of the two runs; the device's real spec
  `mem_bw_gbps`; the host's own T4 band), re-run to confirm PASS,
  commit with the raw tables.
- A FAIL after an unrelated-looking change is the tool doing its job:
  either the change broke the assumption, or the assumption was never
  as solid as recorded — both are findings.
