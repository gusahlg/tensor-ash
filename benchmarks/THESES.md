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
A device without a section — the 4060, NVK, RADV — still measures
everything but reports `info` instead of verdicts; record a section
from a clean run to arm the gate for that GPU.  Numbers below are the
RTX 3070 (GA104) baseline; provenance points at the campaign leg that
measured them (`experiment-branch.md`).

Conventions: the harness burns ~300 ms of large-GEMM work before the
first measurement (clock-state hazard: mode-ordered benches without
shared burn-in overreport later modes), interleaves A/B-style series
where slopes are differenced, and uses GPU timestamps (median of the
sample set) wherever a submission boundary would otherwise pollute the
number.

## T1 — decode is bandwidth-bound

**Claim.** Decode GPU time/token is priced by weight traffic:
`t ≈ weight-bytes/token / achieved-BW`, with achieved bandwidth at
least 80% of the device's spec bandwidth (448 GB/s on GA104, so
≥ 358 GB/s).  For TinyLlama-1.1B f16 the weight bytes/token are
computed from the config at run time:
`2 · (layers·(2·embd² + 2·embd·kv_dim + 3·embd·ffn) + embd·vocab)`
≈ 2.07 GB (norm weights, the embedding row, and KV reads are < 1% and
omitted).

**Measurement.** Load the model in its default configuration
(prepared decode, f16 KV), warm up (prefill + 16 decode steps), then
prefill pp512 and decode tg128; decode GPU ms/token comes from the
single-submission GPU timestamps (`prepared_total`/`graph_total`).
Achieved GB/s = bytes/token over GPU ns/token.  An info row reports
the decode graph census (dispatches, barriers/token — leg 15: 203
dispatches, ~7·22 barriers) and host overhead (wall − GPU).

**Provenance.** Legs 13-17: decode settled at ~170 t/s wall with
~5.2-5.4 ms GPU/token; 2.07 GB / 5.3 ms ≈ 390 GB/s ≈ 87% of 448.
Expectation keys: `mem_bw_gbps`, `t1_min_bw_frac`.

## T2 — decode GEMV efficiency per shape class

**Claim.** Every decode GEMV class runs at ≥ 80% of its bandwidth
floor, where floor = bytes/448 GB/s and bytes = f16 weights + f32
activation row + f32 output row:

| class | K×N | bytes | floor @448 |
|---|---|---|---|
| q/o | 2048×2048 | 8.40 MB | 18.8 µs |
| k/v | 2048×256 | 1.06 MB | 2.4 µs |
| gate/up | 2048×5632 | 23.1 MB | 51.6 µs |
| down | 5632×2048 | 23.1 MB | 51.5 µs |
| lm_head | 2048×32000 | 131.2 MB | 292.9 µs |

**Measurement.** `run_case` per shape (B=1, M=1, f16w, auto-routed —
the routed kernel is reported per row), ML_ITERS (default 50)
timestamped samples, median GPU time, sampled post-timing validation.

**Provenance.** Leg 13/15 forensics: row GEMVs at 81-89% of
bandwidth.  The tiny k/v class is the canary — launch overhead bites
fixed-cost-first.  Keys: `mem_bw_gbps`, `t2_min_bw_frac`.

## T3 — a full compute barrier drains ~7.7 µs

**Claim.** One compute-to-compute barrier inside a recorded graph
costs ~7.7 µs on GA104 (drain + lost overlap), which is what makes
barrier-elimination legs (leg 15: −22 barriers/token ≈ −170 µs) pay.

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

**Provenance.** Leg 15 accounting (~7.7 µs/barrier inferred from the
22-barrier elimination).  Keys: `t3_barrier_us`, `t3_rel_tol`
(±50% — the slope-difference definition is noisy at the µs level).

## T4 — a submission round-trip costs ~30-60 µs

**Claim.** One `vkQueueSubmit` + fence-wait round-trip costs 30-60 µs
on this host — the number that made single-submission graphs (legs
15-17) and the v1.5 PreparedOps + spin-wait work worth building.

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
independently measured sum within 10% of the measured whole.  The sum
should slightly overestimate (standalone runs cannot overlap
independent neighbours the way the graph does); a large positive gap
means lost overlap or an unaccounted op class, a negative gap means
the graph is paying something the pieces don't see.

**Measurement.** Whole: median of 5 from-scratch pp512 prefills'
`prefill_total` GPU timestamps.  Pieces, at the model's own config
shapes on synthetic tensors, mirroring `prefill_ops`' T≥256 branch
per layer (1 attn rmsnorm, q/k/v GEMMs, 2 ropes, 2 KV appends, q
permute, flash, attn permute, o/up/gate/down GEMMs, 2 residual adds,
silu-mul; plus the lm-head tail): each class measured standalone
(GPU-timestamp median), multiplied by its count and layer count.
Barrier term = graph census barriers × T3's measured price (falls
back to the recorded `t3_barrier_us`).  If `prefill_ops` drifts
structurally, this thesis failing is the alarm — update the inventory
in `model_theses.rs` alongside it.

**Provenance.** Leg 16: graph prefill 42.3 ms GPU vs 43.9 ms wall at
pp512.  Key: `t5_rel_gap`.

## T6 — coopmat sustains ≥ 30 TF/s on real prefill shapes

**Claim.** The f16w KHR-coopmat GEMM sustains ≥ 30 TF/s on the REAL
aligned 2048-class prefill shapes (512×2048×2048, 512×5632×2048,
512×2048×5632) — the 34.6 TF/s headline was 4096³, and the delta on
512-row shapes is itself a finding.  4096³ is kept as a regression
reference at ~34.6 TF/s (−15% floor).  A shape routing AWAY from the
coopmat kernel is a broken assumption regardless of rate and fails
the row.

**Measurement.** `run_case` per shape (B=1, f16w, auto-routed),
timestamped medians, routed kernel reported.

**Provenance.** Model-inference leg: 34.6 TF/s at 4096³ (3× the f32
ceiling).  Keys: `t6_min_tflops`, `t6_ref_tflops`, `t6_ref_rel_tol`.

## T7 — token exactness is invariant across execution strategies

**Claim.** Temp-0 generation is bit-identical across every decode
mode (prepared / graph / perop), both KV dtypes (f16 / f32), and
prefill widths T=1/64/300/512 (which exercise the single-token
prefill, fused-MLP T<256, and unfused T≥256 branches).  This is the
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
- New GPU: run `ml_bench thesis --all` once, copy the GA104 section,
  replace the numbers with the measured ones (and the device's real
  `mem_bw_gbps`), commit.
- A FAIL after an unrelated-looking change is the tool doing its job:
  either the change broke the assumption, or the assumption was never
  as solid as recorded — both are findings.
