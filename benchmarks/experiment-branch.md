# Experiment branch log

## Leg 18 — packed decode GEMVs, 64x64 wave-fill coopmat, fused prefill QKV pack — MEASURED (RTX 4060)

Uncommitted session work finalized on `experiment/pack-and-wavefill`.
GPU gate: RTX 4060 (AD107), 136/137 then 137/137 after the RMSNorm
vec4 alignment fix; TinyLlama generate of the pinned 6-id prompt
completed (24 tokens, finite logits).  Kernel benches are GPU
timestamps, 12 iters / 4 warmup, debug ml-bench, no TinyLlama in
the same process as the microbenches.

**Packed-B row GEMVs.** Decode weights pack to `[N/tile][K][tile]` at
load (`pack_f16w_row_tiles`); `MatmulOp::with_packed_b` fail-closes
onto `f16w_row_bda_k16_packed` / `k16_v2_packed`.  Inner-product order
matches the unpacked k16 pair (correctness:
`packed_row_gemv_matches_unpacked_and_cpu`, packed store-fusion).
Unpacked fallback is a panic — the layouts do not alias.

RTX 4060, `examples/bench_packed_gemv.rs`, GPU median, interleaved
unpacked then packed per shape after a 300 ms GEMM burn-in:

| shape | unpacked | packed | packed/unpacked |
|---|---:|---:|---:|
| k/v 1×256×2048 | 6.3 µs | 6.3 µs | 1.00 |
| q/o 1×2048×2048 | 12.5 µs | 12.6 µs | 1.01 |
| up 1×5632×2048 | 24.9 µs | 24.1 µs | 0.97 |
| down 1×2048×5632 | 30.9 µs | 30.5 µs | 0.99 |
| **lm 1×32000×2048** | **3759 µs** | **511 µs** | **0.14** |

lm_head is the reason packing exists: N=32000 makes the unpacked
K-walk a 64 KiB stride, and packed sequential K hits 257 GB/s
(94% of the 4060's 272 GB/s).  Square and deep-K GEMVs are already
at the bandwidth floor unpacked, so llama-ash no longer packs o/down
(they have unpacked siblings).  q/k/v stay packed because the
unpacked copies were dropped in favour of concat `w_qkv`.  gate/up
+ lm_head stay packed (the v2/wide-N wins).  TinyLlama generate of
the 6-id prompt is token-identical after that cut
(`3681,29889,13,13,29906,...`).

**64x64 coopmat wave-fill.** 512x2048 is 64 128-tiles (a short wave
plus a tail on GA104) vs 256 64-tiles.  Heuristic
`coopmat_prefers_m64` (tiles_128 < 96, or 64-aligned-not-128).
Tuner excludes m64 so a measured 128x128 winner cannot override
the static pick.

RTX 4060, f16w, auto route, GPU median:

| shape | kernel | TF/s |
|---|---|---:|
| 512x2048x2048 | `f16w_coopmat_m64n64` | 22.2 |
| 512x5632x2048 | `f16w_coopmat_aligned` | 21.2 |
| 512x2048x5632 | `f16w_coopmat_m64n64` | 22.8 |
| 1024³ | `f16w_coopmat_m64n64` | 21.3 |
| 2048³ | `f16w_coopmat_aligned` | 22.7 |

**LDS double-buffer REJECTED.** Ping-ponging K tiles in shared memory
(128x128 → 37.9 KiB) dropped 2048³ to 6.4 TF/s on the 4060.  Reverted
to single-buffer (18.9 KiB / 9.7 KiB); 2048³ recovered to 22.7 TF/s.
Same occupancy-cliff family as the register-prefetch experiment.

**Prefill QKV concat + pack.** One `w_qkv` GEMM per layer (T=6 and
T≥128 share it — the duplicate `wq`/`wk`/`wv` copies are gone).
f16 T≥128 then `PrefillQkvPack` (RoPE Q to head-major `[H,T,dh]`,
K into Kt, V into V) and flash with token-major O.  Pack requires
`q.shape() == [H,T,dh]`.  CM2 flash row-sums via `coopMatReduceNV`
instead of P@ones.

**RMSNorm vec4 footgun.** Vec4 row walks on `cols=771` (not a
multiple of 4) used `base >> 2` and misread every row after the
first.  Scalar fallback when `cols % 4 != 0`; 768-col pins the
fast path.

**GEMV-chain.** Built, tested, default-off (`ML_DEVICE_SCOPE=1`).
Device-scope memory model scrambled CM2 flash when enabled
globally; llama does not emit `ExecOp::GemvChain`.  Packed B is
rejected.

**Not claimed this leg.** No 3070 pp512/tg128 A/B.  Packed vs
unpacked GEMV microbench needs a `packed_b` cases path.  llama-ash
still keeps packed + unpacked weight copies (prefill GEMM wants
row-major, decode GEMV wants tiled).

## Leg 17 — `experiment/cm2-gemm` (CM2 GEMM with fused store epilogues) — MEASURED (RTX 3070): plain CM2 loses 18-35%, un-demote kept

Coopmat-eligible f16w shapes with a fused epilogue currently DEMOTE to
the SIMT family (`demote_for_op` → `epilogue_fallback_index`), because
the KHR-coopmat1 kernel cannot run per-element code at store time.
llama prefill therefore runs its three residual/gated projections as
PLAIN coopmat matmuls plus 3 standalone `Binary` bandwidth passes per
layer at T >= 256 (`push_mlp_ops`).  NV_cooperative_matrix2's
`coopMatPerElementNV` runs an arbitrary callback over the accumulator
before the store, so a CM2 GEMM keeps tensor-core speed WITH the fused
bias/activation/binary epilogue — expected ≈0% raw-GEMM change (brief:
cm2 ≈ parity with tuned cm1 for plain GEMM), real end-to-end win from
un-demoting.

What landed (all CPU-verifiable; NO GPU runs yet):

- `shaders/matmul_cm2_kernel.glsl` + wrapper `matmul_f16w_cm2.comp`:
  BM=128, BN=64, BK=64 on 128 threads (every matrix dim a multiple of
  64 → inside the exact flexible-dims envelope the init gate already
  validates for cm2 flash; acc = 64 regs/thread).  A f32 quantized to
  f16 via tensor-load decode callback (same numerics as coopmat1's
  staging), B f16, f32 accumulate.  Spec constants 0..3 (variant
  machinery; interior/K flags inert — tensor-layout clamping handles
  edges) + 4..6 shared epilogue via `matmul_epilogue_common.glsl`
  include, applied in a `coopMatPerElementNV` store callback with
  clamped edge-lane reads.  General-shape correct at batch=1 via
  layout clamping; batched dispatches need 16B-aligned batch strides.
- Catalog: `F16wCm2` appended (`f16w_cm2`, tile 128/64/64), slot
  gated on `coopmat2_enabled`.
- Routing: `epilogue_fallback_index` now prefers `f16w_cm2` for
  b_f16 shapes matching the coopmat1 gate (M,N % 128, K % 32,
  M,N >= 256) before falling back to SIMT.  Plain routes untouched.
  Tuner measures `f16w_cm2` only on coopmat-aligned shapes.
- Tests (`tests/correctness/cm2_gemm.rs`, all `#[ignore]`d): plain vs
  dual-rounded reference (aligned/ragged/batched/alpha+accumulate),
  8 epilogue forms + batched-bias vs composed references, the
  un-demote integration case, NaN-poisoned-C overwrite coverage.
  `f16.rs` demote regression updated for the cm2 route.

### GPU validation results (RTX 3070, this session)

Suite: full ignored correctness suite green post-rebase onto
eb81b28 — **129 passed / 0 failed** (all 4 new `cm2_*` tests passed
first run, including ragged clamped-edge stores, the poisoned-C
overwrite, and the un-demote integration case).

**A-load fix (the one real shader change this session)**: the
per-element f32→f16 decode callback forced scalar single-float block
loads and cost ~2x — replaced with a straight f32 tensor load + the
KHR conversion constructor to f16 (identical RNE numerics; callback
kept `#if`-ed out for reference).  4096³ went 15.8 → 22.7 TF/s.

**Plain-GEMM parity check: FAILED — cm2 loses 18-35% everywhere.**
`ml_bench cases` forced-kernel runs, GPU timestamps, burn-in case
(4096³) first, identical case order, both process orderings (stable
to ±0.1 TF/s):

| shape (f16w)      | cm1 TF/s | cm2 TF/s | cm2/cm1 |
|-------------------|---------:|---------:|--------:|
| 512×2048×2048     |    24.3  |    19.9  |   0.82  |
| 512×5632×2048     |    32.2  |    23.2  |   0.72  |
| 512×2048×5632     |    24.8  |    19.2  |   0.78  |
| 4096×4096×4096    |    35.1  |    22.7  |   0.65  |

Tile probes (2, per plan; both reverted): BM128/BN128/BK64 @ 128
threads → 11-12 TF/s (accumulator register blowup at 128
invocations); BM128/BN64/BK128 → 16.5-18.6 TF/s.  BM128/BN64/BK64
stands.  The Bolz cm1≈cm2 parity claim does not reproduce on
GA104/driver 570-class — workgroup-scope coopmat2 GEMM leaves ~1/3
of the KHR-coopmat1 throughput on the floor at every probed shape.

**Fused-epilogue A/B** (`examples/bench_cm2_epilogue.rs`, one
session, burn-in first, single-submission `run_exec_ops` chains):

| case (512-row prefill shape) | fused cm2 | composed cm1+Binary | old SIMT demote |
|------------------------------|----------:|--------------------:|----------------:|
| gate Silu+Mul 512×5632×2048  | 0.620 ms (19.1 TF/s) | 0.494 ms (23.9) | 1.267 ms (9.3) |
| down AddScaled 512×2048×5632 | 0.698 ms (17.0 TF/s) | 0.562 ms (21.0) | 1.141 ms (10.4) |

Two-sided verdict:

- **Un-demote KEPT**: a fused-epilogue op on a coopmat-eligible shape
  now costs ~half of the old SIMT demote (2.0-2.6x faster).  That is
  a real library win for any caller issuing fused ops on these
  shapes.
- **llama-ash NOT rewired**: the composed plain-cm1 + Binary pattern
  it already uses beats fused-cm2 by ~25% (the Binary pass costs far
  less than the cm2-vs-cm1 GEMM gap), so `push_mlp_ops` keeps its
  T >= 256 branch (measured-wins-only).  Plain routes stay on the
  measured coopmat1 winner everywhere; the tuner measures `f16w_cm2`
  on aligned shapes and correctly never picks it.

**Prefill-parity implication (the leg's primary question)**: cm2
GEMM does NOT lift the ~24 TF/s M=512 K-heavy blocker shapes — it
sits 5 TF/s BELOW them.  The remaining path to CUDA-parity pp512
(~16k t/s needs ~33 TF/s average real-shape GEMM) is coopmat1-side:
tile retuning for M=512 K-heavy geometry (512×2048×2048 and
512×2048×5632 run at 69-71% of the 4096³ ceiling on the SAME
kernel — CTA count/wave quantization, not tensor-core limits),
subgroup-scope coopmat tile variants, or K-split forms of cm1.

E2E sanity (llama routing untouched by design — prefill emits plain
matmuls at T >= 256, decode GEMV epilogues never demoted):
interleaved main-vs-branch `bench --pp 512 --tg 128`, 6 pairs both
orderings — see table below; neutral within noise, token gate exact
(`--ids "1,450,7483,310,3444,338" -n 24` reproduces the pinned 24
ids).

| pair | order | main pp512 | branch pp512 | main tg128 | branch tg128 |
|---|---|---|---|---|---|
| 1 | A,B | 12246.90 | 11725.22 | 166.89 | 167.67 |
| 2 | A,B | 11645.29 | 12098.89 | 165.90 | 167.82 |
| 3 | A,B | 12014.04 | 11922.57 | 164.80 | 167.33 |
| 4 | B,A | 11716.82 | 11699.49 | 167.31 | 167.19 |
| 5 | B,A | 11710.66 | 12202.93 | 166.39 | 164.95 |
| 6 | B,A | 11764.22 | 11856.18 | 166.88 | 167.80 |

Medians: pp512 main 11740 vs branch 11890 (+1.3%, noise), tg128
main 166.6 vs branch 167.3 (+0.4%, noise).  Merge-safe: keeps the
un-demote + the cm2 GEMM kernel for fused callers, changes nothing
llama-ash routes through.

## Leg 16 — `experiment/prefill-graph` (prefill as ONE `run_exec_ops` submission)

Prefill previously ran EVERY op as an individual synchronous Executor
call — rms_norm, three qkv matmuls, two ropes, two KV appends, two
head-permute copies, flash attention, and the MLP block, ~14
submit + fence-wait round-trips per layer x 22 layers ≈ 300 per
prefill.  Decode solved this exact problem with the `ExecOp` mixed
graph; this leg gives prefill the same treatment:

- **`ExecOp::FlashAttn`** (the only op prefill needed that the graph
  lacked): wraps the planned flash dispatch verbatim — CM2-first
  routing exactly like standalone `run_flash_attention`, access set
  reads {q, kt, v} writes {out}.  `run_flash_attention_impl` was
  split into `plan_flash_attention` + `submit_one_elementwise`, so
  the immediate and graph paths share one plan.
- **`prefill_with` builds the whole prefill** (all layers + LM-head
  tail) as one `Vec<ExecOp>` and runs it with ONE `run_exec_ops`
  call.  Scratch reuse (xn, q/k/v, q_heads, attn_heads), the x
  ping-pong, and append-then-flash cache dependencies serialize
  through the existing hazard tracker; independent neighbours (the
  three qkv matmuls, the two KV appends) now overlap where the
  per-op path forced full drains.
- **Dedupe/deletions**: MLP emission (`push_mlp_ops`: T>=256 unfused
  coopmat branch / fused-epilogue branch / T==1 normed-GEMV branch)
  and the LM-head tail (`push_lm_head_ops`) are shared between
  `decode_ops` and `prefill_ops`.  Deleted outright: the per-op
  prefill path (`qkv`, `append_kv`, `mlp_block_unfused`, the T>1
  branches of `mlp_block_into`) and the lib's now-unused
  `run_elementwise_2d`.  `mlp_block_into` survives decode-only
  (perop/flash diagnostics modes).  No per-op prefill breakdown
  remains (same convention as decode graph mode); `bench` now logs
  prefill GPU-vs-wall from the graph timestamps instead.

Interleaved A/B (RTX 3070, `bench --pp 512 --tg 128`, alternating
binaries in one session, both orderings; pairs 1-4 of the first run
were invalidated — the B binary was stale [cargo root build had not
relinked the tool], caught and re-run).  8 valid pairs, new wins 8/8:

| pair | order | main pp512 | branch pp512 | main tg128 | branch tg128 |
|---|---|---|---|---|---|
| 1 | A,B | 8559 | 11467 | 170.24 | 168.57 |
| 2 | A,B | 8730 | 11174 | 170.26 | 170.12 |
| 3 | B,A | 9386 | 11147 | 170.40 | 168.64 |
| 4 | B,A | 9713 | 10839 | 169.93 | 170.00 |
| 5 | B,A | 9257 | 11219 | 168.57 | 170.35 |
| 6 | B,A | 8508 | 11169 | 170.80 | 169.63 |
| 7 | B,A | 9693 | 11242 | 169.84 | 169.74 |
| 8 | B,A | 9417 | 11138 | 169.67 | 170.13 |
| mean | | 9158 | **11175 (+22.0%)** | 169.96 | 169.65 (−0.2%, neutral) |

Shape sanity (one run each side): pp128 3494 -> 4346 (+24%);
pp1920 11024 -> 12578 (+14%; TinyLlama's GGUF context_length caps
t_max at 2048, so pp2048+tg is not runnable).

Forensics confirm the submission-overhead theory: graph-mode prefill
is 42.3 ms GPU vs 43.9 ms wall (1.6 ms host: embed/upload, ~360-op
validate+plan+record, logits download); main's per-op wall was
~53-56 ms for the same dispatch chain, i.e. ~10-12 ms of pure
submit + fence-wait overhead now gone.  The GPU-busy 42 ms is the
new floor — reaching CUDA's 16.7k t/s (30.6 ms) from 11.2-11.7k
needs kernel-level wins (flash + GEMM), not submission plumbing.

Correctness: new GPU test `flash_in_graph_matches_standalone_bitwise`
(ExecOp::FlashAttn vs `run_flash_attention`, bitwise).  Suite 114/114
GPU + 45 + 1 unit, clippy/fmt clean on touched files.  Token gate
24/24 byte-identical; T=64 (fused MLP) and T=300 (unfused MLP)
prompts generate identically to main.

## Leg 15 — `experiment/decode-store-fusion` (rope/copy fused into q/k/v GEMV stores)

A fused **store epilogue** on the f16-weights row GEMVs (`STORE_MODE`
spec constant 7, `STORE_DST_F16` 8): the decode q projection RoPE-
rotates its output pairs as it stores (partner column read back from
the already-barriered shared partials for VCOLS=1, lane-local for
VCOLS=2 — bit-exact with the standalone rope), the k projection
rotates AND scatters straight into the `[H, dh, T_max]` Kt cache, and
the v projection scatters into the `[H, T_max, dh]` V cache (f16 RNE
or f32 by cache dtype).  Push constants grow 80 -> 128 bytes (the
Vulkan guaranteed minimum) carrying table/dst/pos pointers + scatter
geometry; position indirection reuses the `PosBuffer` cell, so the
prepared decode graph stays record-once.  Host: `MatmulStore` on
`MatmulOp`, resolution-time validation (M=1, f16 B, no
accumulate/epilogue, `head_dim | N`, table coverage, dst bounds),
store-aware kernel demotion, and precise graph access sets (the
scatter modes write the cache and never C, so the attention's cache
reads barrier correctly).

**Decode graph shrinkage** (TinyLlama: 22 layers): `Rope(q)`,
`RopeScatter(k)`, `CopyStrided(v)` deleted from the decode graph —
12 -> 9 ops/layer, 66 fewer dispatches/token (269 -> 203), and the
k-append hazard barrier is gone (8 -> 7 barriers/layer, −22
barriers/token ≈ −170 µs of drain at the measured ~7.7 µs/barrier).
The standalone Rope / RopeScatter / CopyStrided ExecOps stay — prefill
and the perop path still use them.

Phased interleaved A/Bs (RTX 3070, `llama_ash bench --pp 512 --tg
128`, alternating binaries in one session, warmup prefill + 16 decode
steps burn-in per run; tg128 t/s):

Phase F1+F3 (q-rope + v-scatter) vs main@682f01d — branch 5/6 pairs:

| pair | main | F1+F3 |
|---|---|---|
| 1 | 167.58 | 168.30 |
| 2 | 166.82 | 167.45 |
| 3 | 162.16 | 168.35 |
| 4 | 168.10 | 167.39 |
| 5 | 167.15 | 167.33 |
| 6 | 167.06 | 167.70 |
| mean | 166.48 | **167.75** (+0.8%) |

Phase F2 (k rope-scatter) vs F1+F3 — F2 6/6 pairs:

| pair | F1+F3 | F2 |
|---|---|---|
| 1 | 167.65 | 169.00 |
| 2 | 167.35 | 169.08 |
| 3 | 167.33 | 168.52 |
| 4 | 168.47 | 168.95 |
| 5 | 168.33 | 169.80 |
| 6 | 168.21 | 169.94 |
| mean | 167.89 | **169.22** (+0.8%) |

Final (main vs branch, 8 pairs) — branch 7/8 pairs; pairs 4-6 caught a
clock-throttle window on BOTH sides (the interleave keeps the
comparison valid), where the branch degraded far less — consistent
with lower per-token launch + barrier overhead:

| pair | main pp512 | branch pp512 | main tg128 | branch tg128 |
|---|---|---|---|---|
| 1 | 8211 | 9373 | 168.40 | 170.17 |
| 2 | 9006 | 8712 | 168.25 | 169.35 |
| 3 | 9370 | 8731 | 168.04 | 166.17 |
| 4 | 8067 | 8242 | 151.49 | 154.85 |
| 5 | 8225 | 7131 | 147.89 | 156.65 |
| 6 | 8870 | 8527 | 156.59 | 169.82 |
| 7 | 9023 | 8944 | 168.35 | 168.99 |
| 8 | 9113 | 9263 | 168.06 | 168.86 |
| median | 8938 | 8722 | 168.05 | **168.93** |

Net: tg128 ~166.5-168 -> ~168.5-170 clean-clock (+1.5% across the two
gated phases, both individually positive); pp512 neutral (prefill is
untouched).  **The 175 t/s CUDA-parity goal is NOT reached** — the
fusion removed a quarter of the dispatches and the graph is now
GEMV-dominated; the residual ~6 t/s lives in the seven remaining
hazard barriers/layer and the GEMV kernel time itself (already at
81-89% of bandwidth).

**Phase F4 (QKV weight concat) SKIPPED**, with justification: the
three projection GEMVs already execute as one barrier-free group, so a
single [2048, 2560] GEMV saves only 2 launch slots/layer, not barrier
drain; per-segment stores need a fourth store mode with two dst
pointers + segment bases, pushing the shared push-constant block past
the 128-byte guaranteed minimum for every matmul pipeline in the
library; and the measured yield of pure dispatch-count reduction
(F1+F3: −44 dispatches -> +0.8%) bounds the expected win at well under
+0.5% for materially higher complexity and a portability regression.

Correctness: 7 new GPU tests (fused-vs-composed BITWISE equality for
rope / scatter / rope-scatter on k16 and k16_v2 routes, interior and
ragged N, f16 + f32 caches; pos-cell indirection; validation
rejections incl. loud record-time failure on non-store kernels).
Suite 113/113 + 45 unit, clippy clean.  Token gate 24/24
byte-identical in prepared, graph, and `LLAMA_ASH_KV=f32` modes.

## Leg 14 — `experiment/cm2-flash` (NV_cooperative_matrix2 flash attention)

Tensor-core flash-attention prefill via `GL_NV_cooperative_matrix2`
(the predicted fix for the KHR-coopmat1 flash dead end: workgroup-scope
matrices + `coopMatReduceNV`/`coopMatPerElementNV` keep the
online-softmax M/L/O state matrix-resident, no shared-memory O round
trip). Br=Bc=64 @ 128 threads, f16 operands / f32 accumulate,
tensor-layout dims clamped to the valid KV extent so poisoned cache
tails are never loaded; f32 KV decodes through a per-element callback.
Kernels build only when the extension probe passes
(`VulkanContext::coopmat2_enabled`, `ML_NO_COOPMAT2=1` kill-switch);
`run_flash_attention` routes to them with the SIMT kernels as fallback
and `run_flash_attention_simt` as the A/B hook.

Enablement on RTX 3070 (driver 595.80): all 7 feature bits, max
flexible dim 1024, reserved shmem 8192, flash config supported ->
`enabled=true`.

Kernel A/B (`examples/bench_flash_cm2`, same session, >=300 ms burn-in
per case, SIMT/cm2 iterations interleaved, gpu-timestamp median of 50,
µs):

| case (H=32) | SIMT | cm2 | speedup |
|---|---|---|---|
| dh64 T512 kv_f32 | 458.8 | 81.8 | 5.61x |
| dh64 T512 kv_f16 | 458.6 | 66.1 | 6.93x |
| dh64 T2048 kv_f32 | 4153.2 | 735.4 | 5.65x |
| dh64 T2048 kv_f16 | 4108.4 | 631.1 | 6.51x |
| dh128 T512 kv_f32 | 888.8 | 147.2 | 6.04x |
| dh128 T512 kv_f16 | 883.9 | 114.1 | 7.75x |

The leg-2 reference case (dh64 T2048) drops 4.15 ms -> 0.74 ms, far
past the 2.5 ms target; dh128 — where SIMT was register-bound and only
reached parity with the composed path — now wins 6-7.8x.

**End-to-end** (TinyLlama-1.1B f16, main@1ab581c vs branch, same
session, interleaved main/branch, 6 pairs, `llama_ash bench --pp 512
--tg 128`):

| run | main pp512 | cm2 pp512 | main tg128 | cm2 tg128 |
|---|---|---|---|---|
| 1 | 7079 | 9573 | 165.2 | 167.6 |
| 2 | 8220 | 8584 | 165.0 | 167.1 |
| 3 | 8354 | 9581 | 161.3 | 168.6 |
| 4 | 8144 | 8694 | 165.5 | 168.2 |
| 5 | 7780 | 9404 | 164.5 | 167.8 |
| 6 | 7528 | 9019 | 166.0 | 167.8 |
| median | 7962 | **9211** | 165.1 | 167.8 |

pp512 **+15.7%** (branch faster in 6/6 pairs; ~0.55x CUDA's 16.7k, up
from 0.48x); tg128 +1.6% (decode uses the split-K path, so this is
noise-level positive, not a flash effect). Greedy generation stays
byte-identical (24/24 reference ids). Full GPU suite 106/106 (incl.
the new cm2-vs-SIMT-vs-f64 A/B: dh64 T512/T2048, dh128 T512, f32+f16
KV, GQA/warm-cache/ragged, poisoned tails — all within the provisional
2e-2 f16-operand tolerance on first run, no fixes needed).

## Leg 13 — `experiment/gemv-vec-loads` (VCOLS packed-load row GEMVs)

Parameterized `matmul_row_bda_kernel.glsl` with VCOLS (each lane owns
VCOLS adjacent output columns, B read as `f16vec2`/`f16vec4`, widening
the per-warp B transaction from 64B to 128B/256B per k-step; ragged-N /
unaligned bases fall back to an in-shader scalar path; results bit-exact
across variants). Probed 2/4-column variants of both the 8-slice and the
16-slice f16-weights row GEMVs on the decode-critical shapes.

Per-shape A/B (RTX 3070, `ml_bench cases`, ML_ITERS=500 after a
GEMM burn-in case, identical order, two full cycles — repeatable to
<1 µs; gpu-timestamp median, µs):

| shape (M=1) | row_bda | k16 | v2 | v4 | k16_v2 | k16_v4 |
|---|---|---|---|---|---|---|
| K=2048 N=2048 (q/o_proj) | 31 | 23 | 30 | 30 | **23** | 23 |
| K=5632 N=2048 (ffn_down) | 76 | 58 | 75 | 76 | **58** | 58 |
| K=2048 N=5632 (gate/up) | 58 | 61 | 58 | 57 | **56** | 57 |
| K=2048 N=256 (kv proj) | 16 | **8** | 14 | — | 9 | 12 |
| K=256 N=2048 | 3.7 | 3.5 | 3.7 | 3.7 | 3.5 | 3.7 |
| K=2048 N=32000 (lm_head) | 310 | 319 | 306 | — | **307** | 307 |

Reading: the k16 kernel is already at 81-89% of the 448 GB/s bandwidth
floor on its routed shapes, so packed loads are neutral there. The one
real win is wide-N moderate-K (gate/up), where 16 slices alone regress
~5% vs 8 slices but 16 slices + VCOLS=2 win ~3.3% (58 -> 56 µs).
VCOLS=4 lost or tied everywhere (N=256: 12 vs 8 µs — occupancy
starvation at 128 columns/workgroup); 8-slice v2 never beat its
16-slice sibling.

**Route change**: the m==1 b_f16 wide-N branch (the `else` of the k16
rule) now routes to `f16w_row_bda_k16_v2`; the deep/narrow branch stays
on plain `f16w_row_bda_k16`. **Deleted** (losers, no dead code):
`f16w_row_bda_v2`, `f16w_row_bda_v4`, `f16w_row_bda_k16_v4` wrappers +
catalog entries, and the f16vec4 path in the shared shader. Also fixed
the tuner's m>1 row-kernel exclusion (`ends_with("row_bda")` ->
`contains("row_bda")`, which the k16 names had been slipping past).

**End-to-end** (same session, interleaved base/branch, 8 reps each,
warm-up run discarded): tg128 base 164.8-169.0 (mean 167.1), branch
167.4-171.0 (mean 169.4) — **+1.3%**, branch faster in 7/8 pairs;
pp128 neutral. Below the 175 t/s CUDA-parity goal and below the 2%
branch-level gate; the per-shape gate/up win (+3.3%) is real but only
~90 µs/token of a ~5.9 ms token. Greedy generation stays byte-identical
(24/24 reference ids). 105/105 GPU tests.

Negative result recorded: packed B loads are NOT the remaining decode
gap — the row GEMVs already run at 81-89% of bandwidth. The residual
~8 t/s to CUDA parity lives in barrier drain + non-GEMV kernel work,
not GEMV load width.

## Legs 6-12 — decode/prefill optimization sweep (summary)

TinyLlama-1.1B f16, RTX 3070, llama.cpp CUDA (fa=1) as reference.
Greedy generation stayed byte-identical to the CUDA reference after
every leg; each leg's number is its own clock window (see the PR merge
messages for full context).

| leg | branch | mechanism | measured |
|---|---|---|---|
| 6 | `experiment/prefill-coopmat` | standalone `BinaryOp` bandwidth passes keep T>=256 projections on tensor cores (coopmat cannot fuse epilogues) | pp512 5,435 -> 7,795 t/s (+43%); decode-neutral |
| 7 | `experiment/gemv-k16-widen` | 16-slice GEMV route widened to K>=2048 && N<=2048 (o_proj / q-projection -23%) | tg 103.9 -> 106.7 t/s (+3%) |
| 8 | `experiment/f16-kv-cache` | f16 KV caches: RNE-narrowing appends, kv16 flash variants, cache-dtype-driven selection | speed-neutral at <=2k ctx; 44 vs 88 MB KV |
| 9 | `experiment/decode-attention` | fused split-K decode attention (`run_attn_decode`); chunking tuned to ~32 positions/chunk within [8, 32] | attention 2,752 -> 526 µs/token; tg 107 -> 141 t/s (0.81x CUDA) |
| 10 | `experiment/decode-fusion` | RMSNorm folded into row GEMVs (`with_normed_a`); k-RoPE + Kt append fused into one `RopeScatter` | 16 -> 13 graph ops/layer; tg 141 -> 147 t/s |
| 11 | `experiment/replay-graph` | record-once replayable decode (`prepare_exec_ops` + `PosBuffer` indirection; fixed 32-chunk attn grid) | tg 147 -> 160 t/s (0.91x CUDA); replays bitwise |
| 12 | `experiment/gpu-token-loop` | on-GPU argmax + embedding gather close the token loop; host writes/reads ONE u32 per token | tg 160 -> 165 t/s (tg128 162-166; 0.94x CUDA) |

Standing after leg 12: decode tg128 163-166 t/s vs CUDA 175 (0.94x);
prefill pp512 ~8.0k vs 16.7k t/s (~0.48x). The remaining decode gap is
kernel work + residual barrier drain; the prefill gap is tensor-core
coverage (f16 activations end-to-end, epilogue-capable coopmat).

## Leg 5 — `experiment/token-replay` (whole-token graphs + decode failure analysis)

### Decode: 82.9 → 111.8 t/s in four measured steps

| step | tg128 t/s | mechanism |
|---|---|---|
| baseline (per-op) | 82.9 | ~350 submit+wait per token |
| whole-token graph (`run_exec_ops`) | 87.7 | one submission; bitwise-identical results |
| hazard-aware barriers | 96.4 | fence only real RAW/WAW/WAR; QKV/RoPE/KV-append overlap |
| free reshapes (`Tensor::alias_with_shape`) | 107.1 | GQA views alias q/attn memory; −44 dispatches+barriers |
| 16-slice deep-K GEMV (`f16w_row_bda_k16`) | 111.8 | ffn_down 78→61 µs; deep-K/narrow-N routes |

**Failure analysis that drove it** (LLAMA_ASH_BREAKDOWN per-op GPU timing):
kernels 7.26 ms vs whole-CB 9.97 ms exposed ~2.7 ms of full-barrier drain
(~7.7 µs each); kernel table showed FFN GEMVs at 58-72% of bandwidth
(deep-K serial chains, narrow-N occupancy starvation) while attention was
only 12% of the token. The naive "submission overhead dominates" theory
was half-wrong: one submission alone bought +6%; the rest came from
barrier elimination and kernel work.

### Comparison matrix (TinyLlama-1.1B f16, RTX 3070, llama.cpp CUDA fa=1)

| test | tensor-ash | llama.cpp CUDA | ratio |
|---|---|---|---|
| pp128 | 3,807 t/s | 10,114 t/s | 0.38x |
| pp512 | 5,435 t/s | 16,718 t/s | 0.33x |
| pp1024 | 5,649 t/s | 16,679 t/s | 0.34x |
| pp2032 | 4,261 t/s | 16,393 t/s (pp2048) | 0.26x |
| tg128 | **112.3 t/s** | 175.1 t/s | **0.64x** (was 0.48x) |

Generation remains byte-identical to the CUDA reference (24/24) after
every step. 85/85 GPU tests.

### Remaining path to the goal (tensor-ash > CUDA), ranked

Decode budget now ≈ 8.9 ms: ~6.9 ms kernels + ~1.6 ms barrier drain +
~0.5 ms host. Target 5.7 ms (175 t/s), stretch below.
1. **Op fusion to cut barrier count**: RoPE-into-cache (write the cache
   from the rope kernel; −44 ops), norm+matmul pairs; a decode-attention
   kernel collapsing scores/softmax/PV (−44 ops).
2. **GEMV last 15%**: o_proj at 58% is the worst remaining; qkv batching.
3. **Replayable token graph**: position values via host-visible buffer so
   the command buffer records once (−~0.5 ms host, enables N-token
   pipelining).
4. **Prefill (the wide gap, 0.33x)**: standalone binary elementwise op so
   the big projections keep the coopmat route instead of demoting for
   fused epilogues; then f16 activations end-to-end; then NV_coopmat2
   flash. llama.cpp's pp advantage is pure tensor-core coverage.



## Leg 4 — `experiment/llama-runner` (real-model E2E vs CUDA)

### TinyLlama-1.1B f16 on tensor-ash, compared against llama.cpp CUDA

`tools/llama-ash`: GGUF f16 loader + full TinyLlama forward pass composed
purely from library ops (flash prefill, composed GQA decode, fused
epilogues, RoPE, strided-copy KV caches). **Correctness: greedy generation
is byte-identical to llama.cpp CUDA for all 24 reference tokens** (temp 0,
same prompt ids, same f16 weights).

| metric (RTX 3070) | tensor-ash | llama.cpp CUDA (fa) | ratio |
|---|---|---|---|
| prefill pp512 | 5,525 t/s | ~16,100 t/s | 0.34x |
| decode tg128 | 84.8 t/s | ~176 t/s | 0.48x |

**Honest verdict: correct, not yet faster in practice.** Gap analysis:

1. **Decode is host-submission-bound**: ~350 synchronous dispatches/token
   (22 layers × ~16 ops), each paying ~10 µs submit+wait ⇒ a ~3.5 ms/token
   floor before GPU work. llama.cpp submits one CUDA graph per token.
   Fix: extend PreparedOps replay to elementwise/attention ops so a whole
   token is one pre-recorded command buffer (gameplan item since v1.5).
2. **Fused-epilogue ops demote off the tensor cores** (the routing fix
   that made prefill *work* also costs it): gate/up/down/o_proj carry
   epilogues ⇒ SIMT BDA_V4 instead of coopmat. Fix options: epilogue
   support in the coopmat kernel via shared-staging stores, or a measured
   route choice between fused-SIMT and coopmat+separate-elementwise.
3. **f32 activations**: llama.cpp runs f16 end-to-end; our activations
   double the bandwidth on every op. f16 activation tensors are the
   natural Phase C of the dtype work.
4. Attention: SIMT f32 flash vs their tensor-core FA.

Also fixed on this branch (found BY the real model): auto-routing sent
fused-epilogue ops to `f16w_coopmat_aligned`, which cannot fuse — every
aligned T≥256 prefill with a residual failed. Fused ops now demote to the
epilogue-capable SIMT sibling (auto routes only; explicit ML_KERNEL keeps
its loud failure), with a regression test pinning the demoted route and
numerics. This class of bug is exactly why practice-testing matters: no
synthetic test had combined aligned-f16 shapes with epilogues.



## Leg 2 — `experiment/flash-attention` (branched from main@v2.0.0, 2026-08-14)

### Design: fused causal prefill attention (flash-attention style)

**Why:** the composed prefill path materializes `scores [H, T_q, T_kv]`
and moves it through global memory three times (matmul write, softmax
read+write, PV read). At T=4096, H=32 that is 32·4096² floats = 2 GiB of
score traffic per layer vs the ~100 MiB of Q/K/V/O. An online-softmax
fused kernel keeps score tiles in registers/shared and never leaves the
chip — the classic FlashAttention result, expected to dominate at T ≥ 1k.

**Design (v1, SIMT f32):**
- One workgroup per (head, q-tile). One thread owns one query row: its
  running max `m`, running sum `l`, and the `acc[DH]` output row in
  registers. `DH` is a specialization constant (64/80/96/128 pipelines
  compiled lazily per model geometry).
- K-loop over key tiles staged in shared memory, read directly from the
  existing cache layouts (`Kt [H, dh, T_max]`, `V [H, T_max, dh]`) so the
  fused and composed paths interoperate on the same tensors.
- Per tile: S-row = q·K tile (from shared), causal/prefix mask, tile max,
  one rescale of `l`/`acc` per tile (never per element), P·V accumulate.
  Final `out = acc / l` (all-masked rows store 0).
- Causal semantics match `SoftmaxMask::Causal`: query row `i` attends to
  positions `< pos_base + i + 1`, clamped by `kv_len`; causal tiles fully
  above the diagonal are skipped, so early rows do ~half the work.
- GQA: `kv_head = head / group_size` push constant, no K/V duplication.

**Method:** baseline first (composed path measured per stage at
T=512..4096, dh=64/128), fused kernel must beat the *sum*; correctness
against f64 CPU reference and bit-agreement checks vs the composed path
on shared inputs; paired A/B in one process.

### Leg 2 results: fused flash-attention prefill (v1 shipped)

**Measured A/B (paired, same process, H=32):**

| case | composed sum | flash | speedup |
|---|---|---|---|
| dh64 T1024 | 1.53 ms | 1.46 ms | 1.05x |
| dh64 T2048 | 5.81 ms | 4.30 ms | **1.35x** |
| dh128 T2048 | 7.98 ms | 8.87 ms | 0.90x |
| dh128 T4096 | 32.8 ms | 32.3 ms | 1.02x |

Equally important: flash never allocates the `[H, T_q, T_kv]` score tensor —
**2.1 GiB at H32/T4096** on the composed path — so long-context prefill fits
where the composed path cannot. Correctness pinned against an f64 CPU
reference across five geometries (poisoned cache tails, GQA groups,
warm-cache offsets, ragged sizes, single-row) plus 1e-5 agreement with the
composed path on shared inputs. Suite is 83 GPU tests.

**Optimization attempts, all measured and rejected:**
- *V via warp-uniform L1 reads instead of shared staging*: 2-4x SLOWER —
  per-element uniform global loads serialize through the LSU inside the hot
  FMA loop; shared-memory broadcast is genuinely free, L1 is not.
- *SPLIT=2 (two lanes per query row) for dh128*: 0.61-0.76x — halving rows
  per workgroup doubles K/V staging cost per row, swamping the register
  relief.
- *BK=16 for dh128*: 0.52-0.71x — twice the tiles means twice the online
  rescales (`acc[128]` multiply each) and barriers.

**Verdict:** v1 (BK=32, one lane per row) is the optimum of this design
family on GA104. dh128 SIMT flash is register-bound (~180/thread) and only
reaches parity; the ceiling-breaker there is a cooperative-matrix flash
kernel (tensor-core S and PV tiles) — recorded as the next candidate leg.
Routing guidance in docs: flash for dh64 at T >= ~1k, or whenever score
memory is prohibitive; composed path otherwise.

# Leg 1 log — `experiment/model-inference` (merged as v2.0.0, PR #1)

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

### 2026-08-14 — step 2: FP16 Phase A landed (f16 weights, f32 accumulate)

**What:** `DType::{F32,F16}` on `Tensor` (`uninit_device_f16`), CPU RNE
f16↔f32 conversion in the `&[f32]` upload/download path (host API unchanged),
`shaderFloat16`+`storageBuffer16BitAccess` enabled when present
(`ctx.f16_storage_enabled`), and six new `f16w_*` registry kernels sharing the
existing bodies via `#ifdef B_F16`: large/m128n64k64/k64/small BDA_V4 tiles +
the K-cooperative row GEMV. B loads are `uvec4` packets (8 halves, LDG.128)
unpacked to f32 at shared-staging time, so the proven FFMA inner loop is
byte-identical; only global B traffic halves. Registry slots for `f16w_*`
kernels stay empty on devices without the features; routing, tuning
(`TuneKey.b_f16`, store header v3), split-K2 veto, and explicit-selection
validation all understand storage type. Mismatches error clearly
(`kernel 'x' expects f32 B storage…`).

**Measured (RTX 3070, paired same-process runs):**

| shape | f32 GPU ms | f16w GPU ms | speedup |
|---|---|---|---|
| decode GEMV 1×4096×4096 | 0.158 | 0.084 | **1.88x** |
| llama down-proj 1×4096×11008 | 0.420 | 0.218 | **1.93x** |
| sq1024 (m128n64k64 class) | 0.212 | 0.209 | 1.01x |
| tall 4096×1024×1024 | 0.799 | 0.792 | 1.01x |
| mid384 (k64 class) | 0.019 | 0.018 | ~1.03x |
| sq2048 / batched 8×256³ | 1.485 / 0.032 | 1.498 / 0.032 | ~0.99x / 1.0x |

Exactly the predicted profile: bandwidth-bound decode shapes approach 2x,
compute-bound shapes are neutral (H2F unpack hides under the FFMA pipeline).
One route bug found by the bench and fixed: `to_f16w` initially mapped the
m128n64k64 class onto the large tile (−18% on 1024³); adding the
`f16w_m128n64k64_bda_v4` sibling restored parity, and the route test now pins
the class mirror.

**Verified:** 73/73 GPU tests (5 new f16 tests: rounded-reference matmuls
across all tile classes + ragged edges, route/split-K2 assertions, GEMV +
batched-broadcast epilogue, explicit-mismatch rejection, upload/download
roundtrip), 45 unit tests (RNE encode/decode vectors), clippy clean.

**Direction check:** decode is now ~2x on half the memory footprint — the
local-model lever works. Next: model ops (softmax/rmsnorm/rope/copy) so a
decoder block composes end-to-end, then C ABI v2 exposing dtype + ops + prepared,
then coopmat Phase B for the compute-bound prompt side.

### 2026-08-14 — steps 3-5: internals dedup, C ABI v2, model ops

**Dedup (504059a):** one shared `build_compute_pipeline` (module + spec
constants + pipeline cache) replaced three hand-rolled builders; timestamp
wrap/readback and single-CB submit are now single functions used by both the
sync and prepared paths; the compute→compute barrier helper is shared by graph
and split-K2 recording. ~180 duplicated lines removed, behavior pinned by the
full suite.

**C ABI v2 (b6795d1):** 14 → 32 exports — f16 tensors + capability queries,
epilogue ops (`ta_run_ops`/`ta_run_op_graph`), prepared replay handles
(`create/run/submit/wait/destroy`; the Rust lifetimes instantiate at `'static`
with a documented outlive-the-handle C contract, and destroy fence-waits via
`PreparedOps::Drop`), dispatch-info/tune diagnostics, `ta_executor_create_v2`
with the tune flag. The C smoke example exercises epilogues, f16, and prepared
replay on GPU and exits 0.

**Model ops (5c96580):** `run_softmax_rows` (prefix/causal masks, exact-zero
masked tail), `run_rms_norm` / `run_layer_norm`, `run_rope` (partial rotary,
in-place safe), `run_copy_strided` (transpose/KV-append/head reshape) — a
lazily-built sibling pipeline beside split-K2 reusing the shared builder and
slot machinery. The design bet that a zero-padded KV cache composes exactly
(masked softmax writes true zeros → padded P@V is exact) is pinned by an
end-to-end decode-attention test against a CPU reference. Measured
(`examples/bench_ops.rs`, RTX 3070): rmsnorm ~450 GB/s effective (memory
bound), rope ~85% peak, softmax 0.49 ms @4096² (3-pass; possible later win:
shared-memory exp caching), strided transpose 154 GB/s (scatter-bound;
load-time only). Suite is now 78 GPU tests.

**Direction check:** a decoder block now composes end-to-end in-library.
Remaining: expose the five ops through the C ABI, then coopmat Phase B
(tensor cores) for prompt-side compute.

### 2026-08-14 — steps 6-8: C ABI model ops, TENSOR CORES, decoder-layer proof

**C ABI ops (2351bfe):** the five model ops exported (37 total C entry
points); smoke example checks rms_norm/masked-softmax/transpose on GPU.

**Cooperative matrix (514006d) — the headline result.** KHR coopmat kernel
(`f16w_coopmat_aligned`): 8 subgroups, 128×128×BK32 tile, 4×2 fragments of
16×16×16 per subgroup, A f32→f16 (RNE) at shared staging, B f16 storage,
f32 accumulate, uvec4-staged shared tiles with 16-byte skew (NVIDIA
vk_cooperative_matrix_perf pattern). Strictly aligned by design — coopmat
requires subgroup-uniform control flow, so ragged shapes keep the SIMT f16w
route and the `_aligned` suffix reuses the existing host-side check.
Measured (paired in-process, GPU medians):

| shape | f32 (best DP) | coopmat | speedup | TF/s |
|---|---|---|---|---|
| 4096³ | 11.89 ms | **3.97 ms** | **3.0x** | **34.6** (~87% of TC f32-acc peak) |
| 2048³ | 1.486 ms | 0.539 ms | 2.76x | 31.9 |
| 1024³ | 0.188 ms | 0.093 ms | 2.0x | 23.0 |
| 512³ | 0.037 ms | 0.035 ms | ~1.06x | — (occupancy floor; threshold M,N ≥ 256) |
| prefill 512×4096×4096 | — | 0.579 ms | — | 29.7 |

The library's f32 program ceiling was ~13 TF/s (63-68% of FFMA peak, v1.1
analysis); tensor cores nearly triple it. Correctness pinned by a
dual-rounded-reference test (f16×f16 products are exact in f32, so the
standard `tolerance(k)` holds) covering batch, alpha, and accumulate.
`ML_NO_COOPMAT=1` kill-switch; registry slot empty without the extension.

**Decoder-layer proof (this commit):** one llama-style decode step composed
entirely from library ops — RMSNorm → QKV (f16 weights) → RoPE → KV-cache
append via strided copy → masked attention over the zero-padded cache →
o_proj with fused residual → RMSNorm → SwiGLU MLP with fused gate/residual —
matches an f64 CPU reference (observed error ~1e-5 vs 8×tolerance bound).
Suite: 80 GPU tests.

### 2026-08-14 — coopmat register-prefetch experiment: REJECTED

Tried the NVIDIA-sample software pipeline (fetch tile kt+1 into registers
while the tensor cores work on kt, then store to shared). Clean A/B with
identical case order: **every shape lost badly** — sq4096 4.31 → 6.21 ms
(−44%), sq2048 0.597 → 0.839, sq1024 0.106 → 0.138, prefill 0.579 → 0.736.
Same structural cause as the v1.1 f32 source-level double-buffer dead end:
without a `cp.async` equivalent, the 16 prefetch registers stay live across
the entire fragment-math section, and on top of ~120 accumulator/fragment
registers that hits a register cliff / occupancy loss that outweighs any
load-latency hiding. Reverted; the simple single-buffered loop stays
(baseline re-confirmed at 4.30 ms after revert). Lesson extended: the
register-double-buffer trick that worked *inside* the f32 FFMA inner loop
(tiny 2-deep banks) does not transfer to whole-tile prefetch around coopmat
math. The remaining ~13% to TC peak likely needs subgroup-level tile
re-tuning or NV_coopmat2 features, not source pipelining.

**Direction check:** the mandate is delivered — FP16 in two tiers (storage
and tensor cores), a C ABI that can drive a real decoder, verified building
blocks for llama-class models. Remaining ideas for future legs: coopmat
BK/tile tuning + double buffering (the ~13% gap to TC peak), a fused
flash-attention prefill kernel, lda/offset views to avoid padded-cache
reads at long context, test-scaffolding dedup (~250-400 LOC), GEMM+norm
op-graph integration so a whole layer replays as one PreparedOps command
buffer.
