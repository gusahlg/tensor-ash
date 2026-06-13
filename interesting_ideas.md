# tensor-ash interesting ideas

Distilled from the 971-idea brainstorm in `IDEAS.md`. One investigation agent
per idea, each asked: technically possible on consumer Ampere via Vulkan compute,
and would it move our SGEMM perf needle? Kept only the `PROMISING` (3) and
`MAYBE` (61) verdicts — the other 907 were judged SKIP and live
on in `IDEAS.md` as raw material.

Each entry includes the idea, why it might work, and a concrete next step. Effort:
S=hours, M=day, L=week, XL=research project. Risk: regression/correctness risk if
we tried it.

---

## How to read this file

Angles that contain at least one `PROMISING` are listed first; everything else is
alphabetical. Within an angle, entries are sorted PROMISING → MAYBE, then by
ascending effort (small wins first).

---

## Algorithmic Restructuring / Dispatch / Compiler Ideas — SGEMM @60% on RTX 3070

_2 promising, 19 maybe_

### #245 — **PROMISING** · effort S · risk LOW
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Emit a shape-specialised SPIR-V variant of the BM128/BN128/BK8 kernel that drops every `if (row<M)`/`if (col<N)`/K-tail branch when (M%BM==0, N%BN==0, K%BK==0), and dispatch it from the runtime selector for aligned shapes.

**Rationale:** Our hot loop already has tight FFMA cadence at ~60-68% peak; even predicated bounds-check ifs cost compare+select uniforms, extra registers for masked loads, and inhibit the driver's load coalescing/unrolling around LDG.128. Removing them in the no-tail variant directly reduces instruction count in the inner K loop and the epilogue store, which is exactly where we're FFMA/issue-bound. SPIR-V specialization constants (or a parallel KERNEL_SPECS entry compiled with `#define NO_BOUNDS`) make this a 2-line plumbing change, and the showcase shapes (1024, 2048, 4096) are already multiples of 128, so the win is real on benchmarks we care about.

**Next step (only if PROMISING/MAYBE):** Add a `bm128_bn128_bk8_aligned` KERNEL_SPECS entry whose GLSL `#define TENSOR_ASH_ALIGNED 1` strips the M/N row/col guards and K-tail loop, and route the selector to it when M%128==N%128==K%8==0.

### #278 — **PROMISING** · effort L · risk MED
*Section: Batched / Cross-call Fusion*

**One-line idea:** Replace tile-grid dispatch with Stream-K: launch exactly `numSMs * waves` workgroups, each consuming a contiguous slice of the flattened (M/BM * N/BN * K/BK) iteration space and atomically reducing partial C tiles when a slice straddles a tile boundary.

**Rationale:** Stream-K is the canonical fix for the wave-quantization tail that hurts tensor-ash on shapes where (M/128)*(N/128) is not a clean multiple of 46 SMs — exactly the kind of "non-square / awkward" cases where we currently lose to cuBLAS in 12/26 showcase entries. It is implementable in Vulkan compute today: we already have BDA push constants and an atomic-float reduce path from split-K, so the missing pieces are a work-decomposition prelude in the shader and a small fixup kernel (or in-kernel atomics) for partial tiles. It will not raise our peak-utilisation ceiling on big square shapes (still LDG/LDS/FFMA bound at ~60-68%), but it should meaningfully widen the win-set by killing tail-wave idle SMs, which is the single biggest non-microarchitectural lever left. Risk is MED because atomic-float reduction of partial tiles introduces non-determinism and needs a careful "first WG zeros, last WG writes final" or fixup-kernel design.

**Next step (only if PROMISING/MAYBE):** Prototype a `bk16_streamk` variant that flattens the MN tile grid into a 1D iteration count, dispatches `46 * 2` WGs, uses the existing split-K atomic reduce for partial tiles, and benchmark on the 12 showcase shapes where we currently lose to cuBLAS.

### #220 — MAYBE · effort S · risk MED
*Section: Split-K & K-Reduction Restructuring*

**One-line idea:** Partition K into ~8 chunks, dispatch 8 concurrent SGEMM workgroups per output tile, and have each atomicAdd its partial C-tile contribution to global C.

**Rationale:** Vulkan supports this trivially (VK_EXT_shader_atomic_float gives atomicAdd on f32, already used by our existing split-K scaffolding). However the idea as stated (atomicAdd directly on C) is strictly worse than what we already have: our split-K path already does atomic float reduce, and it has not beaten the Large kernel on showcase shapes because at M=N=1024..4096 with BK=8 we are LDG/LDS/FFMA-bound, not K-parallelism-starved — splitting K just multiplies global C traffic and serializes atomics on the same 128x128 tile across 8 dispatches. The only place this actually helps is small-M/small-N + very large K (tall-skinny / GEMV-ish), where output-tile count is too small to fill 46 SMs; for those shapes adding a shape-gated split-K selector entry could plausibly win.

**Next step (only if PROMISING/MAYBE):** Add a shape-gated KERNEL_SPECS entry that routes only tall-skinny shapes (output-tile count < ~92, K >= 2048) to the existing split-K atomic-reduce kernel and benchmark against Large to confirm a real win before broadening.

### #247 — MAYBE · effort S · risk LOW
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Emit a single SPIR-V module for our SGEMM kernel with BM/BN/BK/TM/TN (and unroll factors) declared as `OpSpecConstant`s, then bake N pipeline variants at pipeline-creation time via `VkSpecializationInfo` so we replace source duplication / 27 KERNEL_SPECS shader files with one SPIR-V + a table of spec-constant tuples.

**Rationale:** This is fully supported on Vulkan/Ampere — `VkSpecializationInfo` + `SpecConstantOp` is standard, and the driver re-runs SPIR-V optimisation (constant-folding loop bounds, unrolling, register allocation) per specialisation, so a spec-constant BK=8 should generate identical SASS to a hard-coded BK=8. So it does NOT directly raise the 60-68% ceiling (we are LDG/LDS/FFMA bound, not dispatch- or codegen-bound), but it is a force-multiplier: it makes shape-specialised autotuning (BM/BN/BK sweeps, TM/TN sweeps, unroll-factor sweeps per M/N/K bucket) cheap to maintain, which is the real path to squeezing the last few % and to keeping the 27-variant table from becoming 200. Risk is low because the kernel source stays one file and we can fall back to the existing hard-coded variants. The only real gotcha is that `shared uvec4 Bs[BM][BK]` array sizes must be expressible from spec-constants — legal in SPIR-V, but glslang/GLSL front-end occasionally refuses array dims from spec-consts, so we may need to author in SPIR-V directly or use `GL_EXT_spec_constant_ops`.

**Next step (only if PROMISING/MAYBE):** Prototype by converting just the current best BM=128/BN=128/BK=8/TM=8/TN=8 kernel to use spec-constants for BK and the inner unroll factor, verify glslang accepts spec-const-sized shared arrays (else hand-edit SPIR-V), and confirm the specialised pipeline matches today's perf within noise on 4096^3 before expanding the sweep.

### #221 — MAYBE · effort M · risk LOW
*Section: Split-K & K-Reduction Restructuring*

**One-line idea:** Run split-K as N independent GEMM dispatches that each write to their own C' slab (shape [splits, M, N]), then a tiny second-pass reduction shader sums the slabs into final C, avoiding any atomic float reduction.

**Rationale:** We already have atomic-float split-K landed as scaffolding but not beating Large; atomic FP32 reductions on Ampere serialize and hurt at the tile-overlap points. A separate-slab + reduction-kernel design is the textbook clean version (used by cuBLAS splitK, CUTLASS) — it trades sizeof(C)*splits VRAM for deterministic, contention-free accumulation, and the reduction pass is bandwidth-bound and cheap (one HBM read+write of C per slab). However, our bottleneck is the BM=128 BN=128 BK=8 tile's LDG/LDS/FFMA throughput, not the reduction; split-K only helps skinny-K shapes where we lack enough tiles to fill 46 SMs. So it's a clean correctness/quality-of-implementation win over atomic split-K but unlikely to lift the geomean — it may unlock specific tall-skinny shapes currently stuck below 60%.

**Next step (only if PROMISING/MAYBE):** Add a `split_k_separate` variant to KERNEL_SPECS that allocates a [splits, M, N] scratch SSBO, dispatches the existing tile kernel with a slab-stride push constant, and chains a trivial 1D reduction shader; benchmark only on K-heavy / small-MN showcase shapes vs the current atomic split-K.

### #228 — MAYBE · effort M · risk MED
*Section: Persistent Kernels & Dispatch Strategy*

**One-line idea:** Launch exactly `46 * waves_per_SM` workgroups whose body loops on `atomicAdd` against a global tile counter, fetching (m_tile, n_tile) indices and computing one 128x128 C-tile per iteration so we amortise launch + overlap C-store with the next A/B prefetch.

**Rationale:** Launch overhead on a 46-SM Ampere with our 128x128 tile is already small for the showcase shapes we win on (e.g. 4096^3 dispatches ~1024 WGs, ~22 waves), so amortisation alone won't move the needle; the real potential lever is the inter-tile pipelining — keeping FFMA busy while the previous tile's C-store and the next tile's first A/B chunk are in flight, which directly attacks our LDG/FFMA overlap ceiling. However, our actual bottleneck is intra-tile LDG/LDS/FFMA scheduling (the 60-68% ceiling), and the scaffolding for persistent kernels already landed without beating Large, suggesting the gain is small unless paired with explicit cross-tile prefetch. Vulkan-feasible: `atomicAdd` on an SSBO uint, push-constant grid shape, no extension needed; SPIR-V workgroup memoryBarrier semantics are sufficient.

**Next step (only if PROMISING/MAYBE):** Extend the existing persistent scaffold to issue the first A/B `LDG.E.128` of tile N+1 into a second register bank before the C-store of tile N completes, and benchmark against Large on the 14 shapes we currently lose.

### #239 — MAYBE · effort M · risk LOW
*Section: Multi-Tile / Heterogeneous*

**One-line idea:** Dispatch two compute pipelines in the same command buffer — a 128x128 BK=8 big-tile kernel covering the M/N-aligned bulk region, and a small-tile (e.g. 64x64 or 32x32) cleanup kernel covering the ragged M/N tails — letting the driver schedule them concurrently on the queue.

**Rationale:** This is plain vanilla Vulkan: two `vkCmdBindPipeline` + `vkCmdDispatch` calls into the same command buffer with disjoint output tile regions, no barrier between them since they write disjoint memory; fully supported on Ampere. For aligned shapes (multiples of 128) it does nothing, but for the 12/26 showcase shapes we currently *lose*, many likely have tails where the big tile wastes work on masked-out lanes — a dedicated small-tile pass on the strip plus a smaller-strict big-tile covering the aligned interior could recover real % of peak on odd shapes. It does not touch our LDG/LDS/FFMA ceiling on the bulk path so it won't lift the 60-68% number on square shapes, but it's a cheap, low-risk win on the "loss" column. KERNEL_SPECS infra already supports multiple variants so the plumbing is mostly selector + dispatch math.

**Next step (only if PROMISING/MAYBE):** Identify the showcase shapes where we currently lose to cuBLAS, check which have non-multiple-of-128 M or N, and prototype a two-dispatch path that runs Large on the floor-aligned interior and a 32x128 / 128x32 edge kernel on the strip, measuring the delta on those specific shapes only.

### #242 — MAYBE · effort M · risk LOW
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Pre-compile ~256 SPIR-V variants where (M,N,K) are baked as compile-time constants (no push-constant loads, fully unrolled K-tail, optimal BM/BN/BK per shape) and dispatch via a hash lookup at call time.

**Rationale:** Technically trivial in Vulkan — we already have 27 KERNEL_SPECS variants and a runtime selector, so extending to per-shape specialization constants or per-shape SPIR-V blobs is a straightforward infra change with near-zero correctness risk. However, our actual bottleneck is LDG/LDS throughput and FFMA pipeline saturation inside the inner tile loop, not dispatch overhead or push-constant cost; baking M/N/K as constants mostly removes a handful of integer ops and lets the K-tail unroll cleanly, which buys maybe 1-3% on odd shapes and ~0% on the already-tuned 1024^3/2048^3 sweet spots. Worth doing for the tail/non-power-of-2 shapes where we currently lose to cuBLAS (12/26 losses), but it won't break the 68% ceiling. Specialization constants are the cheap path; full per-shape SPIR-V is overkill until we have a real codegen story.

**Next step (only if PROMISING/MAYBE):** Add Vulkan specialization constants for M/N/K (and K-tail count) to the existing bk8_bda kernel and measure on the 12 shapes we currently lose, before committing to a 256-blob lookup table.

### #244 — MAYBE · effort M · risk LOW
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Generate per-shape SPIR-V variants where K (and ideally M/N tile counts) are compile-time constants via specialization constants or codegen, so the K-loop fully unrolls and the SPIR-V/driver optimizer can constant-fold addressing and schedule FFMAs more aggressively.

**Rationale:** Vulkan already supports this cleanly via SpecializationInfo (K as spec-const) or offline SPIR-V codegen per shape, and the KERNEL_SPECS table is the right place to plug shape-specialized entries. However, our current inner-K loop is already BK=8 — small and likely unrolled by the driver compiler — so the headline "eliminate loop overhead" win is probably tiny; the real upside is letting the optimizer fold K/strides into addressing math and possibly enabling better LDG.128 scheduling for fixed iteration counts. Given we're LDG/LDS/FFMA-bound near the 60-68% ceiling, this is unlikely to break the ceiling but could give a few % on hot showcase shapes (e.g. 1024^3, 2048^3) and is cheap to try.

**Next step (only if PROMISING/MAYBE):** Add a spec-constant `K_TILES` to the best `bk16_bda_v4` shader, register one shape-specialized variant in KERNEL_SPECS for 1024^3 and 2048^3, and benchmark vs the generic variant to see if the FFMA pipeline tightens.

### #246 — MAYBE · effort M · risk LOW
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Promote lda/ldb/ldc from push-constants to Vulkan specialization constants so the driver folds them into immediates in the compiled SASS, saving one register and the per-iteration IMAD on stride math.

**Rationale:** Vulkan spec constants are the canonical mechanism for exactly this and the driver re-lowers SPIR-V at pipeline creation, so it is trivially possible. But tensor-ash already uses BDA buffer_reference pointers that advance by `+= BK` in the hot loop, so the inner FFMA loop barely touches lda/ldb; savings are confined to outer-tile address setup and the C-store epilogue (a handful of IMADs and maybe 1 register). Given we are bottlenecked by LDG/LDS throughput and FFMA issue, not the IMAD pipe (separate Ampere dispatch port), the speedup is likely <1% and risks pipeline-cache bloat across the 27 KERNEL_SPECS x N shapes. Worth a cheap experiment because the implementation is mechanical and risk is low, but not a high-conviction win.

**Next step (only if PROMISING/MAYBE):** Add `layout(constant_id=...)` spec constants for lda/ldb/ldc in bk16_bda_v4.comp, bake them at pipeline-creation time for one showcase shape (e.g. 4096^3), and diff Nsight SASS + runtime against the push-constant baseline.

### #269 — MAYBE · effort M · risk LOW
*Section: Caching / Layout / Lazy*

**One-line idea:** Pre-pack A once into the BM=128/BK=8 register-tile-friendly swizzled VRAM layout (e.g. block-row-major with interleaved K-strips matched to LDG.128 + uvec4 LDS layout) and reuse that packed buffer across many SGEMM calls when A is reused (LLM weights, repeated K-pass).

**Rationale:** We are LDG/LDS-bandwidth bound at ~60-68% peak, and the inner loop already assumes 128-bit aligned strided loads from A; if A is reused (typical for weight matrices in inference), amortizing a one-time repack into a layout that guarantees coalesced LDG.128 with zero swizzle math and bank-conflict-free LDS stores would cut a few address-math FFMAs and shave LDG replays. But our current kernel already hits buffer_reference LDG.128 cleanly on row-major A, so the win is marginal (maybe 1-3%) unless paired with a layout that also removes the A-side LDS round-trip (e.g. direct register staging) — which is a bigger change. Vulkan-wise it is trivially possible: just an extra "pack" compute pipeline + a cached VkBuffer keyed by (A pointer, shape, layout-version), with the SGEMM kernel selected via KERNEL_SPECS when the caller opts in.

**Next step (only if PROMISING/MAYBE):** Add an opt-in `PackedTensor` handle (cached VkBuffer + layout tag) plus a `pack_a` compute shader that writes A in (BM-block, K-strip, lane-interleaved) order, and a sibling SGEMM variant whose A-loads skip swizzle math, then microbench on a repeated-A workload (e.g. 4096x4096 @ batch 8) to measure delta vs current bk8_bda.

### #275 — MAYBE · effort M · risk MED
*Section: Batched / Cross-call Fusion*

**One-line idea:** Dispatch one persistent compute pipeline whose workgroups loop over a batch of (A_i, B_i, C_i) BDA pointer triples internally, amortising vkQueueSubmit/command-buffer/pipeline-barrier overhead across many small SGEMMs instead of one dispatch per matmul.

**Rationale:** Technically straightforward in Vulkan: push a BDA array (or SSBO) of batch descriptors plus a count, and have each workgroup do `for (b=0; b<B; ++b) { compute tile of C_b; }` — no host roundtrip, no extra extensions needed. However, our current 128x128x8 tile already hits 60-68% peak on showcase shapes where a *single* dispatch saturates 46 SMs, so kernel-launch overhead is not the bottleneck there; the win only materialises for batches of *small* matmuls (e.g. attention-head GEMMs, <=512 dim) where each call under-fills the GPU and ~10-50us of submit overhead dominates. Risk is mostly correctness around per-batch tile-grid remapping and atomic split-K interaction; perf risk is that for the showcase shapes it changes nothing and just adds a code path. Worth doing if/when we target transformer inference (many small GEMMs per layer), not as a path to break the 68% ceiling on big GEMMs.

**Next step (only if PROMISING/MAYBE):** Add a `bk8_bda_batched` variant taking a `BatchDesc[]` BDA buffer and benchmark on a synthetic batch of 32x 256x256x256 FP32 SGEMMs vs 32 separate dispatches to quantify the launch-overhead win before wiring it into KERNEL_SPECS.

### #276 — MAYBE · effort M · risk MED
*Section: Batched / Cross-call Fusion*

**One-line idea:** Sort a batched-GEMM workload by (M,N,K) shape and dispatch a persistent kernel that consumes work-items in sorted order so concurrently-running SMs/warps all execute the same tile recipe and reuse cached push-constant/spec-constant parameters.

**Rationale:** For *batched* SGEMM this is a real and well-known trick — mixing very different shapes in one dispatch causes warp-level divergence on the tile-selection branch and TLB/L2 thrash from interleaved BDA pointers, so sorting by shape and then running a persistent producer/consumer (work-queue in an SSBO with atomicAdd index) is technically possible in Vulkan compute (we already have BDA + atomic SSBOs; persistent scaffolding already exists per the project memory). However, tensor-ash today is a *single-GEMM* showcase library bottlenecked by LDG/LDS/FFMA throughput at 60-68% of peak on one shape at a time — sorting helps the *batched* multi-shape case, not the single-large-GEMM case that defines the current ceiling. It is a useful feature for a future batched-API surface, but it will not lift the single-kernel peak.

**Next step (only if PROMISING/MAYBE):** Defer until a batched-GEMM entry point exists; then add a host-side shape-bucket sort that groups identical (M,N,K) into per-bucket dispatches of the existing best kernel before bothering with a persistent cross-shape work-queue.

### #277 — MAYBE · effort M · risk LOW
*Section: Batched / Cross-call Fusion*

**One-line idea:** When two queued SGEMM calls share the same A matrix, concatenate their B/C buffers logically and launch one BM=128 BN=128 kernel over the widened N dimension instead of two separate dispatches.

**Rationale:** Fusing A-sharing calls amortises A's LDG traffic and shared-memory loads across more BN tiles, which directly attacks our LDG/LDS bottleneck — each A tile already gets reused N/BN times, and doubling N effectively doubles that arithmetic-intensity-relevant reuse for the A side while also killing one launch's dispatch + barrier overhead. The catch is that real workloads rarely present two back-to-back GEMMs with bit-identical A in our current benchmark set; the win is conditional on a batching API existing above the kernel, so this is more about plumbing than kernel work. Implementation is mostly Rust-side: a small fusion pass in the dispatcher that detects identical A pointer + compatible (M, K) and routes to existing Large kernel with concatenated B/C strides — no shader changes needed.

**Next step (only if PROMISING/MAYBE):** Add a `gemm_batch(&[Call])` entry point that groups calls by A buffer-handle + (M,K) and emits a single Large-kernel dispatch with concatenated B/C; measure on a synthetic 2x (4096,4096,4096) shared-A workload to see if launch+L2 savings actually materialise.

### #305 — MAYBE · effort M · risk LOW
*Section: Misc / Out-There*

**One-line idea:** Add an asymmetric BM=128 BN=256 (and/or BM=256 BN=128) tile variant to KERNEL_SPECS and pick it by runtime shape so the longer-reuse axis gets the larger tile.

**Rationale:** Our selector already dispatches multiple tile shapes, so a 128x256 (or 256x128) TM=8 TN=8 variant is a natural extension and is cheap to add to the data-driven table. The risk is that 128x256 BK=8 needs ~12KB more LDS for B and may drop occupancy to 1 block/SM on GA104 (98KB shared per SM, 64K regs); the IDEAS dump already notes 128x256 BK=16 was a dead end due to bank conflicts and occupancy, but BK=8 with uvec4 LDS may still be viable for tall-skinny or short-fat shapes where the symmetric 128x128 tile under-utilizes one axis. Since we are LDG/LDS-bound and ~60-68% of peak, increasing reuse on the dominant axis is exactly the right lever for non-square shapes, and the selector overhead is zero at runtime. The honest expected win is modest (a few percent on specific aspect ratios), not a breakthrough on the showcase geomean.

**Next step (only if PROMISING/MAYBE):** Clone the BM128 BN128 BK8 kernel into a BM128 BN256 BK8 variant (and mirror 256x128), register both in KERNEL_SPECS, and extend the shape selector to pick the asymmetric tile when M/N aspect ratio exceeds ~2x.

### #308 — MAYBE · effort M · risk MED
*Section: Misc / Out-There*

**One-line idea:** When the host queues back-to-back SGEMM dispatches that write disjoint output buffers, omit the inter-dispatch `vkCmdPipelineBarrier`/memory barrier so the GPU can overlap tail of dispatch N with head of dispatch N+1.

**Rationale:** Vulkan compute lets you legitimately skip barriers between dispatches that have no RAW/WAW/WAR hazard, and Ampere's async compute / multi-dispatch overlap can hide tail effects on small or skinny shapes where one CTA wave doesn't fill all 46 SMs. However, our showcase is dominated by single large square GEMMs that already saturate the SMs, so overlap mainly helps the small/skinny tail of the benchmark suite (and multi-GEMM workloads we don't yet target). It's pure host-side scheduling work — no shader changes — so the main risks are correctness regressions if the dependency analyzer is wrong and that the perf win shows up only on a subset of shapes.

**Next step (only if PROMISING/MAYBE):** Add an opt-in `enqueue_batch(&[GemmDesc])` API that walks the list, computes buffer-disjointness on outputs, and emits a single command buffer with barriers only between conflicting pairs; A/B against the current one-barrier-per-dispatch path on a small+skinny shape mix.

### #311 — MAYBE · effort M · risk LOW
*Section: Misc / Out-There*

**One-line idea:** Detect when B (or A) is effectively rank-1 (N==1 or K==1 outer product) and dispatch a dedicated SGEMV / outer-product kernel instead of the BM128xBN128xBK8 GEMM tile.

**Rationale:** When N==1 the GEMM tile wastes 127/128 of its BN lane and is bandwidth-bound on A; a proper SGEMV with warp-reduce along K easily beats it and cuBLAS routes to cublasSgemv anyway, so the "cuBLAS doesn't do this" framing only holds for the true rank-1 outer-product case (K==1, M and N large) — that case is just a scaled column-broadcast (C += alpha * a[:] * b[:]^T) and is memory-bound at ~448 GB/s, where our generic kernel reads B redundantly across BK and pays atomic/loop overhead. Adding shape-based fast paths is a clean win for those slices of our 26-shape showcase without touching the hot kernel, but it doesn't move the 60-68% peak ceiling on the compute-bound shapes that define our headline geomean. Risk is low because dispatch is already shape-driven via KERNEL_SPECS and the fast paths are independent shaders that fall back to the existing kernel on mismatch.

**Next step (only if PROMISING/MAYBE):** Add two specialized shaders (sgemv_n1 and outer_product_k1) plus selector rules in KERNEL_SPECS gated on N==1 / K==1, and benchmark against current Large on synthetic rank-1 shapes.

### #208 — MAYBE · effort L · risk MED
*Section: Fast-Matmul Recursion (Strassen / Winograd / Pan)*

**One-line idea:** Apply a single Strassen level at the host/dispatch layer — recurse C = A*B (N>=2048) into seven (N/2)^3 sub-GEMMs that call our existing BM=128 BN=128 BK=8 kernel, plus 18 elementwise add/sub dispatches for M1..M7 inputs and C-quadrant assembly.

**Rationale:** The 8->7 multiply reduction is a real ~14% FLOP saving, which on a kernel that is FFMA-issue/LDS-bound (not pure DRAM-bound) translates fairly directly into wall-clock — exactly the lever we need to break the 60-68% peak ceiling. However, sub-GEMMs run at (N/2)^3 where our kernel is slightly less efficient (fewer 128x128 tiles, lower SM occupancy on 46 SMs), the seven M_i temporaries plus (A11+A22)/(B11+B22) sums add ~14*(N/2)^2 floats of extra DRAM traffic that eats into the 14% savings, and Strassen FP32 loses ~1 bit of precision per level which may break our cuBLAS-tolerance showcase tests. The fact that Strassen scaffolding already landed but does not yet beat Large kernel is a concrete warning that the add/sub overhead and sub-tile efficiency loss are non-trivial in practice.

**Next step (only if PROMISING/MAYBE):** Profile the existing Strassen scaffolding at N=2048/4096 with NSight to attribute time between the 7 sub-GEMMs and the 18 add/sub kernels, then fuse the elementwise ops into producer/consumer kernels (write A11+A22 straight into the M1 A-operand via BDA) before judging whether the 14% FLOP win survives the bookkeeping.

### #212 — MAYBE · effort L · risk HIGH
*Section: Fast-Matmul Recursion (Strassen / Winograd / Pan)*

**One-line idea:** Choose Strassen recursion depth `d` at dispatch time so each leaf GEMM lands near our BM=128/BN=128 kernel's sweet spot (~1024^3 where we already beat cuBLAS), i.e. `d = floor(log2(N / 1024))` with per-axis variants for non-square shapes.

**Rationale:** Adaptive depth is the right framing — fixed-depth Strassen on top of our Large kernel currently lands only as scaffolding and depth=1 on small problems wastes the 7/8 FLOP saving against scratch+combine overhead, while too-deep recursion drops leaves below our tile sweet spot. The asymptotic 7/8 advantage only materializes when (a) every leaf hits ~60-68% peak (true at 1024^3, false at 256^3) and (b) the 18 add/sub passes don't drown the gain — on Ampere with 936 GB/s HBM and our LDG.128-bound regime, scratch traffic for `M1..M7` plus combines is ~3-4x extra memory per level, so net win requires N >= ~2048 with d=1 and probably N >= ~4096 for d=2. The hard part is not depth selection (trivial) but the FP32 accuracy regression and the scratch allocator / dispatch DAG; depth-as-function-of-shape is a 5-line policy on top of work we have to do anyway.

**Next step (only if PROMISING/MAYBE):** Land a working depth=1 Strassen first (currently only scaffolding), measure crossover N where 7-leaf+18-combine actually beats single Large dispatch on 3070, then encode `d(M,N,K)` as a lookup in the KERNEL_SPECS selector.

### #243 — MAYBE · effort L · risk MED
*Section: Shape-Specialised SPIR-V*

**One-line idea:** Use rspirv to emit shape-specialised SPIR-V SGEMM kernels at first call (BM/BN/BK/TM/TN, unrolled K-tail, baked M/N/K constants, baked strides), cached to disk by shape hash, mirroring cuBLAS's heuristic-fitted kernel zoo.

**Rationale:** Technically possible: rspirv can emit valid SPIR-V, and Vulkan pipeline cache + on-disk blob already supports this pattern; specialization constants give ~80% of the same win for far less code. Our perf ceiling is LDG/LDS throughput and FFMA pipeline, not branch/loop overhead, so the marginal gain over an expanded KERNEL_SPECS table + spec constants is small — mostly K-tail unrolling and removing bounds checks on odd shapes (helps non-multiple-of-128 tails, padding-avoiding kernels). The infra cost is high: rspirv IR builder, register allocator awareness, validation, on-disk cache invalidation across driver versions, and first-call latency spike. Worth it only if we commit to a long-tail shape strategy (LLM inference with weird N/K), not for the 27-variant showcase.

**Next step (only if PROMISING/MAYBE):** Before touching rspirv, exhaust Vulkan specialization constants for BM/BN/BK/TM/TN/M/N/K + unroll hints and measure delta vs current best on 3-5 awkward shapes; only escalate to runtime SPIR-V codegen if spec-constants leave >3% on the table.

### #248 — MAYBE · effort L · risk MED
*Section: Recursive / Hierarchical / Polyhedral*

**One-line idea:** Introduce an explicit 512x512 L2-resident super-tile around the existing 128x128 shared-mem block / 8x8 register block, and jointly autotune the three levels (outer L2, middle SMEM, inner register).

**Rationale:** GA104 has only ~4MB L2 and 46 SMs; an explicit outer 512x512 super-tile would be implemented via workgroup swizzling / dispatch reordering (Z-order or block-row scheduling) so concurrently-running workgroups share L2 lines for A and B — this is a real and known win for SGEMM (cuBLAS does L2 swizzling), and we have not landed a deliberate L2-aware schedule yet (Z-swizzle attempted only on top of the HW scheduler, not as a coarse super-tile remap). However, we are already 60–68% of peak and the dominant bottleneck is LDG/LDS+FFMA inside the 128x128 tile, not DRAM bandwidth; L2 reuse mainly helps very large M,N (e.g. 4096+) and for moderate sizes the win is typically only a few percent. Joint 3-level autotuning is mostly infrastructure work (extend KERNEL_SPECS + dispatch-time swizzle) rather than a new kernel, so risk is bounded but payoff is incremental.

**Next step (only if PROMISING/MAYBE):** Prototype a dispatch-side workgroup-ID remap that groups 4x4 = 16 workgroups (covering 512x512) into L2-friendly tiles and benchmark on 2048^2 / 4096^2 against the current Large kernel before touching SMEM/register dims.

---

## IDEAS_distributed.md — Distributed Systems Patterns Adapted to Vulkan SGEMM

_1 promising, 1 maybe_

### #967 — **PROMISING** · effort L · risk MED
*Section: Realistic Wins from Distributed Patterns*

**One-line idea:** Implement a real Stream-K SGEMM that partitions the MAC loop (not just M/N tiles) across a fixed number of persistent workgroups, with atomic-counter-based work stealing and partial-sum reduction, so irregular shapes saturate all 46 SMs instead of leaving a quantization tail.

**Rationale:** Our 128x128x8 tile leaves big wave-quantization gaps on shapes that are not multiples of 128 in M and N (e.g., the 12/26 cases we currently lose vs cuBLAS); Stream-K is the canonical fix and has documented 5-15pp wins on exactly these irregular sizes, which matches our showcase loss pattern. It is fully feasible in Vulkan compute: we already have the persistent-thread scaffold, BDA push constants, and atomic-float split-K reduction primitives landed (just not winning yet), so Stream-K is essentially "split-K where the split is per-tile and dynamic" plus an atomic work-counter — all standard SPIR-V (`atomicAdd` on a uint counter, plus our existing FP32 atomic reduce for the fix-up tiles). It does not fight our LDG/LDS/FFMA ceiling on already-saturated shapes, but it directly attacks the throughput we leave on the floor on tail shapes, which is the realistic remaining headroom given the 60-68% peak wall.

**Next step (only if PROMISING/MAYBE):** Prototype a `bk8_streamk` variant that reuses the persistent-workgroup launch, adds a global `atomicAdd` MAC-iteration counter, and routes partial-sum tiles through the existing split-K atomic-float reduce path, then benchmark on the 12 irregular-shape losers in the showcase set.

### #968 — MAYBE · effort M · risk LOW
*Section: Realistic Wins from Distributed Patterns*

**One-line idea:** For split-K, give each K-slice workgroup its own dedicated output slot in a `[splitK, M, N]` scratch buffer (no atomics), then launch a second tiny reduction kernel that sums the splitK slots into the final C tile.

**Rationale:** This is the standard "deterministic split-K" pattern (cuBLAS/CUTLASS do it) and trivially possible in Vulkan — just a bigger storage buffer plus a second pipeline dispatch. Our current split-K uses atomic float reduce which serializes contended C tiles and is non-deterministic; CRDT-style per-WG slots remove the contention and make results bit-reproducible. The needle-move is real only on shapes where split-K is already chosen (small M*N, large K — the cases where atomic contention dominates), and our 60-68% ceiling on big GEMMs is untouched; expect single-digit % wins on the small-N tail of the showcase, not on the geomean.

**Next step (only if PROMISING/MAYBE):** Add a `splitk_slotted` kernel variant that writes to a `splitK*M*N` scratch buffer with no atomics plus a 1-WG vectorized reduction pass, gate it behind KERNEL_SPECS for shapes where current atomic split-K is selected, and A/B against the existing atomic path.

---

## Biological / Neuromorphic / Evolutionary / Swarm Ideas for FP32 SGEMM on RTX 3070

_0 promising, 8 maybe_

### #426 — MAYBE · effort S · risk LOW
*Section: Immune System*

**One-line idea:** Maintain a persisted blacklist of (kernel_variant, shape-bucket) pairs that historically regressed vs the current best, so the runtime selector and the auto-tuner skip them even when heuristics rank them highly.

**Rationale:** We already have a 27-entry KERNEL_SPECS table with a shape-based selector, and the dead-ends list in the project memory is literally an informal version of this idea. Formalising it as a small on-disk JSON/TOML "regression memory" keyed by (M,N,K bucket, variant id, driver+GPU id) would prevent the selector and any future evolutionary/auto-tuning loop from re-picking known-bad tiles like 128x256 BK=16 or 3-stage prefetch on 1024^3. It will not raise the 60-68% peak ceiling by one FLOP on its own, but it cheaply protects geomean wins and de-risks every subsequent brainstorm experiment, which is the actual leverage.

**Next step (only if PROMISING/MAYBE):** Add a `kernel_blacklist.toml` consumed by the selector plus a `bench --record-regression` flag that appends (shape-bucket, variant, baseline_gflops, observed_gflops, delta) whenever a candidate underperforms the current best by >2%.

### #395 — MAYBE · effort M · risk LOW
*Section: Evolutionary / Genetic*

**One-line idea:** Encode (BM,BN,BK,TM,TN,num_warps,prefetch_depth) as a 7-gene chromosome and run a small GA on the host that mutates these parameters, recompiles SPIR-V (or selects from a pre-baked set), benchmarks 5 candidates per generation on a calibration shape, and keeps the top 2 to evolve KERNEL_SPECS entries.

**Rationale:** Technically trivial in Vulkan compute - parameters become specialization constants (or per-variant pipelines) and benchmarking is host-side, no GPU autonomy needed; the existing KERNEL_SPECS table + shape selector is essentially the substrate a GA would search. However, the search space here is tiny and already well-trodden: BM/BN/BK/TM/TN have known sweet spots on Ampere (128x128x8 with TM=TN=8), occupancy is dictated by register/shared budgets that mostly admit a handful of valid points, and our 60-68% ceiling is bottlenecked by LDG/LDS/FFMA pipeline behaviour that no tile-shape mutation can fix. A GA is fancy random search over ~hundreds of feasible chromosomes and would mostly rediscover bk16_bda_v4; the real wins come from new kernel structures (cp.async equivalent, async copies, tensor-core TF32 fallback), not new tile sizes. Worth doing as a one-off offline autotuner to fill in non-128x128 shapes in the selector, not as a runtime/idle-dispatch system.

**Next step (only if PROMISING/MAYBE):** Build a small offline host-side autotuner that enumerates a constrained chromosome grid (validity-filtered by register + shared-mem budget), benchmarks each on ~10 representative shapes, and emits an updated KERNEL_SPECS table - skip the "mutate during idle dispatch" runtime part.

### #425 — MAYBE · effort M · risk LOW
*Section: Immune System*

**One-line idea:** When the runtime selector sees a recurring (M,N,K) shape, JIT-spawn several specialized SPIR-V clones of the winning kernel (varying warp count, swizzle, BK, vec layout), benchmark them online, and let the fastest clone proliferate in the persistent VkPipelineCache.

**Rationale:** This is essentially online autotuning dressed up as immune-system metaphor; it's fully possible in Vulkan since SPIR-V specialization constants + VkPipelineCache already let us cheaply materialize variants without recompilation pain, and KERNEL_SPECS is the natural seedbed. However, our bottleneck is LDG/LDS throughput and the ~60-68% FFMA ceiling, not selector quality - we already pick the right of 27 hand-tuned variants per shape, so the upside is only the delta between hand-tuned and shape-specialized (probably 2-5% on a handful of shapes, zero on cold/varied workloads). Risk is low because losing clones just get evicted, but it adds infra complexity and an online benchmark warmup tax that hurts the "first call" latency story.

**Next step (only if PROMISING/MAYBE):** Prototype a small "clone pool" around one shape where we currently lose to cuBLAS (e.g. one of the 12/26 losses), generating 4-8 spec-constant variants of bk16_bda_v4 and measuring whether any clone beats the hand-tuned pick by >3%.

### #433 — MAYBE · effort M · risk LOW
*Section: Predator-Prey / Population Dynamics*

**One-line idea:** Treat each new kernel variant as an "invasive species" routed a small fraction (e.g. 5-10%) of dispatches per shape bucket, track rolling GFLOP/s, and atomically replace the incumbent in KERNEL_SPECS when it dominates by a confidence margin; otherwise evict it.

**Rationale:** This is purely an infrastructure/selector-policy change on top of the existing data-driven KERNEL_SPECS + shape-based runtime selector, so it is trivially possible in Vulkan/Rust without touching SPIR-V. It cannot lift the ~60-68% peak ceiling — it does not address LDG/LDS or FFMA pipeline bottlenecks — but it could give a modest geomean win by auto-promoting better variants per shape bucket and would let us safely A/B-test the half-finished Strassen/persistent/split-K scaffolding currently sitting unused. Risk is LOW because the incumbent always handles the majority of traffic and a guard band prevents noisy regressions from sticking.

**Next step (only if PROMISING/MAYBE):** Add a thin `KernelTournament` layer over the selector that logs per-(shape-bucket, variant) rolling GFLOP/s into an atomic table and swaps the incumbent when a challenger beats it by >2% over N>=50 dispatches.

### #435 — MAYBE · effort M · risk LOW
*Section: Photosynthesis / Energy Harvesting*

**One-line idea:** Use SPIR-V specialization constants (BM/BN/BK/TM/TN, unroll factors, prefetch toggles) baked at pipeline-creation time so each KERNEL_SPECS entry compiles a kernel "tuned" to a workload spectrum (small/medium/large shapes), rather than relying solely on `#define` variants or runtime push constants.

**Rationale:** Specialization constants are fully supported in Vulkan/SPIR-V and let the driver constant-fold + unroll inner loops at pipeline-compile time, which is strictly better than push-constant-driven branching and avoids the source-explosion of 27 hand-rolled variants. However, since we already maintain a data-driven KERNEL_SPECS table with shape-based selection and per-variant compiled SPIR-V, the practical perf delta vs. our current macro-specialized shaders is small — we're bottlenecked by LDG/LDS throughput and FFMA, not by branch overhead. The real win would be code-base simplification (one .comp source, N specializations) and faster iteration on tile sweeps, not raw TFLOPS. Low risk because spec constants are well-understood and we can A/B against the macro variants.

**Next step (only if PROMISING/MAYBE):** Convert the bk16_bda variant to use spec constants for {BM,BN,BK,TM,TN,USE_DOUBLE_BUFFER} and benchmark against the macro-defined twin to confirm zero regression, then collapse the KERNEL_SPECS table.

### #443 — MAYBE · effort M · risk LOW
*Section: Symbiosis / Mutualism*

**One-line idea:** Co-dispatch two specialized kernels (a small/skinny-tile kernel and the BM=128 BN=128 large-tile kernel) in the same submission, with a CPU-side router that partitions the output grid by tile aspect ratio so each kernel only processes the tile regions it handles efficiently.

**Rationale:** We already have a runtime shape-based selector and 27 KERNEL_SPECS variants, so picking *one* kernel per matmul is solved; the symbiosis twist is splitting a *single* matmul into disjoint output-tile regions handled by different kernels in one submit. For pure square shapes this buys nothing (the large tile already wins), but for irregular shapes (e.g. M=1024, N=192, K=4096) where the BM=128 BN=128 tile wastes 33% of N-tiles on padding, a skinny kernel covering the edge column-strip while the large kernel handles the interior could plausibly recover the wasted FFMA. Risk is low because each kernel writes a disjoint C-region (no atomics, unlike split-K), and Vulkan trivially supports two dispatches in one command buffer. The catch: our ~60-68% peak ceiling on the showcase is set by LDG/LDS/FFMA in the *interior*, not by edge waste, so this likely helps the 12/26 *losses* (odd shapes) more than the geomean.

**Next step (only if PROMISING/MAYBE):** Profile the 12 current cuBLAS losses to confirm they are dominated by edge/tail tiles with poor BM=128 BN=128 fit, then prototype a host-side region splitter that issues large-kernel dispatch over the aligned interior and a skinny-kernel (BM=32 BN=32 or BM=64 BN=16) dispatch over the residual strips in one command buffer.

### #472 — MAYBE · effort M · risk LOW
*Section: Wildcards / Misc Biology*

**One-line idea:** Have the library log recently-seen (M,N,K) shapes and, when the GPU is idle, replay them through several KERNEL_SPECS variants to update a persistent shape->best-variant cache that the runtime selector then uses.

**Rationale:** Our selector is already a static shape-based dispatch over 27 KERNEL_SPECS, so wiring in an idle-time autotuner that benchmarks candidates per observed shape and persists winners to disk is straightforward and very plausibly beats hand-tuned heuristics on awkward shapes (skinny, non-power-of-2, the 12/26 cases where we currently lose to cuBLAS). It does nothing for the LDG/LDS/FFMA ceiling, so it cannot raise our 60-68% peak number on shapes the current best kernel already wins; the gain is purely in picking a less-bad variant for outlier shapes. Risk is low because it only changes selection, not kernel code, and a sane fallback to the static table keeps correctness intact. Idle detection in Vulkan is trivial (no submitted work + a timer), and replay is just resubmitting existing pipelines.

**Next step (only if PROMISING/MAYBE):** Add a background autotune task that, on idle, runs the top-3 KERNEL_SPECS candidates against the last N unique shapes and writes a JSON shape->variant cache loaded by the selector at startup.

### #478 — MAYBE · effort M · risk LOW
*Section: Wildcards / Misc Biology*

**One-line idea:** At dispatch time, sample two live "brightness" signals (recent LDG/LDS bytes-per-FFMA and SM occupancy) and bend the KERNEL_SPECS selector toward whichever bottleneck is brighter, picking a more compute-heavy or more bandwidth-heavy tile variant per shape.

**Rationale:** This is just a dressed-up adaptive heuristic on top of our existing data-driven KERNEL_SPECS selector, which today is purely shape-based; adding a two-axis cost model (estimated arithmetic intensity vs. estimated HBM bytes/tile) and picking the variant whose ratio best matches the GA104 roofline is trivially possible in host Rust and costs no kernel work. It won't break the 60-68% ceiling on shapes we already tune well, but it could meaningfully help the 12/26 shapes where we currently lose to cuBLAS by routing tall-skinny / fat-short cases to split-K or smaller-BN variants automatically. Risk is low because it's selector-only and we keep the current table as the fallback; the real cost is building a tiny per-shape benchmark cache or a closed-form cost model.

**Next step (only if PROMISING/MAYBE):** Add a two-term cost function `score(spec, M,N,K) = a*compute_intensity_gap + b*bandwidth_gap` to the selector and offline-fit (a,b) against the existing 27-variant sweep results to see if it picks better variants than the current shape rules.

---

## Cryptography / Number Theory / Ring Algebra Ideas for Vulkan FP32 SGEMM

_0 promising, 1 maybe_

### #537 — MAYBE · effort S · risk LOW
*Section: Best-bet shortlist for actual implementation*

**One-line idea:** Add a debug-flag Freivalds verifier that picks a random vector r, computes A(Br) vs (AB)r in O(N^2) work, and asserts equality within an FP32 tolerance to cheaply catch correctness regressions across all 27 KERNEL_SPECS variants.

**Rationale:** Freivalds is a correctness/debug tool, not a perf lever — it does nothing for the LDG/LDS/FFMA ceiling we are actually bottlenecked by, so it cannot move the needle on the 60-68% peak target. However, given the proliferation of experimental kernels (Strassen scaffolding, split-K atomic reduce, persistent-threads, BDA_V4) where full N^3 reference checks get expensive at 4096+, an O(N^2) probabilistic check behind a `--verify` flag is genuinely useful insurance and effectively free to implement. Risk is LOW since it's gated behind a debug flag and never touches the hot path; the only subtlety is choosing an FP32-aware tolerance (relative epsilon scaled by K) to avoid false positives from non-associative float reduction order differences between kernels.

**Next step (only if PROMISING/MAYBE):** Add a `verify_freivalds(a, b, c, k_iters=3, rel_tol=1e-4*K)` helper in the test harness that runs on CPU after each kernel dispatch when `TENSOR_ASH_VERIFY=1` is set, and wire it into the existing benchmark runner.

---

## Hardware-Abuse / Vulkan-Extension / GPU-Internals Brainstorm

_0 promising, 19 maybe_

### #105 — MAYBE · effort S · risk LOW
*Section: B. Subgroup / Warp-Level Hacks*

**One-line idea:** Use `VK_EXT_subgroup_size_control` plus `requiredSubgroupSize=32` (and the `SUBGROUP_FULL_GROUPS` + `ALLOW_VARYING_SUBGROUP_SIZE` flags) on our SGEMM pipeline to pin the warp width to 32 so any subgroup-shuffle paths (e.g. B-broadcast, split-K reduce) operate without falling back to shared memory.

**Rationale:** On NVIDIA the subgroup size is always 32 on Ampere regardless — the driver does not silently downgrade to 16 on consumer GA104 — so this "fix" addresses a non-problem for our target HW (it would matter on Intel/AMD RDNA where sizes vary). However, the extension is cheap to wire in (one `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` and a feature query), it is a prerequisite for any future warp-shuffle-based kernel (B broadcast via `subgroupShuffle`, cross-lane epilogue reduce, split-K), and explicit declaration removes a class of "what if a future driver chooses 16-wide" portability footguns. By itself it will not move the 60-68% peak needle because we are LDG/LDS/FFMA bound, not shuffle-bound, and our current kernel does not even use subgroup ops.

**Next step (only if PROMISING/MAYBE):** Add a thin helper that queries `subgroupSizeControl` + size range and sets `requiredSubgroupSize=32` on pipeline creation, gated behind a feature flag, so the upcoming shuffle-based B-broadcast / split-K reduce variants can rely on a fixed warp width.

### #157 — MAYBE · effort S · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Use NVAPI/nvidia-smi to force the RTX 3070 into its max P-state (P0) before running the SGEMM bench, preventing the driver from leaving it in compute-mode P2 with reduced memory clock.

**Rationale:** The P2-vs-P0 issue is real on consumer Ampere: NVIDIA's compute driver does pin GeForce cards at P2 for CUDA/compute workloads, which typically caps memory clock ~1000 MHz below P0 and can suppress core boost. Since our kernel sits at 60-68% of FP32 peak and is partly LDG/LDS-throughput bound, a ~5-10% memclk uplift could lift the absolute TFLOPS bench number (without changing % of peak much, since peak also scales). The catch: this is a measurement/benchmarking hygiene fix, not a kernel optimization, and on Linux NVAPI isn't available - we'd use `nvidia-smi -lgc`/`-lmc` or `nvidia-settings` instead. Worth doing once to verify our perf numbers aren't artificially low vs cuBLAS (which may already trigger P0).

**Next step (only if PROMISING/MAYBE):** Before next benchmark, run `nvidia-smi -q -d PERFORMANCE` mid-bench to confirm P-state, and if P2, lock clocks with `nvidia-smi -lgc <max>` / `-lmc <max>` and re-run the showcase to see if cuBLAS-relative geomean shifts.

### #158 — MAYBE · effort S · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Turn off the display output (`xrandr --output ... --off` or switch to a headless TTY) during SGEMM benchmarking so the X compositor / desktop redraw doesn't steal SM cycles or interleave graphics work with our compute queue on the RTX 3070.

**Rationale:** On a single-GPU Linux desktop the compositor (KWin/GNOME-Shell) does cause periodic graphics submissions that share the GPU with compute, and at our 60-68% peak ceiling even a few percent jitter is meaningful for win-rate counts (14/26 vs cuBLAS is decided by small margins). This costs nothing to implement, has no kernel-code risk, and would tighten benchmark variance — but it won't lift the ceiling, since LDG/LDS/FFMA bottlenecks are intrinsic to the kernel, not compositor contention. Best framed as a benchmark-hygiene fix (more credible numbers, possibly flipping a few borderline shapes) rather than a perf-needle win.

**Next step (only if PROMISING/MAYBE):** Add a bench-mode script that drops to a VT (`chvt`) or runs `xrandr --output <name> --off` before the harness and restores after, then re-run the 26-shape sweep to measure variance reduction and any win-rate delta.

### #193 — MAYBE · effort S · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** In the inner-product loop, emit `OpExtInst GLSL.std.450 Fma` explicitly (via `fma()` in GLSL or hand-edited SPIR-V) instead of relying on glslc's `OpFMul` + `OpFAdd`, so the NVIDIA driver guarantees a single FFMA instruction.

**Rationale:** On the NVIDIA Vulkan driver, mul+add usually gets contracted into FFMA at PTX/SASS lowering, but this isn't guaranteed — `precise` qualifiers, debug builds, or compiler heuristics can break fusion, and our FFMA pipeline is one of the cited bottlenecks at the 60-68% ceiling. Swapping the `c += a*b` accumulator to `c = fma(a, b, c)` is a one-line GLSL change in the BM128/BN128/BK8/TM8/TN8 kernel and trivial to A/B; if the driver was already fusing (likely, given we're hitting ~65% of FFMA peak) it's a no-op, but if any of the 64 inner MACs were splitting into MUL+ADD we'd recoup measurable FLOPs. Risk is essentially zero since `fma()` is IEEE-stricter (single rounding), not looser, than mul+add.

**Next step (only if PROMISING/MAYBE):** In the innermost TM*TN accumulation of the current best kernel, replace `c[i][j] += a_reg[buf][i] * b_reg[buf][j]` with `c[i][j] = fma(a_reg[buf][i], b_reg[buf][j], c[i][j])`, dump SPIR-V to confirm `OpExtInst Fma` is emitted, and benchmark the 27-shape sweep.

### #195 — MAYBE · effort S · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Run a saturating warmup dispatch (e.g. a long FFMA-heavy kernel filling all 46 SMs at ~100% occupancy) for a few hundred ms before each timed SGEMM so Ampere's boost clock is already pinned at the top P-state when measurement begins.

**Rationale:** This is purely a benchmark-methodology / measurement-stability change, not a kernel-perf change — Ampere DVFS does ramp clocks based on recent utilization, so a hot warmup absolutely raises the *measured* TFLOPS of short runs and tightens variance, but it does nothing to relieve our actual bottlenecks (LDG/LDS throughput, FFMA pipeline, 60-68% ceiling). It is worth doing as a measurement hygiene step because some of our 14/26 wins vs cuBLAS could be noise from cold-clock starts, and we'd rather report numbers at a stable P-state. Risk is low: only the harness changes, not the kernel. Caveat: if we warm up *too* hard we may instead hit thermal throttling on a small RTX 3070, so the warmup should be intense-but-short (~100-300 ms) rather than seconds.

**Next step (only if PROMISING/MAYBE):** Add a `--warmup-saturate` mode to the bench harness that dispatches a high-occupancy FFMA loop for ~200 ms before the timed run, and compare geomean + stddev vs the current warmup on the 26-shape showcase.

### #100 — MAYBE · effort M · risk MED
*Section: B. Subgroup / Warp-Level Hacks*

**One-line idea:** Replace the As shared-memory staging for A-tile rows with cooperative LDG into per-lane registers plus `subgroupShuffleXor`/`subgroupBroadcast` to fan out the TM=8 A-fragments to all 32 lanes, eliminating the LDS write/read round-trip on the A side.

**Rationale:** Subgroup shuffles are available on Ampere via `GL_KHR_shader_subgroup_shuffle`/`_shuffle_relative` (subgroupSize=32) and physically use the warp shuffle network, so the primitive is sound. But our 128x128 BM*BN tile with TM=TN=8 needs each lane to see 8 A rows across the K=8 strip — that's 128 A values per BK step shared across the 256-thread workgroup (8 warps), and a single warp only holds 32 lanes' worth, so shuffles only cover intra-warp reuse and we'd still need cross-warp staging (i.e. shared memory) unless we restructure tiling to one-warp-per-output-subtile. On Ampere, LDS bandwidth for our current uvec4 path is already near-peak and not obviously the binding bottleneck (FFMA + LDG are), so the upside is modest; however removing half of the LDS traffic and one barrier per K-step is plausibly worth 2-5% if we can keep occupancy. The dead-end list (3-stage prefetch, source double-buffer) suggests we're scheduler-bound, and shuffles introduce a hard cross-lane dep chain that the compiler may schedule worse than LDS — real risk of regression.

**Next step (only if PROMISING/MAYBE):** Prototype a warp-tile variant (BM=32, BN=128, TM=8, TN=8, one warp per A-row-strip) that loads A via LDG into one lane per row and broadcasts with `subgroupBroadcast(a, k%32)`, keeping B in shared, and benchmark vs bk16_bda at 1024^3 / 2048^3 / 4096^3.

### #101 — MAYBE · effort M · risk MED
*Section: B. Subgroup / Warp-Level Hacks*

**One-line idea:** Keep A-tile in shared memory but eliminate Bs entirely: one thread per subgroup loads each B column fragment from BDA, then uses `subgroupBroadcast`/`subgroupShuffle` to fan it out to the other 31 threads that share that N-coordinate in the 8x8 outer-product step.

**Rationale:** On Ampere `shfl.sync` is ~1-cycle and bypasses the LDS pipeline, so swapping LDS.E.128 reads of `Bs` for warp shuffles is plausible and would free shared memory (halve `Bs`, potentially raising occupancy from 2 to 3 blocks/SM) plus halve LDS bank pressure. However, we already use 128-bit LDS via `uvec4`+`uintBitsToFloat`, and our 60-68% ceiling is mostly FFMA-pipeline / register-file bound, not LDS-bound — Nsight on similar kernels shows LDS at ~40-50% utilization, not saturated. The thread mapping must be redesigned so threads sharing an N-coord live in the same subgroup (currently 16x16 thread grid splits each warp across 2 N-coords, so the broadcast partner set is awkward); `subgroupBroadcast` also requires a dynamically uniform lane id, so we'd need `subgroupShuffle` for the general case, which NVIDIA's SPIR-V supports fine. Net: occupancy + LDS-bw side-effect could nudge us 2-5%, but it's not a slam-dunk and risks breaking the working register double-buffer.

**Next step (only if PROMISING/MAYBE):** Prototype a BM=128 BN=128 BK=8 TM=8 TN=8 variant with thread layout reshaped to 32x8 (warp == one row of N-coords), drop `Bs`, replace B loads with one cooperative LDG.E.128 + `subgroupShuffle` fanout, and benchmark at 1024^3 and 4096^3 against bk16_bda_v4.

### #131 — MAYBE · effort M · risk MED
*Section: D. Memory & Buffer Tricks*

**One-line idea:** Bind matrix A as a `samplerBuffer` (R32F texel buffer) and load A-tiles via `texelFetch` so the loads traverse the TEX/TMU read-only cache instead of the L1 data cache used by SSBO/BDA loads, effectively doubling the usable on-SM cache footprint.

**Rationale:** On Ampere, L1 and the texture cache share the same unified L1/SMEM SRAM partition by default, so the "separate physical cache" premise is largely false on GA104 — the win is mostly that TEX loads go through a read-only path with different coalescing/replay behavior and can free pressure on the LSU and L1 tag pipe. Our kernel is currently LDS-bound and FFMA-bound, not L1-miss-bound (BK=8 streams A through L1 with high reuse already), and `samplerBuffer` caps at 16-byte fetches (R32G32B32A32F via `texelFetch`) — same width as our LDG.E.128 BDA path — so we lose nothing in width but gain nothing in bytes/cycle. The realistic upside is reduced contention with B's LDG and possibly fewer L1 evictions for C-accumulator-adjacent traffic; the realistic downside is texel-fetch addressing overhead and losing the BDA fast path that already gives us LDG.128.

**Next step (only if PROMISING/MAYBE):** Prototype a single variant of the current best kernel that swaps only A's load path to `samplerBuffer` + `texelFetch(..., uvec4)` (keep B on BDA), add it as one row in KERNEL_SPECS, and A/B it at 1024^3 and 4096^4096^1024 to see if TEX-path A reduces L1 pressure enough to lift the 60-68% ceiling.

### #136 — MAYBE · effort M · risk MED
*Section: E. Async Compute / Queue Tricks*

**One-line idea:** Launch exactly 46 workgroups (one per SM) that loop forever pulling the next BM=128 BN=128 output tile index from a global atomicAdd counter until all tiles are consumed, amortizing launch + warmup overhead across the whole GEMM.

**Rationale:** Persistent-threads is technically possible in Vulkan compute (atomicAdd on an SSBO counter, while-loop in the shader, dispatch exactly num_SM * waves_per_SM groups) and our repo already has persistent-threads scaffolding from commit 9a70e78, so the plumbing exists. However for large GEMMs (1024^3 and up) launch overhead is sub-millisecond and we are bottlenecked by LDG/LDS/FFMA throughput inside the steady-state tile loop, not by warp warmup or tail effects, so the upside is mostly on small/medium shapes (256-512 range) and shape-mix benchmarks where the existing scaffold "landed but not yet beating Large". Worth one more focused tuning pass (correct grid size, tile-id prefetch, decouple from split-K atomic path) since the code is already there; risk is medium because atomic contention on the work counter and divergent tile tails can regress vs static scheduling.

**Next step (only if PROMISING/MAYBE):** Resurrect the existing persistent-threads variant, dispatch exactly 46*2 workgroups, prefetch next tile-id via atomicAdd one iteration ahead of the K-loop, and benchmark across the 26-shape showcase to see if small/medium shapes gain >5% without regressing 1024^3+.

### #152 — MAYBE · effort M · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Enable `VK_NV_device_diagnostics_config` and capture each KERNEL_SPECS variant in Nsight Compute to read the actual SASS (FFMA/LDG/LDS issue mix, register count, stall reasons) and hand-pick / hand-tune the variant that best matches the HW issue pipeline.

**Rationale:** This is diagnostic tooling, not a runtime optimization — it doesn't change the kernel by itself, but at the 60-68% peak ceiling our remaining gains depend on understanding *why* the driver-emitted SASS stalls (LDS bank conflicts, FFMA-LDG interlock, register reuse cache misses). Nsight Compute on Vulkan already exposes SASS + stall sampling for NVIDIA drivers without this extension; the extension mainly improves crash-dump fidelity, so the actual lever is "use Nsight properly" more than "use this extension." Given we've already hit dead ends (3-stage prefetch regression, double-buffer K-loop) by guessing, having ground-truth SASS would unblock the next 5-10% and is cheap. Risk is LOW because it's purely observational.

**Next step (only if PROMISING/MAYBE):** Capture an Nsight Compute profile of `bk16_bda_v4` and the current Large kernel on 1024^3, dump SASS + warp-stall reasons, and identify the top-2 stall causes before designing any further kernel variants.

### #153 — MAYBE · effort M · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Use `VK_KHR_pipeline_executable_properties` to read the NV driver's reported register-per-thread count for our SGEMM kernel and iteratively tune source/spec-constants/compiler hints until we hit ~64 regs/thread to enable 2 warps/SM (33% occupancy) on Ampere.

**Rationale:** The extension is supported on NVIDIA Vulkan drivers and does expose useful statistics (NumSgprs/NumVgprs-equivalent for NV, register counts, spill info) which gives a concrete observability win we currently lack — right now we are blind to actual register pressure. However, our 128x128 BM/BN BK=8 TM=8 TN=8 kernel inherently needs ~64+ regs just for C accumulators (8x8=64 floats) plus a/b register double-buffer (2*8 + 2*2 uvec4) plus addressing, putting us solidly in the 80-128 reg range; forcing 64 regs/thread would cause heavy spilling and likely regress perf. The real value here is observability (confirm spill, confirm occupancy) rather than the specific "force 64 regs" prescription. For our 60-68% peak ceiling kernel that's LDG/LDS bound, higher occupancy is unlikely to be the unlock — we are already at decent ILP within a warp.

**Next step (only if PROMISING/MAYBE):** Wire up `VK_KHR_pipeline_executable_properties` into the existing pipeline builder to dump register count + spill stats per kernel variant in KERNEL_SPECS, treating it as a diagnostic tool (not a forced-64-reg knob) to validate future micro-tuning.

### #154 — MAYBE · effort M · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Compile several SPIR-V variants of the SGEMM kernel (different BM/BN/BK/TM/TN, unroll factors, prefetch depths) parameterised by specialization constants, then at device init query `VK_KHR_pipeline_executable_properties` / shader stats (register count, occupancy) plus the device's subgroup/SM info and pick the variant with the best predicted occupancy x throughput for the given problem shape.

**Rationale:** We already have a 27-entry KERNEL_SPECS table and a shape-based selector, so "pick variant at runtime" is essentially done; the *new* twist here is using `VK_KHR_pipeline_executable_properties` to read driver-reported register/shared-mem usage and occupancy and let those numbers (not hand-coded heuristics) drive the choice. That is technically possible on Ampere with current NVIDIA Vulkan drivers (the extension is supported and exposes register/spill counts), and specialization-constant variants are standard SPIR-V. It will NOT raise our peak ceiling (we're still LDG/LDS/FFMA bound at 60-68%), but it could squeeze 1-3% by avoiding the wrong tile on edge shapes and making the selector self-tuning across drivers/HW. Worth it as infra hygiene more than as a perf jump.

**Next step (only if PROMISING/MAYBE):** Enable `VK_KHR_pipeline_executable_properties` at device creation, dump per-variant register count / shared-mem / invocations for all KERNEL_SPECS entries at startup, and replace the hand-written shape selector's tie-breaks with an occupancy-weighted score.

### #155 — MAYBE · effort M · risk LOW
*Section: G. Driver / Scheduler Hacks*

**One-line idea:** Expose BM/BN/BK/TM/TN and inner unroll factors as SPIR-V specialization constants so the driver JIT-recompiles a shape-tuned variant the first time each (M,N,K) bucket is dispatched, instead of maintaining 27 hand-written KERNEL_SPECS entries.

**Rationale:** Vulkan spec constants are real and well-supported on Ampere/NVIDIA drivers - they propagate into loop bounds and array sizes at pipeline-create time, enabling full unroll and constant folding (this is exactly how cuBLAS-style shape specialization works under the hood). However, our perf ceiling is LDG/LDS throughput and the FFMA pipeline, not kernel-selection granularity; we already have 27 baked variants covering the shape space, and adding more variants via spec-constants will not break the 60-68% ceiling. The real win is engineering: collapse KERNEL_SPECS into one parameterized shader, cleaner sweep infra, and the ability to micro-tune BK/unroll per exact shape without writing new GLSL. Risk is low because spec-constants are a standard, well-trodden Vulkan feature and pipeline-create cost is one-time per shape bucket (cacheable via VkPipelineCache).

**Next step (only if PROMISING/MAYBE):** Convert the current best BM128/BN128/BK8/TM8/TN8 shader to use `layout(constant_id=...) const uint BM, BN, BK, TM, TN` and benchmark whether the driver-emitted SASS matches the hand-baked variant on 1024^3 and 4096^3.

### #175 — MAYBE · effort M · risk MED
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Offline-repack A and B from row-major into BM x BK and BK x BN panel-packed (cuBLAS/cutlass-style) tile layout at upload time so each kernel global load is one sequential contiguous run, eliminating any partial sector and SMEM transpose work.

**Rationale:** This is a well-known BLAS "pack" optimisation and is trivially possible in Vulkan (CPU-side permute before staging, or a one-time compute pass). However, our current LDG.E.128 path through BDA is already mostly coalesced and we are bottlenecked by FFMA + LDS throughput at the ~60-68% peak ceiling, not by L2/DRAM misses — so the expected uplift is small (a few %) on memory-leaning shapes (skinny K, large N) and ~0 on compute-bound square shapes. It does, however, enable downstream wins: removing the SMEM A-transpose, simpler swizzle, and matching cuBLAS's offline-pack assumption for weight tensors in inference where the repack is amortised across many calls.

**Next step (only if PROMISING/MAYBE):** Add a `pack_A_panels` host-side helper plus one kernel variant (`bk16_packedA`) that consumes the new layout and skips the SMEM transpose, benchmark only on the memory-bound corner of the shape sweep.

### #176 — MAYBE · effort M · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** While the current SGEMM dispatch is still draining on the GPU, have the CPU pre-record the next dispatch's command buffer, update push constants/descriptors, and submit it so the queue is never idle between back-to-back GEMMs.

**Rationale:** This is just standard Vulkan pipelined submission (multiple command buffers in flight, semaphore-chained or same-queue submits) and is fully possible on Ampere/consumer HW — no exotic extension needed. However, it does NOT touch our actual bottleneck: a single large SGEMM (e.g. 4096^3) is hundreds of microseconds of pure FFMA/LDG/LDS work, dwarfing the few-microsecond CPU prep cost, so per-call this saves ~0%. The win is only realised in showcase/benchmark loops or multi-GEMM workloads (transformer stacks: QKV, attn, proj, FFM1, FFM2) where CPU submission latency between dispatches currently leaves bubbles; there it can claw back a few percent of wall-time without affecting the kernel's peak-% number.

**Next step (only if PROMISING/MAYBE):** Profile the showcase loop with Nsight/timestamp queries to measure inter-dispatch GPU idle gaps; if >2% of total time, implement double-buffered VkCommandBuffer recording with a 2-deep submission pipeline.

### #179 — MAYBE · effort M · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Use `VK_KHR_shader_non_semantic_info` + `debugPrintfEXT` from a single instrumented threadgroup of the SGEMM kernel to dump per-PC counters (loads issued, FFMA done, barrier hit) so we can pinpoint which line in the BK=8 main loop is actually stalling.

**Rationale:** `debugPrintfEXT` is real and works on RADV/NV via validation-layer interception, so it is technically possible on consumer Ampere. However it does NOT give SASS-level issue counts (that needs Nsight Compute / SASS sampling); at best we can print our own software counters (e.g. clockARB() deltas via `GL_ARB_shader_clock` or `KHR_shader_clock` around LDG/LDS/FFMA blocks), and only from one or two threads to avoid drowning in output. That's still useful as a poor-man's profiler when Nsight isn't usable from Vulkan compute, and could let us validate hypotheses about LDG vs LDS vs FFMA stalls that drive the 60-68% ceiling, but it won't directly move the needle — it's a *measurement* tool, not an optimization.

**Next step (only if PROMISING/MAYBE):** Add a `#ifdef DEBUG_PRINTF` path in the BK=8 main loop that uses `KHR_shader_clock` to timestamp the LDG-issue, LDS-store, LDS-load, and FFMA blocks, printf from thread (0,0) for a few K iterations, and compare against expected cycle costs to localize the real stall.

### #187 — MAYBE · effort M · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Use `VK_KHR_shader_clock` (`clockARB()`/`clockRealtimeEXT`) to read the SM clock around inner-K FFMA / LDS / LDG regions of our 128x128 BK=8 kernel and iteratively reshape the source until measured per-region cycles drop, effectively reverse-engineering the SASS scheduler's choices.

**Rationale:** `VK_KHR_shader_clock` is widely supported on Ampere via NVIDIA's Vulkan driver and gives a SUBGROUP-scope monotonic clock that's exactly what's used in CUDA microbenchmarks of latency/throughput - so the *measurement* part is genuinely possible and cheap to wire up. The catch is that on a 60-68% peak kernel we are bottlenecked by LDG/LDS throughput and FFMA pipeline saturation, not by mystery scheduling; shader_clock will mostly confirm what NSight / a careful occupancy + bandwidth model already says, and the clock reads themselves perturb the very issue slots we are trying to measure (Heisenberg-style). Where it *could* pay off is as a fast in-repo profiler that ranks the 27 KERNEL_SPECS variants per shape without round-tripping to NSight, and to A/B test prefetch/unroll source patterns that the driver currently silently reorders. Worth a one-day spike to build a tiny "clock-instrumented twin" of bk16_bda, but not a credible standalone path past the 68% ceiling.

**Next step (only if PROMISING/MAYBE):** Add a debug-only variant of the current best kernel that wraps the inner-K body, the LDS load, and the LDG prefetch with `clockRealtimeEXT()` reads into a per-warp SSBO, then diff cycle counts against the production kernel on 1024^3 / 2048^2048x4096 to see whether any single region's measured cycles actually disagrees with our analytical model.

### #200 — MAYBE · effort M · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Compile the BK=8 SGEMM kernel ~20 times with varying SPIR-V spec constants that nudge register pressure (TM/TN unroll factors, prefetch depth, accumulator chunking) and pick the variant with best measured TFLOPs per shape.

**Rationale:** Vulkan spec constants only let us tweak compile-time integers, not directly set NVVM `maxnreg`, so we can't *force* a register count, but we can indirectly steer the driver's allocator by varying unroll depth, scratch arrays, and a_reg/b_vec staging widths — this is exactly the lever NVIDIA's nvcc `--maxrregcount` sweeps exploit. We already have a KERNEL_SPECS table and a shape-based selector, so a sweep harness is a natural extension; given our 60-68% ceiling is partly a register-pressure/occupancy tradeoff, finding a sweet spot per shape could plausibly recover 1-3% on edge shapes (where we lose 12/26 vs cuBLAS). Risk is low because the selector already gates by shape and we keep the existing winner as fallback; the main cost is build-time and pipeline-cache bloat.

**Next step (only if PROMISING/MAYBE):** Add a `cargo xtask sweep-regs` that compiles the current Large kernel with 8-20 spec-const combinations of {prefetch_depth, accum_chunk, inner_unroll} and logs per-shape TFLOPs to pick winners for the KERNEL_SPECS table.

### #202 — MAYBE · effort M · risk LOW
*Section: J. Truly Crazy / "What if?"*

**One-line idea:** Deliberately bloat per-thread register usage (e.g. larger TM/TN tile, more prefetch regs, or explicit spill-free unrolling) so only 1 warp-group fits per SM, dedicating all 64KB regfile + LDS to fewer fatter warps to maximise FFMA ILP per warp.

**Rationale:** We are already register-heavy (TM=TN=8 + a_reg[2][8] + b_vec[2][2] ~ 80-90 regs/thread on a 256-thread block, so occupancy is likely 1-2 blocks/SM already). Pushing further (TM=TN=12 or 16, deeper register prefetch) is the standard "low-occupancy GEMM" recipe that cuBLAS/CUTLASS use on Ampere, and it directly attacks our FFMA-pipeline + LDG/LDS bottleneck by giving each warp more independent FMA chains to hide latency. The 3-stage register prefetch regression and 128x256 BK=16 dead end already show the ceiling is narrow, but a clean BM=128 BN=128 TM=TN=16 (4 warps/block, 1 block/SM) variant is a well-trodden path worth one prototype before declaring 68% the ceiling. Risk is low because it's a new KERNEL_SPECS entry, not a replacement.

**Next step (only if PROMISING/MAYBE):** Add a `bk16_tm16` variant with BM=BN=128, BK=8, TM=TN=16, 64 threads/block (1 warp-group, ~128 regs/thread by design) and benchmark on the 26-shape showcase to see if larger square shapes break past 68%.

---

## Numeric / ML Angle: Crazy Ideas for Vulkan SGEMM Speedup

_0 promising, 2 maybe_

### #17 — MAYBE · effort L · risk HIGH
*Section: Numerical Approximation*

**One-line idea:** Compute C0 = A*B in FP16/BF16 (ideally via VK_KHR_cooperative_matrix tensor cores) then add an FP32 residual correction R = A*B - C0 to recover SGEMM accuracy.

**Rationale:** Technically possible on consumer Ampere: VK_KHR_cooperative_matrix exposes FP16/BF16 tensor cores on GA104 (~81 TFLOPS FP16-with-FP32-accum vs 20.32 FP32 peak), so the low-precision pass is genuinely fast. The problem is that a *true* residual correction is itself a full FP32 SGEMM of A*B (or A*dB with dB the FP16 round-off of B), which costs as much as our current kernel and erases the win unless we accept reduced accuracy or fold the residual into a cheap rank-r update. We would also lose bit-exact cuBLAS-pedantic parity, which is part of our showcase story. This is closer to a research project than a tuning pass; the related Ozaki/3xTF32 split (3 BF16 tensor-core GEMMs to emulate FP32) is the more credible variant on GA104 but is a different idea.

**Next step (only if PROMISING/MAYBE):** Prototype a VK_KHR_cooperative_matrix FP16-accum-FP32 GEMM in isolation and measure raw TFLOPS on GA104; only pursue refinement if the low-precision pass clears ~30 TFLOPS sustained, otherwise drop in favor of Ozaki-3xTF32.

### #69 — MAYBE · effort XL · risk HIGH
*Section: Wild Cards*

**One-line idea:** Emulate FP32 SGEMM by splitting A,B into FP16 hi/lo halves, doing 3-4 FP16 tensor-core matmuls via VK_KHR_cooperative_matrix, and summing the residuals to recover near-FP32 precision (Markidis/TCEC-style).

**Rationale:** This is the ONE class of trick that can break our ~20.32 TFLOPS FP32 peak ceiling on GA104, because Ampere consumer FP16 tensor cores run ~4x faster than FFMA — so even 3-4 FP16 GEMMs plus residual adds can in principle exceed FFMA peak. However, it requires switching from FFMA to VK_KHR_cooperative_matrix (subgroup-shaped fragments, totally different tile/load structure), rewriting the BM=128/BN=128/BK=8 kernel from scratch, and accepting that results are no longer bit-identical to FP32 (typically a few ULPs off), which breaks our cuBLAS-pedantic-math equivalence claim. Effort is XL and risk to the existing 14/26-wins story is HIGH, but if we want a "Huge" tier above Large this is the only realistic angle.

**Next step (only if PROMISING/MAYBE):** Spike a standalone cooperative_matrix FP16 GEMM (single tile, no residual) to confirm the extension is exposed on our driver and measure raw FP16 tensor-core throughput vs FP32 FFMA before committing to the full hi/lo residual rewrite.

---

## Physics / Analog / Field-Theoretic Ideas for Vulkan FP32 SGEMM

_0 promising, 4 maybe_

### #583 — MAYBE · effort S · risk LOW
*Section: Mechanics & Oscillators*

**One-line idea:** Sweep workgroup_x (and BM/BN/BK pad offsets) by small deltas around the current 128x128 tile to detune from L1/L2 cache-line and DRAM-bank "resonant" stride collisions that periodically thrash the memory subsystem.

**Rationale:** This is really just a swizzle/pad sweep dressed up in oscillator language; it is trivially possible in Vulkan (specialization constants over workgroup size + a leading-dim pad), zero new extensions. We are partially LDG/LDS-bound at ~60-68% peak, and Ampere's L2 partitions + DRAM bank-group conflicts are known to penalize specific power-of-two strides, so a small detune (e.g., workgroup_x=129/130, or +4/+8 LDA pad) can plausibly buy 1-3% on pathological shapes (powers of two, multiples of 256). However, we already have a 27-variant KERNEL_SPECS selector and have not seen large gains from minor tile perturbations, and non-power-of-two workgroup_x breaks vector alignment for LDG.128 - so the upside is likely small and shape-specific, not a ceiling-breaker.

**Next step (only if PROMISING/MAYBE):** Add a tiny offline sweep harness that benchmarks the current best kernel with LDA/LDB pad in {0,4,8,16} and workgroup_x perturbations that preserve 128b alignment on the 1024^3 / 2048^2 / 4096^2 shapes, and only land variants that win >=2% without hurting others.

### #593 — MAYBE · effort S · risk LOW
*Section: Exotic / Wild*

**One-line idea:** Pick between a "many-small-WG" kernel (good for tiny shapes, high SM occupancy via many workgroups) and a "single-big-WG" kernel (good for large shapes, max register/shared reuse) using a single device-calibrated matrix-size threshold, framed as a Hawking-Page-style phase transition.

**Rationale:** Stripped of the cosmology metaphor this is just a size-based kernel selector, which we already do via the runtime shape-based selector over KERNEL_SPECS (27 variants). The "novel" piece is reducing it to one scalar threshold computed once per device at init time, which is a trivial refinement of the existing selector and won't change peak throughput — it only matters in the tail where small/large kernels cross over. Won't move the 60-68% peak ceiling on showcase shapes since that's LDG/LDS/FFMA bound, not selection-bound, but could plausibly tidy a few sub-optimal selector decisions in the small-shape regime.

**Next step (only if PROMISING/MAYBE):** Add a one-time init-pass that sweeps a few representative shapes, fits the small-WG vs big-WG crossover point per device, and stores it as a single threshold consulted by the shape selector.

### #563 — MAYBE · effort M · risk LOW
*Section: Thermodynamics & Statistical Mechanics*

**One-line idea:** When launch shape/occupancy crosses a threshold where all 46 SMs would chase the same K-strip in lock-step (maximizing L2 reuse of A and B panels), pick a "collapsed" tile-walk order (e.g. all-SMs-sweep-same-K-block, column-major superblock); otherwise fall back to the standard distributed Z/row-major schedule.

**Rationale:** The BEC metaphor is just shape-dependent L2-reuse-aware scheduling: for tall-skinny or small-N problems where the working set fits L2, forcing all SMs to march through the same B (or A) panel in phase converts B-loads into single-shot L2 hits, which directly attacks our LDG bottleneck — the only knob left below the 68% ceiling that isn't FFMA-bound. We already have a runtime shape-based selector and 27 KERNEL_SPECS, so adding a "collapsed" walk variant is mostly a workgroup-ID remap in the shader plus one selector branch; no new extensions needed. Risk is low because it's a scheduling change, not a math change, but the win is bounded to shapes where L2 reuse is currently being wasted (likely a subset of the 12 losses vs cuBLAS), not the showcase 4096^3 case which is already DRAM-bandwidth-saturated on A panels.

**Next step (only if PROMISING/MAYBE):** Add a `bk16_collapsed` variant that remaps gl_WorkGroupID so all 46 concurrent CTAs share the same BN-column-strip and march together down K, and gate it in the selector when N*K*4 < ~3 MB (fits L2) — measure on the 12 cuBLAS-loss shapes.

### #568 — MAYBE · effort M · risk LOW
*Section: Field Theory & Quantum-Adjacent*

**One-line idea:** Treat the autotuner search over tile schedules (BM/BN/BK/TM/TN, swizzle, split-K factor, persistent vs not) as importance sampling from a Boltzmann distribution exp(-T_measured/tau), annealing tau down to greedily concentrate on fast schedules.

**Rationale:** This is just a dressed-up simulated-annealing / softmax-bandit autotuner over our KERNEL_SPECS axis, which is legitimately useful: we already have 27 hand-picked variants and a runtime shape-based selector, but the discrete (BM,BN,BK,TM,TN,swizzle,split-K) space is much larger and we haven't systematically swept it. It cannot break the ~60-68% FP32 peak ceiling (LDG/LDS/FFMA bound, not schedule-bound), but it could find a couple of shape-specific wins we missed and replace ad-hoc selection logic. The "path integral" framing adds zero physics value over standard Thompson sampling / SA, so the cost is just an offline tuner harness, not kernel changes.

**Next step (only if PROMISING/MAYBE):** Build an offline Python/Rust harness that enumerates legal (BM,BN,BK,TM,TN,split-K) tuples per shape, runs each a few times, and picks via softmax-over-time annealing, then bake winners into KERNEL_SPECS.

---

## PL Theory / Type Systems / Category Theory Ideas for Vulkan FP32 SGEMM

_0 promising, 2 maybe_

### #622 — MAYBE · effort M · risk LOW
*Section: Category Theory*

**One-line idea:** Fuse N independent small GEMMs into one Vulkan dispatch where each workgroup looks up its (M,N,K,A_ptr,B_ptr,C_ptr) tuple from a BDA-resident problem table, so 46 SMs stay saturated instead of two tiny back-to-back grids each filling ~half the GPU.

**Rationale:** This is grouped/batched GEMM (CUTLASS GroupedGemm, cuBLAS gemmBatched) reframed in monoidal-category language - it's real and works. Vulkan supports it trivially via push-constant problem-table index + buffer_reference per-problem A/B/C pointers; no extension needed. But our showcase wins are on *large* GEMMs (M,N>=1024) where one problem already saturates 46 SMs and we're LDG/LDS/FFMA bound, not occupancy bound - so this moves zero needle on the geomean-vs-cuBLAS metric. It only helps the small-shape tail (M,N<=512) where current per-dispatch overhead + tail-effect under-fills the GPU; could win a few of the 12/26 shapes we currently lose.

**Next step (only if PROMISING/MAYBE):** Add a `bk8_grouped` variant that takes a small (<=16-entry) problem-table push-constant of `(M,N,K, A_bda, B_bda, C_bda, tile_offset)` tuples, derives the per-workgroup problem from `gl_WorkGroupID.z` (or a prefix-summed tile_id), and benchmark against the current selector on a synthetic batch of 4-8 small matmuls.

### #626 — MAYBE · effort M · risk LOW
*Section: Category Theory*

**One-line idea:** Fuse the epilogue (bias-add, ReLU/GELU, scaling) directly into the 8x8 C-register tile after the K-loop so the partial accumulators are "streamed" through pointwise ops in-register before the single global store, eliminating any intermediate global writeback.

**Rationale:** The "coinductive stream" framing is just category-theory dressing on classic in-register epilogue fusion, which is genuinely valuable but orthogonal to our LDG/LDS/FFMA bottleneck on raw SGEMM — it won't move the 60-68% peak ceiling on a pure C=A*B benchmark. However, for real inference workloads (the actual point of tensor-ash), folding bias/activation/scale into the 8x8 register tile before the BDA store saves a full C-sized round-trip to HBM per layer, which is a large end-to-end win. Implementation is straightforward: extend KERNEL_SPECS with an epilogue enum and emit the pointwise ops between the K-loop and the vec4 store; risk is low because the hot inner loop is untouched.

**Next step (only if PROMISING/MAYBE):** Add an `Epilogue { None, Bias, BiasRelu, BiasGelu, Scale }` field to KERNEL_SPECS and a templated epilogue block in the Large kernel that runs over c_reg[TM][TN] before the BDA store, then benchmark fused vs. unfused on a representative MLP layer.

---

## Quantum / Superposition Analogies for SGEMM at Hardware Peak

_0 promising, 4 maybe_

### #380 — MAYBE · effort S · risk LOW
*Section: IX. The Honest Reframes*

**One-line idea:** Accept that classical tile-tuning has plateaued at ~60-68% peak and explicitly pivot effort from FFMA/LDS microtuning to algorithm-level changes (Strassen, low-rank, structured sparsity, FP16/TF32 tensor-core fallback for "FP32"-tolerant paths).

**Rationale:** This is a meta-decision rather than a kernel, but it is the honest framing of our current state: BDA/uvec4/reg-double-buffer already captured the easy wins, and the 3-stage prefetch regression plus the dead-end list show diminishing returns from further within-algorithm tuning. The needle-mover is shifting FLOP count (Strassen saves ~12.5% multiplies at N=2048) or precision (tensor-core TF32 is ~4-8x FP32 on GA104 if the API contract allows it), not squeezing another 2% from FFMA scheduling. Cost of writing the decision down and reprioritising the backlog is tiny; risk is only that we deprioritise a real remaining classical win.

**Next step (only if PROMISING/MAYBE):** Add a short "perf ceiling decision" note to the repo declaring 65% peak as the classical target and reordering IDEAS.md so Strassen-real, low-rank, and TF32-tensor-core experiments are scheduled before any further FFMA/LDS micro-tuning ideas.

### #334 — MAYBE · effort M · risk LOW
*Section: I. Superposition / Measurement*

**One-line idea:** Add a second "high-occupancy / low-register" companion kernel (e.g. BM=64 BN=64 TM=4 TN=8, ~2x more warps/SM) next to the current BM=128 BN=128 TM=8 TN=8 "high-reuse / low-occupancy" kernel and let the runtime shape selector pick per (M,N,K), framed as a Heisenberg-style occupancy<->reuse tradeoff.

**Rationale:** Tensor-ash already has a 27-entry KERNEL_SPECS table and shape-based selector, but the bulk of "Large" variants sit at the same 128x128 TM=8 TN=8 reuse point; we have no real low-register/high-occupancy counterpart for skinny or small shapes where the 8x8 register tile starves the SM of warps. A complementary 64x64 TM=4 TN=8 (or 64x128 TM=4 TN=8) kernel is a well-known regime switch on Ampere and cheap to add given the data-driven pipeline. It won't break the 60-68% ceiling on square 1024+ shapes (still LDG/LDS+FFMA bound), but it could plausibly flip several of the 12 losses vs cuBLAS on small/skewed shapes, which is exactly where shape-dependent dispatch pays.

**Next step (only if PROMISING/MAYBE):** Add one BM=64 BN=64 BK=8 TM=4 TN=8 variant (reusing BDA+uvec4 LDS path) to KERNEL_SPECS, extend the selector with an occupancy-vs-reuse threshold on min(M,N), and re-run the 26-shape showcase to count flipped wins.

### #362 — MAYBE · effort M · risk LOW
*Section: VII. Speculation / Many-Worlds Branch Prediction*

**One-line idea:** At dispatch end, each workgroup atomically writes the best-observed (tile-size, BK, swizzle) hint for its (M,N,K) shape into a small persistent SSBO that the host-side selector consults on the next launch, turning the KERNEL_SPECS table into a self-tuning cache.

**Rationale:** Strip the "retrocausal" framing and this is just persistent online autotuning: a per-shape hint buffer that survives between dispatches and biases the runtime shape-based selector. Truly retrocausal (writing into the *past* kernel) is impossible — there is no temporal channel in Vulkan/SPIR-V. But the practical reading (kernel N writes hints, kernel N+1 reads them) is trivial to implement with one SSBO + atomics and would help us actually exploit the 27-variant KERNEL_SPECS table on workloads with repeated shapes (training loops, transformer decode). It will not raise our 60-68% peak ceiling on any single shape — it only helps pick the right existing kernel faster — so the upside is on the showcase geomean / tail shapes, not on peak.

**Next step (only if PROMISING/MAYBE):** Add a small `runtime_hints: HashMap<ShapeKey, KernelId>` populated by per-dispatch CPU-side timing (VK_KHR_timeline_semaphore + timestamp queries) and consulted before the static selector — no shader changes needed for v1.

### #379 — MAYBE · effort L · risk MED
*Section: IX. The Honest Reframes*

**One-line idea:** Reframe the autotuner's objective from per-CTA tile efficiency to maximizing constructive L2/DRAM access overlap across concurrently-resident SMs, i.e. choose block-swizzle, launch order, and BM/BN so neighboring workgroups touch overlapping A/B rows-cols within an L2 residency window instead of thrashing it.

**Rationale:** This is a real and well-known effect on Ampere (L2 is 4MB shared across all SMs, and CUTLASS/cuBLAS explicitly use "threadblock swizzling" / "rasterization order" to maximize L2 reuse across CTAs). On a 46-SM GA104 with 128x128 tiles, roughly 8-16 CTAs are co-resident; if their A-rows and B-cols overlap, you get effectively free reuse from L2. The honest reframe (constructive interference) maps directly onto this and is one of the few remaining levers since we are LDG/LDS-bound and the per-tile FFMA pipeline is mostly saturated. The catch: in Vulkan compute we have far less control over workgroup dispatch order than CUDA (the scheduler picks), so we have to encode the swizzle inside the shader by remapping gl_WorkGroupID->(tile_m,tile_n) — doable but its benefit on GA104 with our 4MB L2 is probably modest (~3-8%), not a path through the 68% ceiling.

**Next step (only if PROMISING/MAYBE):** Add a Hilbert/Morton or CUTLASS-style "log2 group" block-swizzle remap inside bk16_bda_v4 (compute tile_m,tile_n from gl_WorkGroupID.x via a small bit-twiddle) and sweep group sizes {1,2,4,8} on 2048^3 and 4096^3 to see if L2 hit rate (and runtime) improves.

---

## SGEMM via Game Dev / Rendering Wizardry

_0 promising, 1 maybe_

### #790 — MAYBE · effort S · risk LOW
*Section: Bonus / wildcards*

**One-line idea:** At library init, dispatch a tiny dummy matmul for every KERNEL_SPECS variant so all 27 compute pipelines are JIT-compiled and stored in a serialized VkPipelineCache, eliminating first-call shader-compile latency on subsequent shapes.

**Rationale:** This is bog-standard Vulkan: VkPipelineCache + vkCreateComputePipelines + vkGetPipelineCacheData is exactly the game-engine warmup pattern, and we already enumerate 27 variants in the data-driven KERNEL_SPECS table so iteration is trivial. However, it does NOT touch our actual bottleneck (LDG/LDS throughput, FFMA pipeline, 60-68% peak ceiling) -- it only amortizes one-time pipeline creation cost, which is a latency/UX win, not a steady-state throughput win. Worth doing for benchmark hygiene (first-call outliers skew geomean) and for any real inference deployment, but it will not move the perf-vs-cuBLAS needle on warm runs.

**Next step (only if PROMISING/MAYBE):** Add a `warmup_all()` method that creates+caches every KERNEL_SPECS pipeline at Context init and persists VkPipelineCache blob to disk for cross-run reuse.

---
