# tensor-ash IDEAS — Crazy Optimization Brainstorm

Three parallel brainstorming agents produced 326 ideas covering numeric / ML
tricks, hardware abuse / Vulkan extensions, and algorithmic restructuring.
**Aim: maximum creativity, no feasibility filter.** Most ideas are wrong,
half-baked, or impractical — but the goal is a wide net to fish promising
candidates out of later.

Use this as a quarry: when you need a fresh optimisation angle, scan the
relevant section, pick something that resonates, and prototype it.

---

# Numeric / ML Angle: Crazy Ideas for Vulkan SGEMM Speedup

## Mixed Precision Tricks

- **FP16xFP16->FP32 emulation in shader cores**: Pack two FP16 in a u32, do a single FMA on the FP32 unit by manual exponent alignment, claim 2x throughput on memory bandwidth even without tensor cores.
- **TF32-style mantissa truncation**: Truncate FP32 inputs to 10-bit mantissa before multiplying, store accumulator as full FP32; on the 3070 you might be able to skip a clock or two in the multiplier (probably not but worth trying).
- **BF16 emulation by mantissa zeroing**: AND off the low 16 bits of each FP32 operand pre-multiply; if the multiplier short-circuits on trailing zero mantissa bits (it doesn't, but who knows), free speedup.
- **INT8 path with per-tile fp32 scale**: Quantize tiles of A and B to INT8 with one shared scale per 16x16 block; do dot product as INT32, multiply by scale once at the end.
- **INT4 path with packed nibbles**: 4-bit weights packed 8 per u32, integer dot-product, scale at writeback. Loses 5 bits of precision but might be fine for inference.
- **FP8 (E4M3 / E5M2) emulation via LUT**: Decompose each FP8 multiply into one 8-bit table lookup; eight muls per shared-memory load.
- **Posits-8 lookup multiply**: A 256x256 LUT for 8-bit posit multiplication, indexed by concatenated operands; share LUT in LDS.
- **Stochastic rounding on accumulator**: XOR-shift noise injected into the low mantissa bits before each round; lets you use a smaller accumulator (FP24-ish) without bias.
- **Kahan summation but only every 8th add**: Compensated sum for compensating the bottom few bits of the accumulator without paying for it every step.
- **Block-FP**: One shared exponent per 32-element row chunk; multiplies become integer-mantissa multiplies plus a single exponent add at the chunk level.
- **Logarithmic-number-system path**: Take log of A and B once at the load, accumulate via add, exp at writeback. For sgemm-once-then-reused-many-times workloads.

## Numerical Approximation

- **Mitchell logarithm-multiply approximation**: log2(x) ~= mantissa + exponent (linear piecewise), so a*b ~= 2^(log2 a + log2 b). One add instead of one mul.
- **Piecewise-linear multiply LUT**: For one operand range-bin its mantissa to 8 entries; do a*b via shift + small correction add from LUT.
- **Bit-shift multiply for power-of-two-ish operands**: If operand mantissa is within 1.5 ULP of a power of two, just shift the other operand; ~5% of weights in trained nets are close-to-power-of-two.
- **CORDIC rotation for tile updates**: Reinterpret the tile multiply as a series of micro-rotations on (re, im) pairs; nine shifts and adds replace a multiply.
- **Karatsuba on 16-bit halves of FP32**: Split mantissa into hi/lo, three multiplies + adds instead of four. Probably wasteful for f32 multiply hardware but interesting.
- **Goldschmidt-style iterative refinement**: Use one low-precision matmul to estimate, then one residual matmul to correct.
- **Russian peasant multiplication**: Stupid idea but on tiny integer values it's just shift-and-add and might pipeline better.
- **Truncated Booth recoding**: Pre-recode operand B's mantissa into signed digits; skip the multiply slots where digit==0.

## Sparsity

- **2:4 structured sparsity by zeroing low magnitude in each group**: Force two of every four B-mantissas to zero, skip those FMAs in inner loop.
- **Magnitude-thresholded skip**: If |b| < eps_tile, skip the FMA; tile-level thresholds picked from per-tile max.
- **Bloom filter per tile for zero detection**: 32-bit bloom per row that says "this row may contain all-zero blocks", check in shared memory before issuing the FMA.
- **Bitmask of nonzeros per K-chunk**: One u32 mask per 32 K-elements per row of A; use bit-count to skip zero columns entirely.
- **Implicit zero padding for non-multiple-of-tile-size shapes**: But also mark the padded rows in a mask, skip them in the inner loop entirely.
- **Row clustering before multiply**: K-means rows of A into 64 clusters, multiply each cluster's centroid by B once, scatter back. Approximate but cheap.
- **Cosine-similarity-based row deduplication**: If two rows of A have cosine sim > 0.99, multiply once.
- **DropConnect-style random zeroing at multiply time**: With probability p, replace the FMA's b operand with 0; scale up surviving accumulator by 1/(1-p). Lossy but might fit training tolerance.
- **Magnitude-pruned columns of B per tile**: For each tile of B, drop the 25% smallest-magnitude columns and skip; record a mask for write-back.

## Approximate Matmul Algorithms

- **Strassen-Winograd 7-multiply recursion at tile level**: Tile-of-tiles Strassen, accept rounding error.
- **Two-level Strassen**: 49 multiplies instead of 64, recursive at register-tile and shared-tile granularity.
- **Karatsuba-style on 2x2 sub-tiles**: Three multiplies plus additions for a 2x2 block.
- **Pan's 36-mul 3x3 algorithm**: 27 multiplies become 21; tiny speedup but free if it pipelines.
- **AlphaTensor's discovered algorithms**: Use one of the recently-discovered matmul recipes (4x5x5 in 76 mults etc).
- **Random projection sketch (Johnson-Lindenstrauss)**: Replace A with A*R where R is a kxk' random sign matrix, k' << k; multiply A*R*R'*B, with appropriate scaling.
- **Subsampled randomized Hadamard transform of K dimension**: Transform A and B's K axis with Hadamard then subsample; matmul becomes smaller.
- **CountSketch on K dimension**: Hashing sketch reduces K to K' with O(1) operations per K.

## Block-Low-Rank / Factorisation

- **SVD truncation of A**: Factor A = U Sigma V^T with top-r singular values, run two cheaper matmuls.
- **Tile-wise SVD with caching**: Per-tile (e.g. 64x64) SVD with rank-8 truncation; precompute factors and reuse across batch.
- **CUR decomposition**: Pick r columns of A and r rows of B, multiply the sub-matrices; useful when matrices are low-rank-ish.
- **Tensor train decomposition for large K**: K dimension as tensor-train cores; multiplies cascade through cores.
- **Block-diagonal-plus-low-rank decomposition**: A = block-diag + UV; cheap inner block multiplies plus one low-rank correction.

## Randomized / Sketch Methods

- **Monte Carlo column sampling**: Sample columns of A and rows of B proportional to product of L2 norms, do tiny matmul, scale.
- **Randomized SVD of difference**: Use FP16 path for the "main" matmul, then a randomized-SVD-based correction for the residual.
- **Per-tile Bernoulli mask**: Drop 50% of FMAs per tile via random mask, scale by 2. Unbiased estimator.
- **Sparse outer-product summation**: Sample (col_a, row_b) pairs proportional to outer product Frobenius weight.

## Bit-Trick / ULP Skips

- **Skip FMA when fl(a*b) < ULP(accumulator)**: Compare exponent_a + exponent_b vs exponent_acc; skip when too small to matter.
- **Tail-cut threshold per accumulator**: Once accumulator reaches some magnitude, freeze it for the rest of the dot product — typical reductions hit asymptote early.
- **Sign-only multiply for small magnitudes**: If |a*b| is tiny, just add sign(a*b)*ULP(acc); a 1-bit "multiply".
- **Multiply by sign(b) only when |b| < threshold**: Cheap.
- **Skip writes when accumulator hasn't changed by more than ULP**: At tile boundary skip the smem store entirely (probably no speedup but pretty).

## Caching / Memoisation

- **LUT of a*b for 8-bit operands shared in LDS**: 64KB LUT, indexed by 8+8 = 16-bit concat. Maybe shared across the threadblock to amortize load.
- **Per-tile bloom filter for zero detection**: Check before issuing the inner-loop multiply.
- **Hash-cache of recent (row, col) inner products in batched matmul**: When training, batch reuses many A rows; cache the dot product result keyed by row-hash + col-hash.
- **Tile-level dedup across batch**: If two B columns hash to the same bucket, multiply once and broadcast.

## Numerical Slop That ML Tolerates

- **Asymmetric rounding for ReLU-bound outputs**: Round toward zero when downstream activation is ReLU; positive bias gets clipped anyway.
- **Drop the last K iterations of the reduction at random per row**: ML training is noisy; if you do this with consistent stats, optimiser absorbs it.
- **Top-k inner products only**: Per output element, sort top-K |a_k*b_k| and only sum those.
- **Compensate slop with batchnorm/layernorm downstream**: If a layernorm follows, you can scale the entire output, so a constant multiplicative error is free.
- **Re-use stale accumulator across iterations**: If the output is being added to a residual stream, the residual masks small errors; accumulate only every other call.
- **Stochastic depth on tiles**: With prob p skip the entire tile contribution and scale survivors by 1/(1-p).
- **Aggregate-then-multiply for repeated rows**: Sum a-rows-with-the-same-class first, then multiply by B once (only works in classifier-style sgemm).
- **Hogwild-style accumulator without atomics**: Let threads race on the global accumulator; for ML training the noise is fine.

## Wild Cards

- **Sub-byte INT2 path with double-LSB trick**: 2-bit weights, signed ternary {-1, 0, 1}, multiply becomes a conditional add — no actual multiply on the SM at all.
- **Ternary weight matmul as popcount**: Cast as XNOR + popcount of the sign bits, scale by per-row magnitude post hoc.
- **Binary matmul + scalar correction**: Do binary matmul (XNOR/popcount) for fast path, fp32 correction matmul on the residual only.
- **Use the bilateral-symmetric structure of attention QK^T**: For self-attention sgemm, exploit Q=K-ish to skip half the work.
- **Spectral matmul via FFT**: For block-circulant structure, multiply in frequency domain. Niche, but if any weight matrix is circulant-like.
- **Toeplitz-structure decomposition**: Decompose A as low-rank + Toeplitz; Toeplitz part is O(n log n) via FFT.
- **Use the rounding error of low precision as a feature**: Compute FP16 matmul, then compute the error of that as a separate small matmul (a la "block-FP plus residual"), sum.
- **Random tensor decomposition with online learning of the decomposition**: Learn the decomposition rank during training; rank grows when loss plateaus.
- **Symbolic precomputation of constants in the matrix**: If many entries in B equal +/- 1, +/- 0.5 etc, replace multiply with shift.
- **Population-count-based dot product on sign bits**: For sign-only product summary that warns when to compute the full dot.
- **Dithered fixed-point with triangular noise**: Add triangular dither before quantize-to-fixed; reduces correlated rounding error.
- **Compute only the diagonal of large matmuls**: When downstream only needs diag(A*B) (norm, trace stuff), do the row-row dot directly.
- **Operand fusion at sample boundaries in mini-batch**: If two samples in a batch share an A-tile, merge their dot products mid-flight.
- **Use weight magnitude histograms to choose precision per row**: Per-row decision FP16 or FP32 path based on max magnitude of that row.
- **Tile-wise dynamic precision**: Estimate tile output magnitude with a cheap upper bound; if small, use FP16, else FP32.
- **Approximate by polynomial Chebyshev expansion of one operand**: B = sum_i c_i T_i(A); reduces multiplies if c_i decay fast.
- **Iterative refinement with low-precision base**: One FP16 matmul + one FP16 residual matmul = often as accurate as FP32, 2x faster.
- **Compressed-sensing weight reconstruction at multiply time**: Store A in compressed form (DCT coeffs), decompress only the active tile.
- **Walsh-Hadamard pre-rotation of K**: Rotate A and B by Hadamard; many random matrices have lower effective rank after Hadamard, then truncate.
- **Approximate matmul as graph neural net pass**: Treat A and B as bipartite weights, run a single message-passing step; equivalent to matmul but might map better to hardware later.
- **Use INT8 with floating-point exponent metadata**: Group operands by magnitude class, do INT8 mantissa multiplies, fixup exponent at the end (basically BFP but tile-aware).
- **Block-cyclic precision rotation**: Round-robin FP16/FP32 per FMA; on average you get FP24-ish precision at ~1.5x speed.
- **Output noise injection in place of rounding**: Add Gaussian noise to output instead of computing the exact low-bit answer; ML doesn't care.
- **Permute K dimension to put high-magnitude products first**: Greedy reorder so the reduction's most-significant terms come first; lets you early-terminate on insignificance.

## Meta / Process

- **Compute a "matmul ROI" estimate per tile before issuing**: Quick L1-norm upper bound on tile output, skip tile if below epsilon.
- **Two-stage matmul with confidence**: First pass FP16, then refine only the output positions whose magnitudes suggest they matter (top-K by activation downstream).
- **Online learning of which tiles are skippable**: Maintain per-tile expected magnitude; once below threshold consistently, skip permanently.
- **Per-matrix-instance kernel specialization**: Detect repeated calling patterns and JIT a specialised kernel for, e.g., "B has 30% zeros" cases.
- **Use FP32 only on the diagonal blocks, FP16 off-diagonal**: When matrices have structure (e.g. attention with strong diagonal), this is fine.
# Hardware-Abuse / Vulkan-Extension / GPU-Internals Brainstorm
## Target: hand-written FP32 SGEMM on RTX 3070 (Ampere GA104), currently ~60% peak

---

## A. Cooperative Matrix / Tensor Core Exploits

1. **FP16 cooperative_matrix + Kahan/Dekker FP32 correction.** Use `VK_KHR_cooperative_matrix` FP16 tensor cores for the bulk MAC, then run a residual correction pass using FP16 of `(a - fp16(a))` and `(b - fp16(b))` to recover FP32 precision. Costs ~4x ops but at tensor-core rate = still 2x faster than CUDA cores.

2. **Single-pass 3xFP16 simulating FP32 (a la "tf32 trick" but inverted).** Decompose each FP32 as hi/mid/lo FP16 triplets; one tensor-core matmul per cross-term, sum nine products. 9x ops at 8x throughput ≈ net win on Ampere.

3. **FP16 + BF16 mixed cooperative matmul.** Some Ampere paths expose BF16 with the same range as FP32 but 8-bit mantissa; combine FP16 (mantissa) + BF16 (range) tensor-core ops to span FP32 numerics.

4. **Tensor-core matmul as a "preconditioner" and only correct on diff.** Run FP16 tensor-core matmul to get an approximate C0, then run the CUDA-core SGEMM on `(A * B - C0_recovered)` which is small-magnitude → can be done in FP16 again. Two-level refinement.

5. **VK_NV_cooperative_matrix2 / VK_NV_cooperative_matrix.** The NV-specific extension on some drivers exposes more shapes and FP32 accumulate on Ampere — probe pipeline executable properties; even if undocumented for GA104, try the shape table.

6. **`OpCooperativeMatrixLoadNV` with FP16x2 inputs interleaved.** Pack two FP32 matmuls into the lanes of one tensor-core op, then split with a subgroup shuffle. Doubles work per tensor-core fire.

7. **Tensor cores for the K-reduction only.** Use CUDA cores for FMA over a few K-tiles into FP16 partials, then use tensor cores as a wide reduction tree (treating the reduction as a tiny matmul).

8. **Abuse the sparse tensor cores (VK_NV_ray_tracing_motion_blur kindof-related path).** Ampere has 2:4 sparse tensor cores at 2x rate; mask half of B as fake-zero with a structured pattern and reconstruct, getting effectively 4x FP16 rate for FP32 reconstruction.

---

## B. Subgroup / Warp-Level Hacks

9. **Skip shared memory entirely with `subgroupShuffleXor` to broadcast A-tile rows.** All 32 lanes load distinct A elements, then shuffle to broadcast — saves the LDS write/read round-trip for the inner-product broadcast.

10. **`subgroupBroadcast` for B-tile column broadcast** combined with thread-block-level shared mem only for A. Hybrid: A in LDS, B via warp shuffle. Cuts LDS bandwidth in half.

11. **Subgroup-shuffle butterfly reduction for the K-loop final sum** instead of writing partials to LDS. 5-step butterfly within the warp = 1 cycle per step on Ampere.

12. **`subgroupClusteredAdd` for the C accumulator partial sums** when K is split across the warp. Free reduction in hardware.

13. **`SubgroupUniformControlFlow` + `MaximalReconvergence` capabilities** to let the compiler assume no divergence and hoist loads/FMAs past barriers more aggressively.

14. **Override the warp size detection — force subgroup size 32 with `VK_EXT_subgroup_size_control`** to ensure shared-memory-free shuffles work; some pipelines launch with size 16 due to occupancy heuristics.

15. **Use `subgroupQuadBroadcast` (the 2x2 quad ops left over from fragment shaders) inside compute** to do micro-broadcasts of B values to 4 neighbors at once — separate datapath from the main subgroup shuffle.

16. **Lane-rotation trick for a "rolling" K accumulator.** Each thread's accumulator rotates one lane per K-step via `subgroupShuffle(c, (lane+1)%32)`; lets all 32 lanes work on 32 different (m,n) outputs while reusing the same A,B values for free.

17. **"Implicit" double-buffering via two halves of the subgroup.** Lanes 0-15 prefetch K+1 while lanes 16-31 compute K, then swap roles every K iteration. No barriers, no LDS double-buffer.

---

## C. SPIR-V Hand-Assembly / Bypass the Compiler

18. **Hand-written SPIR-V with explicit `OpExtInst` to NV-specific instructions.** glslc throws away subtle scheduling info; write SPIR-V directly via spirv-tools assembler and pin register usage.

19. **Inline PTX via `VK_NV_glsl_shader`'s SPIR-V `OpExtInst NonSemantic.Shader.DebugInfo` hack** to smuggle in raw PTX strings the NV driver back-compiles.

20. **Hand-pack SPIR-V `OpVectorShuffle` patterns** that the driver maps to free-cost permute network on Ampere — picks specific shuffle masks that hit the LD/ST unit's swizzle path.

21. **Forge SPIR-V with `OpDecorate ... Patch` and `Volatile` flags** to disable specific load coalescing/cache pollution heuristics; sometimes the driver under-coalesces — force it.

22. **Use `OpAtomicFAddEXT` with `VK_EXT_shader_atomic_float`** for the C accumulator across multiple workgroups — eliminates the split-K reduction kernel entirely and lets workgroups race; aggregate done in hardware.

23. **Patch the SPIR-V post-compilation to add `RestrictPointer` and `AliasedPointer` decorations.** Glslc is conservative; hand-applied no-alias can free up the scheduler 2x.

24. **`spirv-cross` from generated SPIR-V to GLSL to SPIR-V** with a custom pass between — let one compiler optimize what the other missed.

25. **Inject a custom NIR/LLVM pass via Mesa's RADV when testing AMD, then port the schedule back to NV SPIR-V by hand.** Mesa shows you the dependency graph for free.

26. **Manually unroll into a single huge `OpMatrixTimesMatrix`-style block** with no loop control, then audit `VK_KHR_pipeline_executable_properties` to see the issued SASS — iterate until register count is exactly 255.

---

## D. Memory & Buffer Tricks

27. **256-bit loads via `uvec8` and 512-bit via two interleaved `uvec4` loads back-to-back** to chain into single LD.E.128 SASS instructions; ensures the LD/ST unit issues at peak.

28. **`VK_EXT_descriptor_buffer` with descriptor-buffer-resident pointers** to skip the descriptor set update path. Lower kernel-launch overhead matters for small-tile dispatches.

29. **`VK_KHR_buffer_device_address` raw pointer arithmetic** with `buffer_reference_align(16)` for hand-controlled vectorization; bypass the SPIR-V binding model entirely. Already in some shaders — extend it for B-tile.

30. **`push_descriptor` for A,B,C bindings** to eliminate the descriptor pool allocation per dispatch when streaming many GEMMs.

31. **Async global→shared copy via `VK_NV_memory_decompression` or `VK_EXT_buffer_device_address`-driven memcpy emulation** using a dedicated copy queue running in parallel with compute.

32. **Pinned host-coherent memory (`HOST_COHERENT_BIT | HOST_CACHED_BIT`) for the A matrix** if it streams from CPU — PCIe DMA overlaps with compute.

33. **`VK_EXT_external_memory_host` to import a `MAP_HUGETLB` 2MB-page allocation** and reduce TLB pressure on the GPU side — Ampere has GPU-side IOMMU.

34. **`vkGetMemoryHostPointerPropertiesEXT` to share a CPU-mapped buffer with GPU**, then have the CPU pre-pack A while GPU computes — zero-copy overlap.

35. **Use the BAR1 small-PCIe-window for tiny constants (alpha/beta/sizes)** instead of push constants — sometimes faster on Ampere because it's L2-resident.

36. **`VK_EXT_memory_priority` set to MAX on A/B/C heaps** to force them into the fastest VRAM partition; Ampere has heterogeneous memory partitions.

37. **`VK_EXT_pageable_device_local_memory` + lock** to prevent driver eviction of the matrices during long bench runs.

38. **L2 cache prefetch hints via `OpLoad ... NonTemporal`** on B (streaming) and standard on A (reused). Manual cache management.

39. **L1 bypass for store** of C via `Volatile` / `NonPrivatePointer` so the writeback doesn't pollute L1 with one-shot output.

40. **Texture-unit fetch path for A.** Bind A as a `samplerBuffer` (texel buffer) — uses the TEX cache, which is a separate physical cache from L1 data; doubles your effective L1.

41. **`imageLoad` on a 2D `r32f` texture for B** for the same reason — and 2D textures get the texture-cache's 2D-locality prefetch heuristic for free.

42. **Bindless: `VK_EXT_descriptor_indexing` with `RuntimeDescriptorArray`** so a kernel can sweep an array of matrices for batched GEMM without re-binding.

---

## E. Async Compute / Queue Tricks

43. **Two independent compute queues running two halves of the same GEMM** (split-N). Ampere has multiple async-compute-capable queues; can double SM occupancy in some launch corners.

44. **Async copy queue parallel to compute** for prefetching the next tile from HBM/VRAM while current tile computes. Use `VK_KHR_synchronization2` + timeline semaphores to pipeline tile-by-tile.

45. **Persistent kernel that never returns** — workgroups spin on an atomic work-queue index, picking up tile after tile. Amortizes kernel launch and warp warmup.

46. **GPU-driven dispatch: a controller workgroup writes `vkCmdDispatchIndirect` parameters** based on tile completion, eliminating CPU round-trip latency between GEMMs.

47. **Multiple pipelines for the matrix tail.** Main pipeline handles aligned interior, secondary handles ragged edges — submit both to overlapping queues so the edge case doesn't gate the main case.

48. **Timeline semaphores with non-monotonic signal/wait to fence individual K-slices** — overlap K-slice reductions with the next K-slice MACs.

49. **Persistent-thread kernel with sleep-wait via `glsl.std.450 NClamp` busy-loop** on a producer-written ring buffer — pipeline-stage GEMMs across operations.

50. **`VK_KHR_dynamic_rendering` + render-pass scheduler trick.** Sometimes the graphics/compute scheduler co-issues better when interleaved with a no-op render-pass.

51. **Bind to the high-priority queue (`VK_EXT_global_priority`)** to get scheduler precedence — relevant when the desktop compositor is fighting for SMs.

---

## F. Geometry/Raster Hardware Abuse

52. **Rasterize a quad covering MxN, with B passed via instance attributes and A via vertex texture fetch** — the fixed-function rasterizer iterates over (m,n) at hardware speed, fragment shader computes the dot product. Frees up the SMs from indexing.

53. **Use the tessellation hardware to generate output points** with each tess output as a partial sum. Tess engine has its own throughput separate from compute.

54. **Geometry shader with stream output as a scratch** — abusing the GS's small fast buffer as an extra "shared memory" tier.

55. **Conservative rasterization for tile-coverage testing** when doing sparse / blocked GEMM — hardware does the "is this tile non-zero" test free.

56. **Multi-view rendering (VK_KHR_multiview) for batched GEMM.** Each view = one batch element; the multiview hardware replicates dispatches more cheaply than naive batching.

57. **Render to a 32-bit float framebuffer with blending = ADD** — turn matmul accumulation into framebuffer blending. Ampere ROPs do FP32 add at high rate, parallel to SMs.

58. **Stencil-test as a predication mechanism** for masked GEMM — early-stencil culls dead lanes before they hit the shader.

59. **Variable-rate shading abuse**: lower shading rate on the edge tiles, full rate on interior. Saves work on the (already tiny) ragged border.

60. **Use `VK_KHR_fragment_shading_rate` to do "coarse" passes on a downsampled C** for iterative refinement schemes.

---

## G. Driver / Scheduler Hacks

61. **`VK_NV_device_diagnostics_config` + Nsight-extracted SASS** — read what the driver actually issued, hand-pick the best variant.

62. **`VK_KHR_pipeline_executable_properties` to extract register count and reduce until exactly 64 regs/thread** — that's the sweet spot for 2 warps/SM on Ampere.

63. **Compile multiple variants of the kernel, pick at runtime** based on `vkGetPhysicalDeviceProperties2` shader-stat heuristics. Specialization constants drive the variants.

64. **Specialization constants for tile dims, K-vector, register count** so each (M,N,K) shape gets a custom-compiled kernel without per-call recompile.

65. **`VK_EXT_shader_module_identifier` + pre-warmed pipeline cache** so kernel switch is zero-cost during a batched run.

66. **NVAPI `NvAPI_GPU_SetForcePstate` or undocumented memory-clock pinning** to force max P-state during bench — driver sometimes leaves the card in P2 for compute.

67. **Disable display refresh during bench (`xrandr --output ... --off`)** — the compositor steals SM cycles for desktop redraw on a 3070.

68. **`VK_NV_compute_shader_derivatives`** for fragment-shader-style finite differences inside compute — sometimes enables a different SASS path.

69. **Probe `VK_NV_low_latency2`** for spin-wait overhead reduction on submission.

70. **`VK_NV_dedicated_allocation`** on the result buffer to avoid placement heuristics during allocation.

---

## H. Integer / Bit-Manipulation Cores

71. **Repurpose INT32 ALUs.** Ampere has INT32 datapath separate from FP32; emulate FP32 mul via integer mantissa multiply + exponent add. Half-as-fast per op but adds parallel throughput on top of FP32 MAC.

72. **INT8 dot-product instruction (`OpSDotKHR` / `VK_KHR_shader_integer_dot_product`)** for the integer ALUs while FP32 cores do their thing in parallel — co-issue.

73. **Use the bit-manipulation `findMSB` to derive exponents** for a manual softfloat32, distributing the multiply load between datapaths.

74. **`VK_KHR_shader_float_controls` to force flush-to-zero / denormal-as-zero** on FP32 — saves a small but real number of cycles per FMA when denormals would otherwise stall.

75. **`Float32 RoundingModeRTE` removal** (round-to-nearest-anything) — let the FPU skip rounding cycles where allowed.

---

## I. Numeric/Layout Cross-Pollination

76. **K-blocking such that each K-block fits in the constant cache** — push K-slice of B into the uniform/constant memory.

77. **Generate B via on-the-fly decompression from a halved bandwidth (`fp32_to_bf16` quantization)** — half the global-memory traffic, decompress in registers. K-bound stalls become compute-bound.

78. **Z-order tile layout for A and B** so the cache-locality matches the tile-traversal order. Hardware loves Morton-ordered memory.

79. **Megakernel that does GEMM + activation + bias in one launch**, exposing 3x more work for the scheduler to hide latency over.

80. **Force the L2 to be partitioned for "compute-friendly"** via `VK_NV_memory_decompression` or similar L2 hint extension if exposed.

---

## J. Truly Crazy / "What if?"

81. **Train an ML model to pick kernel parameters** (tile sizes, K-vec, specialization constants) per shape — autotuner controlled by a tiny on-device kernel.

82. **Use the PCIe DMA engine for a Strassen-style merge step on CPU** in parallel with GPU base case. Heterogeneous Strassen.

83. **Two GPUs over NVLink-via-PCIe-P2P (RTX 3070 lacks NVLink, but BAR1-P2P via `VK_EXT_external_memory`)** — if user ever runs dual-3070, split GEMM across boards.

84. **Pre-permute A,B at allocation time** so the in-kernel load pattern is sequential within a cache line and we never miss. Off-line cost, online win.

85. **Speculative-execute the next GEMM while the current one finishes** — overlap CPU prep of the next kernel with GPU tail of current.

86. **Use the video-encoder block (NVENC)** as a totally parallel computation engine for something — perhaps a lookup table accelerator for some pre-/post-processing.

87. **Use NV-Optix / RT cores as a sparse-lookup mechanism** for sparse GEMM patterns (BVH-encoded sparsity).

88. **Shader printf via `VK_KHR_shader_non_semantic_info`** during kernel design to count exact issue counts at each PC.

89. **Map a /dev/nvidia-uvm memory region directly** and have the CPU drop tiles directly into GPU L2 (some Ampere parts expose L2 over the PCIe BAR — try it).

90. **Run a tiny "ghost" workgroup that prefetches** the next K-tile via dummy reads — exists only to warm L2 for the real workgroups.

91. **Persistent kernel with cooperative scheduling: warp A computes, warp B prefetches, warp C reduces, all in the same workgroup.** Manual co-routine pipeline.

92. **Set `VK_KHR_workgroup_memory_explicit_layout` to pack shared mem at byte granularity** — squeeze in extra A,B prefetch buffers via tighter packing.

93. **`OpControlBarrier` with `MemorySemantics = None`** — barrier without memory ordering on the LDS, saving a cycle or two per barrier.

94. **Replace global atomics on a partial-sum with a hand-coded `OpAtomicCompareExchange` lock-free deque** that the driver doesn't recognize as atomic and so doesn't insert L2 fences.

95. **Tile-traversal order that hits the same DRAM page repeatedly** before moving — page-open/close on GDDR6 is real overhead.

96. **Manipulate `VK_KHR_shader_clock`** to measure per-instruction latency in-shader; iteratively tune the schedule by measuring what the SASS scheduler picks.

97. **`gl_HelperInvocationEXT`-style helper lanes in compute** if the driver enables it — pre-warmed lanes for first-iteration prefetch.

98. **Hand-write a SPIR-V variant that uses `OpCopyMemorySized` with large size hints** to coax the driver into emitting LDG.E.128.NOCONSTANT.

99. **Force the matrix into the same SM partition via workgroup-to-SM mapping hints** (`VK_EXT_pipeline_creation_cache_control` + scheduler hacks) — Ampere has 4 SM partitions; pin workgroup to one to avoid cross-partition LDS.

100. **Disable ECC on the VRAM via NVAPI** (3070 has soft-ECC) — gets you the missing GB/s of bandwidth that ECC steals.

101. **Memory-bandwidth-test mode**: detect at runtime whether you're compute-bound or BW-bound, and dispatch a totally different kernel (more reuse vs more parallelism).

102. **Replace the FMA with `OpExtInst Fma` explicitly** — sometimes the glslc-emitted `OpFMul` + `OpFAdd` doesn't fuse on the driver-side.

103. **`OpFMul` with one operand a specialization constant** — driver constant-folds at pipeline-create time, freeing an instruction slot.

104. **Run the warmup pass at 99% utilization to push the GPU to max clock**, then start the timed kernel — Ampere boost behaves better when already at max P-state.

105. **Hijack the geometry shader's small per-primitive memory as a 4th tier of cache** for tiny lookup constants (alpha, beta, mask).

106. **Use `VK_NV_shader_subgroup_partitioned`** for sub-subgroup partitions — gives sub-warp-level reductions, finer than `subgroupClusteredAdd`.

107. **Switch between "compute" and "graphics" submission per GEMM** depending on the GPU's current load on the other path — load balance across the scheduler.

108. **One workgroup acts as a "watchdog" that resets stuck warps** via cooperative signaling — combats long-tail tail latency in persistent kernels.

109. **Profile-guided register allocation**: compile the kernel 20 times with slightly different spec consts that drive register usage, pick the one with the best measured TFLOPs (auto-tuner).

110. **Hand-tuned LD/ST address calculation** to ensure all loads in a warp hit the same DRAM bank pattern — minimizes bank conflicts at the DRAM (not LDS) level.

111. **"Anti-occupancy" trick**: deliberately use *more* registers per thread to push occupancy *down* but keep all SM resources dedicated to fewer, bigger warps — sometimes wins on ILP-rich kernels.

112. **`uniform`-qualified buffer for the row of B that the whole warp shares** — driver may use the uniform datapath / constant cache.

113. **L1.5 / texture cache trick: bind B as both a storage buffer AND a sampler** — sometimes the two cache paths combined exceed either alone.

114. **CPU pre-bakes "schedule programs" (PTX-like)** that the persistent kernel interprets — instead of recompiling for each shape, interpret a compact bytecode.

115. **Use `VK_EXT_mesh_shader` task shader** as a producer of work units feeding compute — mesh shader has lower launch overhead than vkCmdDispatchIndirect.

116. **In-place transpose B during streaming load** using subgroup shuffles, so transposed-B GEMM costs nothing extra over normal GEMM.
# Algorithmic Restructuring / Dispatch / Compiler Ideas — SGEMM @60% on RTX 3070

Brain-dump mode. Repetition acceptable. No filter.

## Fast-Matmul Recursion (Strassen / Winograd / Pan)

1. **Strassen-1 at the top, kernel at the leaves.** One level of Strassen reduces 8 → 7 GEMMs on 4 quadrants. If kernel hits 60% peak today, top-level Strassen gets us ~70% effective throughput at the cost of 18 extra add/sub kernels. Cheap if N≥2048.
2. **Strassen-2 (two recursion levels).** 49 sub-GEMMs of M/4 × N/4. Memory traffic for sums dominates below N=4096 — but at N=8192 it should pay off massively.
3. **Winograd's variant of Strassen.** Same 7 multiplies but fewer additions (15 vs 18). Smaller temp buffer footprint — better L2 residency for the inner GEMMs.
4. **Pan's <2.78 algorithm.** 3×3 → 21 multiplies (vs 27 naive). Heavier add overhead but maybe worth it for square N=3k matrices.
5. **Adaptive Strassen depth.** Pick depth so each leaf GEMM is exactly the kernel's sweet-spot tile (e.g., 1024×1024). Make depth a function of input size.
6. **Strassen-then-naive on irregular shapes.** Pad to next power-of-2 only for the top-level split; pass the unpadded leaves through normal kernel which knows about tails.
7. **In-place Strassen.** Allocate the 7 temp matrices as overlapping aliases of A, B, C tiles. Cuts memory bandwidth by ~3x for the add/sub phases.
8. **Fused Strassen add/sub kernels.** Combine the 18 adds with the next GEMM's prologue: compute (A11+A22) on-the-fly inside the matmul shader load.
9. **Karatsuba-style 2x2 matmul** as building block: 7-mul scheme used at register-tile granularity inside the shader, not the dispatch level. Per-thread does Strassen recursion on its 8x8 fragment.
10. **AlphaTensor-discovered 4x4 schemes.** DeepMind found 47-multiplication 4×4 matmul over Z2; for floats, use Smirnov-style algorithms with 48-49 multiplications. Build register-tile from these.
11. **Hybrid arithmetic-complexity stack.** Top of dispatch tree: Strassen. Middle: Pan. Leaves: tensorcore-shaped naive. Each layer optimised independently.
12. **Smirnov 3x3x6 algorithm.** Rectangular fast algorithm for non-square inner shape. Useful for tall-skinny K dimension.

## Split-K & K-Reduction Restructuring

13. **Split-K with atomic adds to C.** Partition K into 8 chunks, dispatch 8 GEMMs in parallel, each does atomicAdd to the output tile. Trades atomics overhead for K-direction parallelism on small K matrices.
14. **Split-K with separate buffers + reduction kernel.** Each chunk writes to its own slab of C', then a tiny reduction shader sums slabs. No atomic contention.
15. **Workgroup-tree K reduction.** Like NCCL ring-reduce but inside one dispatch. Workgroups arranged in a tree, leaves do GEMM chunks, internal nodes wait via timeline semaphore and sum.
16. **Kogge-Stone parallel prefix for K accumulation.** Re-associate the K loop as a prefix-sum tree to fully expose ILP. Inside-shader version; eliminates the accumulator dependency chain.
17. **Multi-buffered K accumulator.** Keep 4 separate fp32 accumulators per thread, alternate which one we add into per K iteration. Reduces RAW latency to ~1/4. (FMA dependency chain breaker.)
18. **Pairwise-summation K loop.** Tree-reduce K terms in pairs for numerical stability AND parallelism. Helps tail latency hiding.
19. **Two-stage K split.** Outer split-K of 4, inner split-K of 4. 16-way parallel reductions with hierarchical aggregation. Mimics how MPI implementations do all-reduce.
20. **Split-K with persistent reduction WG.** One dedicated workgroup just sits and consumes partial sums via a ring buffer in DRAM. GEMM WGs produce, reducer consumes.

## Persistent Kernels & Dispatch Strategy

21. **Persistent grid.** Launch exactly num_SMs * waves_per_SM workgroups. Each WG runs an internal loop popping work tiles from a global atomic counter. Amortises launch overhead, lets you overlap C-store with next-tile load.
22. **Persistent kernel with work-stealing.** Initial round-robin assignment; idle WGs steal from a heavy WG's queue via atomic compare-exchange.
23. **Mega-kernel = matmul + activation + next matmul prologue.** One persistent kernel does fused FFN. Same launch, multiple operators.
24. **Z-order (Morton-order) tile dispatch.** Issue tiles in Morton sequence so spatially adjacent tiles execute near in time → better L2 reuse on shared A/B rows/cols.
25. **Hilbert curve dispatch.** Even better locality than Morton because Hilbert preserves neighbor-adjacency. Compute index via lookup table in push-constants.
26. **Diagonal dispatch order.** Iterate tiles along anti-diagonals: tile (i,j) where i+j=k. Maximises sharing of A rows and B columns in L2 between concurrently-running WGs on different SMs.
27. **Spiral dispatch.** Tiles emitted in expanding spirals from the matrix center. Empirical L2 win on some architectures.
28. **Custom tile-to-SM affinity.** Use Vulkan device-group / subgroup ID to bind tile coordinates to specific SMs that share L2 slices. Pin reuse to physical L2.
29. **Dispatch indirect with culled tiles.** For sparse outputs (e.g., mask known), build dispatch indirect buffer at runtime, skip zero tiles entirely.
30. **Single dispatch, 3D workgroup grid.** Encode (M-tile, N-tile, K-split) into z-axis. Lets driver pack waves better than three sequential dispatches.

## Multi-Tile / Heterogeneous

31. **Per-region tile size.** Big tile (256×256) in the interior; small tiles (32×32) on the edges. One dispatch with shader branching on tile-coord push-constant.
32. **Two SPIR-Vs, one render pass.** Big-tile pipeline for bulk, small-tile pipeline for tails. Driver schedules both on the same queue with sub-allocation.
33. **Speculative dual-tile dispatch.** Launch BOTH tile-shape A and tile-shape B; whichever finishes first wins; cancel the other via timeline semaphore. Wasteful but predictable latency.
34. **Hot-tile, cold-tile.** Identify which tiles are bandwidth-bound (corners with less reuse) vs compute-bound (interior). Different shader per category.

## Shape-Specialised SPIR-V

35. **One SPIR-V per (M,N,K) hash.** Pre-compile a lookup table of ~256 common shapes. First-call latency hides behind warmup.
36. **Runtime SPIR-V codegen via rspirv.** Build optimal kernel on first call for never-seen shape; cache to disk keyed by shape hash. Like cuBLAS's heuristic-fitted kernels.
37. **Shape-templated unroll factors.** For shapes where K is statically known, fully unroll the K loop. Eliminates loop overhead entirely; SPIR-V optimizer goes wild.
38. **Constant-fold the bounds checks.** When M,N,K are statically known multiples of the tile size, remove all bounds-check ifs at codegen time.
39. **Per-shape pointer-stride bake.** Encode lda/ldb/ldc as compile-time constants in the specialised SPIR-V; saves a register and a multiply per iteration.
40. **Specialisation constants for tile size.** Use SPIR-V SpecConstantOp so one SPIR-V file becomes N shaders at pipeline-creation time with no source duplication.

## Recursive / Hierarchical / Polyhedral

41. **Three-level blocking: 512×512 → 128×128 → 8×8.** Outer = L2 block, middle = shared-mem block, inner = register block. Tune all three jointly with autotuner.
42. **Recursive cache-oblivious matmul.** Split largest dimension in half, recurse. At leaf size = shared-mem capacity, switch to GEMM kernel. CPU technique but transferable to GPU L2.
43. **Polyhedral schedule from isl / Pluto.** Hand-write the iteration polyhedron, let Pluto generate optimal tile + skew. Translate to SPIR-V.
44. **Affine transformation: skewing.** Skew the (i,j,k) iteration space so memory accesses to A and B land on the same L2 set. Reduces conflict misses.
45. **Loop fission of the K loop.** Split K loop into "load phase" loop and "FMA phase" loop in separate basic blocks. Helps SPIR-V scheduler hide loads.
46. **Loop fusion across batched dispatches.** Two consecutive matmuls (e.g., FC1 then FC2) become one fused kernel with intermediate in registers/shared.
47. **Loop peeling for prologue/epilogue.** Peel first and last K iterations; their distinct register pressure lets you do prefetch-on-prologue, drain-on-epilogue patterns.

## Iterative / Mixed Precision

48. **Iterative refinement: FP16 multiply + FP32 correction.** Compute D ≈ A×B in FP16 fast, then compute residual A×B - D in FP32 on the error. 1.5× total work but uses tensor cores for 95% of it.
49. **Bit-banding adaptive precision.** Per-tile, estimate output magnitude; if all values < 2^k for small k, use FP16; if huge dynamic range, use FP32. Lookup table chooses kernel.
50. **Posit / Bfloat-tile speculative.** Speculatively do the matmul in BF16; if any output overflows the "safe" range, redo that tile in FP32.
51. **Block-fp scheme (a la AMX).** Group K-rows of A into blocks sharing one exponent, store mantissas as 8-bit. Re-introduce exponent at end-of-K.

## Transforms / Algebraic Tricks

52. **FFT-based matmul.** For K ≥ 1024, A×B = IFFT(FFT(A_padded) ⊙ FFT(B_padded)) along K. Doesn't actually compute GEMM but a related convolution; usable for circulant matrices.
53. **Number-theoretic transform** instead of FFT for exact-integer-via-block-fp tricks. Avoids floating-point FFT precision issues.
54. **Tensor-train decomposition of A.** If A factors as a TT-train (low TT-rank), replace A×B by chain of small matmuls. Big win on rank-deficient weight matrices.
55. **Hierarchical Tucker decomposition.** Same idea, tree structure of cores. Useful for transformer weights after distillation.
56. **CP decomposition cache.** Decompose A once, reuse for many B's. If rank << M,N, this is asymptotically better.
57. **Random-projection sketch.** Project A to A_s = AS (with sketch S), do A_s × B, then C ≈ AS·S^T·B. Approximate but stupidly fast for tall matrices.
58. **Sparse-pattern detection.** Scan A for >50% zero tiles; switch to CSR-block matmul for those. One dispatch handles mixed sparse/dense.
59. **Two-pass with column-max normalisation.** Pass 1: compute per-column max of |A|. Pass 2: scale A by 1/max, do matmul in narrower range, rescale C. Lets you use FP16 safely.
60. **Two-pass row-norm precompute.** First pass writes row norms to a side buffer; second pass uses them for clever Strassen-stability fixes.

## Caching / Layout / Lazy

61. **Inter-call A-tile cache.** Hash A's pointer + version; if last call also used this A, skip A pre-transpose / pre-pack. Save bandwidth.
62. **Persistent packed A in VRAM.** Pack A into the kernel's native register-tile layout once, reuse across calls. Like cuBLASLt's "preferA" mode.
63. **Lazy multiply graph.** Build a DAG of pending matmuls; defer execution. When forced, fuse adjacent ones via Strassen-add merging.
64. **Recipe-based tensor.** A is stored as "result of X × Y"; defer materialising A; instead compute A·B as X·(Y·B). Cheaper if Y narrow.
65. **Operator fusion: matmul + bias + ReLU + matmul.** Single mega-kernel, intermediate stays in registers/L1. The second matmul's tile dimension matches the first's output tile.
66. **Streaming A from DRAM with prefetch.** Use Vulkan transfer queue to copy next A-block while compute queue chews current one. Async copy hides bandwidth.

## Batched / Cross-call Fusion

67. **Pack many small matmuls into one big block-diagonal.** N×(small) becomes one (big) matmul with block-diagonal structure. Wastes some work but amortises launch.
68. **Persistent batched-GEMM kernel.** One kernel iterates over the batch axis internally, never returns to host. Tile loop nested under batch loop.
69. **Variable-shape batch.** Sort the batch by size; persistent kernel processes them in order so similar-size tiles share warps.
70. **Cross-batch K-fusion.** If batch[0] and batch[1] share A, do one bigger matmul C = A × [B0|B1] instead of two separate calls.
71. **Stream-K dispatch.** Instead of split-K or split-M/N, give each WG a sequential slice of total work. Best load balance for any shape. (NVIDIA paper.)

## Compiler / Codegen Hacks

72. **Hand-write the SPIR-V** for the hot kernel; bypass glslc's register allocator entirely. Use spirv-asm directly.
73. **Custom SPIR-V optimizer pass.** Run spirv-opt with hand-tuned pass list: loop-invariant code motion, scalar replacement of aggregates, FMA combining. Skip dead-code passes that hurt.
74. **AMD's RDNA-style ifetch unrolling.** Manually duplicate the hot K-loop body 4× and rotate accumulator registers. Hides FMA latency at zero ICache cost (until L0 ICache fills).
75. **Outline cold paths.** Move bounds-check fallback path to a separate SPIR-V function; mark with NoInline. Hot loop stays cache-resident.
76. **Function call elimination via specialisation.** Generate fully-inlined kernel variants where ALL function calls are flattened. Increases SPIR-V size but kills jump overhead.
77. **Loop interchange driven by L2-line stride analysis.** Reorder (i,j,k) to minimise stride crossings of 128B L2 lines.
78. **Register tile rotation.** Rotate which register holds which fragment per K iteration to spread the FMA dependency chain across different physical register banks.
79. **Software pipelining with modulo scheduling.** Pre-issue loads N iterations ahead; explicit phase rotation in the K loop body.
80. **Predicated-execution K loop.** Convert the last-iteration branch into a predicate so all warps stay in lockstep.

## Async / Queue Pipelining

81. **Two compute queues, ping-pong tiles.** Queue A computes tile T while Queue B is loading tile T+1. Vulkan timeline semaphore between them.
82. **Transfer queue prefetches next-tile A,B.** Use a separate transfer queue with DMA to overlap HBM read with compute. Treat L2 like a software-managed cache.
83. **Triple-buffered SSBOs.** A0,A1,A2 buffers ringed. While compute reads A0, transfer fills A2, A1 is in flight.
84. **Submit-time tile chaining.** Pre-record N command buffers, each one tile, chain via timeline. Driver can fuse and reorder for occupancy.
85. **Sub-allocator at command-buffer granularity.** Tiny command buffers per tile, recorded once, reused. Cuts host-side overhead to ~nothing.

## Wild / Speculative / Reordering

86. **Two-step: index-only pass, value pass.** Pass 1 writes which K-blocks will dominate (e.g., via row-norm sampling); pass 2 does only the heavy ones first. Useful for early-termination if threshold-based.
87. **Speculative top-k truncation.** Estimate that 95% of contribution comes from top 10% of K-rows; compute those first, check error, only do the rest if needed.
88. **Random-K shuffling.** Permute K dimension; if you only need approximate result for first pass of optimiser step, stop early.
89. **Mixed-radix K loop.** Split K=K1·K2; outer loop runs in one shader, inner loop in another via dispatch indirect. Heterogeneous tile size between inner and outer.
90. **K-split with importance weighting.** Heavy K-chunks get dedicated WGs; light K-chunks get packed together. Like graph partitioning the work.

## Hierarchical Recursive Madness

91. **Strassen at dispatch level + Strassen at register tile.** Two-level multiplicative savings: 7/8 × 7/8 ≈ 0.77 of arithmetic. Need to be VERY careful with error propagation.
92. **Recursive divide-and-conquer with cache-oblivious leaf size.** No autotuner needed for leaf size if leaves are cache-oblivious (≈ Frigo et al.).
93. **Stream-K + Strassen.** Apply Stream-K dispatch policy to Strassen's 7 sub-products. Best of both worlds for load balance + fast-matmul.
94. **Fold the bias-add into the Strassen reconstruction.** The 18 add/sub steps at the end of Strassen already touch every output element; bake bias addition into them.

## Misc / Out-There

95. **GPU-side autotuner.** First-call dispatch runs a tiny probe kernel that measures L2 hit-rate, then chooses the actual GEMM kernel. All within ~10µs.
96. **Reinforcement-learning kernel selector.** Train a tiny MLP to pick tile/dispatch params from shape features. Inference in <1µs on CPU.
97. **JIT-fuse adjacent dispatches at command-buffer record time.** Inspect the recorded command stream, replace consecutive matmuls with a fused mega-kernel.
98. **Asymmetric tile shape selection per axis.** M-direction tile = 128, N-direction = 256, picked by which axis has the longer reuse stride. Anisotropic.
99. **Subgroup-sized tile dispatch.** Each subgroup (32 threads) is one tile. Persistent at subgroup granularity rather than workgroup. Lets you pack 4-8 tiles per WG.
100. **Diamond dispatch order.** Issue tiles whose Manhattan distance from a center tile is monotonically increasing. Maximises L2 reuse around the "hot" diagonals.
101. **Cross-pipeline barrier elision.** Detect that consecutive matmul dispatches don't actually conflict (different output buffers); drop the barrier; let GPU overlap. Run-ahead by one dispatch.
102. **Per-warp K offset.** Different warps in a WG start at different K offsets and meet in the middle. Eliminates the K=0 cold-start phase having all warps stall on the same L2 fetch.
103. **Bidirectional K iteration.** Half the warps iterate K=0→K/2, other half K-1→K/2. Halves the dependency chain length on the accumulator.
104. **Algebraic mat-vec fast paths.** When B is rank-1 (vector outer product), use a custom kernel that doesn't pretend it's a GEMM. cuBLAS doesn't do this; we can.
105. **Permutation pre-multiply.** Find a row/column permutation that puts the heavy elements diagonally → naive matmul has shorter dependency chains. Pre-compute permutation once per A.
106. **Givens rotation pre-conditioning.** For numerically-pathological A, apply small rotations that improve cache-line alignment of the dominant entries.
107. **Bit-reversal permutation of K.** Just like FFT — reorder K so adjacent iterations touch L1-friendly addresses. Free in cost.
108. **Cyclic distribution of K across warps.** Warp w does K = w, w+W, w+2W,... Reduces shared-mem bank conflicts in B-load if B's stride aligns.
109. **Power-of-2-rounded dispatch + masked tail.** Always round dispatch grid up to next power of 2, mask out the tail. Driver loves clean shapes; cost of mask is tiny.
110. **Encode tile coords in the SSBO offset, not push-constant.** Save 4 push-constant bytes per dispatch; lets you cram more useful data (e.g., split-K offset) in the push-constant.

## Compiler-Level / Polyhedral Deeper Dives

111. **Generate kernel via Halide-like scheduling language.** Define algorithm + schedule separately; explore the schedule space; compile to SPIR-V. Build a tiny Halide-for-Vulkan.
112. **TVM-style auto-scheduling.** Use Ansor's cost model to find optimal schedules per shape. Output is SPIR-V.
113. **MLIR linalg-on-tensors → SPIR-V via IREE.** Use IREE's compiler pipeline; replace its kernel with ours for fair comparison; learn from its fusion decisions.
114. **Egraphs for matmul rewriting.** Use egg to find equivalent algebraic forms of the K reduction with shorter critical path.
115. **Symbolic loop bounds with z3.** Feed loop bounds to SMT solver; prove safety of aggressive bounds-check elimination; emit unchecked SPIR-V.

## Cross-Cutting "Take Two and Average"

116. **Two-version A/B test in production.** Always dispatch the current "champion" and a candidate kernel on disjoint quarters of the matrix; compare runtimes; promote winner. Online learning.
117. **Tile shape ensemble.** Dispatch the same tile with 3 different shaders; take whichever returned first; the others are wasted but you got predictable lower-bound latency.
118. **Memoised matmul.** Hash (A,B) tuples; if you've seen them before (e.g., RNN with frozen weights), return cached C. Pathologically narrow but free real-world win.

## Final Pile

119. **Encode the entire matmul algorithm as a giant unrolled SPIR-V.** No loops, just (M·N·K)/tile_size FMAs straight-line. SPIR-V will be MB-sized but the scheduler will go to town. Viable for tiny matmuls (<128).
120. **Generate one SPIR-V per autotuner-discovered configuration.** Persist to a SQLite cache on disk. Library boot time: 50ms. First-call after install: 30ms. Steady-state: instant.


---

# Round 2: Wild creative angles (added later)

Ten more brainstorm agents covering quantum, biological, cryptographic,
physical, programming language theory, visual, game-dev, audio, distributed,
and pure-unhinged angles.  Strict no-feasibility-filter policy.

---

# Quantum / Superposition Analogies for SGEMM at Hardware Peak

Context: Vulkan FP32 SGEMM, RTX 3070, 60% peak, hunting every TFLOP. No feasibility filter — wild, half-baked, possibly physically impossible. Even the broken ones reframe the problem.

---

## I. Superposition / Measurement

1. **Qubit-per-element matmul.** Treat each A[i,k] and B[k,j] as a "qubit" in superposition over the K-axis. One FMA = one Hadamard-then-measurement that collapses the K-dimension. Maps onto warp-wide shuffles where the "measurement basis" is which lane keeps the partial sum.

2. **Delayed collapse accumulators.** Keep the accumulator in superposition of (high, low) Kahan halves until the final write. The compensation term IS the "off-axis" amplitude. Spills to vector regs only on collapse — fewer register pressure spikes.

3. **Wave function = L1 cache state.** Each line is in superposition of "useful" / "evictable" until a load reads it. The collapse rule is "the access pattern wins." Implement as a predicted-access bloom filter that nudges the LRU policy.

4. **Decoherence-as-rounding.** A virtual fp64 partial sum decoheres into fp32 after T accumulation steps (T tuned by chip noise model = chip variance in clock domain). Periodic snap-to-fp32 reduces the working set and might let us fit one more tile in shared memory.

5. **Measurement-basis FMA.** Choose the measurement basis (the K-decomposition order) per warp based on observed lane occupancy. Effectively a basis-rotated dot product where the rotation is free because it's a register name swap.

6. **Many-worlds tile execution.** Speculatively dispatch 4 candidate tile layouts in parallel using different warps inside a workgroup; the first to finish writes the result, the others self-cancel via an atomic predicate. Branch the universe, collapse on first success.

7. **Heisenberg tradeoff for tile size.** You cannot simultaneously know "best occupancy" and "best register reuse" — pick which axis to be sharp on per shape. Use the uncertainty product to motivate a dual-kernel autotuner with shape-dependent regime switching.

---

## II. Tensor Networks (MPS / MERA / PEPS)

8. **MPS approximation for low-rank intermediates.** When A·B contains a low-rank block (common in attention K·V), factor it as a matrix-product state on the fly and stream the bond dimension. Trades exact arithmetic for ~3x fewer FMAs on detected blocks.

9. **MERA-style hierarchical tiling.** Tile the matmul at log levels: 4x4 micro, 32x32 warp, 128x128 block, with renormalization "disentanglers" between levels = the transpose/reshape between shared memory and registers. Frame the existing hierarchical tile cascade as a unitary MERA.

10. **PEPS 2D contraction order.** Treat the C output grid as a 2D tensor network of partial sums. The optimal contraction order is NP-hard, but Monte Carlo PEPS heuristics give near-optimal dispatch orderings of subgroup tiles for irregular shapes.

11. **Bond-dimension throttling.** Cap the rank we accumulate before flushing to global memory — a "truncated SVD" applied to the accumulator before writeback. Lossy, but for inference workloads the bond dimension is often << min(M,N).

12. **DMRG sweep as kernel pass.** Sweep tiles left-to-right then right-to-left over the K-axis, each sweep refining the accumulator. Caches the "environment tensor" in shared memory across passes — could be faster than naive single-pass when K is huge and registers spill anyway.

---

## III. Amplitude Amplification / Grover

13. **Grover-style sparsity hunt.** Amplify the amplitude on B-tiles with nonzero entries, suppress on dense-zero tiles. Implement as a 2-pass kernel: pass 1 builds a bitmap, pass 2 dispatches only "amplified" tiles. Real win for ReLU-sparse activations.

14. **Quantum-inspired importance sampling.** Sample K-axis indices proportional to |A[i,k]|·|B[k,j]| (the "Born rule"), do a partial sum, scale up by 1/p. Gives a stochastic SGEMM that converges fast for low-precision intermediate layers.

15. **Amplitude amplification for tile order.** Iteratively reweight tiles by "likely to hit cache" until convergence — Grover's sqrt(N) over the dispatch queue. Implemented as a learned permutation table per shape class.

16. **Speculative N-way tile decomposition.** Try 8 tile decompositions in parallel warps with a small probe load, collapse to the fastest after one inner iteration. Like Grover's oracle is "did the probe hit cache?"

17. **Diffusion-operator dispatch.** After each wavefront of tile launches, run a "diffusion step" that redistributes pending tiles toward under-utilized SMs. The diffusion matrix is the Grover-style inversion-about-the-mean of the SM occupancy vector.

---

## IV. Quantum Walks

18. **Quantum walk on the dispatch grid.** A walker (next-tile pointer) at each SM moves with amplitudes biased toward neighbors with hot caches. Interference between walkers from neighboring SMs naturally avoids dispatching the same tile twice.

19. **Self-routing dispatch via Szegedy walks.** Model the (M,N)-output grid as a graph, walk it with a unitary that favors tiles sharing rows/cols with the just-completed tile. Free temporal locality without an explicit scheduler.

20. **Coined quantum walk for K-reduction.** The "coin" decides which two partial sums to merge in a parallel reduction tree. A biased coin (skewed by lane-load) makes the reduction tree adapt to register pressure on the fly.

21. **Continuous-time quantum walk = Laplacian smoothing of work assignment.** Solve the heat equation on the dispatch graph at compile time to get steady-state SM loads, use those as a static schedule. Pre-baked nirvana for fixed shapes.

22. **Random-walk hitting-time tile scheduling.** Order tiles by expected first-hit time on a random walk over the dependency DAG. Roughly: process "central" tiles first to maximize downstream reuse.

---

## V. Annealing / Phase / Topology

23. **Quantum annealing for kernel autotune.** Cast (tile-M, tile-N, tile-K, warp-tile, unroll) selection as Ising minimization of measured cycles. Run simulated quantum annealing offline; ship the schedule table. Standard but framed as Ising lets us add couplings (e.g., "if tile-K=8 then unroll=4").

24. **Adiabatic kernel switching.** Smoothly interpolate between two known-good kernels over the first few warp launches; the optimum kernel is the ground state we adiabatically reach. Probably equivalent to a learned mixer, but the framing suggests warm-up phases.

25. **Phase-estimation accumulator.** Replace fixed-point accumulation with phase accumulation around a complex unit circle — the matmul result lives in the argument. Wrap-around is "obviously broken" for FP32 magnitudes, but the phase can encode the small-residual error that Kahan currently chases.

26. **Topological-anyon reduction.** Pretend partial sums are anyons; braiding them = commutative-associative reduction. The braid group structure tells us that ABCD and ACBD give different fp32 results but topologically identical — so we can pick the braid order minimizing register lifetime.

27. **Stabilizer codes for accumulator ECC.** Encode the accumulator across 5 lanes in a [[5,1,3]] code. Detect and correct single-lane bit flips for free using parity FMAs. Useless on RTX 3070 (ECC-off), but the framing gives us a "redundancy budget" we can spend on Kahan instead.

---

## VI. Entanglement / Teleportation Between SMs

28. **Pre-shared entangled tile cache.** Before dispatch, broadcast a small "shared seed" tile of A across all SMs that need it; subsequent loads "teleport" only the per-SM residual. Maps to a global-memory broadcast + per-SM delta-load — could be a real win when A is reused across many SMs (e.g., batched matmul with shared weight).

29. **Bell-pair tile streaming.** Pair each producer SM with a consumer SM. The producer's last write IS the consumer's first read, via a shared L2 line. Requires explicit SM affinity scheduling, but Vulkan subgroups + persistent threads can fake it.

30. **Superdense coding for tile metadata.** Pack 2 bits of tile-flag info into the sign bits of an FP32 transfer — "free" 25% bandwidth gain on metadata-heavy fused ops (bias + activation).

31. **No-cloning theorem as a load-cap.** Forbid two warps in the same workgroup from loading the same tile region. Forces a producer-broadcasts-via-shared pattern; probably what the autotuner already learns but framed crisply.

32. **GHZ-state reductions.** A 3-way reduction across 3 subgroups using a 3-qubit GHZ analogue: one global atomic plus two register-only contributions. The "entanglement" is the agreement protocol on which lane owns the final write.

---

## VII. Speculation / Many-Worlds Branch Prediction

33. **Many-worlds tile ordering.** Spawn 4 micro-kernels with different K-loop orderings on a single output tile. Probe-measure 10% in, kill 3 of them. The "wavefunction collapse" is a barrier + winner-write.

34. **Speculative prefetch via path integral.** Sum over all "future" load orderings weighted by predicted cache reuse. Equivalent to a learned prefetcher, but the path-integral framing suggests we want to penalize destructive interference (two prefetches knocking each other out of cache).

35. **Retrocausal hint injection.** At kernel exit, write back a hint into a small per-SM "future tile" register that the next kernel reads. Lets a kernel "tell its past self" the optimum tile size for the previous shape — autotune feedback at kernel granularity.

36. **Quantum Zeno effect for warp stalls.** Frequent "measurement" (i.e. branch on occupancy counter) freezes warp evolution and prevents bad scheduling decisions from progressing. Probably equivalent to inserting `__syncwarp()` but motivates *exactly when* to insert it.

---

## VIII. HHL / Quantum Linear Algebra / Misc.

37. **HHL-inspired matrix inversion as preconditioner.** When the matmul is part of a linear solve, run a classical HHL-like routine (state prep via QR, eigenvalue inversion via phase, uncompute) — but on the SM. The bottleneck becomes the phase-estimation step, which maps to a Cooley-Tukey FFT and is bandwidth-bound. Reframes: maybe matmul should be a side effect of an FFT.

38. **Quantum-walk-based matmul (Apers et al. style).** No real speedup for dense matmul (Aaronson result), but the bound says: a constant-factor speedup, if it exists, can only come from reducing data movement, not arithmetic. Tells us where to look.

39. **Measurement-based quantum computation = persistent-thread fused kernel.** A fixed "cluster state" = a fused dispatch graph; "measurements" = warp activations. Argues for kernel fusion across attention QKV → softmax → V multiply because the MBQC fabric admits arbitrary computation by routing alone.

40. **Quantum-error-correcting surface code over tile grid.** Lay out output tiles on a 2D surface code; "ancilla qubits" are checksum tiles computed by neighbors. Detects bad accumulations from rare denormals. Probably overkill but suggests: cheap row/column parity FMAs.

41. **Quantum supremacy random-circuit sampling = stress kernel.** A random matmul shape generator that targets the chip's noise spectrum to find tile sizes where the chip's performance has the *most variance*. Then avoid those shapes (or use them as canary autotune probes).

42. **Topological compilation: anyon braiding = wgmma/mma pipeline.** Frame the existing FMA-issue / shared-load / register-rotation pipeline as a braid. Loop-unrolling optimal braid words might give 1-2% from better instruction scheduling.

43. **Quantum Fourier matmul.** FFT-based matmul (Strassen-territory). On RTX 3070 with no tensor cores helping FFT, this loses on f32, but for power-of-2 shapes the asymptotic crossover is around N=8192 if we can fuse the FFT with the multiply.

44. **No-go theorem framing.** Aaronson-style argument: any matmul speedup beyond the arithmetic intensity bound must save *memory traffic*. The quantum lens makes it explicit: stop chasing FMAs, chase loads. Already obvious but worth carving in stone.

45. **Stinespring dilation = persistent threads.** Embed the matmul (a non-unitary channel) into a larger unitary by adding ancilla state = persistent thread blocks that maintain partial-sum buffers across many small matmuls. Useful for shape-irregular workloads where launch overhead dominates.

46. **Decoherence-free subspace = bank-conflict-free shared layout.** Find the shared-memory layout where parallel loads naturally do not interfere — the "noise-free subspace" of the shared memory bank graph. Standard swizzle reframed.

47. **Quantum-walk-based prefetch into texture cache.** Use a Hadamard walk over the K-axis to schedule prefetches into the texture cache (read-only). The Hadamard mixing guarantees we never thrash the same line twice in a window.

48. **Berry phase per tile.** Each tile traversal accumulates a "geometric phase" = the bank-conflict cost of its load pattern. Pick traversal orders that close a loop with zero Berry phase = zero residual conflicts. Probably equivalent to Hilbert-curve ordering but motivates *why* it works.

49. **Quantum cellular automaton schedule.** Each SM has a small CA rule deciding its next tile based on neighbor states (broadcast via L2). Self-organized criticality might give near-optimal load balance at zero runtime cost.

50. **Reverse quantum-walk gradient.** Run an *imaginary-time* quantum walk on the kernel's autotune landscape — converges to the global minimum faster than gradient descent for non-convex tile-search spaces. Equivalent to ITE-style optimization but maps to standard offline tuning.

---

## IX. The Honest Reframes

51. **What quantum gives us as a lens, not a tool:** every quantum analogy that works for SGEMM has a classical equivalent — but the analogy *points at memory traffic and ordering*, not arithmetic. The 40% peak we're leaving is not in math; it's in the order we do the math.

52. **The real prize is interference, not superposition.** Loads from neighboring SMs *destructively interfere* on L2; we want *constructive interference* — coordinated access patterns. Frame the autotuner as maximizing constructive interference on the load fabric.

53. **What if quantum gave us nothing?** Even null result useful: it argues we are at the *classical* arithmetic limit for our chosen algorithm and must change algorithm (Strassen, low-rank, sparse) rather than tune harder.
# Biological / Neuromorphic / Evolutionary / Swarm Ideas for FP32 SGEMM on RTX 3070

Pure creative dump. No feasibility filter. Each idea maps a biological metaphor to a concrete Vulkan/GPU primitive.

---

## Neuromorphic / Spike-Timing

1. **Spike-timing matmul.** Encode each A row as the *phase offset* of a writeback into shared memory; B columns arrive with their own offsets. A multiply becomes a coincidence detection: subgroup `subgroupBallot` fires only when two clocks align within ±N cycles. Wrong for dense GEMM but interesting for ternary {-1,0,1} accumulation.

2. **Refractory periods on shared-mem banks.** After a thread writes to bank `b`, it cannot touch bank `b` for 31 cycles — forces a natural conflict-free swizzle without an explicit XOR. Use a per-bank cooldown counter in shared memory.

3. **Dendritic accumulation tree.** Build the K reduction not as a flat warp-shuffle sum but as a *dendritic branching tree* where each "neuron" (lane) integrates inputs with leak constants. Useful if A is sparse — leaks discard near-zero contributions.

4. **Membrane potential as accumulator.** Treat the FMA accumulator as a leaky integrator: every N MACs, multiply by `(1 - epsilon)` to avoid float accumulation drift. Effectively Kahan summation disguised as a biological process.

5. **Lateral inhibition between tiles.** Adjacent thread tiles compete for L1 bandwidth — a tile that misses too often "inhibits" its neighbours into delaying their next load via a shared counter. Maps to throttling memory pressure via subgroup atomics.

6. **STDP-tuned prefetch distances.** Spike-Timing Dependent Plasticity: each prefetch is a presynaptic spike, the eventual MFA is the postsynaptic spike. Adjust prefetch distance based on the timing gap recorded in a sidechannel buffer between dispatches.

---

## Sparse Coding / Dictionary

7. **Dictionary of tile shapes.** Maintain 8 "atom" kernels (each a different MxNxK tile shape). At runtime, decode the input dimensions into a sparse combination of dictionary atoms and dispatch the matching kernel for each.

8. **Sparse coding of B columns.** Project B onto a learned dictionary of 256 atoms at load time; multiply A against atom outer-products instead of raw B. Pure waste for dense GEMM, possible win if user GEMMs are repeated and structured.

9. **K-SVD kernel selection.** Offline, run K-SVD on observed workload shapes to find the 16 most representative tile configurations. Online, dispatch the nearest match via L2 distance in dimension space.

10. **Atom firing rate = pipeline depth.** Each "atom" in the sparse dictionary corresponds to a different pipeline depth (1, 2, 4, 8 outstanding loads). Pick the atom whose firing rate matches the current bandwidth/compute ratio.

---

## Hebbian / Adaptive Kernel

11. **"Cells that fire together wire together."** Track which (M,N,K) input triples recur in the same process; warm a specialized JIT-compiled kernel for those shapes after the third occurrence. Pheromone-like memory in a persistent SSBO.

12. **Hebbian weight on dispatch order.** If tile (i,j) was followed by tile (i,j+1) in the last 100 dispatches, strengthen that ordering by biasing the workgroup ID mapping. Effectively learns Z-order or Hilbert curves from access logs.

13. **Anti-Hebbian eviction.** Tiles that *don't* fire together get evicted from a persistent shared-memory cache earlier. Maps to a small LRU but biased by co-occurrence rather than recency.

14. **Long-term potentiation in driver.** After 1000 calls with the same shape, "burn in" a specialized kernel into the pipeline cache permanently. Short-term plasticity = uop cache; long-term = SPIR-V variant cache.

---

## Evolutionary / Genetic

15. **GA on tile shape genome.** Encode (BM, BN, BK, TM, TN, num_warps, prefetch_depth) as a 7-gene chromosome. Mutate during idle dispatches, run 5 candidates per generation, keep top 2. Fitness = TFLOPS measured on a calibration shape.

16. **Crossover of two good kernels.** Take BM from kernel A and TN from kernel B, splice into a child SPIR-V variant. Use SPIR-V specialization constants as the gene loci.

17. **Island model.** Run 4 isolated populations of kernel variants on 4 different shape regimes (small/medium/large/skewed). Migrate top performer between islands every 100 dispatches.

18. **Lamarckian tuning.** A kernel that does well *and* its profiler counters look healthy passes both its genotype (params) and learned state (e.g., schedule offsets) to its children. Maps to per-pipeline persistent state in an SSBO.

19. **Mutation rate annealing.** High mutation early in the program lifetime, low later — like neural network learning rate decay but for kernel autotuning.

20. **Coevolution of A-tiling and B-tiling.** Two GA populations: one evolves how A is tiled, another how B is tiled. They are fitness-coupled — a bad B-tiling penalizes the A-tilings it gets paired with.

---

## Ant Colony / Pheromone

21. **Pheromone-trail dispatch ordering.** Each workgroup, upon completing a tile, deposits "pheromone" (an integer atomic add) on the tile coordinate it processed. Next dispatch's WG scheduler reads pheromone map and picks tiles with highest evaporation-adjusted scent — biases toward L2-hot regions.

22. **Evaporation = cache decay.** Pheromone decays exponentially per dispatch. Tiles that haven't been touched in 50ms have no trail, so they're picked uniformly. Matches actual L2 residency timescale.

23. **Multiple pheromone types.** One for "recent FMA hit", one for "recent load miss", one for "recent bank conflict". WGs are attracted by some, repelled by others, mimicking food/poison trails.

24. **Stigmergy for K-loop order.** Each K-block leaves a marker indicating its execution latency. Future WGs reorder their K traversal based on the easiest path through the marker field. Practically: just a TSP-on-K-blocks heuristic in disguise.

25. **Ant nest entry/exit.** A handful of "scout" WGs explore unusual tile orderings while the bulk follow the established pheromone path. Tradeoff exploration/exploitation directly on the GPU.

---

## Slime Mold / Physarum

26. **Physarum K-reduction order.** Treat each K-block as a node, each adjacent-block pair as an edge with conductance proportional to L2 hit rate. Iteratively solve the Physarum optimal-path equations and traverse K in that order.

27. **Adaptive tube thickness = vector width.** Where the slime mold finds a high-flow path, thicken the "tube" by using wider vector loads (vec4 instead of float). Where flow is low, scalar loads. Effectively load-width autotuning per K-stripe.

28. **Physarum tile-graph contraction.** Use the slime-mold pruning algorithm to merge weakly-connected tiles into super-tiles. Useful for irregular dimensions where a single MxN doesn't divide evenly.

29. **Oscillatory flow.** Slime molds reverse flow periodically; map this to alternating the direction of K traversal (forward / backward) every few iterations to balance write-back latency.

---

## Cell Cycle / Mitosis

30. **G1/S/G2/M phased workgroups.** Persistent WGs cycle: G1 = load A tile, S = FMA accumulate, G2 = load next A while writing partial, M = barrier and swap roles. A pipeline parallelism reframing of double-buffering.

31. **Cell-cycle checkpoint.** Before exiting M phase, run a "DNA damage check" — i.e., verify the accumulator hasn't overflowed/NaN'd. If it has, abort and reissue the tile (apoptosis).

32. **Mitotic splitting.** A persistent WG starts owning a 256x256 block; when it accumulates X TFLOPS of work, it "divides" by writing its remaining workload split into a queue for a sibling WG launched on the next dispatch.

33. **Stem-cell WGs.** A pool of "undifferentiated" persistent WGs spawn into specialized kernels (load-heavy, compute-heavy, store-heavy) based on what the queue needs at that instant.

34. **Telomere shortening.** Each WG has a max-iteration counter that decrements; after N iterations it expires and is reissued fresh. Forces L1 cache resets without explicit invalidation.

---

## Bacterial Chemotaxis

35. **Tumbling WGs.** A WG "swims" through tile space following a gradient of recent-cache-hit-rate. If gradient is positive (more hits), continue same direction; if negative, "tumble" (pick random adjacent tile). Maps to a stochastic scheduler atop a tile-quality map.

36. **Run-and-tumble for K-loop.** Inside the K loop, if the last 8 K-blocks gave good FMA throughput, keep marching K=K+1; otherwise, jump to K=K+random. Useful for irregular K patterns.

37. **Quorum sensing for barriers.** WGs broadcast their progress via subgroup atomic counter; only when the count exceeds a quorum threshold do they issue `subgroupBarrier`. Avoids over-synchronizing when most lanes are already done.

38. **Bacterial conjugation = SSBO sharing.** When two WGs pass nearby (work on adjacent tiles), they exchange small "plasmids" of state (best tile config seen so far) via shared SSBO slots.

---

## DNA / Information Storage

39. **DNA-base packing.** Pack 4 ternary or binary matrix values per byte using {A,C,G,T} = 2-bit codes. Use bit-parallel popcount-style ops via `subgroupBallot` for the multiply. Hopeless for FP32 — fun for 1-bit or 2-bit GEMM.

40. **String-matching primitives for repeated patterns.** Identify long runs of repeated values in A; replace them with a "gene" reference and a multiplier. Useful for quantized matrices with many duplicates.

41. **DNA repair = error detection.** Periodically checksum the accumulator using a Hamming-like code; if drift detected, reload from B-side ground truth. Cosmic-ray hardening for long-running GEMMs.

42. **Codon-style instruction packing.** Treat 3 consecutive shared-mem loads as a "codon" that decodes to a specific FMA pattern. SPIR-V codegen could emit a stream of codons matched to instruction-cache lines.

---

## Immune System

43. **Anomaly detection on input shape.** A "T-cell" gatekeeper checks if (M,N,K) matches any "self" patterns; if anomalous, switch to a safe generic kernel; if recognized, dispatch the specialized one.

44. **Antibody pool.** A bank of pre-compiled kernel variants is the antibody repertoire. Each new shape gets matched against the closest antibody by Levenshtein on dimension-string.

45. **Clonal expansion.** When a particular shape is seen many times, "clone" the matching kernel into multiple specialized variants (one per memory layout, one per warp count). The most-effective clone proliferates in the pipeline cache.

46. **Autoimmune avoidance.** Track which kernel variants caused regressions and blacklist them — even if they look good on paper. A "memory T-cell" of past failures.

---

## Forager / Explorer / Exploiter Roles

47. **Three-caste WGs.** 80% Exploiter WGs run the known-best kernel; 15% Forager WGs run a slight variation; 5% Explorer WGs try a wildly different config. Feedback steers next dispatch's mix.

48. **Honeybee waggle dance.** Forager WGs that find a high-performance tile order "dance" by writing their config to a shared SSBO with intensity proportional to fitness; other WGs read and adopt.

49. **Pollinator WGs.** A small set of WGs whose only job is to move good configs between independent dispatch streams (different streams = different "flowers"), enabling cross-pollination of tuning.

---

## Predator-Prey / Population Dynamics

50. **Lotka-Volterra of load vs compute WGs.** Producer WGs that prefetch data are "prey"; consumer WGs that FMA are "predators". Maintain a Lotka-Volterra equilibrium ratio (e.g., 1:4) that's updated dynamically based on the L2 fill level "ecosystem health".

51. **Population crashes trigger barrier.** If the prey/load population is depleted (queue empty), predators starve (idle) — that's a barrier in disguise. The model just lets us choose population sizes more rationally.

52. **Carrying capacity = register pressure.** The number of active WGs per SM is bounded by carrying capacity (register/shared-mem budget). Treat occupancy as an ecological problem — what's the sustainable population?

53. **Invasive species.** A new kernel variant is an "invasive species" — let it compete on a fraction of dispatches; if it dominates, replace the established kernel. Otherwise it gets extinct.

---

## Photosynthesis / Energy Harvesting

54. **Leaf kernel for idle bandwidth.** When the main GEMM is compute-bound, a low-power "leaf" WG opportunistically loads the *next* matrix into L2, storing sugar for future dispatches. Bandwidth-cycle filling.

55. **Chlorophyll = specialization constant.** A specialization constant set at pipeline creation tunes the kernel to the spectrum of expected workloads ("green" matrices = small, "red" = large), like leaves tuned to specific wavelengths.

56. **C4 photosynthesis (bundle sheath cells).** A two-stage GEMM: a "mesophyll" kernel does loose first-pass accumulation, a "bundle sheath" kernel does high-precision refinement. Maps to a low-precision + correction-step algorithm.

---

## Coral / Sponge / Sessile

57. **Coral-growth tile layout.** Tiles are laid out incrementally — each new tile attaches adjacent to a previously hot tile, gradually building a structure that matches access patterns. Online z-order generation.

58. **Sponge filter-feeding kernel.** A pipeline that processes whatever data is currently in L2 — no explicit scheduling, just continuously sweeps shared memory for valid (A,B) pairs and accumulates. Speculative execution as filter feeding.

59. **Coral bleaching alert.** If a tile config starts performing badly (rising temperature = falling TFLOPS), the structure "bleaches" — abandons that layout and regrows from a safer base config.

---

## Murmuration / Flocking / Swarm

60. **Boid rules for WG dispatch.** Each WG's tile choice follows Reynolds rules: (1) cohesion — stay near other WGs to share L2; (2) separation — don't pile on the same SM; (3) alignment — march in the same K direction. Emergent dispatch order without a central scheduler.

61. **Predator avoidance = bank conflict avoidance.** WGs detect "predators" (bank conflicts) via slow timing and veer away. Implemented as a flocking velocity update on tile coordinates.

62. **Murmuration shock waves.** A single WG's slowness propagates as a wave through the flock, causing nearby WGs to slow their issue rate proactively to avoid the same bottleneck.

---

## Symbiosis / Mutualism

63. **Two-kernel symbiosis.** A "small-shape" kernel and a "large-shape" kernel are dispatched together; each consumes the work the other is bad at. Lichen-style: tile dispatcher routes by aspect ratio.

64. **Mitochondria-style energy organelle.** A persistent kernel that just generates random numbers (for stochastic rounding) and feeds them to other WGs via SSBO. Specialization decouples randomness from FMA.

65. **Gut microbiome model.** A swarm of tiny utility kernels (transpose, copy, reduce) live inside the main GEMM pipeline; the main kernel benefits from their diverse services without managing them explicitly.

66. **Coral-zooxanthellae model.** A persistent "host" WG owns shared memory; transient "guest" WGs come in, run a quick FMA pass, and leave — the host accumulates the partial results across many guests.

---

## Octopus / Distributed Cognition

67. **Independent arms = independent warps.** Each of 8 warps in a WG runs its own scheduling logic, picking K-blocks independently. A "central brain" only resolves conflicts at writeback. Maps to subgroup-level autonomy.

68. **Chromatophore camouflage.** WGs change which shared-mem swizzle they use based on local conditions (skewed input, banked layout) — like an octopus matching its environment.

69. **Distributed proprioception.** Each warp keeps a local model of bandwidth/latency it has experienced; the WG aggregates these into a global picture used for the next K-block decision.

---

## Quorum Sensing / Plant Tropism

70. **Quorum-sense barrier release.** WGs increment a shared counter on completion; once the counter hits a quorum threshold (say 75% of active WGs), all remaining ones are signaled to skip ahead. Trades correctness for tail-latency reduction (only viable for approximate GEMM).

71. **Phototropic dispatch.** The grid of WGs "bends" toward SMs with the highest recent occupancy (= light). Implemented as a biased work-stealing queue weighted by SM-id telemetry.

72. **Gravitropic K-traversal.** Heavy (high-bandwidth) loads "fall" first; lighter (cached) loads bubble up. Sort the K loop iteration order by predicted memory cost.

---

## Apoptosis / Lifecycle

73. **Late-WG apoptosis.** If a WG hasn't reported progress within a deadline, kill it (skip its writeback) and reissue its tile in a follow-up dispatch. Tail-latency mitigation via redundancy.

74. **Programmed cell death after N iterations.** Persistent WGs auto-terminate after a fixed iteration budget regardless of progress; their work is repackaged for fresh WGs. Prevents stale resources.

75. **Necrosis vs apoptosis.** Necrosis = sudden death (WG crashes due to OOB), causes "inflammation" (must invalidate cached state). Apoptosis = orderly death (clean exit, cached state remains valid). Encourage the latter via deadline checks.

---

## Mycelial Networks

76. **3D dispatch with mycelial shortcuts.** Standard 3D dispatch grid (M, N, K). Add a sparse graph of "hyphal" connections — random WGs can read each other's partial sums via SSBO, short-circuiting reductions.

77. **Common mycorrhizal network.** Multiple concurrent GEMMs share a single persistent SSBO of "nutrients" (precomputed reciprocals, exp tables, etc.) — like trees sharing fungal networks in a forest.

78. **Fungal scout hyphae.** A small set of WGs do nothing but probe the dispatch space, reporting back via SSBO which tile sizes look promising. The main GEMM follows the trails.

79. **Mycelial decomposition of dead tiles.** When a WG finishes early, its remaining time is repurposed to "decompose" leftover state — e.g., zero out shared memory for the next dispatch. Recycling idle time.

---

## Wildcards / Misc Biology

80. **Crystalline biomineralization (regular tile pattern from "amorphous" inputs).** First pass classifies regions of A/B as "amorphous" (irregular) or "crystalline" (regular); regular regions dispatch a fast structured kernel, amorphous regions a fallback.

81. **Caterpillar pipeline.** N pipeline stages physically advance along the matrix together, each stage a separate dispatch. The "body" moves like a caterpillar — each WG-segment knows only its neighbours. Mimics systolic array on a static scheduler.

82. **Metamorphosis.** Kernel transforms its layout mid-run: starts as "larva" (lots of small WGs, exploration), then forms "pupa" (autotune), emerges as "adult" (one big persistent kernel). Sequential JIT specialization.

83. **Migratory birds.** Long-running GEMM session migrates its hot data across L1/L2/HBM regions periodically to balance wear and access patterns. Like geese rotating who leads the V.

84. **Diapause (animal hibernation).** During long idle windows, the autotuner enters a low-energy mode and just preserves its best-known config rather than continuing to explore. Power-aware ML.

85. **Eusocial castes (ants/bees).** Queen WG owns the global accumulator; workers do FMAs and ship results; drones do nothing but reload constants (e.g., 1/sqrt(2)) on demand. Strict role specialization at the warp level.

86. **Anglerfish bait.** A "lure" kernel runs at very high priority — its sole job is to attract memory traffic into a specific L2 region right before the main kernel needs it. Voluntary prefetching as bait.

87. **Cuttlefish color-change synchrony.** All WGs in a subgroup change their swizzle pattern simultaneously in response to a single broadcast signal — `subgroupBroadcast` triggers a layout pivot mid-kernel.

88. **Penguin huddle.** Cold (cache-cold) WGs migrate to the center of a huddle where there's more shared-memory warmth; hot (cache-hot) ones rotate to the edge. Implemented as a rotation of which WG holds the L2-resident tile.

89. **Sea-turtle natal homing.** A WG remembers (via SSBO) which SM it ran on last time and tries to dispatch there again — assuming residual L1 warmth. SM-id persistence across dispatches.

90. **Slime-mold electrical signaling.** Use atomic-counter "pulses" as long-distance signals between WGs across the GPU — slow but globally coherent, useful for occasional reconfiguration events.

91. **Cellular automata as scheduler.** WGs follow Game-of-Life style local rules (alive/dead based on neighbours' states), and the resulting pattern naturally produces emergent dispatch waves across the grid.

92. **Neural sleep / replay.** During idle periods, the driver "dreams" by replaying recent input shapes through candidate kernel variants and consolidating the winners — offline autotune triggered by idle detection.

93. **Bird-song template matching.** Each kernel variant has a "song" (a signature timing trace); incoming workloads are matched against templates by DTW (dynamic time warping) on dimension sequences.

94. **Symbiont-host gene transfer.** A successful tile-config from kernel A can be horizontally transferred into kernel B's gene pool via SPIR-V specialization constants — mimicking bacterial lateral gene transfer.

95. **Photosynthetic Calvin cycle.** Three-stage looping: fixation (load A), reduction (FMA), regeneration (clear shared mem). Each stage runs in a separate warp, naturally pipelining.

96. **Cnidocyte discharge (jellyfish stinging cell).** A WG that hits a hot tile "stings" — emits a one-shot atomic notification that triggers all neighbours to prefetch ahead. High-priority broadcast on rare events.

97. **Tardigrade cryptobiosis.** When the GPU is throttled (thermal), the autotuner enters cryptobiosis — freezes its state perfectly until the temperature drops, then resumes without re-exploration.

98. **Plant phototaxis with multiple suns.** Two simultaneous gradients (compute throughput and memory throughput) — the dispatch bends toward whichever is brighter, automatically balancing the kernel.

99. **Embryonic gradient morphogens.** Initial WGs deposit a "morphogen" gradient (atomic counter map) that later WGs read to decide their role/layout. Self-organizing tile assignment from a noisy initial state.

100. **Endogenous circadian rhythm.** A 24-hour clock in the driver schedules heavy autotuning during predicted-idle hours; light, conservative kernel choices during predicted-busy hours. ML inference servers with diurnal load patterns.

101. **Sponge totipotency.** Any WG can be reassigned to any role at any time — no fixed specialization. Maps to a uniform persistent-WG dispatch where the role is read from an SSBO at the top of each iteration.

102. **Synchronous firefly flashing.** Pulse-coupled oscillators: each WG has a phase that drifts; nearby WGs nudge each other's phase via subgroup messaging until they all fire FMAs in lockstep. Emergent synchronization without explicit barriers.

---

## Synthesis / Crossovers Across Themes

103. **Evolutionary STDP.** GA evolves the STDP rule itself (prefetch-distance update law), so each workload develops its own learning algorithm. Meta-meta-tuning.

104. **Ant-colony Hebbian hybrid.** Pheromone trails reinforce Hebbian co-firing pairs — joint reinforcement signal for both dispatch order AND kernel choice.

105. **Slime-mold immune system.** Anomalous shapes cause the slime-mold network to retract from suspect tiles, isolating the unfamiliar region while specialized kernels handle it.

106. **Mycelial quorum-sensing barrier.** Global SSBO-backed mycelial graph propagates "ready" signals across the GPU; barrier triggers when quorum is reached on any cluster of WGs. Sub-global synchronization.
# Cryptography / Number Theory / Ring Algebra Ideas for Vulkan FP32 SGEMM

Wild speculation dump. Mathematical rigor optional. Focus: can it map to a Vulkan compute shader primitive?

---

## NTT / FFT-domain matmul

1. **NTT-domain matrix multiplication.** Reinterpret rows of A and columns of B as polynomials over Z/p with p a Solinas / Mersenne-friendly prime; multiply via NTT in O(n log n) per row. Vulkan fit: subgroup-shuffle butterflies map to `subgroupShuffleXor`, but FP32 throughput beats integer NTT here unless we batch heavily.

2. **Bluestein-NTT for arbitrary inner-dim K.** When K isn't a power of two, use Bluestein's chirp-z trick to embed into a length-2^m NTT. Vulkan fit: precomputed chirp table lives in `uniform` buffer, butterflies in shared memory.

3. **Schönhage–Strassen "matmul of matmuls".** Treat each tile as a giant integer, multiply tiles with SSA, recurse. Vulkan fit: cute but the constants kill you below n ≈ 2^18; useful only for symbolic-precision GEMM tests.

4. **Polynomial-ring lift to R[x]/(x^N+1) (negacyclic).** Lifting FP32 to fixed-point and computing in the cyclotomic ring used by CKKS could let us reuse FHE NTT-twiddle tables already in shared memory. Vulkan fit: subgroup-level negacyclic butterfly fits in 32 lanes.

5. **Number-Theoretic Cooley–Tukey radix-32.** Match radix to subgroup size so each butterfly stage is a single `subgroupBroadcast`. Vulkan fit: extremely natural — radix-N FFT where N == subgroupSize is the "one true" FFT for SIMT.

---

## Modular / RNS / CRT

6. **Residue Number System split-K.** Pick three coprime moduli p1·p2·p3 > expected accumulator range; run three independent integer GEMMs in parallel, CRT-reconstruct at the end. Vulkan fit: three dispatches that never sync — perfect for `VK_KHR_synchronization2` graph parallelism.

7. **CRT split-K across multiple queues.** Each compute queue owns one modulus's GEMM; combine on the host or via a tiny epilogue shader. Vulkan fit: maps cleanly to multi-queue submission, hides per-queue command-buffer build cost.

8. **Montgomery-form FP emulation.** Re-encode FP32 mantissas in Montgomery form so the inner `fma` becomes a Montgomery-reduce-fused multiply, eliminating one normalization. Vulkan fit: requires a custom mantissa path; probably slower than the hardware FMA but cool for variable-precision GEMM.

9. **Barrett reduction in the epilogue.** When doing modular GEMM, push Barrett constants into a `push_constant` so the shader does a 64-bit mul-high once per output tile. Vulkan fit: `uvec2`-pair emulation of u64 already works on most GPUs.

10. **Solinas-prime epilogue.** Pick p = 2^k − c with tiny c so the reduction is "shift-and-add-c"; ideal for fast modular accumulation. Vulkan fit: trivial bitshift instructions, fits in one cycle.

11. **Goldilocks prime (p = 2^64 − 2^32 + 1).** Use the field popularized by Plonky2 to get fast 64-bit ops; great for proving GEMM correctness later. Vulkan fit: emulated u64 mul-hi is the bottleneck — but `VK_KHR_shader_integer_dot_product` helps.

12. **Mersenne-prime tile accumulator (p = 2^31 − 1).** Each warp accumulates into a Mersenne field so partial sums never overflow FP32 implicit range. Vulkan fit: replace `fadd` with int32 add + conditional subtract — actually plausible.

---

## Homomorphic encryption / ZK-flavoured

13. **CKKS-style matmul in cipher domain.** If the model is already encrypted, the GEMM kernel operates on ciphertexts (which are vectors of polynomial coefficients) — same NTT primitives as idea #1. Vulkan fit: open research; ciphertext sizes blow up shared memory.

14. **BFV/BGV bootstrap-aware tiling.** Tile sizes chosen so a tile fits within one ciphertext level before bootstrap is needed. Vulkan fit: tile chooser becomes a noise-budget solver in the host code.

15. **Sigma-protocol GEMM proof.** Have the GPU emit a Fiat-Shamir transcript proving `C = A·B` without recomputing; CPU verifies in O(n^2). Vulkan fit: extra epilogue shader hashes the output with a `subgroupBallot`-driven Merkle tree.

16. **Freivalds' algorithm verifier.** Probabilistic O(n^2) check: pick random r, verify A·(B·r) == C·r. Vulkan fit: two extra mat-vec dispatches; cheap insurance against transient bit-flips on consumer GPUs.

17. **Zero-knowledge "blind" GEMM.** Mask A and B with random orthogonal R: compute (A·R)·(R⁻¹·B), recover C. Vulkan fit: useless for perf, but a great differentiating feature for privacy-preserving inference.

18. **Lattice-based pre-conditioner (LLL/BKZ).** Run LLL on B's columns offline so the resulting vectors are short — gives smaller FP32 dynamic range and better cache locality. Vulkan fit: offline-only; the GEMM itself is unchanged.

---

## Boolean / GF(2^n) tricks

19. **GF(2) matmul via XOR-AND.** For binarized weights, matmul reduces to AND+popcount per inner product. Vulkan fit: `subgroupBallot` + `bitCount` — already a single cycle.

20. **GF(2^8) for INT8 GEMM as polynomial mul.** Treat INT8 lanes as GF(2^8) elements, multiply with carryless mul. Vulkan fit: needs `clmul` which Vulkan lacks; emulate with 8 shifts.

21. **Method-of-Four-Russians for boolean GEMM.** Precompute a 16-entry table per 4-bit block, GEMM becomes table lookups. Vulkan fit: 16-entry LUT fits in 64 bytes of shared memory per tile.

22. **Strassen over GF(2).** Strassen's 7-mul recursion works over any ring including GF(2); each "multiply" becomes a popcount AND. Vulkan fit: recursion depth limited by register pressure.

---

## Probabilistic data structures

23. **Bloom filter "is this row zero".** Hash each A-row; if Bloom says "not present in B's nonzero rows," skip the multiply. Vulkan fit: shared-memory Bloom filter of 1024 bits queried per thread.

24. **Count-min sketch of partial sums.** For approximate GEMM, maintain a CMS instead of a dense accumulator; query at the end. Vulkan fit: each subgroup owns one row of the CMS, atomic-add into shared memory.

25. **Cuckoo hash for sparse B.** When B is sparse, store nonzeros in a cuckoo hash; the row-scan becomes a hash lookup per A-element. Vulkan fit: 2-table cuckoo with `subgroupBallot` for collision detection.

26. **HyperLogLog of unique inner-product keys.** Useful for batched GEMM where many (rowA, colB) pairs repeat; HLL estimates the unique workload. Vulkan fit: 1-pass HLL register-update is a single `findMSB` per element.

27. **Quotient filter for memoized (row, col) → dot.** Quotient filter has better cache locality than Bloom and supports deletion. Vulkan fit: one quotient filter per workgroup, lives in shared memory.

---

## Hashing & content-addressed caches

28. **Universal hash of tile contents.** Hash each 16×16 tile of A and B with a Toeplitz hash; identical tiles reuse cached partial sums. Vulkan fit: hash compute is `dot`-over-GF(2); cheap with `subgroupXor`.

29. **Merkle accumulator for hierarchical reduction.** Build a Merkle tree of partial sums so a re-run only recomputes paths whose leaves changed. Vulkan fit: nice for streaming inference where A changes row-by-row.

30. **Cryptographic PRNG for random projections.** Use ChaCha20 in a compute shader to deterministically generate a Johnson–Lindenstrauss sketch; GEMM the sketch instead of the full matrix. Vulkan fit: ChaCha20 has 16 parallel rounds, maps to 16-lane subgroups beautifully.

31. **Hash-based dot-product memoization across batch.** When K-dim is huge but many (row_a_id, col_b_id) repeat in a batch, memoize the dot. Vulkan fit: open-addressed hash table in VRAM with `atomicCompSwap`.

---

## Algebraic structure exploitation

32. **Toeplitz/circulant fast path.** If A is circulant, A·B is a batch of NTT-domain pointwise multiplies — drops K-loop entirely. Vulkan fit: detect circulancy at JIT time; dispatch a different shader.

33. **Schur complement blocked inverse-GEMM.** For (A B; C D)·x decomposition, reuse the Schur complement S = D − C·A⁻¹·B across many right-hand sides. Vulkan fit: precompute S once, GEMM many times — a "preconditioned" SGEMM API.

34. **LU prefactor reuse.** Factor A once with partial-pivot LU; subsequent A·x = b solves are two triangular GEMVs. Vulkan fit: store L and U packed; use subgroup-broadcast for the back-substitution.

35. **QR via Householder for repeated GEMM.** Pre-factor A = QR; matmuls reduce to two triangular ops. Vulkan fit: standard cuBLAS-style code, but Vulkan-native.

36. **Cholesky for SPD weight matrices.** If A is SPD, A = LL^T halves the work for many downstream GEMMs. Vulkan fit: triangular tile-sweep, mostly an algorithmic win.

---

## Side-channel and cache-aware

37. **Side-channel-style cache-line tiling.** Borrow the access patterns from Prime+Probe attacks — they are literally "fit-the-cache" optimization. Vulkan fit: cross-check tile size against the GPU's L1 line size discovered via timing.

38. **Constant-time GEMM as a feature.** Run the same memory-access pattern regardless of data; trivially achieves zero data-dependent timing. Vulkan fit: easy — we already don't branch on data. Sellable as a "privacy-preserving" mode.

39. **Reed–Solomon coded accumulator.** Encode the accumulator state with a (k, n) RS code so that single-bit GPU errors can be corrected mid-kernel. Vulkan fit: trivial parity tile per workgroup; useful on consumer GPUs without ECC.

40. **Erasure-coded split-K with one redundant moduli.** Extend idea #6: use four moduli where any three reconstruct, get free fault tolerance on cosmic-ray-prone consumer GPUs. Vulkan fit: identical to #6 plus a tiny epilogue.

---

## Out-there long-shots

41. **Goldwasser–Micali probabilistic GEMM.** Each output is "correct with high probability"; flip more coins for higher confidence. Vulkan fit: only useful for sketch-based approximate ML; saves bandwidth.

42. **Karatsuba over the polynomial ring of tile elements.** Tile multiplication is itself a polynomial multiplication; recursively Karatsuba it. Vulkan fit: register pressure goes through the roof; useful only at huge tile sizes.

43. **Lattice-based GEMM for post-quantum inference.** Map weights into a Module-LWE-style structured lattice; matmul becomes a polynomial mul in R_q. Vulkan fit: literally the same primitive as #4 (NTT in cyclotomic ring).

44. **Mixed-radix NTT to match Vulkan subgroup sizes (32/64/16).** Per-vendor radix selection so the inner butterfly is always one subgroup. Vulkan fit: requires querying `VkPhysicalDeviceSubgroupProperties.subgroupSize` at init.

45. **GPU-side Fiat–Shamir transcript for "self-attesting" GEMM.** Output includes a hash that the CPU can quickly verify, catching driver bugs. Vulkan fit: free if you already have a reduction epilogue — XOR all outputs into a 256-bit accumulator.

46. **Chinese Remainder split across SMs / workgroups.** Workgroup i computes the result mod p_i; final dispatch CRT-reconstructs. Vulkan fit: variant of #6 with finer granularity — needs cheap inter-workgroup atomic.

47. **Tropical-semiring GEMM (max-plus).** Replace (+, ×) with (max, +) for distance-style problems; same memory pattern, different op. Vulkan fit: `subgroupMax` substitutes for the reduction.

48. **Min-plus matmul for shortest-path GEMM.** Same as #47 but min instead of max — useful for graph-NN inference. Vulkan fit: `subgroupMin`, again one cycle.

49. **Roots-of-unity twiddle table sharing across kernels.** All NTT-based kernels share one 64KB twiddle table in a `uniform` buffer; saves cache and code size. Vulkan fit: descriptor-set 0 lives forever.

50. **AES-NI-style "round-based" GEMM micro-kernel.** Pipeline 10 rounds of `fma`+swap analogously to AES rounds so the compiler unrolls into a single super-instruction. Vulkan fit: just an aggressive `[[unroll]]` plus careful register coloring — but the framing as "AES-style rounds" might inspire a tighter loop.

---

## Best-bet shortlist for actual implementation

- **#16 Freivalds' verifier** — cheap insurance, ships behind a debug flag, demonstrably correct.
- **#39 Reed–Solomon coded accumulator** — sellable feature on consumer GPUs without ECC, low overhead.
- **#5 + #44 Mixed-radix NTT matched to subgroup size** — concrete recipe if we ever add integer GEMM.
- **#28 Universal hash of tile contents** — could give real wins for batched inference with repeated weights.
- **#30 ChaCha20-driven random projection** — sketched GEMM is a known speedup for low-rank workloads.
- **#37 Side-channel-pattern-derived tiling** — purely framing, but might shake loose a new tile size we missed.
# Physics / Analog / Field-Theoretic Ideas for Vulkan FP32 SGEMM

Speculative dump. Physical rigor not guaranteed. Each idea sketches how a GPU compute shader could emulate the analogy to push past 60% peak.

---

## Optical & Photonic

1. **Holographic matmul via FFT-conjugation.** Treat A and B as complex wavefronts; multiplication in Fourier space corresponds to convolution. Pre-tile, run a workgroup-local DFT, multiply spectra, and inverse-transform — useful for structured matrices where the spectrum is sparse.

2. **Photon path tracing through a "weight medium."** Each element c_ij is the total optical intensity arriving at detector (i,j) after rays pass through media of refractive index a_ik·b_kj. A shader emits K rays per output pixel and accumulates intensity, which collapses to standard matmul under linear optics.

3. **Mach-Zehnder interferometer mesh.** Decompose B into unitary phase shifters and beamsplitters (Reck/Clements decomposition); apply A as a sequence of MZI rotations stored in shared memory. Each lane simulates one waveguide — great for matrices that admit cheap unitary factorization.

4. **Coherent optical outer product.** An outer product a·bᵀ is exactly the interference pattern of two plane waves with amplitudes a, b. Workgroups produce rank-1 fringe maps and accumulate them additively into C, mirroring the standard outer-product reformulation but lit by optics intuition.

5. **Spatial light modulator emulation.** Treat the K dimension as a stack of SLM frames; each frame contributes a rank-1 update. A shader streams frames through fast shared memory, modeling SLM refresh as the inner-loop tick.

6. **Hologram caching.** Precompute a phase mask for B once, then "illuminate" with rows of A as plane waves. The phase mask = LUT in shared memory; multiple A's amortize the load.

---

## Analog Devices & Memristors

7. **Memristor crossbar emulation.** B sits as conductances in a virtual crossbar; A's rows act as voltage pulses. Each output column integrates current = ΣV·G. Implement as warp-shuffle-reduced column sums, then add transient settling noise to test robustness.

8. **Settling-time scheduling.** Real crossbars have RC time constants. Schedule tile dispatch by predicted "settling time" so that long-tail tiles overlap with short ones, hiding latency the way real arrays hide RC delay.

9. **Stuck-at-fault aware tiling.** Pretend a fraction of fused-multiply lanes are "stuck" and route around them with a permutation generated from a Latin square. Forces irregular schedules that may avoid bank conflicts.

10. **Conductance quantization for low-precision warmup.** First pass uses 4-bit "G levels" to compute a coarse C; second pass refines. Cheap when one matrix dominates compute time.

---

## Fluid & Continuum Dynamics

11. **Matmul as pipe-network flow.** Tiles = pipes; bandwidth = pipe diameter; output accumulators = reservoirs. Solve max-flow with a quick LP at dispatch time to pick the schedule that saturates HBM bandwidth.

12. **Lattice Boltzmann matmul.** Each lattice site stores partial sums; streaming step shuffles them between neighbors (warp shuffles); collision step applies a relaxation that converges to ΣA·B at steady state. Naturally maps to 2D thread grids.

13. **Vorticity-preserving accumulation.** Order partial sums so that curl-like cancellations happen first, reducing catastrophic cancellation in FP32. Detect via a Laplacian over the A tile.

14. **Smoothed particle hydrodynamics (SPH) accumulation.** Treat each multiply as a particle with kernel-weighted contribution to a grid cell. Useful conceptually for sparse matmul where most multiplies cluster.

15. **Reynolds-number heuristic for tile size.** "Laminar" regime = small tiles, predictable cache; "turbulent" = large tiles, must mix. Pick tile size from an empirical Re analog derived from matrix dimensions and L2 size.

16. **Coriolis-skewed dispatch.** Add an artificial rotational bias to the workgroup index ordering — counterintuitively this can break correlations that cause memory bank conflicts. Free pseudo-random perturbation.

17. **Kelvin waves on the K dimension.** Oscillate the partial-sum traversal order with a low-frequency sinusoid so different SMs hit different cache lines simultaneously, smoothing L2 pressure peaks.

---

## Thermodynamics & Statistical Mechanics

18. **Thermodynamic approximate matmul.** At "high T," sample a subset of K and rescale; lower T progressively to converge. Schedule as a simulated-annealing curriculum where high-T passes are cheap probes that prefetch the right tiles for low-T passes.

19. **Heat-capacity descriptor.** Compute a cheap thermodynamic-flavored statistic (variance · density) of A and B to predict compute time and pick a kernel variant. Avoids profiling overhead.

20. **Free-energy minimization of schedule.** Define F = (compute time) − T·(scheduling entropy). Sweep T to find a schedule that is fast but robust to small input perturbations (i.e., kernel reusable across many matmul shapes).

21. **Bose-Einstein "ground state" SM collapse.** At low effective "occupancy temperature" all SMs settle into the same access pattern, maximizing L2 reuse. Detect when the launch parameters cross the BEC threshold; otherwise pick the warm distributed schedule.

22. **Maxwell's demon prefetcher.** A dedicated "sorter" workgroup inspects the next-tile queue and reorders B columns so consumers always pull from the cheaper memory region — paying its own cost in instructions but reducing global ones.

23. **Tsallis entropy as a structure metric.** Non-extensive entropy of A captures heavy-tailed distributions better than Shannon; high Tsallis → use a stochastic rounding pass that's faster but still accurate enough.

24. **Spin glass = matmul energy landscape.** View ΣA·B as a Hamiltonian over spin configurations of the accumulator sign pattern; warp-cooperative Monte Carlo can find low-frustration accumulation orders that minimize FP cancellation.

25. **Phase transition in tile size.** Sweep tile dim; there is typically a sharp performance cliff where you spill registers. Treat this as a 1st-order transition and stay just before it via a bisection autotune.

---

## Field Theory & Quantum-Adjacent

26. **Path integral over schedules.** Weight each possible tile schedule by exp(−T_sched/τ). Approximate via importance sampling in a kernel autotuner; classical bandit algorithms are a Monte Carlo estimate of this integral.

27. **Feynman-diagram decomposition.** Rewrite ΣA·B as a sum over "diagrams" (which loop order, which fused op). Cheap diagrams = leading order; expensive ones = corrections. Skip high-order diagrams when accuracy budget allows.

28. **Renormalization-group matmul.** Coarse-grain large matrices into block-averaged representations; do a cheap matmul at coarse scale to predict where fine-scale recomputation is needed. Useful for iterative solvers calling SGEMM many times.

29. **Spin-network evolution.** Treat the K-dimension as time steps of a spin chain whose Hamiltonian is encoded by A and B. Trotterized evolution = tile-by-tile matmul with naturally interleaved load and compute.

30. **Action-principle scheduling.** Define an action S = ∫ L(state, schedule)dt. Euler-Lagrange gives a differential equation for the optimal schedule; solve once per kernel-shape class and cache.

31. **Casimir-effect accumulator sharing.** Two SMs working on adjacent tiles can share a tiny "vacuum" buffer in L2 whose contents both will read — emulate the attractive force as a scheduling bias that pulls dependent tiles temporally close.

32. **General-relativistic geodesics on a "matrix metric."** Define a metric where distance = predicted cache miss cost; the geodesic between (start tile) and (end tile) is the optimal accumulation path. Approximate by Dijkstra on a 4D tile graph.

33. **Hawking-radiation noise as stochastic rounding.** Emit FP rounding noise at the boundary of accumulator precision; over many accumulations the bias cancels, mimicking Hawking emission's apparent thermality.

---

## Topology, Symmetry, Localization

34. **Anderson-localization pinning.** Detect rows/columns whose magnitudes dominate (eigenvector-like localization) and pin them to specific SMs across the entire matmul so their tiles never leave L1.

35. **Topological invariant skip-list.** Compute a cheap topological invariant (e.g., a winding-number-flavored count) per tile; tiles with identical invariants likely produce identical contributions modulo a known transform — skip the multiply.

36. **Symmetry breaking detector.** If A is near-symmetric, only compute the upper triangle and reflect; the "near" is quantified by a symmetry-defect order parameter. A 2-pass kernel: cheap detector → specialized symmetric multiplier.

37. **Berry-phase tile rotation.** Rotate the local accumulator basis as you sweep K; the total Berry phase = accumulated rounding error. Choosing a path with zero net Berry phase = numerically stable schedule.

38. **Kosterlitz-Thouless vortex pairing.** Vortex/antivortex pairs in the tile dependency graph cancel; identify and remove them at compile time to eliminate redundant synchronizations.

---

## Mechanics & Oscillators

39. **Coupled-oscillator matmul.** Each accumulator = a damped oscillator driven by A·B contributions; steady-state amplitude = c_ij. Map to a relaxation iteration that converges quadratically when B is well-conditioned — interesting for iterative refinement passes.

40. **Lagrangian dispatch.** Express dispatch overhead as kinetic energy, memory-pressure as potential; minimize ∫L dt over the launch grid. Solve numerically once per machine.

41. **Resonance avoidance.** Find launch dimensions that avoid resonant cache-line collisions; equivalent to dodging eigenfrequencies of the memory subsystem. Sweep workgroup_x ± small δ to detune.

42. **Pendulum-mode tile traversal.** Swing tile index back and forth like a pendulum (zigzag K) so that adjacent SMs hit overlapping cache contents at the turnaround points, doubling L2 reuse.

---

## Diffusion, Waves, Reaction

43. **Diffusion-equation steady-state matmul.** Set up ∂c/∂t = ∇²c with source = A·B; the steady state is the result. Useful when many matmuls share structure — solve once via multigrid, perturb cheaply.

44. **Wave-superposition outer product.** Each rank-1 update is a standing wave; the final C is the superposition. Pre-sort updates by frequency so dominant contributions land first, enabling early-out for accuracy-bounded callers.

45. **Reaction-diffusion tile growth.** "Grow" the active tile front like a Turing pattern; reaction = compute, diffusion = data motion. Self-organizes around hotspots in sparse matrices.

46. **Solitons on the partial-sum line.** Treat each tile's partial sum as a wave packet moving along the K axis; solitons preserve shape under nonlinearity, so a stable accumulation order corresponds to a soliton solution of the schedule equations.

47. **Group-velocity dispersion control.** Order tiles so contributions with similar magnitude arrive simultaneously at the accumulator — analogous to dispersion-compensated optical pulses — to minimize FP precision loss.

---

## Exotic / Wild

48. **Quasi-crystal tiling.** Use a Penrose-like aperiodic tile layout for the workgroup grid; non-repeating means no two tiles share the exact same cache footprint, reducing pathological aliasing.

49. **Fractal recursion (Mandelbrot-style).** Recursively subdivide tiles until a base GEMM kernel handles them; the recursion depth adapts to matrix structure. Cache reuse scales with the fractal dimension.

50. **Causal-set scheduling.** Build a partial order of tile dependencies as a causal set; embed it in a fictitious Minkowski space and pick a foliation that maximizes parallel "now-slices."

51. **Hawking-Page transition heuristic.** Below a certain matrix size, use "AdS-phase" thermal kernel (many small WGs); above, use "black-hole-phase" single big kernel. Threshold computed once per device.

52. **Wheeler delayed-choice prefetch.** Issue speculative loads for two possible next tiles, then commit retroactively once the branch is known — measured-vs-not-measured trick adapted to GPU memory.

53. **Quantum Zeno scheduling.** Frequent "measurements" (barriers) freeze the partial sum evolution; spacing barriers far apart lets accumulation evolve freely. Find the optimal barrier cadence via a Zeno-time analog.

54. **Dark-matter halo of slack.** Reserve a halo of dummy threads around real workgroups whose only job is to keep cache warm — invisible to the result, gravitationally pulling cache lines into place.

55. **Cosmic-string tile boundary.** When two tile regions need to merge, insert a 1-D "string" of synchronization that carries accumulator state along its length, faster than a full barrier.

56. **Inflation-era warmup.** Briefly run with exponentially growing workgroup size during the first few microseconds to "inflate" the cache, then settle into normal operation. Cheap prefetch trick disguised as cosmology.

57. **Black-hole information paradox accumulator.** Information about the order of additions appears lost in FP rounding but can be partially recovered via Kahan compensation = "Hawking radiation" of precision. Use selectively on inner products with extreme dynamic range.

58. **Twistor-space rewrite.** Reformulate certain structured matmuls in twistor-like coordinates where the multiply becomes a contraction over fewer indices. Niche, but for specific block patterns it nukes the K loop.

---

## Bonus speculative

59. **Plasma instability detector.** Watch the runtime perf counters; sudden spikes = instability. Switch kernel variants the way a tokamak switches modes to suppress disruption.

60. **Anyonic braiding for permuted accumulation.** Treat tile pairs as anyons; braid them to permute the accumulation order in topologically protected ways — i.e., guaranteed bit-exact regardless of which path you took.

61. **Quantum-error-correction inspired redundancy.** Compute each tile twice with different orderings and majority-vote in the rare case they disagree by more than ε. Used selectively on numerically dangerous tiles flagged by a cheap pre-pass.

62. **Holographic principle reduction.** The bulk matmul state is encoded on its 2D boundary (the C matrix); maintain only boundary info during compute, reconstructing bulk partial sums on demand. Saves register pressure for huge K.
# PL Theory / Type Systems / Category Theory Ideas for Vulkan FP32 SGEMM

Context: at ~60% of peak on tiled f32 matmul. Brainstorm angle: what PL theory abstractions translate to concrete Vulkan primitives and might unlock the remaining ~40%?

---

## Linear & Substructural Types

1. **Linear-typed shared-memory tiles.** Each LDS tile is a linear resource: the type system forces exactly-once consumption by the FMA loop, so a forgotten `barrier()` after consume becomes a compile error. Maps directly to `groupshared` slot reuse — the compiler can statically reassign one LDS slot to a new tile once the linear handle drops, halving LDS pressure for the same kernel.

2. **Affine register-file tiles.** Affine (use-at-most-once) typing on accumulator fragments matches `VK_KHR_cooperative_matrix` C-matrices which must survive the K-loop. The type system can prove a `coopmatLoadNV → coopmatMulAdd → coopmatStoreNV` chain has no aliasing and emit a single contiguous lifetime, letting the register allocator coalesce.

3. **Substructural LIFO tile stack.** Tiles must be deallocated in reverse allocation order, mirroring how the K-loop double-buffer rotates: ping/pong slots are pushed and popped. Encoding this as a session type means the compiler picks the LDS offset assignment with zero bank conflicts because the access pattern is statically a stack.

4. **Ordered (non-exchange) types for warp lanes.** A type system where lane indices cannot be permuted prevents accidental `subgroupShuffle` that would cross a bank-conflict boundary. Compiler proves swizzle is conflict-free at type-check time.

5. **Borrowed-tile references.** Rust-style `&` references into the tile cache let multiple FMA producers read without copy, while `&mut` is needed for accumulators. Vulkan-level payoff: pin which tiles are in LDS vs. registers vs. global, and the borrow checker forbids stale-cache reads after a `memoryBarrierShared`.

---

## Dependent & Refinement Types

6. **Dependent dims `Mat<M, N>`.** Tile fit (M % BM == 0, N % BN == 0) checked at type-check time eliminates the runtime fallback for ragged edges, halving branches in the hot loop. The Vulkan specialization-constant mechanism is the natural target: dims become spec consts, and the type checker monomorphizes per-shape.

7. **Refinement-typed tile shapes.** `{t : Tile | t.cols * sizeof(f32) ≤ 128 byte cache line}` refinement directly encodes the GPU's L1 line size as a typing invariant. Compiler refuses to compile any tile layout that would straddle lines.

8. **Refinement-typed bounds on magnitude.** If A, B elements are refinement-typed `{x | |x| < 2^k}`, the compiler can prove no NaN/inf in the accumulator and skip the per-FMA flush-to-zero handling, gaining a cycle per FMA on some uarchs.

9. **Dependent-typed split-K factor.** A type `SplitK n` where `n | K` ensures the K-dimension is exactly divisible by the chosen split factor; the type checker rules out kernels needing a tail loop, which is pure VGPR savings.

10. **Indexed monads for tile coordinates.** `Tile (m : Fin BM) (n : Fin BN)` carries proof that indices are in-bounds, so global-memory loads can drop the `imageRobustness` predicate; on Vulkan with `robustBufferAccess` disabled this means one fewer compare instruction per load.

---

## Effect Systems & Monads

11. **Effect-tracked cache misses.** An effect `Miss` is raised when a load misses L1; type-and-effect inference computes the kernel's worst-case `Miss` count, and the optimizer picks the tile size that minimizes this effect-count integer literal.

12. **Memory-tier effect rows.** Effects `{LDS, L1, L2, HBM}` form a row; rows are unioned in sequencing. Kernel signature becomes a contract that the autotuner can verify against profiling data — no run, just inspect the type.

13. **Algebraic effects for async load + sync compute.** An `async_load` effect handler can be implemented as `vkCmdCopyBuffer` to a staging LDS via `VK_KHR_buffer_device_address`, while `compute` is FMA; the handler-based scheduler interleaves them as a single coroutine, recovering the load/compute overlap that hand-written kernels achieve via prefetch.

14. **Reader monad over thread-block configuration.** Threadblock dim, subgroup size, and LDS budget live in a `Reader cfg`, so the same kernel source autotunes by re-monomorphizing without textual changes. Maps to the SPIR-V specialization-constant front-end.

15. **State monad over the accumulator.** `State Acc` makes the K-loop body a pure function `(a, b) → State Acc ()`, and the compiler can prove the state is single-threaded per CU, enabling register coalescing.

16. **Continuation monad for k-strip yields.** CPS-converted kernel yields control after each K-strip; the runtime decides whether to keep the workgroup resident or yield to a different problem. Maps to multiple `vkCmdDispatch` calls fused into a single timeline via `VK_KHR_synchronization2`.

---

## Category Theory

17. **Category of matrices.** Objects: dimension naturals; morphisms: matrices; composition: matmul. Reifying the category lets a meta-optimizer reorder chains `A(BC)` vs `(AB)C` by minimum-cost path — for a single GEMM, this is trivial, but for chains it picks the lowest FLOP order before code-gen.

18. **Monoidal product as block-diag dispatch.** The tensor product `A ⊗ B` of two GEMMs is one block-diagonal GEMM; dispatching `[A; B]` as one large workgroup grid gives better occupancy than two small grids. Vulkan target: a single `vkCmdDispatchIndirect` with concatenated work.

19. **Functor lifting matmul through the cache monad.** Lift `(*) : A → B → C` into the cache monad to track which operands are hot; the lifted operation reorders the contraction to maximize reuse. Practical translation: pick which of M, N, K loops is outer based on which operand's working set fits in LDS.

20. **F-algebra on the partial-result type.** `Partial C` carries the running sum; the K-loop is the catamorphism over the list of K-strips with FMA as the algebra. The compiler can rewrite this cata to a fold-via-vectorize, mapping to subgroup `subgroupAdd` reduction across the K split.

21. **Hylomorphism over the K dimension.** Anamorphism: unfold global tiles → list of K-strips. Catamorphism: fold strips with FMA. Fused hylomorphism = the K-loop never materializes the strip list — exactly what double-buffered prefetch does in hand-rolled kernels.

22. **Coinductive stream of partial accumulators.** Treat the running `C` accumulator as a `Stream C` that emits intermediate sums; downstream operations (e.g. epilogue bias-add, activation) consume the stream pointwise. Maps to in-register epilogue fusion without a global writeback.

23. **Coalgebras as matrix observer.** A `(_, C)`-coalgebra is the "look at one element of C at a time"; the type forces a structured access pattern. Translates to a specific tile-store schedule that hits 128-byte coalesced stores in HBM.

24. **Yoneda lemma → kernel specialization.** `Mat M N ≅ ∀X. (X → Vec M) → Vec N → X→C` says the matrix is determined by its action on test vectors. Practical use: derive a fused matmul+bias kernel from the matmul kernel by partial-evaluating the Yoneda embedding — autogenerated epilogue fusion.

25. **Operad of matmul as 2→1.** Matmul is a binary operation in a symmetric operad of tensors; composing it gives multi-matmul. The operad's associativity coherence is the parenthesization choice for tensor chains — useful for a future BLAS3 frontend.

26. **Profunctor optics for A^T B.** A profunctor lens views a matrix and its transpose simultaneously: a `Lens (Mat M K) (Mat K M)` for transpose. The optic representation lets the compiler choose between in-memory transpose and on-the-fly transposed-load with `subgroupShuffle`, picking whichever fewer instructions.

27. **Lenses for transparent tiling.** A `Lens (Mat M N) (Tile BM BN)` lets the rest of the program ignore tiling. Compiler chooses tile size by analyzing the lens's "weight" — i.e., how often the program peeks into a tile. Concrete payoff: kernel author writes math, lens picks the best LDS layout.

---

## Tagless-Final / Free Monads

28. **Tagless-final matmul AST.** Encode the kernel as `class Matmul repr where matmul : repr (M,K) → repr (K,N) → repr (M,N)`. One `repr` instance emits SPIR-V via `rspirv`, another emits a cost model. Compile-time switching, zero AST runtime overhead.

29. **Free monad on dispatch DSL with auto-fusion.** A `Free DispatchF` AST allows tree rewrites: `bind (load A) (\a → bind (load B) (\b → compute a b))` fuses to a single kernel when no other consumer of `a` or `b` exists. Targets `VK_KHR_compute_shader_derivatives`-free reorderings.

30. **Initial-algebra encoding of accumulator update.** Each K-iter is a tagged FMA in a free F-algebra; running the catamorphism with the "SPIR-V FMA" interpreter gives a straight-line FMA sequence with no overhead. The same cata with a cost interpreter gives a static FLOP count.

31. **Cofree comonad of context-aware tiles.** Each tile carries its entire "context" (which neighbors are nearby in memory) as a `Cofree Stream Tile`. The codegen uses this context to choose `subgroupShuffleUp` vs reload-from-LDS for halo regions of a transposed matmul.

---

## Linear Logic

32. **Exponential `!A` for the B-panel.** In a row-major K-outer schedule, the B-tile is reused by every row of A: encode B as `!Tile` (cacheable, multi-use) and A's row as `Tile` (linear). The type system mandates B be in LDS and A be streamed via registers — exactly the right policy, derived from the modality.

33. **Linear-logic proof search for split-K factorization.** The problem "find a factorization of K into split factors that match the dispatch grid" is a linear-logic provability problem in the multiplicative fragment. A small SMT-style proof search at compile time picks the best split, mapping to `vkCmdDispatch` grid dims.

34. **Multiplicative-additive divisor for output partitioning.** Tile of C = M ⊗ N (multiplicative), but per-workgroup output is `M & N` (additive choice). The MALL fragment models the workgroup's choice of which (m,n) tile to produce; the choice resolution is the grid index.

35. **`?A` modality for write-once accumulators.** The "why-not" modality on the C accumulator forces it to be written exactly once at the end of K-iteration, ruling out partial flushes. Vulkan target: no intermediate `vkCmdPipelineBarrier` between K-strips of the same tile.

---

## Process Calculi & Petri Nets

36. **Petri net of pipeline stages.** Places: {global-A-ready, lds-A-ready, fragment-A-ready, fragment-B-ready, fma-done, store-done}. Transitions: loads, copies, FMA, stores. The net's marking analysis tells you minimum buffer counts; the answer is the ideal `numBuffers` for double/triple-buffered prefetch.

37. **CSP-style channels load → compute → store.** Each stage is a CSP process with typed channels; `||` parallel composition compiles to subgroup-level concurrency. The CSP refinement check verifies the pipeline is deadlock-free, e.g., that the store stage can never starve the compute stage in LDS.

38. **π-calculus mobile channels for warp-to-warp tile passing.** When warps cooperate on a large tile, they pass references over π-calculus channels; the type system forces each channel to be used by exactly two warps. Maps to `subgroupBroadcast` or LDS-backed mailbox.

39. **Session types for the K-loop pipeline.** A session type `!Tile.?Tile.!Acc.end` describes a worker that receives an A-tile, receives a B-tile, sends an accumulator, and stops. Sessions get duality-checked at compile time — no deadlocking pipelines.

40. **Bisimulation-equivalent kernel rewrites.** Two kernels are equivalent if they're bisimilar as labeled transition systems over (load, fma, store). The compiler legally rewrites the standard kernel into a swizzled-K version if and only if bisimulation holds, providing a formal correctness criterion for autotuning.

---

## Recursion Schemes

41. **Anamorphism on tile coordinates.** Unfold from `(0,0)` to all tile coords by recursively splitting the dispatch grid quadtree-style. A Z-order traversal naturally falls out of the ana, giving better L2 reuse than the default linear `gl_WorkGroupID`.

42. **Paramorphism on the K-dimension.** A para sees both the recursive sub-result and the original K-strip; useful for epilogue fusion that needs both the running accumulator and the final K-strip's raw values. Translates to a kernel variant where the last K-strip emits both `C[i,j]` and `B[K-1,j]` as fused side effects.

43. **Apomorphism for early-exit kernels.** An apo can short-circuit the corecursion; in matmul this is the "if accumulator overflowed" early-exit, which a refinement-typed accumulator never needs. Vulkan payoff: no need for `VK_KHR_shader_terminate_invocation`.

44. **Histomorphism for K-loop with memo.** A histo gives the K-iteration access to all previous partial sums. Practical: detect when a B-row pattern recurs and fuse multiple K-strips into one FMA via SIMD-pack.

---

## Curry-Howard & Proof-Theoretic

45. **A fast matmul is a constructive proof of `∃C. C = AB`.** The proof's normal form is the kernel; cut-elimination on the proof corresponds to inlining helpers in the SPIR-V module. Practical use: implement the codegen as proof-normalization, get whole-program inlining for free.

46. **Bidirectional type-checking on tile shapes.** Push constraints down (output tile shape ⇒ workgroup tile shape), pull dimensions up (per-thread fragment shape ⇒ subgroup tile shape). The bidirectional algorithm settles all tile dimensions simultaneously, replacing the manual autotuner.

47. **Curry-Howard with linear logic = a fast matmul is a proof in linear logic.** Specifically a proof of `A ⊗ B ⊸ C` where the A and B operands are each consumed once. The cut-free proof corresponds to a kernel that never re-reads any tile from global memory — the "ideal" data-reuse target.

48. **Realizability semantics for refinement types.** Read `{x : f32 | |x| < ∞}` as a Kleene realizer; the realizer is the executable code that checks the bound. With abstract interpretation, the realizer compiles away, leaving the bare FMA.

---

## Higher-Categorical / Type-Theoretic

49. **Matrices as ∞-groupoids (HoTT).** Path-types between matrix shapes are reshape/transpose isomorphisms; the univalence axiom lets you swap a transposed-stored matrix for a transposed-load schedule for free. Practical: the compiler treats `A^T` and `transpose(A)` as definitionally equal, removing dead transposes.

50. **2-category of GPU kernels.** Objects: shapes. 1-cells: kernels. 2-cells: kernel rewrites (e.g., swap loop nest). The 2-categorical horizontal composition is kernel fusion; vertical composition is sequencing — autotuning becomes finding the optimal 2-cell.

51. **Differential lambda calculus on the matmul.** The derivative of matmul w.r.t. A is the backward kernel; sharing the AST between forward and backward via the differential calculus halves codegen work. Maps to two SPIR-V entry points sharing 90% of their SSA.

52. **Type theory of containers for sparse matmul.** A container `(S, P)` with shape `S` and positions `P` gives a unified type for dense, sparse, blocked-sparse. One generic kernel parametrized over the container compiles to specialized versions per `(S,P)` — fewer LoC, no perf loss.

---

## Concrete Compiler-Design Angles

53. **Polyhedral model meets dependent types.** Encode the tile schedule as a Presburger formula; the type checker validates the schedule. Output is a fused, tiled, vectorized SPIR-V — essentially `Polly` for Vulkan.

54. **Tilings as a normal form in a rewrite system.** Write all tilings as terms in a confluent term-rewriting system; normalize to canonical form. Two tilings reaching the same normal form are equi-perf, so the autotuner samples one per equivalence class — radically pruning the search space.

55. **Region-based memory for tile lifetimes.** Each LDS tile lives in a typed region; region inference picks the LDS-budget partition. Maps to the SPIR-V `Workgroup` storage class with size determined at type-check time.

56. **Abstract interpretation of bank-conflict potential.** A small abstract domain (stride mod 32) lets the compiler statically detect bank conflicts in LDS access. Reject conflict-prone layouts at compile time — guaranteed conflict-free SGEMM kernels.

57. **Symbolic execution of the K-loop for occupancy.** Run the SPIR-V symbolically with VGPR counters; the symbolic state at function exit tells you exact register pressure. Use as the autotuner's primary objective rather than measured occupancy.

58. **Total functions ⇒ no kernel divergence.** A totality checker on the kernel body proves there's no divergent branching, eliminating `subgroupAll`/`subgroupAny` reconvergence overhead. Curry-Howard angle: total = proof, divergent = invalid proof.

59. **Datatype-generic programming for layouts.** A single generic kernel `matmul : ∀ layout. Layout layout ⇒ Mat layout → Mat layout → Mat layout` derives row-major, column-major, Z-order, Hilbert-curve specializations via type-class dispatch. Pick the best at link time per problem shape.

60. **Modal type system distinguishing host & device.** A `□A` modality means "available on both host and device"; spec constants and pipeline state live in `□`. The modal discipline makes the host-device boundary explicit and removes implicit `vkCmdPushConstants` correctness errors.
# Visual / Perceptual / Codec-Inspired SGEMM Ideas

Context: FP32 SGEMM at ~60% peak. Want to claw back every TFLOP using tricks borrowed from image/video/graphics pipelines. Many of these are speculative or lossy; flagged where relevant.

---

## Frequency-domain transforms

1. **JPEG-style 8x8 DCT block matmul.** Precompute DCT of A and B tiles into shared memory. In DCT domain the energy concentrates in low-frequency coefficients; truncate the tail and do a rank-k matmul of the surviving coefficients, then inverse-DCT into the C accumulator. Map: Vulkan subgroup matrix ops handle the small DCT butterflies; specialization constants pick the truncation rank.

2. **Walsh-Hadamard transform tiles.** Cheaper than DCT (only adds/subtracts), still energy-compacting for natural weight distributions. Pre-WHT the A tile into LDS, multiply in transform domain, inverse-WHT on accumulator flush.

3. **Wavelet (Haar) decomposition of B.** Recursively split B into LL/LH/HL/HH subbands; the LL band carries most energy so multiply against it first and treat HH as a correction term computed only where C magnitude exceeds a threshold.

4. **Daubechies-4 lifting steps in registers.** Lifting scheme reduces multiply count by ~30%; apply it as a pre-pass in shared memory so the matmul itself sees a sparser tile.

5. **Number-theoretic transform (NTT) over a Mersenne prime field.** Exact (not lossy) frequency-domain multiply; useful if you're willing to convert FP32 to fixed-point and back. Replace 4 mul-adds with 2 in the frequency domain for large tiles.

6. **Strassen-in-frequency.** Apply Strassen's recursion to DCT-domain tiles where the additive structure of Strassen aligns nicely with the DC coefficient handling.

---

## Texture-compression style storage

7. **BC1 (DXT1) weight packing.** Cluster each 4x4 weight block into 2 endpoints + 2-bit interpolation indices; decode in-shader during the inner loop. 4:1 compression ratio means more of B fits in L2; bandwidth-bound matmuls become compute-bound.

8. **BC4 single-channel block compression for FP32.** Store min/max + 3-bit indices per 4x4 patch. Decode is 1 LUT lookup + 1 lerp per element; the L2 hit rate jump can pay for the decode.

9. **ASTC-like variable-rate block compression.** Tiles with high variance get 8x8 blocks at 4bpp; quiet tiles get 12x12 at 2bpp. A pre-pass classifies tiles by gradient magnitude and emits the right shader variant.

10. **BC6H (HDR) for accumulator spill.** When you have to evict partial sums from registers to LDS, compress with a half-float endpoint scheme so LDS pressure drops.

11. **BC7 mode-adaptive endpoints.** BC7's 8 modes are great for matrices with varying local structure; pick mode per tile via a histogram of B's gradients computed at load time.

---

## Sparsity / RLE / entropy coding

12. **Run-length encoding of zero runs in B.** ReLU-derived activations and pruned weights produce long zero stretches; store (length, value) pairs and skip multiplies entirely. Map: warp-level prefix sum decodes RLE positions into thread IDs.

13. **Huffman-coded weight values.** Quantize B to 256 buckets then Huffman-code; decode in a small per-warp table. Pairs well with bandwidth-bound problems.

14. **CSR-on-tile.** Each 32x32 tile gets a tiny CSR encoding if its density < 50%; subgroup ballot picks the sparse path at runtime via specialization constants.

15. **Bitmap mask + dense values.** One u32 mask per row + packed nonzeros. Subgroup vote/ballot routes lanes to the right input element with no branching.

---

## Vector quantization / palette tricks

16. **Vector-quantized weight palette.** k-means cluster all 4-element subvectors of B into 256 codewords; store 1-byte indices instead of 16 bytes. Inner loop becomes `palette[idx[k]] * A[i,k]` — and crucially the palette lookups can be pre-multiplied by A's column to make it a giant lookup-add.

17. **Product quantization of B columns.** Split each column into 4 subvectors, quantize each independently. Inner product becomes 4 table lookups per output element.

18. **Pre-multiplied palette LUTs.** For repeated matmuls with the same B, precompute `palette[c] * A_tile` once and reuse across all subsequent multiplications — turns multiplies into pure adds.

19. **Texel cache as palette LUT.** Stash the palette in an actual `VK_IMAGE_TYPE_1D` texture; texture cache + hardware filtering may beat LDS for random-access lookups.

---

## Mipmapping / multiresolution

20. **Mip-pyramid on B.** Build a Gaussian pyramid of B (level 0 = full, level 1 = 2x downsampled, etc). Compute C at the coarsest level first; refine only tiles where the gradient suggests detail matters. Anytime algorithm.

21. **Trilinear interpolation between mip levels.** Sample B as if it were a 3D texture in (i, j, mip); hardware trilinear filtering does the blend for free.

22. **Anisotropic-filter style biasing.** Rows of A that vary slowly along k can use a coarser mip of B than rows that vary fast. Per-row mip-bias is a sampler parameter — already implemented in silicon.

23. **Foveated matmul.** Pick a "fovea" region of C (the rows/cols you care most about, e.g., top-k logits) and compute those at full precision; periphery uses a coarse mip. Great for inference with sparse output interest.

---

## Inter-frame / temporal / batch coherence

24. **Inter-frame delta encoding for batched matmuls.** When you have B_0, B_1, B_2... in a batch, encode B_{t+1} = B_t + delta_t with sparse delta. The matmul A * B_{t+1} = A*B_t + A*delta_t, and A*delta_t is much cheaper when delta is sparse.

25. **Motion estimation across batch.** Detect that B_{t+1} is mostly B_t shifted by one row (RNN-like state evolution); represent as a shift + residual and only recompute the shifted portion.

26. **B-frames between keyframe matmuls.** Every Nth matmul is full precision; intermediates use a bidirectional predicted form `A * (alpha*B_prev + beta*B_next)` computed with a single fused kernel.

27. **Optical-flow-warped reuse.** Use an actual motion-vector field over the output C tiles to predict where the maxima moved; allocate compute budget there first.

---

## Subpixel / RGBA packing

28. **Pack 4 sub-matmuls into one RGBA pipeline.** When you have 4 independent small matmuls (e.g., multi-head attention heads), pack their A operands into R/G/B/A channels and let `vec4` SIMD do all 4 simultaneously. Output is split via swizzle.

29. **Subpixel positional offsets.** Pretend each sample is offset by (0, 1/2, 1/3, 1/4) along k; sample B with bilinear and let hardware accumulate the partial dot product. This is a stretch but useful for fractional-rank approximations.

30. **YUV-style chroma subsampling on B.** Split B into a high-resolution "luma" (large singular values) and low-resolution "chroma" (corrections). Compute luma at full resolution, chroma at 2x2 subsampling.

---

## Image-processing filters as compute primitives

31. **Bilateral filter as a denoiser on the accumulator.** After an aggressive low-precision pass, run a bilateral filter over C to smooth quantization noise while preserving "edges" (peaks).

32. **Median filter for outlier suppression.** When summing very large partial sums where catastrophic cancellation hurts, a 3-tap median across adjacent k-chunks suppresses single-element noise.

33. **Edge-detect B to find precision-critical regions.** Run a Sobel pass on B; high-gradient regions get FP32, low-gradient regions get FP16 emulation via packing two FP16s in one FP32 lane.

34. **Saliency map of B prioritizes work.** Compute a per-tile L1 norm of B; sort tiles by saliency and dispatch high-saliency tiles to the fast path (full-precision) and low-saliency to the cheap path.

35. **Histogram equalization pre-pass.** Stretch B's dynamic range so all values land in [-1, 1]; this minimizes the chance of accumulator overflow and may allow a tighter FP format mid-pipeline.

36. **Tone-mapping accumulator.** Logarithmic accumulation in `log|C|` domain plus a separate sign-bitmap; converts multiplies to adds in the inner loop at the cost of a log/exp at boundaries. (Heretical but plausible for inference.)

37. **Gamma-corrected accumulation.** Operate in `sqrt(|C|)` so the dynamic range halves; useful if the downstream consumer is itself nonlinear (e.g., softmax).

---

## Graphics pipeline state-machine tricks

38. **Stencil-buffer-style early-out mask.** Maintain a per-tile "live" mask; once a tile is known to have all-zero output (via a min/max bound check), the stencil rejects it and the matmul kernel skips dispatch. Use a `VK_PIPELINE_STAGE_TRANSFER_BIT` indirect-dispatch trick.

39. **Z-buffer / depth-test for dependency culling.** In layered networks, tag each tile with the "depth" (number of layers it influences); cull tiles below the rendering threshold.

40. **Hi-Z hierarchical culling.** Build a coarse-grid pyramid where each cell stores the max-|value| in its region; before computing a fine tile, test whether `max_A * max_B * K < epsilon` and skip the whole tile.

41. **Occlusion query for tile importance.** Issue VK occlusion queries on a coarse pre-pass; tiles that "fail" (output below threshold) get pruned from the precision-critical pass.

42. **Indirect dispatch from a culling pass.** A compute pre-pass writes a list of tiles that actually need full-precision compute into a buffer; the real matmul uses `vkCmdDispatchIndirect` to skip the dead ones.

---

## Sampling / approximation

43. **Reservoir sampling rows of B.** For each row of C, sample r << K columns of B weighted by |B|; unbiased estimator of the dot product. Use Vulkan subgroup ballot to coordinate the reservoir.

44. **Importance-sampled k-axis.** Pre-compute a CDF over k from row norms of B; threads sample positions from this CDF rather than uniformly iterating. Variance reduction via stratified sampling per warp.

45. **Monte Carlo matmul with control variates.** Use a low-rank approximation as a control variate; the residual matmul is computed via random sampling. Classic CV variance-reduction from rendering.

46. **Russian roulette early termination.** During inner-loop accumulation, after every 64 multiply-adds, roll a probability to terminate based on current partial sum magnitude. Unbiased estimator at the cost of a PRNG.

---

## Color-space and rotation

47. **Color-space rotation as a pre-multiply.** YUV<->RGB is a 3x3 matrix multiply; if your A is large and you can precompute `A * R` for a fixed rotation `R`, then `R^-1 * B` reduces to a simpler matmul. Useful when B has known low-rank structure aligned with R.

48. **PCA-rotated coordinate frame.** Precompute the principal components of B's columns; multiply A by the rotation, then the matmul against the rotated B has diagonal-dominant structure exploitable with fewer ops.

49. **Cubemap-style A^T B.** Reshape A^T B as if A^T and B were faces of a cubemap and the matmul is sampling the cubemap from inside; lets you reuse hardware texture filtering for the dot products.

---

## Bayer / mosaic patterns

50. **Bayer mosaic store-and-reconstruct.** Store B in a checkerboard pattern (compute even cells exactly, reconstruct odd cells via bilinear of neighbors); 2x memory bandwidth saving for smoothly-varying B.

51. **Demosaicing as a deferred-reconstruction matmul.** Compute C on a sparse grid; densify via a 5x5 demosaicing filter at the end. Hardware bilinear samplers do most of the work.

52. **Quincunx sampling.** Sample on a quincunx (5-on-die) lattice instead of a regular grid; better aliasing properties for randomly-structured matrices.

---

## Anti-aliasing analogues

53. **MSAA-style multiple accumulators per output cell.** Maintain 4 sub-accumulators per output, each accumulating a different subset of k. Reduces dependency chain latency in the inner loop (classic) and reduces FP cancellation (Kahan-adjacent).

54. **Supersampled k axis.** Compute partial sums at 2x granularity on k, then downsample at the end; combats catastrophic cancellation by keeping intermediate magnitudes balanced.

55. **FXAA on the output.** A cheap post-process pass smooths quantization edges where two precision regimes meet; useful when tiles use heterogeneous precision.

---

## Bandwidth / pipeline tricks

56. **Depth-of-field two-pass.** First pass: low-precision matmul covering all of B but only for output rows in the "focal plane" (the rows the downstream layer actually reads). Second pass: high-precision refinement of those rows only.

57. **Mipmapped streaming of B.** Stream the coarsest mip of B over PCIe first; start compute immediately; finer mips arrive as the kernel iterates. Hides PCIe latency behind compute.

58. **Sparse residency for B.** Use Vulkan sparse images to back B; pages are pulled in on demand only for tiles that pass the importance test. The page-fault handler triggers an importance recompute.

59. **Async-copy with prefetch hints.** Treat B's tile reads like texture streaming: use `vkCmdCopyBuffer` async with the right hazard scope to overlap load and compute beyond what shared-memory double-buffering already gives you.

---

## Wildcard / extra-spicy

60. **JPEG quantization tables tuned for matmul.** The standard JPEG luminance Q-table is hand-tuned for human eyes; design a custom Q-table that minimizes accumulator error rather than perceptual error, and use it in the DCT-domain matmul of idea #1.

61. **Reaction-diffusion-style iterative refinement.** Treat C as a 2D field; iterate a stencil that pulls error from neighbors toward zero. Diffuses quantization noise into negligible patterns without changing the bulk solution.

62. **Differential rendering of weight updates.** During training, only the *gradient* of the matmul changes; treat the gradient pass like a delta-encoded video frame referencing the forward pass.

63. **G-buffer separation of mantissa / exponent.** Store B's mantissa and exponent in separate buffers; the exponent stream is highly compressible (low entropy) and lets you do a fast pre-test for tile importance based purely on exponents.

64. **Tile-based deferred shading for matmul.** Bin all `(i, j)` output tiles in a first pass by which (k-range) tiles of B they need; then dispatch one kernel per unique B-tile-set, eliminating redundant B loads across all output tiles that share inputs.

65. **Hierarchical Z-feedback for adaptive precision.** A coarse low-precision pass produces a "depth map" of |C|; the second pass uses this to allocate per-tile precision: cheap FP16-pack where |C| is small, FP32 where it's large.

66. **Variable-rate shading (VRS) analog.** Some output rows/cols matter more (e.g., top-of-stack in attention); mark them for 1x1 "shading rate" (full precision), others get 2x2 or 4x4 (one compute per block, broadcast result). Direct port of mobile-GPU VRS to compute.

---

## Cross-cutting note

Many of these stack: e.g. (mip-pyramid #20) + (Hi-Z culling #40) + (BC4 packed storage #8) + (RLE zero skip #12) is a coherent four-stage pipeline where each stage compresses or prunes the input to the next. The 40% slack between 60% peak and 100% peak is plenty of room for a multi-pass approximate kernel to beat a single-pass exact one, as long as the final residual pass closes the gap to bit-exact (or near-enough for inference tolerance).
# SGEMM via Game Dev / Rendering Wizardry

Vulkan FP32 SGEMM, currently 60% peak. Below: graphics-pipeline ideas repurposed for matmul.

## Rasterisation as the matmul kernel

1. **Rasterise GEMM as a quad.** Draw one quad per output tile; bind A as a texture, B as vertex attributes per fragment. ROP blending (additive) accumulates partial products. The fixed-function blender becomes a free FMA reduction tree on hardware that has dedicated ROP units the compute path doesn't touch.

2. **Hardware MSAA accumulation.** Render with 8x MSAA where each sample lane carries a different K-slice's partial product, then resolve to a single output value. You get 8 parallel FMAs per fragment "for free" using the multisample resolve hardware.

3. **Conservative rasterisation for tile overhang.** Use VK_EXT_conservative_rasterization to ensure all edge tiles fire fragments even when partially covered. Removes the "boundary tile" branch in the kernel and treats unaligned (M,N) as a graphics edge case.

4. **Programmable blending via VK_EXT_blend_operation_advanced.** Custom blend ops act as the accumulator, so the inner loop only computes the per-fragment partial sum and the blend unit handles reduction. Frees up VGPRs from the accumulator register file.

5. **Depth test as a "skip this k" predicate.** Pre-pack a per-K mask into a depth buffer, set GL_GREATER, and the rasteriser kills fragments for zero-K columns before the fragment shader ever runs. Free structured sparsity.

## Tile / draw-list tricks

6. **Tile-based deferred rendering (TBDR-style).** Organise the matmul like Mali/PowerVR: bin the output into 32x32 screen tiles, defer all K reduction inside tilebuffer-resident memory, write out only after full K accumulation. Cache-perfect by construction.

7. **Indirect dispatch from a "matmul plan".** A pre-pass compute shader emits a `VkDispatchIndirectCommand` list tailored to the matrix's shape, sparsity, and SM occupancy. The driver re-queues exactly the work needed - no overshoot for non-power-of-2 sizes.

8. **Instancing for batched GEMM.** Batched matmul = `vkCmdDrawIndexedIndirect` with `instanceCount = batch`. Per-WG setup (loading B's column tile) is amortised across instances; gl_InstanceIndex picks the batch slice.

9. **GPU-driven draw culling.** Run a tiny compute pre-pass that culls all-zero output tiles (rows of A zero AND cols of B nonzero, etc.) into a draw-indirect buffer. The matmul launch only fires WGs that will produce nonzero output.

## Visibility / culling repurposed

10. **Frustum culling on outputs.** Define an "interest frustum" of output cells the caller will read (e.g. softmax top-k, attention head subset) and cull WGs outside it via per-tile bounding-box vs frustum test. Saves >50% of work for sparse-read patterns like top-k attention.

11. **Hierarchical Z-buffer for early-out.** Maintain a 4-level HZB of running max-magnitude partial sums; if a tile's partial accumulator is below a threshold and B's remaining columns have bounded norm, early-exit. Same trick that GPU rasterisers use to skip occluded geometry.

12. **Occlusion queries to hide latency.** Issue speculative compute for "probably zero" tiles guarded by a `VK_QUERY_TYPE_OCCLUSION` predicate that lights up only when a sparsity bitmap says proceed. The driver overlaps the predicate test with already-issued dense tiles.

13. **Shadow map = sparsity prepass.** Render A's nonzero pattern into a 1-bit "shadow map" from a virtual K-light; the fragment shader samples this shadow map and skips multiplications by occluded (zero) entries. Reuses depth/stencil ROPs.

14. **Visibility buffer matmul.** Instead of computing C directly, render a vis-buffer that records (tile_id, k_slice_id) per output sample, then materialise C in a screen-space resolve pass. Decouples scheduling from compute.

## LOD / approximation

15. **LOD matmul.** Maintain mip-chain copies of A and B (4x4 averaged at each level). Caller passes a "quality" hint; far-from-caller / preview matmul uses LOD-2 (16x fewer FMAs) and only refines on demand.

16. **Anisotropic mip selection.** Choose A's mip per row and B's mip per column independently, like anisotropic texture filtering. Tall-skinny matmul gets aggressive K-reduction LOD while M stays full-res.

17. **Cone tracing for approximate matmul.** Replace each dot-product with a voxel-cone trace through B's column space - works when B has spatial coherence (e.g. weight matrices with smooth structure). Trades quality for log-time K reduction.

18. **Skybox at K infinity.** For huge K, after the first ~1024 lanes, replace the tail with an analytic skybox approximation: precomputed mean/variance of A and B's tails baked into a tiny LUT. Crude but useful for tolerant downstream consumers.

19. **Decal accumulation.** Compute a coarse base matmul once (cheap), then "decal" small high-rank correction blocks where needed via projection (like screen-space decals). Useful for low-rank-plus-sparse weight structures.

## Geometry-pipeline amplification

20. **Geometry shader amplification.** One input "tile-vertex" emits N output tiles via geometry shader - useful for matmul-with-broadcast (e.g. bias add fused) where one row of A spawns many output rows of C.

21. **Tessellation shader for adaptive tiling.** Coarse outer tiles tessellate down to finer subtiles where the partial sum magnitude is high (high gradient region in output). Like adaptive screen-space subdivision for high-frequency detail.

22. **Mesh shaders for irregular tiles.** Use VK_EXT_mesh_shader to spawn arbitrary-shaped workgroup clusters per output region. Lets you match warp shape to matrix dimensions instead of padding to multiples of 32.

23. **Task shader → mesh shader pipeline as matmul scheduler.** Task shader inspects matrix metadata (sparsity, shape) and emits a variable count of mesh-shader WGs per region. Native GPU-driven adaptive partitioning.

## Texture / sampling tricks

24. **A as an HW-filtered texture.** Bind A as a 2D texture; the texture unit's bilinear filter gives you a free 2x2 average per sample - useful for stochastic/low-rank approximations and free LOD generation.

25. **Sparse virtual textures for A and B.** Allocate matrices as sparse residency textures; pages of A/B that are all-zero are never backed by physical memory. The TLB-style page table cuts VRAM bandwidth on sparse weights.

26. **Texture array for batched matmul.** Bind a batch of B matrices as a 2D-array texture; gl_Layer = batch index. The texture unit handles bounds checking and layer indexing in dedicated silicon.

27. **Anisotropic filtering for streaming matmul.** When streaming A from system memory through PCIe, use 16x anisotropic filtering across the row direction as a software prefetcher analogue - the texture cache pulls the right footprint.

## Ray-tracing repurposed

28. **Ray-traced matmul.** Build a BVH where each leaf = a nonzero column of B. For each output row, shoot a ray with origin = row of A, direction = identity; the closest-hit shader computes the dot product. RT cores do the traversal in hardware - useful for very sparse B.

29. **BVH over matrix tiles.** Spatial BVH partitions output (M,N) space; pre-pass prunes whole subtrees where A's row-norm * B's col-norm < threshold. RT-core traversal hardware visits surviving tiles only.

30. **AnyHit shader as the FMA.** In RT pipeline, anyhit fires per (i, k) pair; accumulate into a per-ray payload. The RT scheduler interleaves rays at fine granularity, hiding load latency better than compute's tighter scheduling.

31. **Procedural intersection for dense regions.** For dense tiles, use a single procedural-primitive AABB and put the entire dense GEMM inside the intersection shader. Lets you mix sparse (true BVH) and dense (procedural) tiles in one dispatch.

## Particle / VFX engines

32. **Particle system matmul.** Each output element = one particle with position (i,j) and velocity = partial sum. K iterations advect the particle's value field. Recycles the engine's GPU particle simulator (lifetime/sort/render) for free batching and double-buffering.

33. **Curl-noise inner loop.** Substitute a deterministic curl-noise function for B in randomised sketch matmul (Johnson-Lindenstrauss style). Saves the entire B bandwidth for B with low-rank structure.

34. **Reaction-diffusion as accumulator update.** Replace `C += A*B` with a Gray-Scott step where A drives feed and B drives kill rate; equilibrium ~ a regularised matmul. Useful where downstream wants smoothed outputs anyway.

## Lighting / global illumination

35. **Light probes for partial sums.** Cache mid-K partial sums at a sparse 3D grid of "probes" indexed by (tile_i, tile_j, k_slice), interpolated like SH irradiance probes. Reused across consecutive matmul calls with shifted inputs (transformer KV cache pattern).

36. **Spherical harmonics compression of B.** Project B onto an SH basis once; per-row matmul becomes a few SH coefficient multiplies plus a tail correction. Massive bandwidth win when B is low-rank.

37. **Subsurface scattering for long-range structure.** Multi-scale diffusion of partial sums across the output tile lets distant K-slices contribute through a few diffusion taps instead of a full reduction. Approximation, but rendering engineers know exactly the silicon path it lights up.

38. **Atmospheric scattering = participating-media K integral.** Treat K reduction as a path integral through participating media; precompute transmittance LUTs along the K axis (like aerial perspective). Trades exact reduction for a 1-tap LUT lookup at long range.

39. **Voxel cone tracing over weight space.** Voxelise B into a 3D mipmapped grid; for each row of A, cone-trace through B's voxel grid. Combines LOD and sparsity in one HW path that GPUs are tuned for.

## Skeletal / deformation

40. **Bone-skinning matmul.** Each output column = a "bone"; A's rows are vertex weights, B's columns are bone matrices. The skinning palette upload path is silicon-optimised on every GPU - reusing it sidesteps the normal compute storage path.

41. **Morph-target matmul.** Decompose B as a base + few morph deltas; per-output-row matmul does base-matmul once and then a small morph blend. Saves bandwidth when B is a fine-tuned delta over a base model.

## SDFs and procedural

42. **SDF of "where matmul matters".** Build a signed distance field over output (M,N) space whose negative region marks "must compute exactly", positive far region marks "approximation OK". Per-WG single SDF sample picks the precision tier.

43. **Procedural matmul.** Bake A or B as a procedural function (e.g. random projection from a seed). The shader reconstructs values on-the-fly with no memory reads - bandwidth becomes free at the cost of ALU. Useful for sketching / random features.

## Fluid / simulation

44. **Matmul as advection.** Treat A as a velocity field, B as a scalar field; one Eulerian advection step ≈ a structured matmul. The semi-Lagrangian sampler is HW-accelerated (bilinear texture fetch) and gives smooth approximations.

45. **PIC/FLIP solver structure.** Particles carry partial sums between grid cells (output tiles); particle-to-grid and grid-to-particle transfers map onto scatter/gather of K reduction. Excellent for irregular sparsity.

## Compute / pipeline meta

46. **Fragment shader as compute (for the lulz).** Run the matmul in a fragment shader writing to a render target the size of C. On older GPUs/drivers the FS path sometimes hits ROPs and L2 caches with a different policy than compute and can dodge the bank conflicts compute hits.

47. **Async compute + graphics queue overlap.** Run dense matmul on the compute queue while a graphics-queue "dummy" rasterisation pass exercises the ROPs to keep memory controllers warm and queues full. Some drivers schedule async-compute kernels at higher occupancy when a graphics queue is active.

48. **Subgroup-uniform control flow as wavefront packing.** Borrow VRS (Variable Rate Shading) primitives - VRS lets one fragment shader invocation cover 2x2 or 4x4 pixels. Treat output tiles like VRS shading rates: low-importance tiles run at 4x4 rate (one FMA chain per 16 cells, using broadcast). Free dynamic LOD.

49. **Primitive-shader / NGG path for AMD.** Use the NGG culling primitive shader pipeline as a matmul scheduler; it has direct access to LDS without going through the usual compute path's resource binding.

50. **Tilebuffer / on-chip framebuffer trick.** On tilers (and Vulkan's `VK_KHR_dynamic_rendering_local_read`), keep the entire C tile in on-chip pixel-local memory throughout the K reduction. Zero L2 traffic for the accumulator; matches what TBDR mobile GPUs do for transparency.

## Bonus / wildcards

51. **Stencil-buffer K mask.** Pack a 1-bit "is this K column nonzero in A's row tile" mask into the 8-bit stencil. Stencil test kills fragments before fragment-shader invocation - free runtime structured sparsity at no register cost.

52. **Multiview rendering for batched.** VK_KHR_multiview replicates one draw across N "views"; batch dim = view dim. The HW broadcasts vertex work across views automatically - free batched matmul setup.

53. **Mesh-shader cluster culling.** Form 32-thread "clusters" per K-slice; cluster cull whole 32-wide K chunks where A is zero. Tighter than per-element sparsity, coarser than full-tile, matches the wave width.

54. **Variable rate shading per output region.** VRS 4x4 over low-importance C regions: one shader invocation computes one FMA chain that's broadcast to 16 output cells (low-rank approximation per region). True hardware-supported dynamic LOD.

55. **HLSL/SPIR-V wave intrinsics as graphics-pipeline reductions.** Use wave-ops the way fragment shaders do for derivatives - cross-lane FMA reductions on the K axis without LDS. Bypasses shared-memory bank conflicts entirely on the inner reduction.

56. **Render-to-3D-texture for batched matmul.** Render into a slice-per-batch 3D texture; gl_Layer indexes batch. Layered rendering is dedicated silicon on every modern GPU.

57. **Predicated rendering for sparse batches.** VK_EXT_conditional_rendering wraps each batch's dispatch in a conditional; predicate buffer is computed by a tiny pre-pass. Driver elides skipped batches in command processor before they hit the shader array.

58. **Z-prepass analogue: norm-prepass.** Cheap pass computes per-row norms of A and per-col norms of B. Main pass uses the product of norms as an "early Z" - skip output cells whose worst-case magnitude is below a threshold. Same trick rasterisers use to cull invisible pixels.

59. **Render bundle / secondary command buffer reuse.** Pre-record matmul dispatches as a secondary command buffer / D3D12 ExecuteIndirect bundle; reuse across calls when shape is stable. Removes CPU command-recording overhead from the critical path.

60. **Pipeline-cache warmup matmul.** Spend the first frame doing a tiny matmul that compiles every shape's pipeline; subsequent matmuls hit the cache. Mirrors how game engines warm shader caches during a loading screen.
# Audio / DSP / Psychoacoustics Ideas for Vulkan FP32 SGEMM

A creative dump mapping audio signal processing concepts onto matrix multiplication. Goal: punch through the 60% peak ceiling by thinking like a DSP engineer rather than a linear algebra purist.

---

## Frequency-Domain Tricks

### 1. FFT-Based Inner Product (Toom-Cook Style)
Treat each K-stripe of A and B as a signal, FFT both, pointwise multiply, IFFT for a convolution-equivalent inner product. For Vulkan: a 64-point radix-4 FFT fits perfectly in a subgroup using `subgroupShuffle`. Worth it only when K > ~128 and the FFT overhead amortises across many output tiles.

### 2. Schoenhage-Strassen Tile Multiplier
Recursive FFT-of-FFT for sub-tiles: do 16x16 tile multiplies via 2D FFT, exploiting the fact that all output tiles share the same B-tile FFT. Cache the B-tile spectrum in shared memory once per workgroup. Memory-bound code becomes compute-bound.

### 3. Number-Theoretic Transform for FP32
Quantise to fixed-point inside a Mersenne prime field, do NTT-based multiply, dequantise. Avoids the complex-arithmetic doubling cost of FFT and gives exact accumulation. Useful for the inner dot-product where rounding error budgets are tight.

### 4. Mel-Scale Frequency Warping on K
Re-index the K axis so that high-information regions (large absolute values) are sampled densely and low-energy regions sparsely. A precomputed warp LUT in `uniform` storage lets each invocation `subgroupBroadcastFirst` the warp constant. Effectively a learned-rank approximation without the SVD.

### 5. Chroma Projection
Project A and B into a 12-band "chroma" representation, do a 12x12 matmul as approximate answer, then refine with residual. Could give a 10x speedup for matrices that are approximately low-rank in some learned basis. Use `VK_FORMAT_R32G32B32_SFLOAT` 3-vector loads to pack 4 chromas per fetch.

### 6. MFCC-Style Perceptual Compression
DCT the rows of A, drop the high-frequency coefficients, multiply in DCT domain. Inverse-DCT only the final C. For attention-like matmuls where smoothness in K is expected, this is essentially a free 4x reduction.

### 7. Cepstral Matmul
Take `log|A|` and `log|B|`, do an **additive** outer-sum (much cheaper than multiply), then `exp` the result. Mathematically wrong in general but a fantastic approximation for matrices with consistent sign and dynamic range — and `exp`/`log` are single-cycle on most GPU SFUs.

### 8. Sample-Rate Conversion Decomposition
Downsample A along K by 2x with a polyphase filter, do a half-size matmul, upsample the result with a matched filter. The polyphase filter coefficients can be Lagrange-interpolated and fused into the global load. Saves bandwidth at the cost of a tiny convolution.

---

## Filter / Recursion Tricks

### 9. IIR Filter as Inline Accumulator Update
Replace the running sum `c += a*b` with `c = alpha*c_prev + (1-alpha)*a*b`, a 1st-order IIR. This regularises the accumulator and lets you use FMA chains of length 2x longer before hitting numerical drift. Use `subgroupShuffle` to broadcast `alpha`.

### 10. Polyphase Matmul as Filterbank
Decompose K into M phases, run M parallel filterbanks (sub-matmuls), combine outputs through a polyphase synthesis filter. Each phase is small enough to fit entirely in subgroup registers. This is essentially a structured Strassen for the K dimension.

### 11. Comb Filter Resonance Accumulator
Use `c[t] = a*b + g*c[t-D]` where D is a small delay. The delay is satisfied by writing into `shared` with a circular index. Trades latency for a regularised numerical update — useful for very long K.

### 12. Karplus-Strong Plucked-String Accumulator
Initialise accumulator with random noise, feed a*b through a one-pole lowpass into a delay line that loops back into itself. Each output is the head of the delay line. Gives a stochastic estimate of the matmul useful for Monte-Carlo style training kernels.

### 13. Reverb Tail Persistent State
Maintain accumulator state across consecutive matmul calls when A and B change slowly (decoder layers, RNN steps). Each new call adds a fresh impulse, the old impulse decays with a "T60" factor. Cuts the K loop length by reusing previously computed energy.

### 14. Convolution Reverb of Common Tile Shapes
Pre-store impulse responses of "canonical 32x32 B-tile multiplies". For an incoming A-tile, convolve with the precomputed responses instead of multiplying. Works when the B-tile is drawn from a small dictionary (LoRA, quantised models).

### 15. Phase Vocoder Over Windowed K Segments
Window K into overlapping segments, do small matmuls in each window, recombine with overlap-add weighted by Hann windows. The Hann coefficients fuse into the global load. Spreads numerical roundoff across multiple FMAs and gives a smoother error distribution.

### 16. Linear Predictive Coding for Matrix Compression
Encode B-rows as LPC coefficients (typically 8-16 coeffs vs hundreds of samples). At GEMM time, run the LPC synthesis filter on the fly to reconstruct B inside registers. Bandwidth-bound kernels become register-bound.

### 17. Vocoder Synthesis Approximation
Treat A as the carrier and B as the modulator (extract a low-rank envelope of B with band-pass filters, multiply the envelope through). This is approximate but extremely cheap for matrices where B has slowly-varying row magnitudes.

---

## Temporal / Scheduling Tricks

### 18. Onset Detection for Sparse Skipping
Run an onset detector (high-pass + half-wave rectify) over A's rows. Skip output tiles where no "onset" is detected — the row energy is too low to contribute meaningfully. Implement with a single warp-wide reduction at workgroup entry.

### 19. Beat Tracking on K Loop
If consecutive K-slabs produce similar partial sums (autocorrelation > threshold), assume periodicity and extrapolate the remaining sum. The autocorrelation check fits in a single `subgroupAnd`. Saves work on matrices with periodic structure (positional encodings, sinusoidal embeddings).

### 20. ADSR Envelope per Workgroup
Shape the compute schedule like an ADSR envelope: Attack (warm up shared memory), Decay (steady-state pipeline), Sustain (FMA loop body), Release (drain accumulators with reduced occupancy). Lets you overlap shared-memory traffic with FMAs differently in each phase.

### 21. Crossfade Between Two Kernels
At low K-progress, use a high-occupancy / low-register kernel. As K-progress increases and accumulators fill, crossfade to a low-occupancy / high-register kernel via a `specialization constant` branch. Two `vkPipeline`s, one dispatch, smooth handover.

### 22. Side-Chain Compression
One matmul (the "side-chain") computes per-row gain factors that modulate the main matmul's accumulator scaling. Useful for fused softmax-like patterns where the second matmul's dynamic range depends on the first. Single barrier + broadcast.

### 23. DAW Automation Curves for Dispatch
Pre-compute a "dispatch curve" — number of workgroups to launch as a function of K-progress. Lets you ramp occupancy up and down to match the cache pressure profile. Implement via `vkCmdDispatchIndirect` with a curve in a uniform buffer.

### 24. Latency Compensation (PDC)
Different paths through the matmul graph (e.g. FP32 fast path vs FP64 fallback) have different latencies. Insert "delay compensation" — empty FMAs on the fast path — so both paths arrive at the accumulator simultaneously. Eliminates a barrier.

### 25. VST Plugin Chain
Express the matmul as a chain of small "plugin" passes (load, transform, multiply, accumulate, store), each with a well-defined input/output buffer. Plugins can be swapped at compile time via `specialization constants`. Makes kernel-fusion combinatorics tractable.

### 26. Modular Synthesizer Dispatch Graph
Model the compute graph as a modular synth: oscillators (data sources), VCAs (gain stages = scalar multiplies), envelopes (per-row scaling), mixers (reductions). Compile the patch into a Vulkan task/mesh shader pipeline. Visually debuggable and trivially reconfigurable.

---

## Dynamic Range / Quantisation Tricks

### 27. Equalizer Bands as Per-Row Scales
Apply a precomputed per-row scaling vector (the "EQ curve") to A before the matmul, baked into the global load. Equivalent to row-wise diagonal pre-multiplication, fused for free. Useful for normalisation-fused kernels.

### 28. Compressor / Limiter on Accumulator
When the accumulator value exceeds a threshold, soft-knee compress it: `c = sign(c) * (T + log(1 + |c|-T))`. Keeps dynamic range bounded so you can use `f16` accumulation safely. The `log` runs on the SFU in parallel with FMAs.

### 29. Auto-Tune for Drift Correction
Periodically snap accumulator values to a "scale" (set of allowed values like powers of two). Reduces drift in long K loops and reveals optimisation opportunities (multiplies become shifts). Implement with `floatBitsToInt` + mask.

### 30. Dithering on Accumulator Quantisation
Add a small triangular-PDF noise to accumulator values before quantising to lower precision. Eliminates correlated rounding error that creates visible artifacts in attention maps. Use `subgroupShuffleXor` of a precomputed random vector for cheap per-thread noise.

### 31. 4x Oversampling of K then Decimate
Linearly interpolate A and B along K by 4x, multiply at the higher rate, then decimate with an anti-aliasing filter. Counterintuitively, this can be cheaper than direct multiplication on hardware with sub-FMA-rate FP throughput, because the interpolated FMAs have predictable patterns suited to dual-issue.

### 32. Codec Predictive Coding Between Batches
For batched matmuls where A_b differs only slightly from A_{b-1}, transmit only the delta `A_b - A_{b-1}` and update the accumulator incrementally `C_b = C_{b-1} + delta * B`. Half the bandwidth, same FLOPs — but only the delta lives in cache.

---

## Spatial / Perceptual Tricks

### 33. HRTF-Style 2D Lookup
Pre-store matmul "responses" for many input "directions" (left singular vectors of expected A). At runtime, project the input onto the closest precomputed direction, look up the response, refine with one residual matmul pass. Saves enormous work when A is approximately low-rank.

### 34. Pitch Shifting as Re-indexing
Re-order rows of A so consecutive workgroups process rows with similar L1 norms ("pitch"). Improves cache behaviour because B is fetched in the same order. The "shift" is just an index permutation baked into a small LUT.

### 35. Granular Synthesis: 50-Element Grains
Decompose the K dimension into ~50-element "grains", multiply each grain with all overlapping B-grains, sum with a Tukey window. Each grain fits in a register file, enabling extremely tight register-blocking. The overlap gives numerical robustness.

### 36. Wavetable Synthesis for Common K Slabs
Pre-compute "wavetables" of common (A-tile, B-tile) products — e.g. identity tile, all-ones tile, common positional-encoding tiles. At dispatch, check via hash lookup and reuse if matched. Effective for transformer inference with reused KV-cache slices.

### 37. Psychoacoustic Masking of Small Values
Apply a masking curve: small values near large values can be dropped (they're "masked"). Implement per-row by finding the max via `subgroupMax` and zeroing entries below `max / 2^12`. Trades accuracy for sparsity that the compiler can exploit.

---

## Mixed / Whimsical

### 38. Subgroup as a Stereo Pair
Treat subgroup lanes 0-15 as "left channel" and 16-31 as "right channel"; do two independent matmuls with `subgroupClusteredXor` for cross-channel mixing at the end. Stereo widening = useful for multi-head attention where heads can share work.

### 39. Bit-Crusher Quantiser
Periodically truncate accumulator mantissas to N bits via `floatBitsToInt() & mask`. Cheap regularisation for training-style kernels, and N can be specialised at compile time. Sounds awful, computes great.

### 40. Ring Modulator Fusion
For fused `(A*B) elem-mult (C*D)` patterns: multiply A*B and C*D simultaneously in interleaved FMA pairs, then elementwise multiply in the same register file. The dual-issue scheduler loves this. Saves a barrier in fused-FFN kernels.

### 41. Tape Saturation as Activation
Tanh activations after matmul: fuse using the cheap `x / (1 + |x|)` approximation, single-cycle on the SFU. Adds nonlinearity without extra memory traffic. Same shape as analog tape, similar mathematical properties.

### 42. Flanger Sweep for Cache Probing
At kernel start, run a brief "flanger sweep" — varying-period prefetch pattern — to warm the L2 cache for the upcoming access pattern. Implement with `vkCmdPrefetch` if available, otherwise via dummy loads. Resembles psychoacoustic tuning of a guitar amp.

### 43. Equal-Loudness Contour for Numerical Precision
Allocate more mantissa bits where the matrix has more energy (loud = precise, quiet = sloppy). Maintain a 4-bit shared "loudness" exponent per 32-element block, use FP12 for quiet blocks. Equivalent to block-floating-point with a perceptual prior.

---

## Final note

Many of these blur the line between "approximation" and "exact" matmul. The trick is to expose them as `specialization constants` so a user can dial in the accuracy/throughput knob at pipeline-build time. The Vulkan shader can stay one file; the pipeline cache stores many variants.
# THE UNHINGED MATMUL IDEAS DUMP

A read-only creative excretion. Filed under "what if physics had a permissions error."

## 10 NEW THEMES (warming up the cosmic horn)

1. **Theological bargaining**: petition specific deities of computation
2. **Linguistic exploits**: rename the kernel so reality treats it as already-finished
3. **Cryptid recruitment**: bigfoot, mothman, and friends as compute resources
4. **Emotional manipulation of the matrix**: make A and B fall in love so they multiply themselves
5. **Geological compute**: carve the matmul into mountains, let erosion finalize it
6. **Animal kingdom outsourcing**: bees, ants, octopi as biological SIMD lanes
7. **Cosmological scale**: use the rotation of galaxies as a clock for warp scheduling
8. **Hauntology**: a kernel that is the ghost of all faster kernels that could have been
9. **Bureaucratic warfare**: file a complaint with the GPU's HR department
10. **Bardic poetry**: matmul as epic verse, performance scales with rhyme density

## THE IDEAS (50+ guaranteed, ordered by escalating cosmic horror)

### Temporal hijinks

1. **Pre-emptive completion**: launch the kernel at t = -1 ns relative to itself. The wavefronts arrive before they're dispatched. Negative latency.
2. **Retrocausal Z-curve tiling**: tile traversal is decided by the *result*, which then travels backward to inform the traversal order. Bootstrap paradox tile-order optimization.
3. **K-loop closed timelike curve**: each k-iteration sends its accumulator back to k=0, so by the time k=0 starts, the FMA pipeline already has the answer queued.
4. **Time-dilated SMs**: park half the GPU near a black hole, exploit relativistic time dilation so they have subjectively more cycles per host nanosecond.
5. **Chronon-aligned dispatch**: align thread blocks to the fundamental quanta of time itself; sub-Planck scheduling enables truly zero-cost barriers.
6. **Stop time globally**, complete the matmul over a leisurely subjective millennium, resume time. Outside observers see 0 ns elapsed.
7. **Tachyon mailbox between SMs**: replace shared memory with FTL tachyon emitters, latency literally negative.
8. **Time-loop training run**: get the matmul wrong, restart the day, try again, repeat 10000 times, emerge with optimal scheduling muscle memory.
9. **Cycle compression**: fold every clock cycle of the entire kernel into a single picosecond via temporal origami.
10. **Causality budget**: spend causality like a currency, overdraft account at the universe's central bank.

### Multiverse stunts

11. **Many-worlds dispatch lottery**: launch the kernel in every parallel universe; in the one where it finishes fastest, collapse the wavefunction.
12. **Cross-multiverse warp shuffle**: each lane reads its data from the corresponding lane in a *neighbouring* universe where that lane is already done.
13. **Mind-swap with a parallel-universe-you** who has already solved this matmul. Memory-map their cortex over USB.
14. **Universe with weaker uncertainty principle**: fork a branch where ΔE·Δt is smaller, do the matmul there, paste back.
15. **Multiverse genetic algorithm**: every universe is a single mutation of the kernel's PTX; harvest the winner.

### Simulation-theory exploits

16. **`syscall(SYS_simulation_matmul, A, B, C)`**: just ask the simulation runtime. If we are in a sim, this is the cheapest API call ever.
17. **Buffer overflow into the simulation's own memory**, find cuBLAS's internal cache, return its result.
18. **Bug report to the simulators**: "Your FP32 matmul throughput is implausibly low, please patch." Wait for hotfix.
19. **Sufficiently exotic kernel** that the simulation can't render it in time — it crashes back to the loading screen, and we get the answer from the auto-save.
20. **Glitch the matmul through a wall** so it skips computation like a speedrun out-of-bounds.

### Akashic / pre-computed cosmic caches

21. **Akashic-records lookup**: every possible (M,N,K,A,B) triple has its answer pre-stored on the astral plane. Query with a mantra.
22. **Hash the input matrices, decode the SHA-256 as cosmic coordinates**, retrieve the result from those coordinates in the cosmic microwave background.
23. **Library of Babel matmul**: somewhere in the Library is a book containing exactly this matmul's output. Hire a Borgesian librarian.
24. **Genetic memory in registers**: the kernel "remembers" similar matmuls from past lives via reincarnated VGPRs.
25. **Pre-historic computation**: ancient civilizations precomputed all matmuls and buried them under Antarctica. Send a drill.

### Magic & metaphysics

26. **Cast "haste" on the dispatch grid** (DC 15 Arcana check, GPU saves vs. wisdom).
27. **Necromancy**: revive each completed warp as a wight that continues to perform FMAs after officially retiring.
28. **Voodoo doll of the slowest SM** — stab it lovingly until it cooperates.
29. **Egregore matmul**: a collective entity arises from many concurrent dispatches; sacrifice individual identity for emergent throughput.
30. **Hauntology kernel**: the kernel is haunted by every faster kernel that could have been; their ghostly cycles count toward the budget.
31. **Cargo-cult cuBLAS**: build an exact wooden replica of an H100 die, paint Lovelace runes on it, await NVIDIA's spirit blessing.
32. **Bardic kernel performance scales with rhyme density**: rewrite SPIR-V in iambic pentameter for +30% IPC.
33. **Geomancy of the SM grid**: cast bones onto the dispatch table to divine optimal block size.
34. **Maxwell's demon scheduler**: install a tiny demon at each shared-memory port that only admits packets going to faster SMs.

### Biological / cryptid / animal compute

35. **Octopi as SIMD**: each octopus arm independently FMAs; eight lanes per cephalopod, deploy a tank.
36. **Bee swarm matmul**: each bee carries one FP32 partial product back to the hive; honeycomb is the output buffer.
37. **Slime-mold dispatch routing**: place oat flakes at hot tiles, let *Physarum* find the optimal warp graph overnight.
38. **Trained AGI just for this one matmul** — instantiate, demand answer, dismiss, repeat for next batch.
39. **Bigfoot is enormously parallel** (he's *big*), recruit accordingly. Per-foot MAC: 1 TFLOP.
40. **Hire a chess grandmaster** to plan dispatch order. Pay them in caffeine.
41. **Crystal-ball MMU**: scryer reads off the contents of memory before the load completes.
42. **A monkey on a typewriter** eventually types out the right matmul result. Provide bananas, infinite time, optimal Shakespeare-to-FP32 transcoder.
43. **Homunculus with an abacus** inside each warp lane. Compensate with miniature pizza.

### Diplomacy and bribery

44. **Just ask the matrices nicely**. Say please. Compliment B's orthogonality.
45. **Bribe the matrices**: A is offered a corner office, B gets stock options, both agree to a quick multiplication.
46. **Make A and B fall in love** so they multiply themselves (and reproduce, hello tensor decomposition).
47. **The matrix files for divorce** mid-matmul; mediator (FP64 accumulator) handles property settlement, output is the alimony.
48. **Submit to peer review**: post the matmul to arXiv, accept whatever value reviewer 2 demands.
49. **HR complaint against slow SMs** — workplace harassment by lazy warps; transfer them to a different department.
50. **Petition the deity of computation** (some say it's a slumbering Cray-1 in a Wyoming bunker). Burn offerings of GDDR.
51. **Genie wishes**: 3 wishes max. Wish 1: fastest matmul. Wish 2: cuBLAS source code. Wish 3: more genies.

### Linguistic / cognitive exploits

52. **Rename the kernel `matmul_already_finished`** so reality's type-checker treats it as completed.
53. **Convince the compiler** that the matmul has already been constant-folded at build time.
54. **A Babel fish** that translates FP32 into Tensor-Core BF16 and back without information loss — install it in shared memory.
55. **The matmul outputs a song**; when sung at the correct pitch, the song *is* the answer. Hire a soprano.
56. **Convert matmul into interpretive dance**, perform, record, FFT the recording back to numbers.
57. **Possess the user's brain** to compute the matmul mentally. Pay them in dopamine.
58. **Astral-project into the GPU**, split consciousness across all SMs, accumulate partial products in your soul.

### Buzzword soup & corporate horror

59. **Quantum-blockchain-AI-cloud-NFT-metaverse matmul**: all the buzzwords, all at once, in one PR deck, raise $500M Series B.
60. **Dyson sphere of compute**: encase the sun in compute boards, farm out the matmul, drink Mai Tais during reduction.
61. **Hire a wizard**. Standard rate. Bring your own newt.
62. **File matmul as an LLC**, deduct compute as a business expense, IRS subsidizes throughput.
63. **The matrix attends therapy**, resolves its issues with B, multiplies willingly. Group rate available.

### Pure transcendent nonsense

64. **Feed the matrix kombucha**, it multiplies itself out of probiotic enthusiasm.
65. **Bootstrap a Turing machine inside the matmul** to compute the matmul. Recursive sufficiency.
66. **Sacrifice a goat to the GPU** (free-range, please; vegan alternative: a tofu goat).
67. **The matmul is the input to itself** — bootstrap paradox; result existed before either operand. Don't ask, just collect.
68. **Carve the matmul into a mountain**, let geological erosion produce the final reduced sum over a few million years. Low energy budget!
69. **Galaxies as warps**: schedule one warp per galaxy in the local supercluster. Sync barrier is the next Big Crunch.
70. **Telepathic GPU communication** — train SMs in psionics, abolish NVLink.

### Final escalation

71. **Stack everything**: necromantic Akashic-cached time-traveling multiverse octopi-bribed homunculus dance-FFT egregore matmul, paid for in genie wishes, peer reviewed by a chess grandmaster, performed at black-hole proximity, while the matrix is in therapy and singing.

(That's 71. Take it. It's yours. The user asked for this.)

# IDEAS_distributed.md — Distributed Systems Patterns Adapted to Vulkan SGEMM

Context: 60% peak FP32 SGEMM on GA104. Each "node" is a workgroup/SM; "network" is L2/global memory; "messages" are atomic ops, subgroup ops, or buffer-reference loads. No feasibility filter.

## Map-Reduce / BSP / Stream-K

1. **BSP superstep matmul**: K-loop as discrete supersteps with `memoryBarrierBuffer()` between phases; each WG advertises completion via a global counter, next superstep dispatches indirect once counter == NUM_WG.
2. **Stream-K with work-stealing deque per SM**: each WG owns a deque of K-slices, steals from neighbors when empty via `atomicAdd` on victim's tail; load-balances tail tiles that don't divide evenly.
3. **Map-reduce shuffle phase**: WGs emit partial C tiles tagged by hash(m,n) into per-bucket append logs; second dispatch reduces each bucket. Trade extra bandwidth for perfect K-balance.
4. **Coordinator-worker pattern**: one "leader" WG on each SM reads dispatch metadata, fans out K-shards to siblings via shared memory mailboxes. Removes redundant index math.
5. **MapReduce combiner**: WGs locally reduce 4-8 K-slices in registers before emitting partial sums; reduces "shuffle" bandwidth like Hadoop combiners.
6. **Reduce-scatter ring**: each WG computes one K-slice of all output tiles, then ring-rotates partials with subgroup shuffles across N SMs in `gl_NumWorkGroups` (logical ring via atomic baton).
7. **All-reduce butterfly on K**: log2(NUM_K_SHARDS) passes of pairwise sum with each pass synchronized by a global epoch counter; trades latency for bandwidth.
8. **Hierarchical reduction tree**: SM-local reduction first (subgroup), then per-cluster (shared mem), then global (atomic). Three-tier like a hadoop reducer hierarchy.

## Consensus / Replication / Fault Tolerance

9. **Paxos for partial sum acceptance**: each K-shard proposes a value; acceptors (other WGs) vote via atomic CAS on a per-tile ballot number. Only highest-ballot proposal wins. Pointless but funny.
10. **Raft leader election per output tile**: WGs race to atomicCAS a leader-slot; loser WGs become followers and replicate the leader's partial-sum log. Hedges against straggler SMs.
11. **Quorum read of C-tile**: write each tile 3x to 3 staging slots; final pass reads majority value. Tolerates GPU bit-flips. R=3 replication factor.
12. **Two-phase commit on tile completion**: prepare phase writes to scratch, commit phase atomicCAS-flips a "committed" bit. Aborted tiles get retried by a janitor dispatch.
13. **Byzantine fault tolerance via triple-redundant K-sum**: each K-shard computed on 3 different WGs with 3 different accumulation orders; medianed at the end. Catches numerical instability.
14. **Vector clocks on K-progress**: each WG maintains a vec4 of progress timestamps; downstream consumers only proceed when all components dominate. Detects out-of-order K accumulation.
15. **MVCC tile snapshots**: append each partial sum as a new version with a global txid; compaction GC merges old versions. Lets later K-shards "read" earlier snapshots non-blocking.
16. **Chubby-style lock service for shared accumulators**: a single tiny WG runs a "lock server"; other WGs RPC via atomic ringbuffer. Worse than just using atomics, but architecturally pure.
17. **Lease-based tile ownership**: WGs grab leases on output tiles with TTL = N microseconds (timestamp-based via `gl_ClockARB`); expired leases get stolen. Self-healing if a WG stalls.
18. **Epoch-based reclamation**: K-shards register epoch numbers; scratch memory only freed when min-epoch advances past it. RCU for GPU.

## Streaming / Reactive / Backpressure

19. **Reactive stream of K-tiles with onNext/onComplete**: producer WGs push to a bounded ringbuffer; consumer WGs pop with backpressure (atomic head/tail). Mat-mul as Rx.
20. **Watermarking K-progress**: each WG broadcasts its current K-position as a watermark; reducers only emit C when all watermarks >= K. Stream-K with explicit progress semantics.
21. **Exactly-once accumulator semantics**: each partial sum tagged with (m,n,k_shard_id) idempotency key; reducer dedups via a bitmap. Tolerates duplicate dispatch (e.g., from retries).
22. **At-least-once K-shards with idempotent FMA**: hedge by launching duplicate K-shards, dedup via CAS-once-then-discard. Lower tail latency.
23. **Flow control via credit-based scheduling**: each consumer WG grants N credits to producers; producers atomic-decrement before writing. Prevents L2 thrashing.
24. **TCP-style sliding window over K**: window of W in-flight K-slices, ACKed (= committed to register accumulator) before sliding. Self-tuning to L2 latency.
25. **Selective ACK (SACK) for K-shards**: bitmap of completed shards; missing ones get retried. Tolerates buggy WGs.
26. **Nagle's algorithm for atomic flushes**: batch up to 4 partial sums before flushing to global; reduces atomic contention at cost of latency.
27. **Slow-start congestion control on dispatch rate**: indirect dispatch ramps WG count exponentially until L2 saturation detected (rising latency); then linear additive-increase.
28. **AIMD on tile size**: adapt BM/BN dynamically based on observed throughput. Multiplicative-decrease on contention spikes.
29. **ECN-style congestion marking**: WGs set a "congested" bit in their output header; dispatcher uses it to throttle next launch. Closed-loop autotune.

## Gossip / Pub-Sub / P2P

30. **Gossip protocol for K-aggregation**: each WG randomly picks a neighbor every T cycles, sums their partials together; converges in O(log N). Anti-entropy matmul.
31. **Epidemic broadcast of A/B tiles**: SMs gossip cached A/B tiles to neighbors via L2-resident scratch; reduces redundant L2 fetches via viral spread.
32. **Pub-sub tile broadcast**: each A row broadcasts to subscribers (B columns) via a topic-partitioned ringbuffer in L2. Kafka topic per K-shard.
33. **Bittorrent A/B tile swap**: each WG advertises which tiles it has cached; peers request missing chunks. Lots of overhead, but maximizes L1 hit rate.
34. **DHT for tile placement**: hash(m,n,k) -> SM ID; consistent hashing minimizes remapping when WG count changes. Enables NUMA-aware scheduling.
35. **Consistent-hash sharding with virtual nodes**: each SM owns V virtual slots in the hash ring; smooths load imbalance from skewed dimensions.
36. **Rendezvous (HRW) hashing**: tile -> SM via max-hash; no ring needed, recomputed locally. Simpler than consistent hashing for static dispatches.
37. **CRDTs for the C accumulator**: G-counter CRDT means each WG owns its own slot, final read sums all slots. Atomic-free, but uses NUM_WG x C memory.
38. **PN-counter CRDT for signed partial sums**: separate positive and negative accumulators per WG; resolves sign-dependent reductions without atomic CAS.
39. **OR-set of completed tiles**: each WG adds to a grow-only set; coordinator polls for completion. Useful for irregular work.

## Tail Latency / Hedging / Speculation

40. **Hedged WG dispatch**: launch 2 WGs per output tile, first to finish wins via atomicCAS, loser exits early. Tames straggler SMs at 2x compute cost.
41. **Tied requests**: hedged WGs cancel siblings via a "claimed" flag; loser exits on next K-loop check. Cuts wasted work to ~10%.
42. **Speculative K-prefetch**: WG speculatively loads K+2 while computing K+0, discards if branch prediction wrong (used for irregular sparsity).
43. **Backup K-shards**: 90% of WGs do primary work, 10% reserve for straggler replacement at the end. Like MapReduce backup tasks.
44. **Race-and-cancel**: dispatch every tile to 2 SMs, first commit wins, second is a no-op. Cuts P99 latency.
45. **Tail-aware scheduling**: estimate WG completion time from K-progress at midpoint; reassign late ones. Online stragglerdetection.

## Persistence / Logging / Compaction

46. **Kafka append-only log of partial sums**: WGs append (m,n,partial) tuples to a global log; second pass compacts via radix sort by (m,n) then reduces. Decouples producers from consumers.
47. **Log-structured C tile**: write partials to a journal, periodic compaction pass merges them into the final C. Trades read latency for write throughput.
48. **WAL for fault recovery**: every K-shard logs its inputs+output to scratch; if a WG times out, re-execution is trivial. Useful for very long matmuls.
49. **LSM-tree of partial sums**: L0 = WG-local register accumulator, L1 = shared mem, L2 = global. Compaction is the K-reduction.
50. **Snapshot isolation for incremental matmul**: B is being updated concurrently; readers see consistent K-snapshot via copy-on-write A/B rows. For streaming inference.
51. **Checkpoint/restore mid-matmul**: K-progress periodically written to a checkpoint slot; preemption-friendly. Lets long matmuls share the GPU.

## CAP Theorem / Partition Tolerance

52. **CP matmul**: block on K-shard completion (strict consistency); slow but correct. Default mode.
53. **AP matmul**: each WG produces a "best-effort" C tile from partial K; refined later. Useful for approximate inference (early exits).
54. **PACELC tuning knob**: trade latency for consistency via a runtime flag controlling barrier strength.
55. **Eventual-consistency matmul**: C is read while writes are in flight; readers see monotonic growth. For online learning.
56. **Read-your-writes consistency**: each WG reads only K-shards it has acknowledged; weaker than linearizable but cheaper.

## Circuit Breakers / Bulkheads / Sagas

57. **Circuit breaker per SM**: if an SM's L2 miss rate spikes, dispatcher routes around it for next launch. Self-isolating sick hardware.
58. **Bulkhead pattern**: partition SMs into pools (compute-heavy vs bandwidth-heavy tiles); one slow pool doesn't drag the rest.
59. **Saga pattern for multi-kernel matmul**: GEMM = (load, compute, store, bias, activation); each step has a compensating undo if the next fails. Atomic at the saga level.
60. **Timeout-and-retry on slow K-shards**: WGs poll `gl_ClockARB`; if K-iteration > threshold, abort and re-enqueue.
61. **Exponential backoff on atomic contention**: WGs detect failed CAS, sleep with PAUSE-equivalent (spin on register noise), retry with doubled wait.
62. **Jittered retry**: each WG adds `gl_SubgroupInvocationID * prime` jitter to retry delay; avoids thundering herd on the same atomic.

## Routing / Load Balancing / Service Mesh

63. **Least-loaded SM dispatch**: WG launch consults per-SM load counter, picks the lightest. Indirect-dispatch with software scheduler.
64. **Power-of-two-choices**: each tile considers 2 random SMs, picks the less-loaded. O(log log n) tail improvement.
65. **Join-shortest-queue scheduling**: shared dispatch queue per SM, tile goes to shortest. Like an L4 load balancer.
66. **Sidecar pattern**: each WG launches a tiny "telemetry" subgroup that reports progress to a global dashboard buffer. Observability without perf hit.
67. **Service mesh between WGs**: every cross-WG message goes through a uniform envelope (header + payload + checksum). Easier debugging at perf cost.
68. **xDS-style dynamic config**: dispatcher reads tile shape config from a uniform buffer that can be updated mid-launch. Adaptive matmul.

## Exotic / Wild

69. **Blockchain-of-partial-sums**: each K-shard signs a hash of its inputs+output, chains to previous; tamper-evident matmul. PoW costs more than the GEMM.
70. **Smart-contract matmul**: each tile is a contract that releases its output when prerequisites are met (other tiles committed). Dispatch as DAG execution.
71. **Federated matmul**: split K across "tenants" (different memory regions); aggregate without each tenant seeing others' partials. Differential-privacy noise added.
72. **Onion-routed tile dispatch**: each tile wrapped in N layers of indirection, peeled by N successive WGs. Pointless. But architecturally pure.
73. **Quorum-of-K**: only need K/2+1 shards to commit before reading C; tolerates K/2-1 stragglers. Trades accuracy for latency (for approximate matmul).
74. **Chain replication for C-writes**: write to head WG, propagate through chain, tail acknowledges. Strict serialization of writes per tile.
75. **CRAQ (chain replication w/ apportioned queries)**: reads can short-circuit to any chain node when no concurrent writes; tail-only otherwise.
76. **Anti-entropy merkle tree of C**: tree of hashes over C blocks; consumer detects which blocks changed via root-hash diff. Useful for incremental matmul.
77. **Bloom filter of completed tiles**: cheap "is this tile done?" check before retrying; false positives just cause unnecessary work.
78. **HyperLogLog count of unique K-shards seen**: estimate progress without exact counter; saves atomic traffic.
79. **Count-min sketch of per-tile work**: track work distribution probabilistically for load balancing.
80. **Tangle/DAG of partial sums (IOTA-style)**: each new partial confirms 2 previous; no global ordering needed. Probably terrible.

## Networking-Specific Mappings

81. **Multicast tree for A-tile broadcast**: source WG writes A-tile, log2(NUM_WG) hops propagate to all consumers via L2; avoids redundant global loads.
82. **IP fragmentation for huge K-shards**: split a K-shard into MTU-sized pieces (matching L2 cache line), reassemble at destination. Pointless for FP32 but academic.
83. **TCP Reno fast-retransmit**: detect missing K-shard via 3 duplicate "next-shard" ACKs, retransmit. Useful if some WGs OOM.
84. **QUIC-style 0-RTT matmul**: start computing speculatively before all A/B is loaded; rollback if mismatch. Lower latency.
85. **BGP route advertisement**: each SM advertises which tiles it has cached; dispatcher uses shortest path. Pointless but fun.
86. **OSPF link-state for L2 partitions**: each SM knows latency to each L2 slice; dispatcher routes minimally. Real on multi-die GPUs.
87. **MPLS label-switching on tile IDs**: prepend a 32-bit label to each tile message, routers (compaction passes) switch labels without inspecting payload. Faster reduction.
88. **SDN-style separate control/data plane**: one dispatch dictates the "flow rules" (per-tile SM assignment), workers just execute. Cleaner architecture.

## CAP / Failure-Mode Theatre

89. **Chaos monkey kernel**: randomly kills WGs mid-execution to validate restart logic. Production hardening.
90. **Network partition simulation**: artificially block subgroup ops between subsets of WGs; test split-brain recovery.
91. **Latency injection**: insert random spin loops to find perf cliffs. Stress-test scheduler.
92. **Game day**: run matmul with 30% of SMs disabled; measure graceful degradation.

## Process-Algebra Cute

93. **Actor model per WG**: WGs send messages (atomic appends) to mailboxes; computation is pure message handling. Erlang-on-Vulkan.
94. **CSP channels on K**: each K-shard reads from an unbuffered channel; producers block until consumer ready. Forces strict serialization.
95. **Pi-calculus tile mobility**: tiles migrate between WGs by passing channel names; the dispatcher is a name server. Pure overhead.

## Realistic Wins from Distributed Patterns

96. **Stream-K (real)**: actually try a full Stream-K implementation with workgroup-stealing; known to win 5-15pp on irregular shapes.
97. **Per-WG accumulator slots (CRDT G-counter)**: skip atomicAdd entirely for split-K; sum slots in a final 1-WG reduction. Often a real win on small N.
98. **Hedged tail-tile launch**: for the last 5% of tiles, double-launch with first-to-commit semantics. Real P99 latency improvement.
99. **L2-resident pub-sub for A-broadcast**: pin A-panel to L2 via explicit prefetch, all WGs in the column subscribe; cuts global bandwidth on tall-skinny.
100. **Adaptive BSP supersteps**: barrier between K-chunks of dynamic size based on L2 miss rate observed by previous superstep. Closed-loop autotune.
