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

