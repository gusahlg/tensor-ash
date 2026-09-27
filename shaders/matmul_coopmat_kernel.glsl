#pragma use_vulkan_memory_model
// Tensor-core GEMM via KHR cooperative matrix: A f32 storage
// (converted to f16 while staging into shared memory), B f16 storage,
// f32 accumulate on 16x16x16 subgroup fragments.
//
// Strictly aligned: the host rejects dispatches unless M % BM == 0,
// N % BN == 0, K % 32 == 0 (BM/BN default 128; the wave-fill wrapper
// uses 64).  Cooperative-matrix ops require subgroup-uniform control
// flow — the divergent bounds tricks of the SIMT kernels are invalid
// here, and `cooperativeMatrixRobustBufferAccess` is not required by
// this device, so out-of-bounds loads would be UB rather than zeros.
//
// Compile-time inputs (set by the .comp wrapper):
//     A_F16 (optional): A is stored as IEEE half; staging becomes a
//     straight uvec4 copy (the f32 path's packHalf2x16 conversion
//     disappears).  Compute is unchanged — the shared tile holds
//     halves either way.
//     C_F16 (optional): C is stored as IEEE half; the f32 accumulator
//     fragments convert (RNE) at store time, and ACCUMULATE reloads
//     prior C through the same half view.
//     RASTER_M_FAST (optional): column-major tile order (see main()).
//     ACC_F16 (optional): the accumulator fragments are float16_t
//     (HMMA.*.F16: twice the f32-accumulate tensor rate on GeForce
//     Ampere/Ada, half the accumulator registers).  Every MMA rounds
//     the running sum to f16 — the numerics of cuBLAS
//     CUBLAS_COMPUTE_16F, which llama.cpp's CUDA backend uses for f16
//     weight GEMMs.  Converted to f32 at store time.
//     ACC_F16_PROMOTE (optional, exclusive with ACC_F16): f16 partial
//     fragments accumulate PROMOTE_KT K-tiles (PROMOTE_KT * BK K
//     elements) at the f16 rate, then are added into f32 accumulator
//     fragments and reset.  Error is bounded by the f16 rounding of a
//     chunk partial instead of the full running sum, at the cost of
//     half again the f32 accumulator registers.
//
// Default layout: 256 threads = 8 subgroups of 32.  Workgroup tile
// 128x128, BK = 32 (a wrapper may define BK 64; the host then checks
// K % 64 through the catalog's tile_k).  Subgroups tile the output (BM/WM) x SG_N, each
// owning a WM x WN slice of 16x16 fragments.  The 64x64 wrapper uses
// 128 threads / 4 subgroups / 32x32 slices.  Shared tiles are staged
// as uvec4 (8 halves) with a 16-byte row skew, per NVIDIA's
// vk_cooperative_matrix_perf shmem kernel.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifdef SPEC_TILE
// Generator module (M1): every tile parameter is a specialization
// constant (ids 16-23), so ONE SPIR-V module yields any legal
// BM x BN x BK / WM x WN / THREADS / skew configuration.  Arrays and
// loop bounds below are spec expressions; glslang keeps the spec-bounded
// [[unroll]] loops rolled and the driver unrolls them after
// specialization.  Host-side legality (catalog.rs): THREADS =
// (BM/WM)*(BN/WN)*32, BM*BK/8 and BK*BN/8 multiples of THREADS,
// WM/WN/BK multiples of 16.
layout(constant_id = 16) const uint BM = 128u;
layout(constant_id = 17) const uint BN = 128u;
layout(constant_id = 18) const uint BK = 32u;
layout(constant_id = 19) const uint WM = 64u;
layout(constant_id = 20) const uint WN = 32u;
layout(constant_id = 21) const uint THREADS = 256u;
layout(constant_id = 22) const uint A_SKEW = 1u;
layout(constant_id = 23) const uint B_SKEW = 1u;
const uint SG_N = BN / WN;
layout(local_size_x_id = 21, local_size_y = 1, local_size_z = 1) in;
#else
#ifndef BM
#define BM 128u
#endif
#ifndef BN
#define BN 128u
#endif
#ifndef THREADS
#define THREADS 256u
#endif
#ifndef WM
#define WM 64u
#endif
#ifndef WN
#define WN 32u
#endif
#ifndef SG_N
#define SG_N 4u
#endif
#ifndef BK
#define BK 32u
#endif
// Shared rows carry a 16-byte (one uvec4 / 8 halves) skew to dodge
// bank conflicts on the fragment loads.
const uint A_SKEW = 1u;
const uint B_SKEW = 1u;
layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;
#endif

#if defined(ACC_F16) && defined(ACC_F16_PROMOTE)
#error "ACC_F16 and ACC_F16_PROMOTE are exclusive"
#endif
#ifdef ACC_F16
#define ACC_T float16_t
#else
#define ACC_T float
#endif
#ifdef ACC_F16_PROMOTE
#ifndef PROMOTE_KT
#define PROMOTE_KT 4u
#endif
#endif

layout(constant_id = 0) const bool ACCUMULATE = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE = true;
// Alignment is a host-checked precondition, so these two are inert;
// they exist to keep the shared KernelVariant machinery uniform.
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE = false;

const uint LM = 16u;
const uint LN = 16u;
const uint LK = 16u;
const uint FRAG_M = WM / LM;
const uint FRAG_N = WN / LN;

const uint A_ROW_V4 = BK / 8u + A_SKEW; // 5 (BK=32) or 9 (BK=64) uvec4 per A row
const uint B_ROW_V4 = BN / 8u + B_SKEW; // 17 (BN=128) or 9 (BN=64) uvec4 per B row

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer H8ReadOnly {
    uvec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32V4ReadWrite {
    vec4 v[];
};
#ifdef C_F16
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F16CReadWrite {
    float16_t v[];
};
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    uint  K;
    uint  batch_stride_a;
    uint  batch_stride_b;
    uint  batch_stride_c;
    uint  flags;
    float alpha;
    F32ReadOnly  a_ptr;
    F32ReadOnly  b_ptr;
    F32ReadWrite c_ptr;
    F32ReadOnly  d_ptr;
    F32ReadOnly  bias_ptr;
    float beta;
    uint bias_batch_stride;
} pc;

shared uvec4 Ash[BM * A_ROW_V4];
shared uvec4 Bsh[BK * B_ROW_V4];

// A tile: BM x BK halves.  f32 A packs 8 K-consecutive loads (two
// vec4) into 8 halves; f16 A is a straight 16-byte copy (row starts
// stay uvec4-aligned because K % 32 == 0 is a host-checked
// precondition).  Whole-tile LDS double-buffering was measured at
// 6.4 TF/s on 2048^3 (4060) vs the single-buffer ~20+ TF/s class —
// 128x128 x2 exceeds the 48 KiB occupancy budget.
void load_a_tile(uint a_base, uint block_row, uint k_base, uint tid) {
#ifdef A_F16
    H8ReadOnly a_v8 = H8ReadOnly(uint64_t(pc.a_ptr));
#else
    F32V4ReadOnly a_v4 = F32V4ReadOnly(uint64_t(pc.a_ptr));
#endif
    const uint slots = BM * (BK / 8u);
    [[unroll]] for (uint i = 0u; i < slots / THREADS; ++i) {
        const uint slot = tid + i * THREADS;
        const uint row = slot / (BK / 8u);
        const uint col8 = slot % (BK / 8u);
        const uint g_index = a_base + (block_row * BM + row) * pc.K + k_base + col8 * 8u;
#ifdef A_F16
        Ash[row * A_ROW_V4 + col8] = a_v8.v[g_index >> 3u];
#else
        const vec4 lo = a_v4.v[g_index >> 2u];
        const vec4 hi = a_v4.v[(g_index >> 2u) + 1u];
        Ash[row * A_ROW_V4 + col8] = uvec4(
            packHalf2x16(lo.xy), packHalf2x16(lo.zw),
            packHalf2x16(hi.xy), packHalf2x16(hi.zw));
#endif
    }
}

// B tile: BK x BN halves, straight 16-byte copies.
void load_b_tile(uint b_base, uint block_col, uint k_base, uint tid) {
    H8ReadOnly b_v8 = H8ReadOnly(uint64_t(pc.b_ptr));
    const uint slots = BK * (BN / 8u);
    [[unroll]] for (uint i = 0u; i < slots / THREADS; ++i) {
        const uint slot = tid + i * THREADS;
        const uint row = slot / (BN / 8u);
        const uint col8 = slot % (BN / 8u);
        const uint g_index = b_base + (k_base + row) * pc.N + block_col * BN + col8 * 8u;
        Bsh[row * B_ROW_V4 + col8] = b_v8.v[g_index >> 3u];
    }
}

void main() {
    const uint batch = gl_WorkGroupID.z;
#ifdef RASTER_M_FAST
    // CTAs launch x-fastest; remap the linear launch index so that
    // consecutive CTAs walk M (column-major tile order).  The CTAs in
    // flight then share one B column block, so B streams from DRAM
    // about once instead of once per row block — the wide-N / short-M
    // case where x-fastest order re-reads B (M/BM) times.  Uniform per
    // workgroup, so coopmat control flow stays uniform.
    const uint tile_lin = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    const uint block_row = tile_lin % gl_NumWorkGroups.y;
    const uint block_col = tile_lin / gl_NumWorkGroups.y;
#else
    const uint block_row = gl_WorkGroupID.y;
    const uint block_col = gl_WorkGroupID.x;
#endif
    const uint tid = gl_LocalInvocationIndex;
    const uint sg = gl_SubgroupID;

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;
    const uint c_base = batch * pc.batch_stride_c;

    // Subgroup slice of the BM x BN tile: (BM/WM) x SG_N grid.
    const uint warp_row0 = (sg / SG_N) * WM;
    const uint warp_col0 = (sg % SG_N) * WN;

    coopmat<ACC_T, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> acc[FRAG_M][FRAG_N];
    [[unroll]] for (uint i = 0u; i < FRAG_M; ++i)
        [[unroll]] for (uint j = 0u; j < FRAG_N; ++j)
            acc[i][j] = coopmat<ACC_T, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(0.0);
#ifdef ACC_F16_PROMOTE
    // f16 chunk partials, promoted into `acc` every PROMOTE_KT K-tiles.
    coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> part[FRAG_M][FRAG_N];
    [[unroll]] for (uint i = 0u; i < FRAG_M; ++i)
        [[unroll]] for (uint j = 0u; j < FRAG_N; ++j)
            part[i][j] = coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(0.0);
#define MMA_DST part
#else
#define MMA_DST acc
#endif

    const uint num_k = pc.K / BK;
#ifdef ACC_F16_PROMOTE
    // Chunk loop: uniform bounds keep the promotion in subgroup-uniform
    // control flow (a coopmat requirement).
    for (uint kc = 0u; kc < num_k; kc += PROMOTE_KT) {
    const uint kt_end = min(kc + PROMOTE_KT, num_k);
    for (uint kt = kc; kt < kt_end; ++kt) {
#elif defined(K_UNROLL)
    // Literal K-tile unroll (generator body axis): the driver unrolls the
    // pre-unrolled literal kernels' runtime K loop 8x with exit checks but
    // leaves a spec-constant module's loop rolled; this restores the
    // straight-line K-tiles with uniform early exits.
    for (uint kt0 = 0u; kt0 < num_k; kt0 += K_UNROLL) {
    [[unroll]] for (uint ku = 0u; ku < K_UNROLL; ++ku) {
    const uint kt = kt0 + ku;
    if (kt >= num_k) break;
#else
    for (uint kt = 0u; kt < num_k; ++kt) {
#endif
        const uint k_base = kt * BK;
        load_a_tile(a_base, block_row, k_base, tid);
        load_b_tile(b_base, block_col, k_base, tid);
        barrier();

        [[unroll]] for (uint ks = 0u; ks < BK / LK; ++ks) {
            coopmat<float16_t, gl_ScopeSubgroup, LM, LK, gl_MatrixUseA> a_frag[FRAG_M];
            [[unroll]] for (uint i = 0u; i < FRAG_M; ++i) {
                coopMatLoad(
                    a_frag[i],
                    Ash,
                    (warp_row0 + i * LM) * A_ROW_V4 + ks * (LK / 8u),
                    A_ROW_V4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }
            [[unroll]] for (uint j = 0u; j < FRAG_N; ++j) {
                coopmat<float16_t, gl_ScopeSubgroup, LK, LN, gl_MatrixUseB> b_frag;
                coopMatLoad(
                    b_frag,
                    Bsh,
                    (ks * LK) * B_ROW_V4 + (warp_col0 + j * LN) / 8u,
                    B_ROW_V4,
                    gl_CooperativeMatrixLayoutRowMajor);
                [[unroll]] for (uint i = 0u; i < FRAG_M; ++i)
                    MMA_DST[i][j] = coopMatMulAdd(a_frag[i], b_frag, MMA_DST[i][j]);
            }
        }
        barrier();
    }
#if defined(K_UNROLL) && !defined(ACC_F16_PROMOTE)
    }
#endif
#ifdef ACC_F16_PROMOTE
    [[unroll]] for (uint i = 0u; i < FRAG_M; ++i) {
        [[unroll]] for (uint j = 0u; j < FRAG_N; ++j) {
            acc[i][j] += coopmat<float, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(part[i][j]);
            part[i][j] = coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(0.0);
        }
    }
    }
#endif

    // Store: fragment rows are 16 floats = 4 vec4 (or 16 halves),
    // N % BN == 0 keeps every row start 16-byte aligned.  Alpha and
    // accumulate act on fragment elements before the store; with
    // C_F16 the f32 accumulator converts (RNE) as the last step.
#ifdef C_F16
    F16CReadWrite c_h = F16CReadWrite(uint64_t(pc.c_ptr));
#else
    F32V4ReadWrite c_v4 = F32V4ReadWrite(uint64_t(pc.c_ptr));
#endif
    const uint row0 = block_row * BM + warp_row0;
    const uint col0 = block_col * BN + warp_col0;
    [[unroll]] for (uint i = 0u; i < FRAG_M; ++i) {
        [[unroll]] for (uint j = 0u; j < FRAG_N; ++j) {
#ifdef ACC_F16
#ifdef C_F16
            if (ALPHA_IS_ONE && !ACCUMULATE) {
                // Plain C = A@B with f16 C: the f16 accumulator IS the
                // stored value (f16 -> f32 -> f16 would be exact).
                coopMatStore(acc[i][j], c_h.v,
                             c_base + (row0 + i * LM) * pc.N + col0 + j * LN,
                             pc.N, gl_CooperativeMatrixLayoutRowMajor);
                continue;
            }
#endif
            // Epilogue math in f32, exactly like the f32-accumulate path.
            coopmat<float, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> r =
                coopmat<float, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(acc[i][j]);
#define ACC_OUT r
#else
#define ACC_OUT acc[i][j]
#endif
            if (!ALPHA_IS_ONE) {
                for (uint e = 0u; e < ACC_OUT.length(); ++e) ACC_OUT[e] *= pc.alpha;
            }
#ifdef C_F16
            const uint element = c_base + (row0 + i * LM) * pc.N + col0 + j * LN;
            if (ACCUMULATE) {
                coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> prior16;
                coopMatLoad(prior16, c_h.v, element, pc.N, gl_CooperativeMatrixLayoutRowMajor);
                ACC_OUT += coopmat<float, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(prior16);
            }
            coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> out16 =
                coopmat<float16_t, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator>(ACC_OUT);
            coopMatStore(out16, c_h.v, element, pc.N, gl_CooperativeMatrixLayoutRowMajor);
#else
            const uint element = (c_base + (row0 + i * LM) * pc.N + col0 + j * LN) / 4u;
            if (ACCUMULATE) {
                coopmat<float, gl_ScopeSubgroup, LM, LN, gl_MatrixUseAccumulator> prior;
                coopMatLoad(prior, c_v4.v, element, pc.N / 4u, gl_CooperativeMatrixLayoutRowMajor);
                ACC_OUT += prior;
            }
            coopMatStore(ACC_OUT, c_v4.v, element, pc.N / 4u, gl_CooperativeMatrixLayoutRowMajor);
#endif
#undef ACC_OUT
        }
    }
}
