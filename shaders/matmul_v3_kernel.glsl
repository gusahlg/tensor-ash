// =====================================================================
//  matmul_v3_kernel.glsl  -  Double-buffered GEMM with STATIC stage
//  indices.
//
//  Why this file exists
//  --------------------
//  The v2 kernel uses `(kt & 1u)` as the SMEM stage index.  Per a
//  SPIR-V audit, those dynamic indices are what prevent the NVIDIA
//  Vulkan driver from reordering the next-strip global load with the
//  current-strip FFMA pipeline: the optimizer cannot prove that the
//  prefetched stage and the compute stage don't alias.  The v3 kernel
//  fixes that by manually peeling the outer loop into pairs of
//  iterations, so each compute / prefetch combo refers to stage `0u`
//  and stage `1u` by literal constants.  The compiler then knows the
//  prefetched writes (stage 1) and the compute reads (stage 0) target
//  disjoint shared memory and is free to interleave them with the
//  inner FFMAs.
//
//  Host contract
//  -------------
//  This kernel REQUIRES:
//    - INTERIOR_ONLY = true   (M, N tile-aligned)
//    - K_MULTIPLE   = true    (K is a multiple of BK)
//    - K is a multiple of 2 * BK  (i.e. an even number of full strips)
//
//  The host (record_matmul_commands) is responsible for only dispatching
//  to v3 when all three hold.  A v3 fallback path is not provided.
// =====================================================================

#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive   : require

layout(local_size_x = (BN / TN), local_size_y = (BM / TM), local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE    = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE  = true;
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE    = false;

const uint WG_X    = BN / TN;
const uint WG_Y    = BM / TM;
const uint THREADS = WG_X * WG_Y;
const uint A_TILE  = BM * BK;
const uint B_TILE  = BK * BN;
const uint A_PER_T = A_TILE / THREADS;
const uint B_PER_T_V4 = B_TILE / 4u / THREADS;
const uint BN_V4   = BN / 4u;
const uint TN_V4   = TN / 4u;

layout(set = 0, binding = 0, std430) readonly  buffer A_BUF { float a[]; };
layout(set = 0, binding = 1, std430) readonly  buffer B_BUF { float b[]; };
layout(set = 0, binding = 2, std430)           buffer C_BUF { float c[]; };

layout(push_constant) uniform PC {
    uint  M;
    uint  N;
    uint  K;
    uint  batch_stride_a;
    uint  batch_stride_b;
    uint  batch_stride_c;
    uint  flags;
    float alpha;
} pc;

// Two-stage shared memory.  Stages are indexed by *literal* 0u and 1u
// in main() so the compiler can prove non-aliasing across them.
shared float As[2][BM][BK + 1u];
shared uvec4 Bs[2][BK][BN_V4];

void load_a_strip(uint stage, uint a_base, uint block_row, uint k_base,
                  uint tid) {
    // INTERIOR_ONLY path: unconditional load.
    [[unroll]] for (uint i = 0u; i < A_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BK;
        const uint col   = idx % BK;
        const uint g_row = block_row * BM + row;
        const uint g_col = k_base + col;
        As[stage][row][col] = a[a_base + g_row * pc.K + g_col];
    }
}

void load_b_strip(uint stage, uint b_base, uint block_col, uint k_base,
                  uint tid) {
    // INTERIOR_ONLY path: unconditional vec4 load + uvec4 STS.
    [[unroll]] for (uint i = 0u; i < B_PER_T_V4; ++i) {
        const uint idx4 = tid + i * THREADS;
        const uint row  = idx4 / BN_V4;
        const uint col4 = idx4 % BN_V4;
        const uint g_row = k_base + row;
        const uint g_col_base = block_col * BN + col4 * 4u;
        const uint base = b_base + g_row * pc.N + g_col_base;
        const vec4 v = vec4(b[base], b[base + 1u], b[base + 2u], b[base + 3u]);
        Bs[stage][row][col4] = uvec4(
            floatBitsToUint(v.x),
            floatBitsToUint(v.y),
            floatBitsToUint(v.z),
            floatBitsToUint(v.w));
    }
}

#if TN_RAW >= 4
#define COMPUTE_STRIP(STAGE, ACC, A_ROW0, B_COL0_V4)                       \
    [[unroll]] for (uint k = 0u; k < BK; ++k) {                            \
        float a_reg_local[TM];                                             \
        [[unroll]] for (uint i = 0u; i < TM; ++i)                          \
            a_reg_local[i] = As[STAGE][A_ROW0 + i][k];                     \
        vec4 b_vec_local[TN / 4u];                                         \
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)                    \
            b_vec_local[j4] = uintBitsToFloat(Bs[STAGE][k][B_COL0_V4 + j4]); \
        [[unroll]] for (uint i = 0u; i < TM; ++i)                          \
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)                \
                ACC[i][j4] = fma(vec4(a_reg_local[i]),                     \
                                  b_vec_local[j4],                          \
                                  ACC[i][j4]);                              \
    }
#else
#define COMPUTE_STRIP(STAGE, ACC, A_ROW0, B_COL0_V4)                       \
    [[unroll]] for (uint k = 0u; k < BK; ++k) {                            \
        float a_reg_local[TM];                                             \
        [[unroll]] for (uint i = 0u; i < TM; ++i)                          \
            a_reg_local[i] = As[STAGE][A_ROW0 + i][k];                     \
        float b_reg_local[TN];                                             \
        [[unroll]] for (uint j = 0u; j < TN; ++j) {                        \
            const uint c_ = b_col0 + j;                                    \
            b_reg_local[j] = uintBitsToFloat(Bs[STAGE][k][c_ / 4u][c_ % 4u]); \
        }                                                                  \
        [[unroll]] for (uint i = 0u; i < TM; ++i)                          \
            [[unroll]] for (uint j = 0u; j < TN; ++j)                      \
                ACC[i][j] = fma(a_reg_local[i], b_reg_local[j], ACC[i][j]); \
    }
#endif

void main() {
    const uint batch     = gl_WorkGroupID.z;
    const uint block_row = gl_WorkGroupID.y;
    const uint block_col = gl_WorkGroupID.x;

    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;
    const uint c_base = batch * pc.batch_stride_c;

    const uint num_full_k = pc.K / BK;
    // num_pairs guaranteed by the host to be at least 1 and exact.
    const uint num_pairs  = num_full_k >> 1u;

    const uint a_row0    = ty * TM;
    const uint b_col0    = tx * TN;
    const uint b_col0_v4 = b_col0 / 4u;

#if TN_RAW >= 4
    vec4 acc[TM][TN / 4u];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            acc[i][j4] = vec4(0.0);
#else
    float acc[TM][TN];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j = 0u; j < TN; ++j)
            acc[i][j] = 0.0;
#endif

    // Prologue: load strip 0 into stage 0.
    load_a_strip(0u, a_base, block_row, 0u, tid);
    load_b_strip(0u, b_base, block_col, 0u, tid);
    barrier();

    // Main pair loop.  Each iteration:
    //   (a) prefetch odd strip into stage 1, compute even strip from stage 0,
    //   (b) prefetch next-even strip into stage 0, compute odd strip from stage 1.
    // The literal `0u` / `1u` stage indices give the compiler an exact
    // alias map: the prefetched writes target one buffer while the
    // FMAs read from the other.
    [[dont_unroll]] for (uint kp = 0u; kp < num_pairs - 1u; ++kp) {
        const uint odd_k_base = (2u * kp + 1u) * BK;
        load_a_strip(1u, a_base, block_row, odd_k_base, tid);
        load_b_strip(1u, b_base, block_col, odd_k_base, tid);
        COMPUTE_STRIP(0u, acc, a_row0, b_col0_v4);
        barrier();

        const uint next_even_k_base = (2u * kp + 2u) * BK;
        load_a_strip(0u, a_base, block_row, next_even_k_base, tid);
        load_b_strip(0u, b_base, block_col, next_even_k_base, tid);
        COMPUTE_STRIP(1u, acc, a_row0, b_col0_v4);
        barrier();
    }

    // Final pair: process the last two strips with no further prefetch.
    {
        const uint last_odd_k_base = (2u * (num_pairs - 1u) + 1u) * BK;
        load_a_strip(1u, a_base, block_row, last_odd_k_base, tid);
        load_b_strip(1u, b_base, block_col, last_odd_k_base, tid);
        COMPUTE_STRIP(0u, acc, a_row0, b_col0_v4);
        barrier();
        COMPUTE_STRIP(1u, acc, a_row0, b_col0_v4);
        barrier();
    }

    // ---- Epilogue: alpha-scale (+ accumulate) + store. ----
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

#if TN_RAW >= 4
    [[unroll]] for (uint i = 0u; i < TM; ++i) {
        const uint row_off = c_base + (row_base + i) * pc.N + col_base;
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
            vec4 v = acc[i][j4];
            if (!ALPHA_IS_ONE) v = alpha * v;
            if (ACCUMULATE) {
                v.x = c[row_off + j4*4u + 0u] + v.x;
                v.y = c[row_off + j4*4u + 1u] + v.y;
                v.z = c[row_off + j4*4u + 2u] + v.z;
                v.w = c[row_off + j4*4u + 3u] + v.w;
            }
            c[row_off + j4*4u + 0u] = v.x;
            c[row_off + j4*4u + 1u] = v.y;
            c[row_off + j4*4u + 2u] = v.z;
            c[row_off + j4*4u + 3u] = v.w;
        }
    }
#else
    [[unroll]] for (uint i = 0u; i < TM; ++i) {
        const uint row_off = c_base + (row_base + i) * pc.N + col_base;
        [[unroll]] for (uint j = 0u; j < TN; ++j) {
            const float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
            if (ACCUMULATE) {
                c[row_off + j] = c[row_off + j] + v;
            } else {
                c[row_off + j] = v;
            }
        }
    }
#endif
}
