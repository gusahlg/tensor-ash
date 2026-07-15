// =====================================================================
//  matmul_splitk2_kernel.glsl  —  Two-stage split-K, stage 1.
//
//  Same K-windowed GEMM as matmul_splitk_kernel.glsl, but instead of
//  atomically reducing partial tiles into C, each (tile, k_split)
//  workgroup plain-stores its partial into a scratch buffer laid out
//  as `num_k_splits` consecutive MxN planes per batch element:
//
//      scratch[(batch * num_k_splits + k_split) * M*N + row * N + col]
//
//  A second tiny kernel (matmul_f32_splitk2_reduce.comp) then sums the
//  planes into C.  Trades scratch bandwidth + one extra dispatch for
//  the removal of all atomic contention — deterministic, no
//  VK_EXT_shader_atomic_float requirement, and the partial stores are
//  full-throughput STG instead of RED.
//
//  `pc.c_ptr` points at the SCRATCH buffer here, not C.  Every scratch
//  cell inside the M*N planes is written by exactly one workgroup, so
//  no pre-zeroing is needed.  Alpha is applied in stage 1 (linear, so
//  scaling partials equals scaling the sum); the reducer is a pure sum.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW  (TN_RAW >= 4)
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = (BN / TN), local_size_y = (BM / TM), local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE    = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE  = true;
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE    = false;

const uint WG_X    = BN / TN;
const uint WG_Y    = BM / TM;
const uint THREADS = WG_X * WG_Y;
const uint B_TILE_V4  = (BK * BN) / 4u;
const uint B_PER_T_V4 = B_TILE_V4 / THREADS;
const uint A_TILE = BM * BK;
const uint A_PER_T = A_TILE / THREADS;
const uint BN_V4 = BN / 4u;
const uint TN_V4 = TN / 4u;

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
    F32ReadWrite c_ptr;   // scratch, not C
} pc;

shared float As[BM][BK + 1u];
shared uvec4 Bs[BK][BN_V4];

void load_a_tile_scalar(uint a_base, uint block_row, uint k_base, uint k_lo, uint k_hi,
                        uint tid, bool m_full) {
    F32ReadOnly a_s = pc.a_ptr;
    [[unroll]] for (uint i = 0u; i < A_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BK;
        const uint col   = idx % BK;
        const uint g_row = block_row * BM + row;
        const uint g_col = k_base + col;
        float v = 0.0;
        if ((m_full || g_row < pc.M) && g_col >= k_lo && g_col < k_hi) {
            v = a_s.v[a_base + g_row * pc.K + g_col];
        }
        As[row][col] = v;
    }
}

void load_b_tile_scalar(uint b_base, uint block_col, uint k_base, uint k_lo, uint k_hi,
                        uint tid, bool n_full) {
    F32ReadOnly b_s = pc.b_ptr;
    [[unroll]] for (uint i = 0u; i < B_PER_T_V4; ++i) {
        const uint idx4  = tid + i * THREADS;
        const uint row   = idx4 / BN_V4;
        const uint col4  = idx4 % BN_V4;
        const uint g_row = k_base + row;
        const uint g_col_base = block_col * BN + col4 * 4u;
        const bool row_in = g_row >= k_lo && g_row < k_hi;
        const uint row_off = b_base + g_row * pc.N;
        float v0 = 0.0;
        float v1 = 0.0;
        float v2 = 0.0;
        float v3 = 0.0;
        if (row_in && (n_full || (g_col_base + 0u) < pc.N))
            v0 = b_s.v[row_off + g_col_base + 0u];
        if (row_in && (n_full || (g_col_base + 1u) < pc.N))
            v1 = b_s.v[row_off + g_col_base + 1u];
        if (row_in && (n_full || (g_col_base + 2u) < pc.N))
            v2 = b_s.v[row_off + g_col_base + 2u];
        if (row_in && (n_full || (g_col_base + 3u) < pc.N))
            v3 = b_s.v[row_off + g_col_base + 3u];
        Bs[row][col4] = uvec4(
            floatBitsToUint(v0),
            floatBitsToUint(v1),
            floatBitsToUint(v2),
            floatBitsToUint(v3));
    }
}

void main() {
    const uint num_k_splits = max(1u, pc.flags >> 16u);
    const uint k_split_id   = gl_WorkGroupID.z % num_k_splits;
    const uint batch        = gl_WorkGroupID.z / num_k_splits;
    const uint block_row    = gl_WorkGroupID.y;
    const uint block_col    = gl_WorkGroupID.x;

    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;

    const bool m_full = ((block_row + 1u) * BM) <= pc.M;
    const bool n_full = ((block_col + 1u) * BN) <= pc.N;

    // Each split owns [k_lo, k_hi) of K, divided as evenly as possible.
    const uint k_per_split  = pc.K / num_k_splits;
    const uint k_remainder  = pc.K % num_k_splits;
    const uint k_lo = k_split_id * k_per_split + min(k_split_id, k_remainder);
    const uint k_hi = k_lo + k_per_split + (k_split_id < k_remainder ? 1u : 0u);

    const uint a_row0    = ty * TM;
    const uint b_col0    = tx * TN;
    const uint b_col0_v4 = b_col0 / 4u;

    vec4 acc[TM][TN / 4u];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            acc[i][j4] = vec4(0.0);

    const uint kt_lo = k_lo / BK;
    const uint kt_hi = (k_hi + BK - 1u) / BK;

    for (uint kt = kt_lo; kt < kt_hi; ++kt) {
        const uint k_base = kt * BK;
        load_a_tile_scalar(a_base, block_row, k_base, k_lo, k_hi, tid, m_full);
        load_b_tile_scalar(b_base, block_col, k_base, k_lo, k_hi, tid, n_full);

        barrier();

        // Identical inner FMA loop as the v4 kernel, including register
        // double buffering — the hot path is byte-for-byte the same.
        float a_reg[2][TM];
        vec4  b_vec[2][TN / 4u];
        [[unroll]] for (uint i = 0u; i < TM; ++i)
            a_reg[0][i] = As[a_row0 + i][0];
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            b_vec[0][j4] = uintBitsToFloat(Bs[0][b_col0_v4 + j4]);

        [[unroll]] for (uint k = 0u; k < BK - 1u; ++k) {
            const uint cur = k & 1u;
            const uint nxt = (k + 1u) & 1u;
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[nxt][i] = As[a_row0 + i][k + 1u];
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[nxt][j4] = uintBitsToFloat(Bs[k + 1u][b_col0_v4 + j4]);
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[cur][i]), b_vec[cur][j4], acc[i][j4]);
        }

        {
            const uint last = (BK - 1u) & 1u;
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[last][i]), b_vec[last][j4], acc[i][j4]);
        }

        barrier();
    }

    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;
    const uint mn = pc.M * pc.N;
    const uint plane = (batch * num_k_splits + k_split_id) * mn;

    // Plain (non-atomic) partial store into this split's scratch plane.
    F32ReadWrite out_s = pc.c_ptr;
    [[unroll]] for (uint i = 0u; i < TM; ++i) {
        const uint g_row = row_base + i;
        if (!m_full && g_row >= pc.M) continue;
        const uint row_off = plane + g_row * pc.N;
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
            vec4 v = acc[i][j4];
            if (!ALPHA_IS_ONE) v = alpha * v;
            [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                const uint g_col = col_base + j4 * 4u + s;
                if (!n_full && g_col >= pc.N) continue;
                out_s.v[row_off + g_col] = v[s];
            }
        }
    }
}
