// =====================================================================
//  matmul_streamk_dp_kernel.glsl  —  DP-flat kernel for Stream-K's
//  bulk dispatch.
//
//  2D dispatch: gl_WorkGroupID = (block_col, block_row).  The
//  dispatch shape matches the regular BDA_V4 aligned kernel so the
//  NVIDIA work distributor's L2-friendly swizzle still applies.
//  Workgroups whose flat `tile_id = block_row * n_tiles + block_col`
//  exceeds `dp_tiles_total` early-out; the SK-tail kernel covers
//  those tiles.  The load + MAC + epilogue path is otherwise
//  byte-for-byte identical to the BDA_V4 aligned kernel.
//
//  This kernel is the "bulk" half of the hybrid Stream-K dispatch.
//  The SK-tail kernel (matmul_streamk_kernel.glsl) handles the
//  wave-quantization tail with persistent threads + atomicAdd.
//
//  Restrictions:
//    * M, N tile-aligned, K is a multiple of BK.
//    * batch == 1.
//    * accumulate == false (host pre-zeroes C in hybrid mode).
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW   (TN_RAW must be >= 4)
//
//  The push constant layout matches the SK-tail kernel so a single
//  pipeline layout / push constant range covers both kernels.  Fields
//  unused by this kernel (`iters_per_wg_sk`, `rem_sk`,
//  `dp_tiles_total`, `g_sk`, `total_iters_sk`) are still read by the
//  shader compiler but optimised away — they're not referenced.
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = (BN / TN), local_size_y = (BM / TM), local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE   = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE = true;

const uint WG_X    = BN / TN;
const uint WG_Y    = BM / TM;
const uint THREADS = WG_X * WG_Y;
const uint A_TILE_V4  = (BM * BK) / 4u;
const uint B_TILE_V4  = (BK * BN) / 4u;
const uint A_PER_T_V4 = A_TILE_V4 / THREADS;
const uint B_PER_T_V4 = B_TILE_V4 / THREADS;
const uint BK_V4 = BK / 4u;
const uint BN_V4 = BN / 4u;
const uint TN_V4 = TN / 4u;

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32V4ReadWrite {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};

// Push constant layout matches matmul_streamk_kernel.glsl so the host
// can use a single pipeline layout across both kernels.
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
    uint  iters_per_tile;
    uint  iters_per_wg_sk;
    uint  rem_sk;
    uint  n_tiles;
    uint  dp_tiles_total;
    uint  g_sk;
    uint  total_iters_sk;
} pc;

shared float As[BM][BK + 1u];
shared uvec4 Bs[BK][BN_V4];

void load_a_tile_v4(uint a_base, uint block_row, uint k_base, uint tid) {
    F32V4ReadOnly a_v4 = F32V4ReadOnly(uint64_t(pc.a_ptr));
    [[unroll]] for (uint i = 0u; i < A_PER_T_V4; ++i) {
        const uint idx4  = tid + i * THREADS;
        const uint row   = idx4 / BK_V4;
        const uint col4  = idx4 % BK_V4;
        const uint g_row = block_row * BM + row;
        const uint g_col_base = k_base + col4 * 4u;
        const uint addr_idx = a_base + g_row * pc.K + g_col_base;
        const vec4 v = a_v4.v[addr_idx >> 2u];
        const uint sc = col4 * 4u;
        As[row][sc + 0u] = v.x;
        As[row][sc + 1u] = v.y;
        As[row][sc + 2u] = v.z;
        As[row][sc + 3u] = v.w;
    }
}

void load_b_tile_v4(uint b_base, uint block_col, uint k_base, uint tid) {
    F32V4ReadOnly b_v4 = F32V4ReadOnly(uint64_t(pc.b_ptr));
    [[unroll]] for (uint i = 0u; i < B_PER_T_V4; ++i) {
        const uint idx4  = tid + i * THREADS;
        const uint row   = idx4 / BN_V4;
        const uint col4  = idx4 % BN_V4;
        const uint g_row = k_base + row;
        const uint g_col_base = block_col * BN + col4 * 4u;
        const uint addr_idx = b_base + g_row * pc.N + g_col_base;
        const vec4 v = b_v4.v[addr_idx >> 2u];
        Bs[row][col4] = floatBitsToUint(v);
    }
}

void main() {
    // 2D dispatch: gl_WorkGroupID.x = block_col, .y = block_row.
    // Matches the schedule shape that the base aligned kernel uses so
    // the NVIDIA work distributor's L2-friendly swizzle still applies.
    // Workgroups whose tile_id falls in the SK tail are dispatched but
    // skip themselves; the SK-tail kernel handles those tiles.
    const uint block_col = gl_WorkGroupID.x;
    const uint block_row = gl_WorkGroupID.y;
    const uint tile_id   = block_row * pc.n_tiles + block_col;
    if (tile_id >= pc.dp_tiles_total) return;

    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    const uint a_base = 0u;
    const uint b_base = 0u;
    const uint c_base = 0u;

    const uint num_full_k = pc.iters_per_tile;

    const uint a_row0    = ty * TM;
    const uint b_col0    = tx * TN;
    const uint b_col0_v4 = b_col0 / 4u;

    vec4 acc[TM][TN / 4u];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            acc[i][j4] = vec4(0.0);

    for (uint kt = 0u; kt < num_full_k; ++kt) {
        const uint k_base = kt * BK;
        load_a_tile_v4(a_base, block_row, k_base, tid);
        load_b_tile_v4(b_base, block_col, k_base, tid);

        barrier();

        // Register-level inner-k double buffer.  Same code as BDA_V4.
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

    // Strict-aligned plain-store epilogue.
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    F32V4ReadWrite c_v4 = F32V4ReadWrite(uint64_t(pc.c_ptr));
    [[unroll]] for (uint i = 0u; i < TM; ++i) {
        const uint row_off = c_base + (row_base + i) * pc.N + col_base;
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
            vec4 v = acc[i][j4];
            if (!ALPHA_IS_ONE) v = alpha * v;
            const uint c_addr = row_off + j4 * 4u;
            c_v4.v[c_addr >> 2u] = v;
        }
    }
}
