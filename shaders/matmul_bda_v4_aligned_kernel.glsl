// =====================================================================
//  matmul_bda_v4_aligned_kernel.glsl
//
//  Strict-aligned clone of matmul_bda_v4_kernel.glsl.  Assumes at
//  compile time that:
//     pc.M % BM == 0   (every workgroup is interior on the M axis)
//     pc.N % BN == 0   (every workgroup is interior on the N axis)
//     pc.K % BK == 0   (no K-tail iteration)
//
//  Removes the bounds-checked scalar load helpers and the edge epilogue
//  paths entirely — the SPIR-V binary contains only the LDG.E.128 /
//  LDS.E.128 / FFMA hot path and the STG.E.128 epilogue.  Source-level
//  removal (rather than spec-constant fold) is the experiment: if it
//  beats the spec-const path, that's evidence the driver isn't fully
//  pruning the dead branches; if it ties, the spec-const fold was
//  already perfect and the variant is harmless.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW   (TN_RAW must be >= 4)
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = (BN / TN), local_size_y = (BM / TM), local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE   = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE = true;
// Spec constants 2 and 3 (INTERIOR_ONLY, K_MULTIPLE) are intentionally
// omitted — this variant hard-codes them to true at source level so the
// glslang front-end never emits the bounds-checked branches.  The
// pipeline-create code still writes 4 spec-const map entries; entries
// referencing IDs not present in the shader are ignored per the Vulkan
// spec, so the existing wiring keeps working unchanged.

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
    const uint batch     = gl_WorkGroupID.z;
    const uint block_row = gl_WorkGroupID.y;
    const uint block_col = gl_WorkGroupID.x;

    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;
    const uint c_base = batch * pc.batch_stride_c;

    // Strict-aligned: pc.K is a multiple of BK, so the K-loop is exact.
    const uint num_full_k = pc.K / BK;

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

        // Register-level inner-k double buffer: identical to bda_v4.
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

    // Strict-aligned epilogue: pc.N is a multiple of BN (>= 32) so the
    // vec4 STG.E.128 path is always safe — no edge or scalar fallback.
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    F32V4ReadWrite c_v4 = F32V4ReadWrite(uint64_t(pc.c_ptr));
    F32ReadWrite   c_s  = pc.c_ptr;
    [[unroll]] for (uint i = 0u; i < TM; ++i) {
        const uint row_off = c_base + (row_base + i) * pc.N + col_base;
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
            vec4 v = acc[i][j4];
            if (!ALPHA_IS_ONE) v = alpha * v;
            const uint c_addr = row_off + j4 * 4u;
            if (ACCUMULATE) {
                v.x = c_s.v[c_addr + 0u] + v.x;
                v.y = c_s.v[c_addr + 1u] + v.y;
                v.z = c_s.v[c_addr + 2u] + v.z;
                v.w = c_s.v[c_addr + 3u] + v.w;
                c_s.v[c_addr + 0u] = v.x;
                c_s.v[c_addr + 1u] = v.y;
                c_s.v[c_addr + 2u] = v.z;
                c_s.v[c_addr + 3u] = v.w;
            } else {
                c_v4.v[c_addr >> 2u] = v;
            }
        }
    }
}
