// =====================================================================
//  matmul_bda_kernel.glsl  —  buffer_reference (LDG.128) variant of the
//  GEMM template.
//
//  The descriptor-binding variants compile `a[idx]` to a chain of
//  OpAccessChain + OpLoad through a `StorageBuffer` pointer, which on
//  NVIDIA's Vulkan driver lowers to scalar LDG.E.32 instructions even
//  when the four reads are immediate-offset adjacent.  This kernel
//  instead receives the GPU virtual addresses of A, B, C through push
//  constants and dereferences them through `GL_EXT_buffer_reference`
//  block pointers typed as `vec4[]`.  The SPIR-V then carries the
//  `BufferReference` decoration and the driver emits LDG.E.128 on the
//  hot load path.
//
//  Host contract: requires `bufferDeviceAddress` to be enabled on the
//  Vulkan device and every input buffer to be created with
//  `SHADER_DEVICE_ADDRESS` usage.  Both are arranged by `VulkanContext`
//  whenever the device advertises the feature (always true on
//  Vulkan-1.2-capable discrete GPUs).
//
//  Wrapper preprocessor inputs:
//      BM, BN, BK, TM, TN, TN_RAW
//  The host promises the same INTERIOR_ONLY / K_MULTIPLE specialization
//  constants as the descriptor variant.
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
const uint A_TILE  = BM * BK;
const uint B_TILE  = BK * BN;
const uint A_PER_T = A_TILE / THREADS;
const uint B_PER_T = B_TILE / THREADS;
// Vec4-counted derivations.
const uint A_TILE_V4  = A_TILE / 4u;
const uint B_TILE_V4  = B_TILE / 4u;
const uint A_PER_T_V4 = A_TILE_V4 / THREADS;
const uint B_PER_T_V4 = B_TILE_V4 / THREADS;
const uint BK_V4 = BK / 4u;
const uint BN_V4 = BN / 4u;

// Buffer-reference typed pointers.  The 16-byte alignment promise is
// what tells the NVIDIA driver it's safe to issue a single LDG.E.128
// per access — without that, the driver falls back to LDG.E.32x4.
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32V4ReadWrite {
    vec4 v[];
};
// Scalar-typed alias for bounds-checked / edge loads.
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
} pc;

#include "matmul_epilogue_common.glsl"

shared float As[BM][BK + 1u];
shared float Bs[BK][BN];

// ----------------------------------------------------------------------
//  Vec4 cooperative loaders for the INTERIOR_ONLY + K_MULTIPLE path.
//  Each thread loads a single vec4 per outer-iter, which lowers to one
//  LDG.E.128 instruction on NVIDIA.
// ----------------------------------------------------------------------
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
        const uint sc = col4 * 4u;
        Bs[row][sc + 0u] = v.x;
        Bs[row][sc + 1u] = v.y;
        Bs[row][sc + 2u] = v.z;
        Bs[row][sc + 3u] = v.w;
    }
}

// ----------------------------------------------------------------------
//  Scalar cooperative loaders.  Used on edge tiles (INTERIOR_ONLY=false
//  or K_MULTIPLE=false), which can't guarantee 16-byte alignment after
//  the bounds check.  Functionally identical to the descriptor kernel.
// ----------------------------------------------------------------------
void load_a_tile_scalar(uint a_base, uint block_row, uint k_base,
                        uint tid, bool m_full, bool k_full) {
    F32ReadOnly a_s = pc.a_ptr;
    [[unroll]] for (uint i = 0u; i < A_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BK;
        const uint col   = idx % BK;
        const uint g_row = block_row * BM + row;
        const uint g_col = k_base + col;
        float v = 0.0;
        if ((m_full || g_row < pc.M) && (k_full || g_col < pc.K)) {
            v = a_s.v[a_base + g_row * pc.K + g_col];
        }
        As[row][col] = v;
    }
}

void load_b_tile_scalar(uint b_base, uint block_col, uint k_base,
                        uint tid, bool n_full, bool k_full) {
    F32ReadOnly b_s = pc.b_ptr;
    [[unroll]] for (uint i = 0u; i < B_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BN;
        const uint col   = idx % BN;
        const uint g_row = k_base + row;
        const uint g_col = block_col * BN + col;
        float v = 0.0;
        if ((k_full || g_row < pc.K) && (n_full || g_col < pc.N)) {
            v = b_s.v[b_base + g_row * pc.N + g_col];
        }
        Bs[row][col] = v;
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

    const bool m_full = ((block_row + 1u) * BM) <= pc.M;
    const bool n_full = ((block_col + 1u) * BN) <= pc.N;

    const uint num_full_k = pc.K / BK;
    const bool has_k_tail = !K_MULTIPLE && ((pc.K % BK) != 0u);

    const uint a_row0 = ty * TM;
    const uint b_col0 = tx * TN;

    float acc[TM][TN];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j = 0u; j < TN; ++j)
            acc[i][j] = 0.0;

    // ---- Main K loop: full BK strips, no K bounds check ----
    for (uint kt = 0u; kt < num_full_k; ++kt) {
        const uint k_base = kt * BK;
        if (INTERIOR_ONLY && K_MULTIPLE) {
            load_a_tile_v4(a_base, block_row, k_base, tid);
            load_b_tile_v4(b_base, block_col, k_base, tid);
        } else {
            load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, true);
            load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, true);
        }

        barrier();

        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[a_row0 + i][k];
            [[unroll]] for (uint j = 0u; j < TN; ++j)
                b_reg[j] = Bs[k][b_col0 + j];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
        }

        barrier();
    }

    // ---- K-tail loop ----
    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, false);
        load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, false);

        barrier();

        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[a_row0 + i][k];
            [[unroll]] for (uint j = 0u; j < TN; ++j)
                b_reg[j] = Bs[k][b_col0 + j];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
        }

        barrier();
    }

    // ---- Epilogue: alpha-scale (+ accumulate) + store. ----
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    if (INTERIOR_ONLY || (m_full && n_full)) {
        F32ReadWrite c_s = pc.c_ptr;
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint row_off = c_base + (row_base + i) * pc.N + col_base;
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) v = c_s.v[row_off + j] + v;
                if (EPI_ANY) v = epi_apply(v, row_off + j, col_base + j);
                c_s.v[row_off + j] = v;
            }
        }
    } else {
        F32ReadWrite c_s = pc.c_ptr;
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint g_row = row_base + i;
            if (g_row >= pc.M) continue;
            const uint row_off = c_base + g_row * pc.N;
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const uint g_col = col_base + j;
                if (g_col >= pc.N) continue;
                float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) v = c_s.v[row_off + g_col] + v;
                if (EPI_ANY) v = epi_apply(v, row_off + g_col, g_col);
                c_s.v[row_off + g_col] = v;
            }
        }
    }
}
