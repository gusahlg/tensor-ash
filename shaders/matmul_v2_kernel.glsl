// =====================================================================
//  matmul_v2_kernel.glsl - SOTA-style GEMM kernel body.
//
//  Inputs from wrapper: BM, BN, BK, TM, TN, TN_RAW (preprocessor int
//  alias of TN).  Designed for 128x128 / BK=8 / TM=TN=8 (matching the
//  cuBLAS ampere_sgemm_128x128_nn layout) but written so other tile
//  shapes can be plugged in.
//
//  Optimizations on top of the baseline kernel:
//    * BK is small (8) so two SMEM regions for double-buffering fit
//      inside the 48 KB per-WG Vulkan limit.  Result: 2 WGs/SM on
//      Ampere instead of 1.
//    * Explicit double-buffered K-loop: tile (k+1) is loaded into the
//      alternate SMEM region while tile k is computed.  One barrier
//      per strip instead of two.
//    * Bs is stored as `shared uvec4`, which on NVIDIA Vulkan (driver
//      >=525) emits LDS.E.128 and quarters the shared-mem instruction
//      count on the Bs read.  Reads cast through `uintBitsToFloat`.
//    * The Bs cooperative store uses an explicit `vec4` write
//      (constructed from 4 contiguous global floats) so the global
//      load is a single LDG.32x4 transaction per thread.
//    * As is column-padded (+1) for the column-strided inner read, but
//      kept row-major float so broadcast reads of `As[a_row0+i][k]`
//      stay conflict-free.
// =====================================================================

#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive   : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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

// Two-stage shared-memory regions.
//
//   As[stage][row][k]      row-major + 1 col pad (broadcast read works
//                          conflict-free for the 16x16 WG layout).
//   Bs[stage][k][col4]     uvec4 column-packed; one element is a vec4
//                          of contiguous N-cols. Decode with
//                          `uintBitsToFloat`.
shared float As[2][BM][BK + 1u];
shared uvec4 Bs[2][BK][BN_V4];

void load_a_strip(uint stage, uint a_base, uint block_row, uint k_base,
                  uint tid, bool m_full, bool k_full) {
    [[unroll]] for (uint i = 0u; i < A_PER_T; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BK;
        const uint col   = idx % BK;
        const uint g_row = block_row * BM + row;
        const uint g_col = k_base + col;
        float v = 0.0;
        if ((INTERIOR_ONLY || m_full || g_row < pc.M)
            && (INTERIOR_ONLY || k_full || g_col < pc.K)) {
            v = a[a_base + g_row * pc.K + g_col];
        }
        As[stage][row][col] = v;
    }
}

void load_b_strip_v4(uint stage, uint b_base, uint block_col, uint k_base,
                     uint tid) {
    // INTERIOR_ONLY && K_MULTIPLE path.  Each thread does one vec4
    // global load and one uvec4 shared store: the canonical 256-thread
    // 8x128 strip is 1024 floats == 256 vec4s, exactly one per thread.
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

void load_b_strip_scalar(uint stage, uint b_base, uint block_col, uint k_base,
                         uint tid, bool n_full, bool k_full) {
    [[unroll]] for (uint i = 0u; i < B_TILE / THREADS; ++i) {
        const uint idx   = tid + i * THREADS;
        const uint row   = idx / BN;
        const uint col   = idx % BN;
        const uint g_row = k_base + row;
        const uint g_col = block_col * BN + col;
        float v = 0.0;
        if ((INTERIOR_ONLY || k_full || g_row < pc.K)
            && (INTERIOR_ONLY || n_full || g_col < pc.N)) {
            v = b[b_base + g_row * pc.N + g_col];
        }
        const uint col4 = col / 4u;
        const uint sub  = col % 4u;
        // Component write into the uvec4 storage; not as fast as the
        // vec4 path but keeps the layout consistent.
        Bs[stage][row][col4][sub] = floatBitsToUint(v);
    }
}

void load_a_b(uint stage, uint a_base, uint b_base,
              uint block_row, uint block_col, uint k_base,
              uint tid, bool m_full, bool n_full, bool k_full) {
    load_a_strip(stage, a_base, block_row, k_base, tid, m_full, k_full);
    if (INTERIOR_ONLY && K_MULTIPLE) {
        load_b_strip_v4(stage, b_base, block_col, k_base, tid);
    } else {
        load_b_strip_scalar(stage, b_base, block_col, k_base, tid, n_full, k_full);
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

    const bool m_full = block_row < pc.M / BM;
    const bool n_full = block_col < pc.N / BN;

    const uint num_full_k = pc.K / BK;
    const bool has_k_tail = !K_MULTIPLE && ((pc.K % BK) != 0u);

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

    if (num_full_k != 0u) {
        // Prologue: load strip 0 into stage 0.
        load_a_b(0u, a_base, b_base, block_row, block_col, 0u, tid,
                 m_full, n_full, true);
        barrier();

        // Pipelined K loop.
        [[dont_unroll]] for (uint kt = 1u; kt < num_full_k; ++kt) {
            const uint cur_stage  = (kt - 1u) & 1u;
            const uint next_stage = kt & 1u;
            const uint next_k_base = kt * BK;

            load_a_b(next_stage, a_base, b_base, block_row, block_col,
                     next_k_base, tid, m_full, n_full, true);

            [[unroll]] for (uint k = 0u; k < BK; ++k) {
                float a_reg[TM];
                [[unroll]] for (uint i = 0u; i < TM; ++i)
                    a_reg[i] = As[cur_stage][a_row0 + i][k];
#if TN_RAW >= 4
                vec4 b_vec[TN / 4u];
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    b_vec[j4] = uintBitsToFloat(Bs[cur_stage][k][b_col0_v4 + j4]);
                [[unroll]] for (uint i = 0u; i < TM; ++i)
                    [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                        acc[i][j4] = fma(vec4(a_reg[i]), b_vec[j4], acc[i][j4]);
#else
                float b_reg[TN];
                [[unroll]] for (uint j = 0u; j < TN; ++j) {
                    const uint c = b_col0 + j;
                    b_reg[j] = uintBitsToFloat(Bs[cur_stage][k][c / 4u][c % 4u]);
                }
                [[unroll]] for (uint i = 0u; i < TM; ++i)
                    [[unroll]] for (uint j = 0u; j < TN; ++j)
                        acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
#endif
            }

            barrier();
        }

        // Compute the last prefetched strip.
        const uint last_stage = (num_full_k - 1u) & 1u;
        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[last_stage][a_row0 + i][k];
#if TN_RAW >= 4
            vec4 b_vec[TN / 4u];
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[j4] = uintBitsToFloat(Bs[last_stage][k][b_col0_v4 + j4]);
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[i]), b_vec[j4], acc[i][j4]);
#else
            float b_reg[TN];
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const uint c = b_col0 + j;
                b_reg[j] = uintBitsToFloat(Bs[last_stage][k][c / 4u][c % 4u]);
            }
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
#endif
        }
        barrier();
    }

    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_b(0u, a_base, b_base, block_row, block_col, k_base, tid,
                 m_full, n_full, false);
        barrier();
        [[unroll]] for (uint k = 0u; k < BK; ++k) {
            float a_reg[TM];
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[i] = As[0u][a_row0 + i][k];
#if TN_RAW >= 4
            vec4 b_vec[TN / 4u];
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[j4] = uintBitsToFloat(Bs[0u][k][b_col0_v4 + j4]);
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[i]), b_vec[j4], acc[i][j4]);
#else
            float b_reg[TN];
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const uint c = b_col0 + j;
                b_reg[j] = uintBitsToFloat(Bs[0u][k][c / 4u][c % 4u]);
            }
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j = 0u; j < TN; ++j)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
#endif
        }
        barrier();
    }

    // ---- Epilogue: alpha-scale (+ accumulate) + store. ----
    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    if (INTERIOR_ONLY || (m_full && n_full)) {
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint row_off = c_base + (row_base + i) * pc.N + col_base;
#if TN_RAW >= 4
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
#else
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) {
                    c[row_off + j] = c[row_off + j] + v;
                } else {
                    c[row_off + j] = v;
                }
            }
#endif
        }
    } else {
#if TN_RAW >= 4
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint g_row = row_base + i;
            if (g_row >= pc.M) continue;
            const uint row_off = c_base + g_row * pc.N;
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
                vec4 v = acc[i][j4];
                if (!ALPHA_IS_ONE) v = alpha * v;
                [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                    const uint g_col = col_base + j4*4u + s;
                    if (g_col >= pc.N) continue;
                    float val = v[s];
                    if (ACCUMULATE) val = c[row_off + g_col] + val;
                    c[row_off + g_col] = val;
                }
            }
        }
#else
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint g_row = row_base + i;
            if (g_row >= pc.M) continue;
            const uint row_off = c_base + g_row * pc.N;
            [[unroll]] for (uint j = 0u; j < TN; ++j) {
                const uint g_col = col_base + j;
                if (g_col >= pc.N) continue;
                const float v = ALPHA_IS_ONE ? acc[i][j] : alpha * acc[i][j];
                if (ACCUMULATE) {
                    c[row_off + g_col] = c[row_off + g_col] + v;
                } else {
                    c[row_off + g_col] = v;
                }
            }
        }
#endif
    }
}
