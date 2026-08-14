// =====================================================================
//  matmul_bda_v4_kernel.glsl  —  BDA kernel with uvec4 shared storage
//  for Bs to attempt LDS.E.128 on the compute hot-path.
//
//  Same global-load fast path as matmul_bda_kernel.glsl: per-thread
//  vec4 loads through `buffer_reference` blocks emit LDG.E.128.
//
//  Difference from matmul_bda_kernel.glsl:
//   * Bs is `shared uvec4` typed; the load function packs the 4
//     contiguous globals into one uvec4 store, and the inner compute
//     loop reads `Bs[k][col4]` as a `uvec4` and `uintBitsToFloat`s it.
//     On NVIDIA driver 525+ this lowers to LDS.E.128 — verified in
//     /tmp/gemm_sota_research.md.
//   * As stays scalar / row-major: the broadcast read pattern is
//     already conflict-free and per-thread reads are interspersed with
//     FMAs by the SPIR-V unroll pass.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW
//     TN_RAW must be >= 4 for the vec4 inner read; smaller-TN tiles
//     should use matmul_bda_kernel.glsl instead.
//     B_F16 (optional): B is stored as IEEE half.  Loads convert to
//     f32 while staging into Bs, so the FMA inner loop and stores are
//     byte-identical to the f32 build; only global B traffic halves.
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#ifdef B_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

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
const uint A_TILE_V4  = A_TILE / 4u;
const uint B_TILE_V4  = B_TILE / 4u;
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
#ifdef B_F16
// Eight halves per element: one LDG.128 carries a full 16-byte packet.
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer H8ReadOnly {
    uvec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
const uint B_TILE_V8  = B_TILE / 8u;
const uint B_PER_T_V8 = B_TILE_V8 / THREADS;
const uint BN_V8 = BN / 8u;
#endif

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

#include "matmul_epilogue_common.glsl"

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

#ifdef B_F16
void load_b_tile_v4(uint b_base, uint block_col, uint k_base, uint tid) {
    H8ReadOnly b_v8 = H8ReadOnly(uint64_t(pc.b_ptr));
    [[unroll]] for (uint i = 0u; i < B_PER_T_V8; ++i) {
        const uint idx8  = tid + i * THREADS;
        const uint row   = idx8 / BN_V8;
        const uint col8  = idx8 % BN_V8;
        const uint g_row = k_base + row;
        const uint g_col_base = block_col * BN + col8 * 8u;
        const uint addr_idx = b_base + g_row * pc.N + g_col_base;
        const uvec4 h = b_v8.v[addr_idx >> 3u];
        const vec2 f0 = unpackHalf2x16(h.x);
        const vec2 f1 = unpackHalf2x16(h.y);
        const vec2 f2 = unpackHalf2x16(h.z);
        const vec2 f3 = unpackHalf2x16(h.w);
        Bs[row][col8 * 2u + 0u] = uvec4(
            floatBitsToUint(f0.x), floatBitsToUint(f0.y),
            floatBitsToUint(f1.x), floatBitsToUint(f1.y));
        Bs[row][col8 * 2u + 1u] = uvec4(
            floatBitsToUint(f2.x), floatBitsToUint(f2.y),
            floatBitsToUint(f3.x), floatBitsToUint(f3.y));
    }
}
#else
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
#endif

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
    // Per-uvec4 cooperative load.  Each thread fills one full uvec4 in
    // Bs by performing up to four bounds-checked scalar global reads
    // and packing them.  This avoids the component-write race that
    // would happen if we let four neighbour threads each store .x/.y/
    // .z/.w of the same shared element.
    //
    // The four scalar reads are explicit (rather than a for-s loop
    // with `v[s]=...`) so the SPIR-V compiler observes a single
    // assignment to each `vec4` component on every path, which avoids
    // a lurking codegen issue we hit with the loop form.
#ifdef B_F16
    F16ReadOnly b_s = F16ReadOnly(uint64_t(pc.b_ptr));
#else
    F32ReadOnly b_s = pc.b_ptr;
#endif
    [[unroll]] for (uint i = 0u; i < B_PER_T_V4; ++i) {
        const uint idx4  = tid + i * THREADS;
        const uint row   = idx4 / BN_V4;
        const uint col4  = idx4 % BN_V4;
        const uint g_row = k_base + row;
        const uint g_col_base = block_col * BN + col4 * 4u;
        const bool row_in = k_full || g_row < pc.K;
        const uint row_off = b_base + g_row * pc.N;
        float v0 = 0.0;
        float v1 = 0.0;
        float v2 = 0.0;
        float v3 = 0.0;
        if (row_in && (n_full || (g_col_base + 0u) < pc.N))
            v0 = float(b_s.v[row_off + g_col_base + 0u]);
        if (row_in && (n_full || (g_col_base + 1u) < pc.N))
            v1 = float(b_s.v[row_off + g_col_base + 1u]);
        if (row_in && (n_full || (g_col_base + 2u) < pc.N))
            v2 = float(b_s.v[row_off + g_col_base + 2u]);
        if (row_in && (n_full || (g_col_base + 3u) < pc.N))
            v3 = float(b_s.v[row_off + g_col_base + 3u]);
        Bs[row][col4] = uvec4(
            floatBitsToUint(v0),
            floatBitsToUint(v1),
            floatBitsToUint(v2),
            floatBitsToUint(v3));
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

    // Vectorized B loads need row starts aligned to one packet: four
    // floats for f32 B, eight halves for f16 B.
#ifdef B_F16
    const bool b_rows_packet_aligned = (pc.N & 7u) == 0u;
#else
    const bool b_rows_packet_aligned = (pc.N & 3u) == 0u;
#endif

    const uint a_row0    = ty * TM;
    const uint b_col0    = tx * TN;
    const uint b_col0_v4 = b_col0 / 4u;

    vec4 acc[TM][TN / 4u];
    [[unroll]] for (uint i = 0u; i < TM; ++i)
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            acc[i][j4] = vec4(0.0);

    for (uint kt = 0u; kt < num_full_k; ++kt) {
        const uint k_base = kt * BK;
        if (INTERIOR_ONLY && K_MULTIPLE) {
            load_a_tile_v4(a_base, block_row, k_base, tid);
            load_b_tile_v4(b_base, block_col, k_base, tid);
        } else if (K_MULTIPLE) {
            // An M/N edge need not scalarize every interior workgroup. K is
            // tile-aligned here, so A rows are 16-byte aligned; B additionally
            // needs a four-float row stride. Keeping this behind the existing
            // specialization flag leaves odd-K codegen identical.
            if (m_full)
                load_a_tile_v4(a_base, block_row, k_base, tid);
            else
                load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, true);
            if (n_full && b_rows_packet_aligned)
                load_b_tile_v4(b_base, block_col, k_base, tid);
            else
                load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, true);
        } else {
            load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, true);
            load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, true);
        }

        barrier();

        // Register-level inner-k double buffer.  Per the peak-gap
        // analysis: with a single `a_reg[TM]` SSA value the NVIDIA
        // Vulkan NVVM backend respects the strict load -> FMA chain and
        // emits 8 LDS + 1 LDS-vec4 + 8 FMAs as a single ordered block.
        // Splitting the registers into `[2][TM]` gives the prefetched
        // k+1 loads a different SSA destination than the in-use k
        // values, which lets the compiler interleave LDS+FMA inside the
        // 4-cycle FFMA pipeline.  Confirmed with spirv-dis: the
        // prefetched OpLoad %v4uint moves above the previous iter's
        // last OpExtInst Fma.
        float a_reg[2][TM];
        vec4  b_vec[2][TN / 4u];
        [[unroll]] for (uint i = 0u; i < TM; ++i)
            a_reg[0][i] = As[a_row0 + i][0];
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            b_vec[0][j4] = uintBitsToFloat(Bs[0][b_col0_v4 + j4]);

        [[unroll]] for (uint k = 0u; k < BK - 1u; ++k) {
            const uint cur = k & 1u;
            const uint nxt = (k + 1u) & 1u;
            // Prefetch k+1 into the alternate register bank.
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[nxt][i] = As[a_row0 + i][k + 1u];
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[nxt][j4] = uintBitsToFloat(Bs[k + 1u][b_col0_v4 + j4]);
            // Compute current iter from the cur bank.
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[cur][i]), b_vec[cur][j4], acc[i][j4]);
        }

        // Final iter: just compute, nothing left to prefetch.
        {
            const uint last = (BK - 1u) & 1u;
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[last][i]), b_vec[last][j4], acc[i][j4]);
        }

        barrier();
    }

    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, false);
        load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, false);

        barrier();

        // Register-level inner-k double buffer.  Per the peak-gap
        // analysis: with a single `a_reg[TM]` SSA value the NVIDIA
        // Vulkan NVVM backend respects the strict load -> FMA chain and
        // emits 8 LDS + 1 LDS-vec4 + 8 FMAs as a single ordered block.
        // Splitting the registers into `[2][TM]` gives the prefetched
        // k+1 loads a different SSA destination than the in-use k
        // values, which lets the compiler interleave LDS+FMA inside the
        // 4-cycle FFMA pipeline.  Confirmed with spirv-dis: the
        // prefetched OpLoad %v4uint moves above the previous iter's
        // last OpExtInst Fma.
        float a_reg[2][TM];
        vec4  b_vec[2][TN / 4u];
        [[unroll]] for (uint i = 0u; i < TM; ++i)
            a_reg[0][i] = As[a_row0 + i][0];
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            b_vec[0][j4] = uintBitsToFloat(Bs[0][b_col0_v4 + j4]);

        [[unroll]] for (uint k = 0u; k < BK - 1u; ++k) {
            const uint cur = k & 1u;
            const uint nxt = (k + 1u) & 1u;
            // Prefetch k+1 into the alternate register bank.
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                a_reg[nxt][i] = As[a_row0 + i][k + 1u];
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[nxt][j4] = uintBitsToFloat(Bs[k + 1u][b_col0_v4 + j4]);
            // Compute current iter from the cur bank.
            [[unroll]] for (uint i = 0u; i < TM; ++i)
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                    acc[i][j4] = fma(vec4(a_reg[cur][i]), b_vec[cur][j4], acc[i][j4]);
        }

        // Final iter: just compute, nothing left to prefetch.
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

    if (INTERIOR_ONLY) {
        // INTERIOR_ONLY guarantees pc.N is a multiple of BN >= 32,
        // so it's a multiple of 4 and every per-row `row_off` is a
        // multiple of 4 - the vec4-store / STG.E.128 path is safe.
        F32V4ReadWrite c_v4 = F32V4ReadWrite(uint64_t(pc.c_ptr));
        F32ReadWrite c_s = pc.c_ptr;
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
                }
                if (EPI_ANY)
                    v = epi_apply4(v, c_addr >> 2u, (col_base + j4 * 4u) >> 2u, batch);
                if (ACCUMULATE) {
                    c_s.v[c_addr + 0u] = v.x;
                    c_s.v[c_addr + 1u] = v.y;
                    c_s.v[c_addr + 2u] = v.z;
                    c_s.v[c_addr + 3u] = v.w;
                } else {
                    c_v4.v[c_addr >> 2u] = v;
                }
            }
        }
    } else if (m_full && n_full) {
        // Edge dispatch where THIS workgroup happens to land inside
        // M and N: we still can't issue STG.E.128 because pc.N may not
        // be a multiple of 4, so cross-row strides break vec4
        // alignment.  Fall back to scalar stores.  Correctness >
        // throughput on edge tiles.
        F32ReadWrite c_s = pc.c_ptr;
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint row_off = c_base + (row_base + i) * pc.N + col_base;
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
                vec4 v = acc[i][j4];
                if (!ALPHA_IS_ONE) v = alpha * v;
                [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                    const uint c_addr = row_off + j4 * 4u + s;
                    float val = v[s];
                    if (ACCUMULATE) val = c_s.v[c_addr] + val;
                    if (EPI_ANY) val = epi_apply(val, c_addr, col_base + j4 * 4u + s, batch);
                    c_s.v[c_addr] = val;
                }
            }
        }
    } else {
        F32ReadWrite c_s = pc.c_ptr;
        [[unroll]] for (uint i = 0u; i < TM; ++i) {
            const uint g_row = row_base + i;
            if (g_row >= pc.M) continue;
            const uint row_off = c_base + g_row * pc.N;
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
                vec4 v = acc[i][j4];
                if (!ALPHA_IS_ONE) v = alpha * v;
                [[unroll]] for (uint s = 0u; s < 4u; ++s) {
                    const uint g_col = col_base + j4 * 4u + s;
                    if (g_col >= pc.N) continue;
                    float val = v[s];
                    if (ACCUMULATE) val = c_s.v[row_off + g_col] + val;
                    if (EPI_ANY) val = epi_apply(val, row_off + g_col, g_col, batch);
                    c_s.v[row_off + g_col] = val;
                }
            }
        }
    }
}
