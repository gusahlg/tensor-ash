// =====================================================================
//  matmul_bda_v4t_kernel.glsl  —  BDA_V4 + transposed-A shared staging
//
//  Sibling of matmul_bda_v4_kernel.glsl.  B staging, epilogue, stores,
//  push constants, and spec constants are identical.  A is stored
//  TRANSPOSED in shared memory (k-major, AsT[k][m]) so each thread's
//  TM A values are contiguous and load as TM/4 uvec4 (LDS.128) instead
//  of TM scalar LDS.32.  Issue-slot bound:
//     v4  64x64 TM=8/TN=4: 8 LDS.32 + 1 LDS.128 / 32 FFMA  (~78%)
//     v4t same tile:       2 LDS.128 + 1 LDS.128 / 32 FFMA (~91%)
//     v4  128x128 TM=8/TN=8: 8+2 / 64 FFMA (~86%)
//     v4t same tile:         2+2 / 64 FFMA (~94%)
//
//  TM must be a multiple of 4 (vec4 A staging).  BM/4 >= 8 for every
//  shipped tile, so the XOR column swizzle stays in range.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW
//     B_F16 (optional): same as the v4 body.
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#ifdef B_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// TM is `#define`d as an unsigned literal (`8u`); glslang's preprocessor
// rejects the `u` suffix in `#if`, so the gate uses TM_RAW (plain int,
// same pattern as TN_RAW in the wrappers).
#if (TM_RAW % 4) != 0
#error "matmul_bda_v4t_kernel requires TM multiple of 4 (vec4 A staging)"
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
const uint TM_V4 = TM / 4u;

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

// Transposed A tile: element (k, m) lives at
//   AsT[k][ast_col(k, m/4)] component (m % 4)
// so a thread's TM consecutive rows are one or two uvec4s (LDS.128).
// No +1 padding: the XOR swizzle (not a column pad) is what kills
// the 32-bank conflicts.
shared uvec4 AsT[BK][BM / 4u];
shared uvec4 Bs[BK][BN_V4];

// XOR column swizzle.  BM/4 >= 8 on every shipped tile, so XOR with a
// value < 8 keeps the column inside the same aligned group of 8
// (BM=64 -> 16 columns, BM=128 -> 32 columns).
uint ast_col(uint k, uint m4) {
    return m4 ^ ((k >> 2u) & 7u);
}

// A staging unit: 4 consecutive rows x 4 consecutive k.  A thread loads
// one vec4 from each of its four rows (lanes with consecutive `u` read
// consecutive 16-byte chunks of the same row -> coalesced LDG.128),
// transposes the 4x4 block in registers and writes four WHOLE uvec4s
// (one per k) — no per-component shared stores, so no read-modify-
// write and nothing for the SPIR-V optimizer to choke on.
//
// Warp write pattern: 8 consecutive lanes share m4 and have c = 8p..8p+7
// (BK_V4 >= 8), so ast_col = m4 ^ (c & 7) hits 8 distinct uvec4 bank
// groups -> conflict-free 128-bit shared stores.
const uint A_UNITS = (BM / 4u) * BK_V4;
const uint A_UNITS_PER_T = (A_UNITS + THREADS - 1u) / THREADS;

void load_a_tile_v4(uint a_base, uint block_row, uint k_base, uint tid) {
    F32V4ReadOnly a_v4 = F32V4ReadOnly(uint64_t(pc.a_ptr));
    [[unroll]] for (uint i = 0u; i < A_UNITS_PER_T; ++i) {
        const uint u = tid + i * THREADS;
        if ((A_UNITS % THREADS) != 0u && u >= A_UNITS) break;
        const uint m4 = u / BK_V4;
        const uint c  = u % BK_V4;
        const uint k0 = c * 4u;
        const uint g_row0 = block_row * BM + m4 * 4u;
        const uint base = a_base + g_row0 * pc.K + k_base + k0;
        const vec4 r0 = a_v4.v[base >> 2u];
        const vec4 r1 = a_v4.v[(base + pc.K) >> 2u];
        const vec4 r2 = a_v4.v[(base + 2u * pc.K) >> 2u];
        const vec4 r3 = a_v4.v[(base + 3u * pc.K) >> 2u];
        AsT[k0 + 0u][ast_col(k0 + 0u, m4)] = floatBitsToUint(vec4(r0.x, r1.x, r2.x, r3.x));
        AsT[k0 + 1u][ast_col(k0 + 1u, m4)] = floatBitsToUint(vec4(r0.y, r1.y, r2.y, r3.y));
        AsT[k0 + 2u][ast_col(k0 + 2u, m4)] = floatBitsToUint(vec4(r0.z, r1.z, r2.z, r3.z));
        AsT[k0 + 3u][ast_col(k0 + 3u, m4)] = floatBitsToUint(vec4(r0.w, r1.w, r2.w, r3.w));
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
    // Same 4x4 units as the vector path, with bounds-checked scalar
    // loads (+0.0 outside M/K) and whole-uvec4 transposed stores.
    F32ReadOnly a_s = pc.a_ptr;
    [[unroll]] for (uint i = 0u; i < A_UNITS_PER_T; ++i) {
        const uint u = tid + i * THREADS;
        if ((A_UNITS % THREADS) != 0u && u >= A_UNITS) break;
        const uint m4 = u / BK_V4;
        const uint c  = u % BK_V4;
        const uint k0 = c * 4u;
        vec4 col0 = vec4(0.0);
        vec4 col1 = vec4(0.0);
        vec4 col2 = vec4(0.0);
        vec4 col3 = vec4(0.0);
        [[unroll]] for (uint r = 0u; r < 4u; ++r) {
            const uint g_row = block_row * BM + m4 * 4u + r;
            const bool row_in = m_full || g_row < pc.M;
            const uint row_off = a_base + g_row * pc.K + k_base + k0;
            float v0 = 0.0;
            float v1 = 0.0;
            float v2 = 0.0;
            float v3 = 0.0;
            if (row_in && (k_full || (k_base + k0 + 0u) < pc.K)) v0 = a_s.v[row_off + 0u];
            if (row_in && (k_full || (k_base + k0 + 1u) < pc.K)) v1 = a_s.v[row_off + 1u];
            if (row_in && (k_full || (k_base + k0 + 2u) < pc.K)) v2 = a_s.v[row_off + 2u];
            if (row_in && (k_full || (k_base + k0 + 3u) < pc.K)) v3 = a_s.v[row_off + 3u];
            col0[r] = v0;
            col1[r] = v1;
            col2[r] = v2;
            col3[r] = v3;
        }
        AsT[k0 + 0u][ast_col(k0 + 0u, m4)] = floatBitsToUint(col0);
        AsT[k0 + 1u][ast_col(k0 + 1u, m4)] = floatBitsToUint(col1);
        AsT[k0 + 2u][ast_col(k0 + 2u, m4)] = floatBitsToUint(col2);
        AsT[k0 + 3u][ast_col(k0 + 3u, m4)] = floatBitsToUint(col3);
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
    const uint a_row0_v4 = a_row0 / 4u;
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

        // Register-level inner-k double buffer, same SSA split as v4:
        // prefetch k+1 into the alternate bank, compute k, last iter
        // separate.  A is now vec4 (LDS.128) per TM/4 rows.  k is a
        // compile-time constant in the unrolled loop, so ast_col folds
        // to `(a_row0_v4 + i4) ^ CONST`.
        vec4 a_vec[2][TM / 4u];
        vec4 b_vec[2][TN / 4u];
        [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
            a_vec[0][i4] = uintBitsToFloat(AsT[0][ast_col(0u, a_row0_v4 + i4)]);
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            b_vec[0][j4] = uintBitsToFloat(Bs[0][b_col0_v4 + j4]);

        [[unroll]] for (uint k = 0u; k < BK - 1u; ++k) {
            const uint cur = k & 1u;
            const uint nxt = (k + 1u) & 1u;
            // Prefetch k+1 into the alternate register bank.
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                a_vec[nxt][i4] = uintBitsToFloat(
                    AsT[k + 1u][ast_col(k + 1u, a_row0_v4 + i4)]);
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[nxt][j4] = uintBitsToFloat(Bs[k + 1u][b_col0_v4 + j4]);
            // Compute current iter from the cur bank.
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                [[unroll]] for (uint s = 0u; s < 4u; ++s)
                    [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                        acc[i4 * 4u + s][j4] = fma(
                            vec4(a_vec[cur][i4][s]),
                            b_vec[cur][j4],
                            acc[i4 * 4u + s][j4]);
        }

        // Final iter: just compute, nothing left to prefetch.
        {
            const uint last = (BK - 1u) & 1u;
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                [[unroll]] for (uint s = 0u; s < 4u; ++s)
                    [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                        acc[i4 * 4u + s][j4] = fma(
                            vec4(a_vec[last][i4][s]),
                            b_vec[last][j4],
                            acc[i4 * 4u + s][j4]);
        }

        barrier();
    }

    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, false);
        load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, false);

        barrier();

        // Register-level inner-k double buffer (K-tail copy of the
        // main loop; same SSA split so the NVVM backend can interleave
        // the prefetched LDS.128 with FFMA).
        vec4 a_vec[2][TM / 4u];
        vec4 b_vec[2][TN / 4u];
        [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
            a_vec[0][i4] = uintBitsToFloat(AsT[0][ast_col(0u, a_row0_v4 + i4)]);
        [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
            b_vec[0][j4] = uintBitsToFloat(Bs[0][b_col0_v4 + j4]);

        [[unroll]] for (uint k = 0u; k < BK - 1u; ++k) {
            const uint cur = k & 1u;
            const uint nxt = (k + 1u) & 1u;
            // Prefetch k+1 into the alternate register bank.
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                a_vec[nxt][i4] = uintBitsToFloat(
                    AsT[k + 1u][ast_col(k + 1u, a_row0_v4 + i4)]);
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                b_vec[nxt][j4] = uintBitsToFloat(Bs[k + 1u][b_col0_v4 + j4]);
            // Compute current iter from the cur bank.
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                [[unroll]] for (uint s = 0u; s < 4u; ++s)
                    [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                        acc[i4 * 4u + s][j4] = fma(
                            vec4(a_vec[cur][i4][s]),
                            b_vec[cur][j4],
                            acc[i4 * 4u + s][j4]);
        }

        // Final iter: just compute, nothing left to prefetch.
        {
            const uint last = (BK - 1u) & 1u;
            [[unroll]] for (uint i4 = 0u; i4 < TM_V4; ++i4)
                [[unroll]] for (uint s = 0u; s < 4u; ++s)
                    [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                        acc[i4 * 4u + s][j4] = fma(
                            vec4(a_vec[last][i4][s]),
                            b_vec[last][j4],
                            acc[i4 * 4u + s][j4]);
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
