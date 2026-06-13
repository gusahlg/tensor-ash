// =====================================================================
//  matmul_streamk_kernel.glsl  —  Hybrid DP+SK persistent-grid SGEMM.
//
//  Single kernel, two modes selected per workgroup:
//
//  Mode A (Data-parallel):  workgroups [0, dp_tiles_total) each own
//    exactly one output tile and run the full K loop end-to-end with
//    a plain vec4 STG.E.128 epilogue.  Identical FMA hot path to the
//    BDA_V4 kernel — no atomics, no per-tile bookkeeping.
//
//  Mode B (Stream-K tail):  workgroups [dp_tiles_total,
//    dp_tiles_total + g_sk) split the iter-space of the leftover
//    `tail_tiles = T - dp_tiles_total` tiles evenly.  Tiles touched
//    by two WGs (the seam) reduce via hardware atomicAdd; tiles fully
//    owned by one WG use a plain store.
//
//  This is the CUTLASS Stream-K design: amortize the per-tile
//  bookkeeping cost of the persistent grid only on the wave-quantization
//  tail.  Pure Stream-K (dp_tiles_total=0, all WGs in mode B) and pure
//  DP (g_sk=0, all WGs in mode A) are corner cases of the same code
//  path.
//
//  Restrictions (kept tight for v1):
//   * M, N must be tile-multiples (no bounds checks on the load path).
//   * K must be a multiple of BK.
//   * batch == 1 (the persistent grid does not span batches).
//   * accumulate=false; the kernel always writes C := alpha * A*B.  The
//     host MUST zero-fill C before the dispatch because the seam path
//     atomicAdds into C and relies on the initial value being 0.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW
//     TN_RAW must be >= 4 for the vec4 inner read.
//
//  Inner shape: mac_loop and epilogue are inlined directly into main()
//  so the [TM][TN/4] per-thread accumulator stays register-resident
//  across the tile boundary in mode B (DP-mode tiles only run one
//  iteration of the outer loop, so the placement is moot there).
//
//  Atomic implementation: VK_EXT_shader_atomic_float hardware
//  atomicAdd(float, float).  Maps to Ampere RED.E.ADD.F32.  The host
//  rejects the dispatch when the extension is unavailable.
// =====================================================================

#extension GL_EXT_control_flow_attributes  : require
#extension GL_GOOGLE_include_directive     : require
#extension GL_EXT_buffer_reference         : require
#extension GL_EXT_buffer_reference2        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_float      : require

layout(local_size_x = (BN / TN), local_size_y = (BM / TM), local_size_z = 1) in;

// Specialization constants 0/1 match the standard matmul layout so the
// host can reuse its 4-entry spec map without changes; 2/3 are ignored.
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
// Coherent float view of C for hardware atomicAdd.
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer F32CoherentRW {
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
    // Hybrid Stream-K schedule (precomputed on host).
    uint  iters_per_tile;    // K / BK
    uint  iters_per_wg_sk;   // total_iters_sk / g_sk  (floor)
    uint  rem_sk;            // total_iters_sk - iters_per_wg_sk * g_sk
    uint  n_tiles;           // N / BN
    uint  dp_tiles_total;    // floor(T / G) * G  (WGs [0, dp_tiles_total) are mode A)
    uint  g_sk;              // SK persistent grid size  (WGs [dp_tiles_total, dp_tiles_total+g_sk) are mode B)
    uint  total_iters_sk;    // tail_tiles * iters_per_tile
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

// Hardware float atomicAdd via VK_EXT_shader_atomic_float
// (shaderBufferFloat32AtomicAdd).  Maps directly to the Ampere
// RED.E.ADD.F32 SASS instruction.  Skip the no-op when val == 0.0 to
// avoid pointless atomic traffic on zero chunks.
void atomic_add_f32(uint addr, float val) {
    if (val == 0.0) return;
    F32CoherentRW c_f = F32CoherentRW(uint64_t(pc.c_ptr));
    atomicAdd(c_f.v[addr], val);
}

void main() {
    const uint wg  = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationIndex;
    const uint tx  = tid % WG_X;
    const uint ty  = tid / WG_X;

    // batch == 1 hard requirement: a_base, b_base, c_base are all 0.
    const uint a_base = 0u;
    const uint b_base = 0u;
    const uint c_base = 0u;

    const uint a_row0    = ty * TM;
    const uint b_col0    = tx * TN;
    const uint b_col0_v4 = b_col0 / 4u;

    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;

    // Per-WG iter range.  Mode A: aligned to one full tile.  Mode B:
    // even slice of the SK tail iter-space.
    uint start_iter;
    uint end_iter;

    if (wg < pc.dp_tiles_total) {
        // Mode A (DP): tile_id = wg, full K loop.
        start_iter = wg * pc.iters_per_tile;
        end_iter   = start_iter + pc.iters_per_tile;
    } else {
        // Mode B (Stream-K tail).
        const uint sk_wg = wg - pc.dp_tiles_total;
        if (sk_wg >= pc.g_sk) return;
        const uint sk_extra      = min(sk_wg, pc.rem_sk);
        const uint sk_start_local = sk_wg * pc.iters_per_wg_sk + sk_extra;
        const uint sk_end_local   =
            sk_start_local + pc.iters_per_wg_sk + (sk_wg < pc.rem_sk ? 1u : 0u);
        if (sk_start_local >= pc.total_iters_sk) return;
        // Translate to global iter space (tail tiles start at dp_tiles_total).
        const uint sk_offset = pc.dp_tiles_total * pc.iters_per_tile;
        start_iter = sk_start_local + sk_offset;
        end_iter   = sk_end_local   + sk_offset;
    }

    uint it = start_iter;
    while (it < end_iter) {
        const uint tile_id      = it / pc.iters_per_tile;
        const uint tile_iter_lo = tile_id * pc.iters_per_tile;
        const uint k_lo         = it - tile_iter_lo;
        const uint tile_end_it  = min(end_iter, tile_iter_lo + pc.iters_per_tile);
        const uint k_hi         = tile_end_it - tile_iter_lo;

        const uint block_row = tile_id / pc.n_tiles;
        const uint block_col = tile_id - block_row * pc.n_tiles;

        // Per-tile accumulator.  Kept local so the NVIDIA SPIR-V backend
        // keeps the 16 vec4s in registers across the BK-strip loop.
        vec4 acc[TM][TN / 4u];
        [[unroll]] for (uint i = 0u; i < TM; ++i)
            [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4)
                acc[i][j4] = vec4(0.0);

        // Inlined mac_loop: identical inner FMA loop to BDA_V4.
        for (uint kt = k_lo; kt < k_hi; ++kt) {
            const uint k_base = kt * BK;
            load_a_tile_v4(a_base, block_row, k_base, tid);
            load_b_tile_v4(b_base, block_col, k_base, tid);

            barrier();

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

        // Inlined epilogue.  Single owner = plain vec4 STG.E.128;
        // seam tile = hardware atomicAdd of the partial.
        const bool is_full_owner = (k_lo == 0u) && (k_hi == pc.iters_per_tile);
        const uint row_base = block_row * BM + a_row0;
        const uint col_base = block_col * BN + b_col0;

        if (is_full_owner) {
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
        } else {
            [[unroll]] for (uint i = 0u; i < TM; ++i) {
                const uint row_off = c_base + (row_base + i) * pc.N + col_base;
                [[unroll]] for (uint j4 = 0u; j4 < TN_V4; ++j4) {
                    vec4 v = acc[i][j4];
                    if (!ALPHA_IS_ONE) v = alpha * v;
                    atomic_add_f32(row_off + j4 * 4u + 0u, v.x);
                    atomic_add_f32(row_off + j4 * 4u + 1u, v.y);
                    atomic_add_f32(row_off + j4 * 4u + 2u, v.z);
                    atomic_add_f32(row_off + j4 * 4u + 3u, v.w);
                }
            }
        }

        it = tile_end_it;
    }
}
