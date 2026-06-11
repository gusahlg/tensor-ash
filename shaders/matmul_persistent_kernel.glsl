// =====================================================================
//  matmul_persistent_kernel.glsl  -  Persistent-threads variant of the
//  BDA v4 kernel.
//
//  Instead of one workgroup per output tile, the host dispatches a
//  fixed number of workgroups (= min(num_tiles, 2 * sm_count)) and each
//  workgroup loops, atomically pulling the next unprocessed tile off a
//  global counter until none remain.  This amortises the launch
//  overhead and (in theory) keeps the SMs warm across small dispatches
//  where the regular grid would otherwise generate only a few tiles.
//
//  Same global-load fast path as matmul_bda_v4_kernel.glsl: per-thread
//  vec4 loads through `buffer_reference` blocks emit LDG.E.128, and the
//  inner Bs is stored as uvec4 to elicit LDS.E.128.
//
//  Compile-time inputs (set by the .comp wrapper):
//     BM, BN, BK, TM, TN, TN_RAW
//     TN_RAW must be >= 4 for the vec4 inner read.
//
//  Push-constant changes vs. the regular BDA kernel:
//     * counter_ptr: device address of a single u32 atomic counter
//       (the recording layer zeroes it before dispatch).
//     * grid_x, grid_y: number of tiles along N and M respectively,
//       so we can decode tile_idx -> (batch, block_row, block_col)
//       inside the shader rather than having the host pass each.
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
// Single 32-bit atomic counter shared across every workgroup in this
// dispatch.  The host zeroes it via a TRANSFER copy before the kernel
// runs, and re-zeroes it on every submit.
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer AtomicCounter {
    uint v;
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
    AtomicCounter counter_ptr;
    uint  grid_x;
    uint  grid_y;
    uint  num_tiles;
} pc;

shared float As[BM][BK + 1u];
shared uvec4 Bs[BK][BN_V4];
// One slot for the workgroup-leader to broadcast the next tile index
// it claimed from the atomic counter, so the rest of the threads in the
// workgroup don't need to atomicAdd themselves.
shared uint  s_tile_idx;

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

// Compute one output tile.  Identical math to matmul_bda_v4_kernel.glsl
// — only the prologue (decoding tile_idx -> (batch, block_row,
// block_col)) is different.
void compute_tile(uint tile_idx) {
    // Linear tile index -> (batch, block_row, block_col).
    // The grid is laid out as gx * gy tiles per batch, batch is the
    // outermost dimension so neighbouring tile_idx values land on the
    // same batch (=> share the same a_base/b_base/c_base, which keeps
    // the L2 hot for sequential tile picks).
    const uint tiles_per_batch = pc.grid_x * pc.grid_y;
    const uint batch     = tile_idx / tiles_per_batch;
    const uint in_batch  = tile_idx - batch * tiles_per_batch;
    const uint block_row = in_batch / pc.grid_x;
    const uint block_col = in_batch - block_row * pc.grid_x;

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
        } else {
            load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, true);
            load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, true);
        }

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

    if (has_k_tail) {
        const uint k_base = num_full_k * BK;
        load_a_tile_scalar(a_base, block_row, k_base, tid, m_full, false);
        load_b_tile_scalar(b_base, block_col, k_base, tid, n_full, false);

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

    const float alpha = ALPHA_IS_ONE ? 1.0 : pc.alpha;
    const uint row_base = block_row * BM + a_row0;
    const uint col_base = block_col * BN + b_col0;

    if (INTERIOR_ONLY) {
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
                    c_s.v[row_off + g_col] = val;
                }
            }
        }
    }
}

void main() {
    AtomicCounter ctr = AtomicCounter(uint64_t(pc.counter_ptr));
    const uint tid = gl_LocalInvocationIndex;
    // Persistent loop: one thread per workgroup claims the next tile
    // via an atomicAdd, broadcasts it via shared memory, and the whole
    // workgroup either processes the tile (if in range) or exits.
    //
    // Pre-claim: every WG starts with `num_warm` worth of work.  We
    // skip that optimisation here for clarity — the atomicAdd is on
    // the worst-case path so the win/loss is dominated by how many
    // claims we end up doing, not by saving one per WG.
    while (true) {
        if (tid == 0u) {
            s_tile_idx = atomicAdd(ctr.v, 1u);
        }
        barrier();
        const uint my_tile = s_tile_idx;
        // Re-read the broadcast value before the next barrier so every
        // thread caches its own copy and we can post-barrier overwrite
        // s_tile_idx in the next iter.
        barrier();
        if (my_tile >= pc.num_tiles) break;
        compute_tile(my_tile);
    }
}
