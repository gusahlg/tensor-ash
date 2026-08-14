// Fused causal prefill attention with online softmax (FlashAttention
// pattern), f32 activations.
//
// One workgroup per (q-tile, head); one thread owns one query row —
// its running max `m`, running sum `l`, and the acc[DH] output row
// stay in registers, and score tiles never touch global memory.  K
// and V tiles are staged in shared memory straight from the cache
// layouts the composed path uses (`Kt [H_kv, dh, T_max]`,
// `V [H_kv, T_max, dh]`), so both paths interoperate on the same
// tensors.  GQA maps `kv_head = head / group_size`.
//
// Row semantics match `SoftmaxMask::Causal`: query row `i` attends to
// positions `< min(kv_len, pos_base + i + 1)`.  Tiles fully above a
// workgroup's causal frontier are never visited, so early q-tiles do
// proportionally less work.  All-masked rows store zeros.
//
// Compile-time inputs (set by the .comp wrapper): DH (head
// dimension) and optionally SPLIT (1 or 2).  SPLIT=2 pairs adjacent
// lanes on one query row, halving the per-thread accumulator so the
// register-heavy dh128 variant keeps occupancy: each lane owns
// DH/SPLIT accumulator lanes and score dots are completed with one
// subgroupShuffleXor per key.

#ifdef KV_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

#ifndef SPLIT
#define SPLIT 1       // lanes cooperating per query row (1 or 2)
#endif
#ifndef BK
#define BK 32u        // key/value tile depth (wrapper may override)
#endif
const uint THREADS = 128u;
const uint SPLITU = uint(SPLIT);
const uint BQ = THREADS / SPLITU; // query rows per workgroup
const uint DHS = DH / SPLITU;     // accumulator lanes per thread

layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};
#ifdef KV_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
#define KV_READER F16ReadOnly
#else
#define KV_READER F32ReadOnly
#endif

layout(push_constant) uniform PC {
    uint t_q;
    uint t_max;
    uint kv_len;
    uint pos_base;
    uint group_size;
    float scale;
    uint _pad0;
    uint _pad1;
    F32ReadOnly q_ptr;
    F32ReadOnly kt_ptr;
    F32ReadOnly v_ptr;
    F32ReadWrite out_ptr;
} pc;

shared float Ksh[DH * BK]; // [d][j]
shared float Vsh[BK * DH]; // [j][d]

void main() {
    const uint head = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    const uint q0 = gl_WorkGroupID.x * BQ;
    const uint row = q0 + tid / SPLITU;
    const uint half_id = tid % SPLITU;
    const uint d0 = half_id * DHS; // this lane's slice of the head dim
    const bool live = row < pc.t_q;

    const uint kv_head = head / pc.group_size;
    const uint q_base = (head * pc.t_q + min(row, pc.t_q - 1u)) * DH;
    const uint kt_base = kv_head * DH * pc.t_max;
    const uint v_base = kv_head * pc.t_max * DH;

    // Causal frontier for this row and for the whole workgroup.
    const uint valid_row = live ? min(pc.kv_len, pc.pos_base + row + 1u) : 0u;
    const uint last_row = min(q0 + BQ, pc.t_q) - 1u;
    const uint valid_max = min(pc.kv_len, pc.pos_base + last_row + 1u);
    const uint num_tiles = (valid_max + BK - 1u) / BK;

    float m = -3.402823466e38;
    float l = 0.0;
    float acc[DHS];
    [[unroll]] for (uint d = 0u; d < DHS; ++d) acc[d] = 0.0;

    for (uint tile = 0u; tile < num_tiles; ++tile) {
        const uint k0 = tile * BK;

        // Cooperative staging.  Kt rows are contiguous along T, so
        // consecutive threads read consecutive positions; V rows are
        // contiguous along dh.
        for (uint i = tid; i < DH * BK; i += THREADS) {
            const uint d = i / BK;
            const uint j = i % BK;
            const uint k_pos = k0 + j;
            KV_READER kt = KV_READER(uint64_t(pc.kt_ptr));
            Ksh[i] = k_pos < pc.t_max ? float(kt.v[kt_base + d * pc.t_max + k_pos]) : 0.0;
        }
        for (uint i = tid; i < BK * DH; i += THREADS) {
            const uint j = i / DH;
            const uint d = i % DH;
            const uint k_pos = k0 + j;
            KV_READER vv = KV_READER(uint64_t(pc.v_ptr));
            Vsh[i] = k_pos < pc.t_max ? float(vv.v[v_base + k_pos * DH + d]) : 0.0;
        }
        barrier();

        // Score row for this thread's query against the tile.
        float s[BK];
        [[unroll]] for (uint j = 0u; j < BK; ++j) s[j] = 0.0;
        for (uint d = 0u; d < DHS; ++d) {
            const float qd = pc.q_ptr.v[q_base + d0 + d];
            [[unroll]] for (uint j = 0u; j < BK; ++j) {
                s[j] = fma(qd, Ksh[(d0 + d) * BK + j], s[j]);
            }
        }
#if SPLIT == 2
        // Adjacent lanes hold the two half-dots of the same row.
        [[unroll]] for (uint j = 0u; j < BK; ++j) {
            s[j] += subgroupShuffleXor(s[j], 1u);
        }
#endif
        float tile_max = -3.402823466e38;
        [[unroll]] for (uint j = 0u; j < BK; ++j) {
            s[j] = (k0 + j) < valid_row ? s[j] * pc.scale : -3.402823466e38;
            tile_max = max(tile_max, s[j]);
        }

        // Online update: one rescale of l/acc per tile.  Rows whose
        // frontier is entirely below this tile skip it (per-thread
        // branch, no barrier inside).
        if (tile_max > -3.0e38) {
            const float m_new = max(m, tile_max);
            const float correction = exp(m - m_new);
            l *= correction;
            [[unroll]] for (uint d = 0u; d < DHS; ++d) acc[d] *= correction;
            [[unroll]] for (uint j = 0u; j < BK; ++j) {
                const float p = exp(s[j] - m_new);
                l += p;
                [[unroll]] for (uint d = 0u; d < DHS; ++d) {
                    acc[d] = fma(p, Vsh[j * DH + d0 + d], acc[d]);
                }
            }
            m = m_new;
        }
        barrier();
    }

    if (live && valid_row > 0u) {
        const float inv = l > 0.0 ? 1.0 / l : 0.0;
        const uint out_base = (head * pc.t_q + row) * DH + d0;
        [[unroll]] for (uint d = 0u; d < DHS; ++d) {
            pc.out_ptr.v[out_base + d] = acc[d] * inv;
        }
    }
}
