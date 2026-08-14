// Split-K decode attention, stage 1: partial online-softmax attention
// over one KV-position chunk for one kv head's GQA group.
//
// Decode attends ONE query row per head to the `kv_len` cached
// positions.  The composed path (scores matmul + row softmax + P@V)
// reads the full padded T_max extent and round-trips a scores tensor
// through global memory; this kernel reads only the valid prefix and
// keeps scores in registers.  Parallelism across the sequence comes
// from split-K: workgroup (chunk, kv_head) covers positions
// `[chunk*chunk_size, min((chunk+1)*chunk_size, kv_len))` and writes
// one partial (acc[DH], m, l) per query row to scratch; stage 2
// (op_attn_decode_combine.comp) merges the chunks exactly.
//
// Thread layout: 128 threads = ROWS(8) query rows x LANES(16)
// position lanes.  Lane L of row r walks positions p0+L, p0+L+16, ...
// with private online-softmax state (m, l, acc[DH]) in registers; the
// 16 lanes of a row then merge via a subgroupShuffleXor butterfly
// (all offsets < 16 stay inside a row's 16-lane block on any
// subgroup size >= 16), so the reduction never touches shared memory
// and is deterministic.  Rows past the group size idle.
//
// Layouts match the composed path and the flash kernel:
// q [kv_heads, group, DH] contiguous, Kt [kv_heads, DH, T_max],
// V [kv_heads, T_max, DH].  Scratch is
// [kv_heads, num_chunks, group, DH+2] f32 with m at [DH], l at [DH+1].
//
// Compile-time inputs (set by the .comp wrapper): DH (head
// dimension) and optionally KV_F16 (f16 K/V caches, read via
// float()).  Positions >= kv_len are never read, so cache-tail
// garbage (even f16 inf) cannot leak in.
//
// Position indirection (replayable command buffers): when
// `pos_ptr != 0` it points at one u32 position `p` and the effective
// kv_len becomes `pc.kv_len + p` (record with kv_len = 1 so the
// effective length is `p + 1`).  The recorded grid then covers a
// FIXED chunk count; chunks whose start lies at or past the effective
// kv_len run an empty position loop and write the online-softmax
// identity (m ~ -inf, l = 0, acc = 0), which stage 2 merges as an
// exact no-op.

#ifdef KV_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const uint ROWS = 8u;   // max GQA rows per kv head (group <= ROWS)
const uint LANES = 16u; // position lanes cooperating per row

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
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer U32ReadOnly {
    uint v[];
};

layout(push_constant) uniform PC {
    uint kv_len;
    uint num_chunks;
    uint group;
    uint t_max;
    float scale;
    uint _pad0;
    F32ReadOnly q_ptr;
    F32ReadOnly kt_ptr;
    F32ReadOnly v_ptr;
    F32ReadWrite scratch_ptr;
    U32ReadOnly pos_ptr; // optional indirect position (0 = unused)
} pc;

shared float Qsh[ROWS * DH];

void main() {
    const uint chunk = gl_WorkGroupID.x;
    const uint kv_head = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    const uint row = tid / LANES;
    const uint lane = tid % LANES;
    const bool live = row < pc.group;

    // Stage the group's q rows once; every position reuses them.
    for (uint i = tid; i < pc.group * DH; i += 128u) {
        Qsh[i] = pc.q_ptr.v[kv_head * pc.group * DH + i];
    }
    barrier();

    const uint kv_len =
        pc.kv_len + (uint64_t(pc.pos_ptr) != 0ul ? pc.pos_ptr.v[0] : 0u);
    const uint chunk_size = (kv_len + pc.num_chunks - 1u) / pc.num_chunks;
    const uint p0 = chunk * chunk_size;
    const uint p1 = min(p0 + chunk_size, kv_len);
    const uint kt_base = kv_head * DH * pc.t_max;
    const uint v_base = kv_head * pc.t_max * DH;

    // Private online-softmax state; an empty lane keeps the identity
    // (m = -inf-ish, l = 0, acc = 0), which merges as a no-op.
    float m = -3.402823466e38;
    float l = 0.0;
    float acc[DH];
    [[unroll]] for (uint d = 0u; d < DH; ++d) acc[d] = 0.0;

    if (live) {
        KV_READER kt = KV_READER(uint64_t(pc.kt_ptr));
        KV_READER vv = KV_READER(uint64_t(pc.v_ptr));
        for (uint p = p0 + lane; p < p1; p += LANES) {
            // Kt rows are contiguous along T, so the 16 lanes of a row
            // read 16 consecutive positions per d step.
            float s = 0.0;
            [[unroll]] for (uint d = 0u; d < DH; ++d) {
                s = fma(Qsh[row * DH + d], float(kt.v[kt_base + d * pc.t_max + p]), s);
            }
            s *= pc.scale;
            const float m_new = max(m, s);
            const float corr = exp(m - m_new);
            const float p_exp = exp(s - m_new);
            l = l * corr + p_exp;
            [[unroll]] for (uint d = 0u; d < DH; ++d) {
                acc[d] = fma(p_exp, float(vv.v[v_base + p * DH + d]), acc[d] * corr);
            }
            m = m_new;
        }
    }

    // Butterfly merge of the 16 lane-partials of each row.  The merge
    // of two online states is exact: rescale both to the shared max.
    // All threads participate (shuffles need active lanes); dead rows
    // merge identity states and never write.
    [[unroll]] for (uint offset = LANES / 2u; offset >= 1u; offset >>= 1u) {
        const float m_other = subgroupShuffleXor(m, offset);
        const float l_other = subgroupShuffleXor(l, offset);
        const float m_new = max(m, m_other);
        const float c_self = exp(m - m_new);
        const float c_other = exp(m_other - m_new);
        l = l * c_self + l_other * c_other;
        [[unroll]] for (uint d = 0u; d < DH; ++d) {
            const float acc_other = subgroupShuffleXor(acc[d], offset);
            acc[d] = acc[d] * c_self + acc_other * c_other;
        }
        m = m_new;
    }

    // All 16 lanes hold the identical merged row state; they write the
    // acc vector cooperatively (coalesced), lane 0 adds (m, l).
    if (live) {
        const uint base = ((kv_head * pc.num_chunks + chunk) * pc.group + row) * (DH + 2u);
        [[unroll]] for (uint d = lane; d < DH; d += LANES) {
            pc.scratch_ptr.v[base + d] = acc[d];
        }
        if (lane == 0u) {
            pc.scratch_ptr.v[base + DH] = m;
            pc.scratch_ptr.v[base + DH + 1u] = l;
        }
    }
}
