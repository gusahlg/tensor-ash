// One output row per workgroup-row, one output column per lane, with
// eight K-slice warps cooperating on the same 32 columns. A single-warp
// version ran a lone 1x4096x4096 GEMV on only ~128 warps (~180 GB/s on
// a 46-SM GA104); slicing K eight ways multiplies the resident warps
// without changing the fully coalesced per-warp B row access. The
// shared-memory reduce runs in a fixed order, so results stay
// deterministic. Correct for arbitrary M through gl_WorkGroupID.y.
//
// Compile-time inputs (set by the .comp wrapper):
//     B_F16 (optional): B is stored as IEEE half; loads convert to f32
//     before the FMA, halving global B traffic (the GEMV bottleneck).
//     VCOLS (optional, B_F16 only, 2): each lane owns VCOLS adjacent
//     output columns and reads B as f16vec2, widening the per-warp B
//     transaction from 64B to 128B per k-step.  The workgroup covers
//     32*VCOLS columns, so the grid shrinks by VCOLS (the .comp
//     wrapper's catalog tile_n must be 32*VCOLS).  (VCOLS=4 via
//     f16vec4 was measured neutral-to-losing on every decode shape
//     and removed.)
//     Per output column the K-order and the fixed-order slice reduce
//     are identical to VCOLS=1, so results stay bit-exact across
//     variants.  Ragged N (or an unaligned B base) falls back to an
//     in-shader scalar path with per-column bounds checks.
//
// NORM_A (push-constant flags bit 1): fold an RMSNorm of the A row
// into the GEMV.  Each workgroup first cooperatively computes
// inv_rms = inversesqrt(mean(a^2) + eps) over its K-length A row, then
// every A element is read as `a[k] * inv_rms * norm_w[k]`.  The norm
// weight rides the (otherwise unused) bias_ptr slot and eps rides
// beta, so NORM_A excludes the bias epilogue and the AddScaled binary
// (the host validates).  Redundant across the column workgroups of one
// row, but K extra reads per workgroup are trivia next to B.
//
// STORE_MODE (spec constant 7, B_F16 only): fused store epilogue for
// the decode q/k/v projections — RoPE-rotate the output row and/or
// scatter it straight into a strided KV-cache layout at C-store time,
// replacing the separate rope / append dispatch (and its hazard
// barrier).  0 = plain C store, 1 = rotate + store to C (q, in
// place), 2 = scatter to store_dst_ptr (v append), 3 = rotate +
// scatter (k -> Kt cache).  Full rotary only (rot_dim == head_dim);
// the host validates M == 1, batch == 1, head_dim | N, and excludes
// ACCUMULATE and every EPI_* form.  The rope partner of column c is
// c^1: the adjacent lane's slice sums for VCOLS=1 (reread from the
// already-barriered shared partials, same fixed-order reduce), the
// lane-local pair for VCOLS=2 — bit-exact either way with a
// standalone rope over the unfused GEMV output.
// STORE_DST_F16 (spec constant 8): scatter destination stores IEEE
// half (RNE via the float16_t conversion), for f16 KV caches.

#extension GL_EXT_control_flow_attributes : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#ifdef B_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#ifndef KSLICES
#define KSLICES 8
#endif
const uint KSLICES_U = uint(KSLICES);

#ifndef VCOLS
#define VCOLS 1
#endif
#if VCOLS != 1 && !defined(B_F16)
#error "VCOLS > 1 is implemented for the B_F16 path only"
#endif
#if VCOLS != 1 && VCOLS != 2
#error "VCOLS must be 1 or 2"
#endif
const uint VCOLS_U = uint(VCOLS);

layout(local_size_x = 32, local_size_y = KSLICES, local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE = true;
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE = false;

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};
#ifdef B_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 2) restrict writeonly buffer F16WriteOut {
    float16_t v[];
};
#if VCOLS == 2
#define f16vec_load f16vec2
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F16VecReadOnly {
    f16vec2 v[];
};
#endif
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer U32ReadOnly {
    uint v[];
};

layout(push_constant) uniform PC {
    uint M;
    uint N;
    uint K;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
    uint flags;
    float alpha;
    F32ReadOnly a_ptr;
    F32ReadOnly b_ptr;
    F32ReadWrite c_ptr;
    F32ReadOnly d_ptr;
    F32ReadOnly bias_ptr;
    float beta;
    uint bias_batch_stride;
    // Store-epilogue fields (STORE_MODE != 0 only; zero otherwise).
    F32ReadOnly store_table_ptr; // [T_max, head_dim/2, 2] (cos, sin)
    F32ReadWrite store_dst_ptr;  // scatter destination base
    U32ReadOnly store_pos_ptr;   // optional indirect position (0 = unused)
    uint store_pos_base;
    uint store_pos_scale;    // dst element advance per position
    uint store_stride_head;  // dst element stride per head
    uint store_stride_dim;   // dst element stride per head lane
    uint store_head_dim;
} pc;

#include "matmul_epilogue_common.glsl"

#ifdef B_F16
layout(constant_id = 7) const uint STORE_MODE = 0u;
layout(constant_id = 8) const uint STORE_DST_F16 = 0u;

uint store_position() {
    const uint p = uint64_t(pc.store_pos_ptr) != 0ul ? pc.store_pos_ptr.v[0] : 0u;
    return pc.store_pos_base + p;
}

void store_scatter(uint pos, uint c, float value) {
    const uint index = pos * pc.store_pos_scale
        + (c / pc.store_head_dim) * pc.store_stride_head
        + (c % pc.store_head_dim) * pc.store_stride_dim;
    if (STORE_DST_F16 != 0u) {
        F16WriteOut dst = F16WriteOut(uint64_t(pc.store_dst_ptr));
        dst.v[index] = float16_t(value);
    } else {
        pc.store_dst_ptr.v[index] = value;
    }
}

// Rotate one element of the interleaved (even, odd) pair — identical
// fma order to op_rope_body.glsl, so fused and standalone rope stay
// bit-exact.
float store_rope(uint pos, uint c, float value, float partner) {
    const uint dim = c % pc.store_head_dim;
    const uint angle = (pos * (pc.store_head_dim / 2u) + dim / 2u) * 2u;
    const float cos_t = pc.store_table_ptr.v[angle];
    const float sin_t = pc.store_table_ptr.v[angle + 1u];
    return (dim & 1u) == 0u
        ? fma(value, cos_t, -partner * sin_t)
        : fma(partner, sin_t, value * cos_t);
}
#endif

#if VCOLS == 1
shared float partial[KSLICES][32];
#define PARTIAL_SCRATCH(s, l) partial[s][l]
#else
shared float partial[KSLICES][32][VCOLS];
#define PARTIAL_SCRATCH(s, l) partial[s][l][0]
#endif

void main() {
    const uint batch = gl_WorkGroupID.z;
    const uint row = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint slice = gl_LocalInvocationID.y;
    const uint col = gl_WorkGroupID.x * (32u * VCOLS_U) + lane * VCOLS_U;
    // No early return: every thread must reach the barrier. Dead lanes
    // (column out of range) contribute a zero partial.
    const bool live = INTERIOR_ONLY || col < pc.N;

    const uint a_base = batch * pc.batch_stride_a + row * pc.K;
    const uint b_base = batch * pc.batch_stride_b + col;
#ifdef B_F16
    F16ReadOnly b = F16ReadOnly(uint64_t(pc.b_ptr));
#else
    F32ReadOnly b = pc.b_ptr;
#endif
    // Fused RMSNorm preamble (see header).  The flag is a push
    // constant, so the branch (and its barriers) is workgroup-uniform.
    const bool norm_a = (pc.flags & 2u) != 0u;
    float a_scale = 1.0;
    if (norm_a) {
        const uint tid = slice * 32u + lane;
        const uint wg = KSLICES_U * 32u;
        float sumsq = 0.0;
        for (uint k = tid; k < pc.K; k += wg) {
            const float a = pc.a_ptr.v[a_base + k];
            sumsq = fma(a, a, sumsq);
        }
        PARTIAL_SCRATCH(slice, lane) = sumsq;
        barrier();
        [[unroll]] for (uint step = wg >> 1u; step > 0u; step >>= 1u) {
            if (tid < step) {
                const uint other = tid + step;
                PARTIAL_SCRATCH(slice, lane) += PARTIAL_SCRATCH(other >> 5u, other & 31u);
            }
            barrier();
        }
        a_scale = inversesqrt(PARTIAL_SCRATCH(0u, 0u) / float(pc.K) + pc.beta);
        // Every thread has read partial[0][0]; safe to reuse below.
        barrier();
    }
#if VCOLS == 1
    float acc = 0.0;
    if (live) {
        if (norm_a) {
            for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                acc = fma(
                    pc.a_ptr.v[a_base + inner] * a_scale * pc.bias_ptr.v[inner],
                    float(b.v[b_base + inner * pc.N]),
                    acc
                );
            }
        } else {
            for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                acc = fma(
                    pc.a_ptr.v[a_base + inner],
                    float(b.v[b_base + inner * pc.N]),
                    acc
                );
            }
        }
    }
    partial[slice][lane] = acc;
    barrier();

    if (slice == 0u && live) {
        float sum = partial[0][lane];
        [[unroll]] for (uint s = 1u; s < KSLICES_U; ++s) sum += partial[s][lane];
        const uint c_index = batch * pc.batch_stride_c + row * pc.N + col;
        float value = ALPHA_IS_ONE ? sum : pc.alpha * sum;
        if (ACCUMULATE) value += pc.c_ptr.v[c_index];
        if (EPI_ANY) value = epi_apply(value, c_index, col, batch);
#ifdef B_F16
        if (STORE_MODE != 0u) {
            const uint pos = store_position();
            if (STORE_MODE != 2u) {
                // Rope partner (col^1 = lane^1): the adjacent lane's
                // slice sums, reread from the shared partials (nothing
                // writes them after the barrier above).
                float partner = partial[0][lane ^ 1u];
                [[unroll]] for (uint s = 1u; s < KSLICES_U; ++s) partner += partial[s][lane ^ 1u];
                if (!ALPHA_IS_ONE) partner *= pc.alpha;
                value = store_rope(pos, col, value, partner);
            }
            if (STORE_MODE == 1u) {
                pc.c_ptr.v[c_index] = value;
            } else {
                store_scatter(pos, col, value);
            }
        } else {
            pc.c_ptr.v[c_index] = value;
        }
#else
        pc.c_ptr.v[c_index] = value;
#endif
    }
#else
    float acc[VCOLS];
    [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) acc[v] = 0.0;
    // The vector path needs every lane's VCOLS columns to share one
    // aligned f16vec load per k-step: N and the batch stride must be
    // VCOLS-multiples and the B base pointer f16vec-aligned.  All
    // three tests are workgroup-uniform; N % VCOLS == 0 also makes
    // per-lane liveness all-or-nothing (col and N are both
    // VCOLS-multiples), so `live` covers the whole vector.
    const bool vec_ok = pc.N % VCOLS_U == 0u
        && pc.batch_stride_b % VCOLS_U == 0u
        && (uint64_t(pc.b_ptr) & uint64_t(2u * VCOLS_U - 1u)) == 0ul;
    if (vec_ok) {
        if (live) {
            F16VecReadOnly bv =
                F16VecReadOnly(uint64_t(pc.b_ptr) + uint64_t(b_base) * 2ul);
            const uint n_vec = pc.N / VCOLS_U;
            if (norm_a) {
                for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                    const float a =
                        pc.a_ptr.v[a_base + inner] * a_scale * pc.bias_ptr.v[inner];
                    const f16vec_load w = bv.v[inner * n_vec];
                    [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
                        acc[v] = fma(a, float(w[v]), acc[v]);
                    }
                }
            } else {
                for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                    const float a = pc.a_ptr.v[a_base + inner];
                    const f16vec_load w = bv.v[inner * n_vec];
                    [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
                        acc[v] = fma(a, float(w[v]), acc[v]);
                    }
                }
            }
        }
    } else if (live) {
        // Ragged-N / unaligned fallback: scalar loads, per-column
        // bounds checks (only the last live lane can be partial).
        if (norm_a) {
            for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                const float a =
                    pc.a_ptr.v[a_base + inner] * a_scale * pc.bias_ptr.v[inner];
                [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
                    if (INTERIOR_ONLY || col + v < pc.N) {
                        acc[v] = fma(a, float(b.v[b_base + inner * pc.N + v]), acc[v]);
                    }
                }
            }
        } else {
            for (uint inner = slice; inner < pc.K; inner += KSLICES_U) {
                const float a = pc.a_ptr.v[a_base + inner];
                [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
                    if (INTERIOR_ONLY || col + v < pc.N) {
                        acc[v] = fma(a, float(b.v[b_base + inner * pc.N + v]), acc[v]);
                    }
                }
            }
        }
    }
    [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) partial[slice][lane][v] = acc[v];
    barrier();

    if (slice == 0u && live) {
        // Both column sums up front: the rope store needs the pair.
        float sums[VCOLS];
        [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
            float sum = partial[0][lane][v];
            [[unroll]] for (uint s = 1u; s < KSLICES_U; ++s) sum += partial[s][lane][v];
            sums[v] = ALPHA_IS_ONE ? sum : pc.alpha * sum;
        }
        [[unroll]] for (uint v = 0u; v < VCOLS_U; ++v) {
            const uint c = col + v;
            if (!INTERIOR_ONLY && c >= pc.N) break;
            const uint c_index = batch * pc.batch_stride_c + row * pc.N + c;
            float value = sums[v];
            if (ACCUMULATE) value += pc.c_ptr.v[c_index];
            if (EPI_ANY) value = epi_apply(value, c_index, c, batch);
            if (STORE_MODE != 0u) {
                const uint pos = store_position();
                // Rope partner (c^1) is the lane-local other column.
                if (STORE_MODE != 2u) value = store_rope(pos, c, value, sums[v ^ 1u]);
                if (STORE_MODE == 1u) {
                    pc.c_ptr.v[c_index] = value;
                } else {
                    store_scatter(pos, c, value);
                }
            } else {
                pc.c_ptr.v[c_index] = value;
            }
        }
    }
#endif
}
