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
//
// NORM_A (push-constant flags bit 1): fold an RMSNorm of the A row
// into the GEMV.  Each workgroup first cooperatively computes
// inv_rms = inversesqrt(mean(a^2) + eps) over its K-length A row, then
// every A element is read as `a[k] * inv_rms * norm_w[k]`.  The norm
// weight rides the (otherwise unused) bias_ptr slot and eps rides
// beta, so NORM_A excludes the bias epilogue and the AddScaled binary
// (the host validates).  Redundant across the column workgroups of one
// row, but K extra reads per workgroup are trivia next to B.

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
#endif

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
} pc;

#include "matmul_epilogue_common.glsl"

shared float partial[KSLICES][32];

void main() {
    const uint batch = gl_WorkGroupID.z;
    const uint row = gl_WorkGroupID.y;
    const uint lane = gl_LocalInvocationID.x;
    const uint slice = gl_LocalInvocationID.y;
    const uint col = gl_WorkGroupID.x * 32u + lane;
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
        partial[slice][lane] = sumsq;
        barrier();
        [[unroll]] for (uint step = wg >> 1u; step > 0u; step >>= 1u) {
            if (tid < step) {
                const uint other = tid + step;
                partial[slice][lane] += partial[other >> 5u][other & 31u];
            }
            barrier();
        }
        a_scale = inversesqrt(partial[0][0] / float(pc.K) + pc.beta);
        // Every thread has read partial[0][0]; safe to reuse below.
        barrier();
    }
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
        pc.c_ptr.v[c_index] = value;
    }
}
