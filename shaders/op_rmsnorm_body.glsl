// Row normalization: RMSNorm by default, LayerNorm behind spec
// constant 0.
//
//   RMSNorm:   out = x * w / sqrt(mean(x^2) + eps)
//   LayerNorm: out = (x - mean(x)) * w / sqrt(var(x) + eps) + b
//
// One workgroup per row, 256 threads, fixed-order shared-memory
// reductions for determinism. Bandwidth-bound: one read + one write
// of the row plus the weight row.
//
// Compile-time inputs (set by the .comp wrapper):
//     IO_F16 (optional): input and output are stored as IEEE half
//     (weight and bias stay f32); every read widens to f32, all
//     arithmetic is f32, and the store narrows with RNE.

#ifdef IO_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const bool LAYER_NORM = false;

layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32V4ReadWrite {
    vec4 v[];
};
#ifdef IO_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 2) restrict buffer F16ReadWrite {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 8) restrict readonly buffer F16V4ReadOnly {
    f16vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 8) restrict buffer F16V4ReadWrite {
    f16vec4 v[];
};
#define IO_READER F16ReadOnly
#define IO_WRITER F16ReadWrite
#define IO_V4_READER F16V4ReadOnly
#define IO_V4_WRITER F16V4ReadWrite
#else
#define IO_READER F32ReadOnly
#define IO_WRITER F32ReadWrite
#define IO_V4_READER F32V4ReadOnly
#define IO_V4_WRITER F32V4ReadWrite
#endif

layout(push_constant) uniform PC {
    uint rows;
    uint cols;
    float eps;
    uint _pad;
    IO_READER in_ptr;
    IO_WRITER out_ptr;
    F32ReadOnly weight_ptr;
    F32ReadOnly bias_ptr; // LayerNorm only; 0 otherwise
} pc;

shared float red[256];

float reduce_sum(float value, uint tid) {
    red[tid] = value;
    barrier();
    [[unroll]] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) red[tid] += red[tid + stride];
        barrier();
    }
    const float total = red[0];
    barrier();
    return total;
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.rows) return;
    const uint base = row * pc.cols;
    const float inv_n = 1.0 / float(pc.cols);
    // Vec4 row walks need cols % 4 == 0 so each row start stays
    // element-aligned in the vec4 view.  Odd widths (the 771-col
    // reference test) keep the scalar path.
    const bool vec_ok = (pc.cols & 3u) == 0u;

    float mean = 0.0;
    if (LAYER_NORM) {
        float local = 0.0;
        if (vec_ok) {
            IO_V4_READER in4 = IO_V4_READER(uint64_t(pc.in_ptr));
            const uint n4 = pc.cols >> 2u;
            const uint base4 = base >> 2u;
            for (uint col4 = tid; col4 < n4; col4 += 256u) {
                local += dot(vec4(in4.v[base4 + col4]), vec4(1.0));
            }
        } else {
            for (uint col = tid; col < pc.cols; col += 256u) {
                local += float(pc.in_ptr.v[base + col]);
            }
        }
        mean = reduce_sum(local, tid) * inv_n;
    }

    float local_sq = 0.0;
    if (vec_ok) {
        IO_V4_READER in4 = IO_V4_READER(uint64_t(pc.in_ptr));
        const uint n4 = pc.cols >> 2u;
        const uint base4 = base >> 2u;
        for (uint col4 = tid; col4 < n4; col4 += 256u) {
            const vec4 centered = vec4(in4.v[base4 + col4]) - vec4(mean);
            local_sq = fma(centered.x, centered.x, local_sq);
            local_sq = fma(centered.y, centered.y, local_sq);
            local_sq = fma(centered.z, centered.z, local_sq);
            local_sq = fma(centered.w, centered.w, local_sq);
        }
    } else {
        for (uint col = tid; col < pc.cols; col += 256u) {
            const float centered = float(pc.in_ptr.v[base + col]) - mean;
            local_sq = fma(centered, centered, local_sq);
        }
    }
    const float inv_rms = inversesqrt(reduce_sum(local_sq, tid) * inv_n + pc.eps);

    if (vec_ok) {
        IO_V4_READER in4 = IO_V4_READER(uint64_t(pc.in_ptr));
        IO_V4_WRITER out4 = IO_V4_WRITER(uint64_t(pc.out_ptr));
        F32V4ReadOnly w4 = F32V4ReadOnly(uint64_t(pc.weight_ptr));
        const uint n4 = pc.cols >> 2u;
        const uint base4 = base >> 2u;
        for (uint col4 = tid; col4 < n4; col4 += 256u) {
            const vec4 x = vec4(in4.v[base4 + col4]);
            vec4 value = (x - vec4(mean)) * inv_rms * w4.v[col4];
            if (LAYER_NORM) {
                F32V4ReadOnly b4 = F32V4ReadOnly(uint64_t(pc.bias_ptr));
                value += b4.v[col4];
            }
#ifdef IO_F16
            out4.v[base4 + col4] = f16vec4(value);
#else
            out4.v[base4 + col4] = value;
#endif
        }
    } else {
        for (uint col = tid; col < pc.cols; col += 256u) {
            float value = (float(pc.in_ptr.v[base + col]) - mean) * inv_rms * pc.weight_ptr.v[col];
            if (LAYER_NORM) value += pc.bias_ptr.v[col];
#ifdef IO_F16
            pc.out_ptr.v[base + col] = float16_t(value);
#else
            pc.out_ptr.v[base + col] = value;
#endif
        }
    }
}
