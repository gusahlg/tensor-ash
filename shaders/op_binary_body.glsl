// Standalone binary elementwise op over contiguous tensors:
//   mode 0:  out = a + beta * b     (residual add)
//   mode 1:  out = silu(a) * b     (SwiGLU gating)
//
// Exists so large matmuls can keep their tensor-core route (which
// cannot fuse epilogues) and pay one cheap bandwidth pass instead of
// demoting the whole GEMM to the SIMT family.  In-place safe for
// either operand.
//
// Compile-time inputs (set by the .comp wrapper):
//     IO_F16 (optional): a, b, and out are stored as IEEE half; the
//     arithmetic stays f32 and the store narrows with RNE.

#ifdef IO_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) restrict buffer F32ReadWrite {
    float v[];
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
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32V4ReadWrite {
    vec4 v[];
};
#define IO_READER F32ReadOnly
#define IO_WRITER F32ReadWrite
#define IO_V4_READER F32V4ReadOnly
#define IO_V4_WRITER F32V4ReadWrite
#endif

layout(push_constant) uniform PC {
    uint n;
    uint mode;
    float beta;
    uint _pad;
    IO_READER a_ptr;
    IO_READER b_ptr;
    IO_WRITER out_ptr;
} pc;

float combine(float a, float b) {
    return pc.mode == 0u ? a + pc.beta * b : (a / (1.0 + exp(-a))) * b;
}

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const uint base = i * 4u;
    if (base + 4u <= pc.n) {
        const vec4 a = vec4(IO_V4_READER(uint64_t(pc.a_ptr)).v[i]);
        const vec4 b = vec4(IO_V4_READER(uint64_t(pc.b_ptr)).v[i]);
        const vec4 value = vec4(
            combine(a.x, b.x), combine(a.y, b.y),
            combine(a.z, b.z), combine(a.w, b.w));
#ifdef IO_F16
        IO_V4_WRITER(uint64_t(pc.out_ptr)).v[i] = f16vec4(value);
#else
        IO_V4_WRITER(uint64_t(pc.out_ptr)).v[i] = value;
#endif
        return;
    }
    if (base >= pc.n) return;
    for (uint k = 0u; k < 4u; ++k) {
        const uint j = base + k;
        if (j >= pc.n) break;
        const float value = combine(float(pc.a_ptr.v[j]), float(pc.b_ptr.v[j]));
#ifdef IO_F16
        pc.out_ptr.v[j] = float16_t(value);
#else
        pc.out_ptr.v[j] = value;
#endif
    }
}
