// Generic strided 3D copy: dst[z][y][x] = src[z][y][x] under
// independent element strides and base offsets. One kernel covers
// transpose/permute (swap strides), KV-cache append (dst offset +
// column stride), head reshaping, and sub-matrix extraction.
//
// Source and destination must not overlap (no temporal ordering
// between invocations).
//
// Bandwidth-trivial by design; extent[0] is the fastest-varying axis,
// so pick the layout that makes it contiguous in at least one of
// src/dst for coalescing.
//
// Position indirection (replayable command buffers): when
// `pos_ptr != 0` it points at one u32 position `p` and the effective
// dst_offset becomes `dst_offset + p * pos_scale`, so a recorded
// KV-append dispatch replays correctly as the host bumps `p`.
//
// Storage variants (set by the .comp wrapper): SRC_F16 reads IEEE
// half sources (widening exactly), DST_F16 narrows the destination
// with RNE; the four wrappers cover every f32/f16 pairing.

#if defined(DST_F16) || defined(SRC_F16)
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
#ifdef DST_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict buffer F16WriteOut {
    float16_t v[];
};
#endif
#ifdef SRC_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadIn {
    float16_t v[];
};
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer U32ReadOnly {
    uint v[];
};

layout(push_constant) uniform PC {
    uint extent_x;
    uint extent_y;
    uint extent_z;
    uint src_offset;
    uint src_stride_x;
    uint src_stride_y;
    uint src_stride_z;
    uint dst_offset;
    uint dst_stride_x;
    uint dst_stride_y;
    uint dst_stride_z;
    uint pos_scale; // dst_offset advance per indirect position
    F32ReadOnly src_ptr;
    F32ReadWrite dst_ptr;
    U32ReadOnly pos_ptr; // optional indirect position (0 = unused)
} pc;

// The destination is written through this view when DST_F16 is set;
// the push-constant layout stays identical (a pointer is a pointer).

void main() {
    const uint id = gl_GlobalInvocationID.x;
    const uint plane = pc.extent_x * pc.extent_y;
    const uint total = plane * pc.extent_z;
    if (id >= total) return;

    const uint x = id % pc.extent_x;
    const uint y = (id / pc.extent_x) % pc.extent_y;
    const uint z = id / plane;

    const uint p = uint64_t(pc.pos_ptr) != 0ul ? pc.pos_ptr.v[0] : 0u;
    const uint src = pc.src_offset + x * pc.src_stride_x + y * pc.src_stride_y + z * pc.src_stride_z;
    const uint dst = pc.dst_offset + p * pc.pos_scale
        + x * pc.dst_stride_x + y * pc.dst_stride_y + z * pc.dst_stride_z;
#ifdef SRC_F16
    F16ReadIn in16 = F16ReadIn(uint64_t(pc.src_ptr));
    const float value = float(in16.v[src]);
#else
    const float value = pc.src_ptr.v[src];
#endif
#ifdef DST_F16
    F16WriteOut out16 = F16WriteOut(uint64_t(pc.dst_ptr));
    out16.v[dst] = float16_t(value);
#else
    pc.dst_ptr.v[dst] = value;
#endif
}
