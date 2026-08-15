// Rotary position embedding over [T, H, dh] activations.
//
// Rotates interleaved pairs (x[2i], x[2i+1]) within the first
// `rot_dim` lanes of each head vector by angles from a precomputed
// [T_max, rot_dim/2, 2] cos/sin table (uploaded once as a plain f32
// tensor). Lanes at or past rot_dim pass through untouched (partial
// rotary, e.g. Phi). Out-of-place safe and in-place safe: each pair
// is read and written exactly once by one invocation.
//
// One invocation per rotated pair: global id over T * H * rot_dim/2.
//
// Compile-time inputs (set by the .comp wrapper):
//     ROPE_SCATTER (optional): instead of mirroring the input layout,
//     every output element (token, head, dim) lands at
//     `dst_offset + token*dst_stride_token + head*dst_stride_head +
//      dim*dst_stride_dim` — the fused rope-plus-KV-append path, which
//     writes the rotated k row straight into the [H_kv, dh, T_max]
//     Kt-cache layout.  Input and destination must not overlap.
//     DST_F16 (ROPE_SCATTER only): destination stores IEEE half (RNE
//     via the float16_t conversion), for f16 KV caches.
//     IO_F16 (mirror layout only): input and output are stored as
//     IEEE half (f16 activations); reads widen to f32, the rotation
//     stays f32 (f32 cos/sin table), and stores narrow with RNE.
//
// Position indirection (replayable command buffers): when
// `pos_ptr != 0` it points at one u32 position `p`; the effective
// pos_base becomes `pc.pos_base + p` and (ROPE_SCATTER) the effective
// dst_offset becomes `pc.dst_offset + p * pc.pos_scale`, so a recorded
// dispatch replays correctly as the host bumps `p` between tokens.

#if defined(IO_F16) && defined(ROPE_SCATTER)
#error "IO_F16 is implemented for the mirror-layout rope only"
#endif
#if defined(DST_F16) || defined(IO_F16)
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
#ifdef IO_F16
layout(buffer_reference, std430, buffer_reference_align = 2) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 2) restrict buffer F16ReadWrite {
    float16_t v[];
};
#define IO_READER F16ReadOnly
#define IO_WRITER F16ReadWrite
#else
#define IO_READER F32ReadOnly
#define IO_WRITER F32ReadWrite
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer U32ReadOnly {
    uint v[];
};

#ifdef ROPE_SCATTER
layout(push_constant) uniform PC {
    uint tokens;
    uint heads;
    uint head_dim;
    uint rot_dim;   // even, <= head_dim
    uint pos_base;  // absolute position of token 0 (decode step offset)
    uint dst_offset;
    uint dst_stride_token;
    uint dst_stride_head;
    uint dst_stride_dim;
    uint pos_scale; // dst_offset advance per indirect position
    F32ReadOnly in_ptr;
    F32ReadWrite dst_ptr;
    F32ReadOnly table_ptr; // [T_max, rot_dim/2, 2] = (cos, sin) pairs
    U32ReadOnly pos_ptr;   // optional indirect position (0 = unused)
} pc;

void store_dst(uint index, float value) {
#ifdef DST_F16
    F16WriteOut out16 = F16WriteOut(uint64_t(pc.dst_ptr));
    out16.v[index] = float16_t(value);
#else
    pc.dst_ptr.v[index] = value;
#endif
}
#else
layout(push_constant) uniform PC {
    uint tokens;
    uint heads;
    uint head_dim;
    uint rot_dim;   // even, <= head_dim
    uint pos_base;  // absolute position of token 0 (decode step offset)
    uint _pad;
    IO_READER in_ptr;
    IO_WRITER out_ptr;
    F32ReadOnly table_ptr; // [T_max, rot_dim/2, 2] = (cos, sin) pairs
    U32ReadOnly pos_ptr;   // optional indirect position (0 = unused)
} pc;
#endif

void main() {
    const uint half_rot = pc.rot_dim / 2u;
    const uint pairs = pc.tokens * pc.heads * half_rot;
    const uint id = gl_GlobalInvocationID.x;
    if (id >= pairs) return;

    const uint pair = id % half_rot;
    const uint head = (id / half_rot) % pc.heads;
    const uint token = id / (half_rot * pc.heads);

    const uint p = uint64_t(pc.pos_ptr) != 0ul ? pc.pos_ptr.v[0] : 0u;

    const uint vec_base = (token * pc.heads + head) * pc.head_dim;
    const uint i0 = vec_base + pair * 2u;
    const float x0 = float(pc.in_ptr.v[i0]);
    const float x1 = float(pc.in_ptr.v[i0 + 1u]);

    const uint angle = ((pc.pos_base + p + token) * half_rot + pair) * 2u;
    const float c = pc.table_ptr.v[angle];
    const float s = pc.table_ptr.v[angle + 1u];

    const float r0 = fma(x0, c, -x1 * s);
    const float r1 = fma(x0, s, x1 * c);

#ifdef ROPE_SCATTER
    const uint dst_base = pc.dst_offset + p * pc.pos_scale
        + token * pc.dst_stride_token + head * pc.dst_stride_head;
    const uint d0 = dst_base + (pair * 2u) * pc.dst_stride_dim;
    store_dst(d0, r0);
    store_dst(d0 + pc.dst_stride_dim, r1);
#elif defined(IO_F16)
    pc.out_ptr.v[i0] = float16_t(r0);
    pc.out_ptr.v[i0 + 1u] = float16_t(r1);
#else
    pc.out_ptr.v[i0] = r0;
    pc.out_ptr.v[i0 + 1u] = r1;
#endif

    // Pass-through lanes past rot_dim, distributed over the pair ids
    // of this head vector so no second dispatch is needed.
    if (pc.rot_dim < pc.head_dim) {
        const uint tail = pc.head_dim - pc.rot_dim;
        for (uint lane = pair; lane < tail; lane += half_rot) {
            const uint idx = vec_base + pc.rot_dim + lane;
#ifdef ROPE_SCATTER
            store_dst(dst_base + (pc.rot_dim + lane) * pc.dst_stride_dim, pc.in_ptr.v[idx]);
#else
            pc.out_ptr.v[idx] = pc.in_ptr.v[idx];
#endif
        }
    }
}
