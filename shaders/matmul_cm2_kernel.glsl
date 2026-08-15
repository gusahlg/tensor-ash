// Tensor-core GEMM on GL_NV_cooperative_matrix2 (workgroup-scope
// cooperative matrices, tensor addressing) WITH the fused store
// epilogue the KHR-coopmat1 kernel cannot implement.
//
// Same ABI as every other matmul kernel: the `MatmulPushConstants`
// prefix through `bias_batch_stride`, spec constants 0..3 for the
// KernelVariant machinery and 4..6 for the shared epilogue
// (`matmul_epilogue_common.glsl` — bias / activation / +beta*D / *D).
// A is f32 storage quantized to f16 in a tensor-load decode callback
// (same numerics as the coopmat1 kernel's staging conversion), B is
// f16 storage (`f16w_`), C f32, all accumulation f32.
//
// Why this exists next to `f16w_coopmat_aligned`: plain-GEMM
// throughput MEASURED 18-35% behind coopmat1 on GA104 (22.7 vs 35.1
// TF/s at 4096^3 — the Bolz cm1~cm2 parity claim does not reproduce
// here), so plain routes never pick this kernel.  Its value is
// `coopMatPerElementNV`: an arbitrary per-element callback over the
// accumulator before the store, so the bias/activation/binary
// epilogues fuse here — a fused op on a coopmat-eligible shape costs
// ~half of the old SIMT demote (0.62 vs 1.27 ms on the 512x5632x2048
// gate case).  Note the composed plain-cm1 + Binary pattern is still
// ~25% faster than fused-cm2, which is why llama-ash keeps composing.
//
// Tile geometry: BM=128, BN=64, BK=64 on 128 threads (measured best;
// BN=128 collapses to ~12 TF/s on accumulator pressure at 128
// invocations, BK=128 loses ~15%).  Every
// declared matrix dimension is a multiple of 64, so this stays inside
// the exact flexible-dimensions envelope the context init validates
// for the cm2 flash kernels (f16 x f16 -> f32, 128 invocations,
// granularities dividing 64 — NVIDIA reports 32x16x16 on RTX).
// Per-thread accumulator footprint: 128x64 f32 / 128 threads = 64
// registers; A/B staging is compiler-managed.
//
// Bounds: tensor layouts clamp — out-of-range loads return the clamp
// constant 0.0 (so a ragged K tail contributes exact zeros and ragged
// M/N tiles multiply garbage-free), and out-of-range stores are
// dropped.  Unlike the coopmat1 route this body is general-shape
// correct WITHOUT the `_aligned` host precondition.  One caveat the
// router honors: tensor base addresses (buffer + element offset) must
// be 16-byte aligned, so batched dispatches need 4-float / 8-half
// batch strides — guaranteed by the aligned-shape routing constraint
// (M, N % 128, K % 32); explicit `ML_KERNEL=f16w_cm2` selections on
// ragged *batched* shapes are experimental.
//
// Requires --target-env=vulkan1.3 (SPIR-V 1.6) and the
// VK_NV_cooperative_matrix2 device features (registry slot stays
// empty otherwise).  Compile-time inputs from the .comp wrapper:
// BM, BN, BK.

#pragma use_vulkan_memory_model
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const bool ACCUMULATE = false;
layout(constant_id = 1) const bool ALPHA_IS_ONE = true;
// Inert: tensor-layout clamping already handles edges and the K tail
// with no divergent control flow; declared to keep the shared
// KernelVariant machinery uniform.
layout(constant_id = 2) const bool INTERIOR_ONLY = false;
layout(constant_id = 3) const bool K_MULTIPLE = false;

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32V4ReadOnly {
    vec4 v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32ReadWrite {
    float v[];
};
// Single-float "block" decoded by `decodeAF32`: the tensor load
// quantizes f32 A elements to the f16 matrix component type (same
// pattern as the flash kernels' f32-KV decode; needs BlockLoads,
// which the coopmat2 gate requires).
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer ABlockF32 {
    float v;
};

// Bit-identical prefix of `MatmulPushConstants` (the layout carries
// the full struct; trailing store-epilogue fields are not read here).
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
    F16ReadOnly  b_ptr;
    F32ReadWrite c_ptr;
    F32ReadOnly  d_ptr;
    F32ReadOnly  bias_ptr;
    float beta;
    uint bias_batch_stride;
} pc;

// Shared epilogue semantics + spec constants 4..6 (EPI_BIAS, EPI_ACT,
// EPI_BINARY): the scalar `epi_apply` below is the single source of
// truth for bias/activation/binary math across every fusing kernel.
#include "matmul_epilogue_common.glsl"

float16_t decodeAF32(const in ABlockF32 blk, const in uint32_t blockCoords[2],
                     const in uint32_t coordInBlock[2]) {
    return float16_t(blk.v);
}

// A-load strategy: a straight f32 tensor load (vectorizable) followed
// by the KHR conversion constructor to f16 — the per-element decode
// callback above forces scalar single-float block loads and measured
// ~2x slower end-to-end (kept for reference).
#define A_LOAD_CONVERT 1

// coopMatPerElementNV store callback: the fused epilogue, applied to
// the accumulator right before the (bounds-clamped) tensor store.
// Row/col arrive as arguments, so global bias/D indices are exact.
// Edge-tile lanes past M/N clamp their *read* indices to stay
// in-bounds (plain buffer loads have no layout net); the clamped
// store below then drops those lanes, so the values never land.
float epiStore(const in uint32_t r, const in uint32_t c, const in float e,
               const in uint32_t row0, const in uint32_t col0, const in uint32_t batch) {
    const uint g_row = min(row0 + r, pc.M - 1u);
    const uint g_col = min(col0 + c, pc.N - 1u);
    const uint c_idx = batch * pc.batch_stride_c + g_row * pc.N + g_col;
    return epi_apply(e, c_idx, g_col, batch);
}

void main() {
    const uint batch = gl_WorkGroupID.z;
    const uint row0 = gl_WorkGroupID.y * BM;
    const uint col0 = gl_WorkGroupID.x * BN;

    // Layout dimensions are the true problem extents: loads past M/K/N
    // clamp to the constant 0.0 (a zero A/B operand contributes an
    // exact zero to the product), stores past M/N are dropped.
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlA =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlA = setTensorLayoutDimensionNV(tlA, pc.M, pc.K);
    tlA = setTensorLayoutStrideNV(tlA, pc.K, 1);
    tlA = setTensorLayoutClampValueNV(tlA, 0u);

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlB =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlB = setTensorLayoutDimensionNV(tlB, pc.K, pc.N);
    tlB = setTensorLayoutStrideNV(tlB, pc.N, 1);
    tlB = setTensorLayoutClampValueNV(tlB, 0u);

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlC =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlC = setTensorLayoutDimensionNV(tlC, pc.M, pc.N);
    tlC = setTensorLayoutStrideNV(tlC, pc.N, 1);
    tlC = setTensorLayoutClampValueNV(tlC, 0u);

    const uint a_base = batch * pc.batch_stride_a;
    const uint b_base = batch * pc.batch_stride_b;
    const uint c_base = batch * pc.batch_stride_c;

    coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> acc =
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator>(0.0);

    const uint num_k = (pc.K + BK - 1u) / BK;
    [[dont_unroll]]
    for (uint kt = 0u; kt < num_k; ++kt) {
        const uint k0 = kt * BK;
#if A_LOAD_CONVERT
        coopmat<float, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a_raw;
        coopMatLoadTensorNV(a_raw, pc.a_ptr.v, a_base,
            sliceTensorLayoutNV(tlA, row0, BM, k0, BK));
        const coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a_mat =
            coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA>(a_raw);
#else
        coopmat<float16_t, gl_ScopeWorkgroup, BM, BK, gl_MatrixUseA> a_mat;
        coopMatLoadTensorNV(a_mat, pc.a_ptr.v, a_base,
            sliceTensorLayoutNV(tlA, row0, BM, k0, BK), decodeAF32);
#endif
        coopmat<float16_t, gl_ScopeWorkgroup, BK, BN, gl_MatrixUseB> b_mat;
        coopMatLoadTensorNV(b_mat, pc.b_ptr.v, b_base,
            sliceTensorLayoutNV(tlB, k0, BK, col0, BN));
        acc = coopMatMulAdd(a_mat, b_mat, acc);
    }

    // Same op order as every other kernel: alpha-scale, ACCUMULATE,
    // then the fused epilogue on the final value.
    if (!ALPHA_IS_ONE) {
        acc *= pc.alpha;
    }
    if (ACCUMULATE) {
        coopmat<float, gl_ScopeWorkgroup, BM, BN, gl_MatrixUseAccumulator> prior;
        coopMatLoadTensorNV(prior, pc.c_ptr.v, c_base,
            sliceTensorLayoutNV(tlC, row0, BM, col0, BN));
        acc += prior;
    }
    if (EPI_ANY) {
        coopMatPerElementNV(acc, acc, epiStore, row0, col0, batch);
    }
    coopMatStoreTensorNV(acc, pc.c_ptr.v, c_base,
        sliceTensorLayoutNV(tlC, row0, BM, col0, BN));
}
