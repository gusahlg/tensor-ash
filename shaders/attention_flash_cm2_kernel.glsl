// Fused causal prefill attention on tensor cores via
// GL_NV_cooperative_matrix2 (workgroup-scope cooperative matrices).
//
// Same semantics and push-constant block as the SIMT kernel
// (`attention_flash_kernel.glsl`): `q`/`out` are `[H, T_q, dh]`, `Kt`
// is `[H_kv, dh, T_max]`, `V` is `[H_kv, T_max, dh]`; query row `i`
// attends to positions `< min(kv_len, pos_base + i + 1)`; GQA maps
// `kv_head = head / group_size`; all-masked rows store zeros.
//
// Design (per the NV_cooperative_matrix2 brief and llama.cpp's
// `flash_attn_cm2.comp`): one workgroup of 128 threads per
// (q-tile, head) with Br=Bc=64 flexible-dimension matrices.  The
// whole online-softmax state — running max M, running sum L, and the
// O accumulator — stays in (compiler-managed) matrices, so O never
// round-trips through shared memory: `coopMatReduceNV` gives
// fragment-resident row max/smear and `coopMatPerElementNV` gives
// row/col-aware exp + causal masking.  This is the documented fix for
// the KHR-coopmat1 dead end (shared O accumulate + softmax phase
// serialization).
//
// Numerics: Q/K/V operands are f16 (Q converts after scaling; f32 KV
// converts in a per-element decode callback), all accumulation (S, L,
// M, O) is f32.  Tensor-layout dimensions are clamped to the valid KV
// extent with clamp-constant 0.0, so poisoned cache tails past
// `kv_len` are never even loaded — `0 * huge` NaNs are impossible by
// construction.  Masked scores use -FLT_MAX/2, not -inf, so `Mold-M`
// can never produce NaN (llama.cpp's precaution).
//
// Compile-time inputs (set by the .comp wrapper): DH (head dimension,
// multiple of 16), optionally KV_F16 (f16 K/V caches: direct tensor
// loads instead of the f32 decode callback), and optionally IO_F16
// (f16 q and out — f16 activations: q loads as halves and converts to
// f32 before the scale, out narrows the f32 O accumulator with RNE at
// store time; every intermediate stays exactly as in the f32-IO
// variant).  Requires
// --target-env=vulkan1.3 (SPIR-V 1.6) and the
// VK_NV_cooperative_matrix2 device features.

#pragma use_vulkan_memory_model
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

const uint Br = 64u; // query rows per workgroup
const uint Bc = 64u; // key/value positions per tile

layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F32ReadOnly {
    float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F32ReadWrite {
    float v[];
};
#ifdef IO_F16
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F16IoReadOnly {
    float16_t v[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) restrict buffer F16IoReadWrite {
    float16_t v[];
};
#define IO_READER F16IoReadOnly
#define IO_WRITER F16IoReadWrite
#else
#define IO_READER F32ReadOnly
#define IO_WRITER F32ReadWrite
#endif
#ifdef KV_F16
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer F16ReadOnly {
    float16_t v[];
};
#define KV_READER F16ReadOnly
#else
#define KV_READER F32ReadOnly
// Single-float "block" decoded by `decodeKvF32` below; the tensor
// load converts f32 cache elements to the f16 matrix component type.
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer KvBlockF32 {
    float v;
};
#endif

// Identical block to the SIMT kernel; only the KV pointer types
// change per variant (same 8-byte references either way).
layout(push_constant) uniform PC {
    uint t_q;
    uint t_max;
    uint kv_len;
    uint pos_base;
    uint group_size;
    float scale;
    uint _pad0;
    uint _pad1;
    IO_READER q_ptr;
    KV_READER kt_ptr;
    KV_READER v_ptr;
    IO_WRITER out_ptr;
} pc;

// -FLT_MAX/2: large enough to never win a row max, finite enough that
// `Mold - M` stays NaN-free (never a global const: uintBitsToFloat is
// not a constant initializer).
#define NEG_FLT_MAX_OVER_2 uintBitsToFloat(0xFEFFFFFFu)

// coopMatReduceNV combine ops (both operands and result f32).
float maxReduce(const in float x, const in float y) { return max(x, y); }
// Row-smear: reduce keeps column 0 and broadcasts it across the row,
// which also resizes Br x Bc state onto Br x DH accumulators.
float smearReduce(const in float x, const in float y) { return x; }

// coopMatPerElementNV element ops (row/col arrive as arguments — the
// exact (row, col) knowledge coopmat1 fragments hide).
float Exp(const in uint32_t r, const in uint32_t c, const in float e) { return exp(e); }
float Max(const in uint32_t r, const in uint32_t c, const in float a, const in float b) {
    return max(a, b);
}

// Causal + prefix + ragged-edge mask in one callback: element (r, c)
// of the S tile at (q0, k0) is valid iff its query row exists and its
// key position is below that row's causal frontier.
float applyMask(const in uint32_t r, const in uint32_t c, const in float e,
                const in uint32_t q0, const in uint32_t k0) {
    const bool valid = (q0 + r < pc.t_q)
        && (k0 + c < min(pc.kv_len, pc.pos_base + q0 + r + 1u));
    return valid ? e : NEG_FLT_MAX_OVER_2;
}

#ifndef KV_F16
// f32 KV caches: per-element decode to f16 (needs the BlockLoads
// feature; f16 caches load directly).
float16_t decodeKvF32(const in KvBlockF32 blk, const in uint32_t blockCoords[2],
                      const in uint32_t coordInBlock[2]) {
    return float16_t(blk.v);
}
#endif

void main() {
    const uint head = gl_WorkGroupID.y;
    const uint q0 = gl_WorkGroupID.x * Br;
    const uint kv_head = head / pc.group_size;

    // Workgroup causal frontier: rows past t_q are clamped away, so
    // the loop bound (and every branch below) is workgroup-uniform —
    // required for workgroup-scope cooperative matrix ops.
    const uint last_row = min(q0 + Br, pc.t_q) - 1u;
    const uint valid_max = min(pc.kv_len, pc.pos_base + last_row + 1u);
    const uint num_tiles = (valid_max + Bc - 1u) / Bc;

    // Tensor layouts clamp out-of-bounds reads to 0.0.  K/V dimensions
    // are bounded by `valid_max` (<= kv_len), so the poisoned tail and
    // the ragged last tile read exact zeros.
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlQ =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlQ = setTensorLayoutDimensionNV(tlQ, pc.t_q, DH);
    tlQ = setTensorLayoutStrideNV(tlQ, DH, 1);
    tlQ = setTensorLayoutClampValueNV(tlQ, 0u);

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlK =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlK = setTensorLayoutDimensionNV(tlK, DH, valid_max);
    tlK = setTensorLayoutStrideNV(tlK, pc.t_max, 1);
    tlK = setTensorLayoutClampValueNV(tlK, 0u);

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlV =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlV = setTensorLayoutDimensionNV(tlV, valid_max, DH);
    tlV = setTensorLayoutStrideNV(tlV, DH, 1);
    tlV = setTensorLayoutClampValueNV(tlV, 0u);

    // Q: load (f16 storage widens to f32 first), scale in f32,
    // convert to the f16 A operand (post-scale logits are safely in
    // f16 range).
    coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> Q;
#ifdef IO_F16
    {
        coopmat<float16_t, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> Qh;
        coopMatLoadTensorNV(Qh, pc.q_ptr.v, head * pc.t_q * DH,
            sliceTensorLayoutNV(tlQ, q0, Br, 0, DH));
        Q = coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator>(Qh);
    }
#else
    coopMatLoadTensorNV(Q, pc.q_ptr.v, head * pc.t_q * DH,
        sliceTensorLayoutNV(tlQ, q0, Br, 0, DH));
#endif
    Q *= pc.scale;
    coopmat<float16_t, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseA> Qf16 =
        coopmat<float16_t, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseA>(Q);

    // Online-softmax state, all f32, all matrix-resident.
    coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> O =
        coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> L =
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> M =
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(NEG_FLT_MAX_OVER_2);

    const uint kt_base = kv_head * DH * pc.t_max;
    const uint v_base = kv_head * pc.t_max * DH;

    [[dont_unroll]]
    for (uint tile = 0u; tile < num_tiles; ++tile) {
        const uint k0 = tile * Bc;

        // Kt is already [dh, T] — no transpose view needed.
        coopmat<float16_t, gl_ScopeWorkgroup, DH, Bc, gl_MatrixUseB> K_T;
#ifdef KV_F16
        coopMatLoadTensorNV(K_T, pc.kt_ptr.v, kt_base,
            sliceTensorLayoutNV(tlK, 0, DH, k0, Bc));
#else
        coopMatLoadTensorNV(K_T, pc.kt_ptr.v, kt_base,
            sliceTensorLayoutNV(tlK, 0, DH, k0, Bc), decodeKvF32);
#endif

        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> S =
            coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
        S = coopMatMulAdd(Qf16, K_T, S);

        // Three-zone masking, workgroup-uniform: tiles fully below
        // every row's frontier skip the callback; tiles fully above
        // are never visited (loop bound).  Only diagonal-straddling
        // or ragged tiles pay for the per-element pass.
        if (k0 + Bc > pc.pos_base + q0 + 1u || k0 + Bc > pc.kv_len || q0 + Br > pc.t_q) {
            coopMatPerElementNV(S, S, applyMask, q0, k0);
        }

        // Online softmax, entirely matrix-resident.  Every real query
        // row sees a valid column in tile 0, so M is finite from the
        // first tile and later fully-masked columns underflow to
        // exactly 0 in P.  (Rows past t_q compute garbage that the
        // clamped store below never writes.)
        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> rowmax, P, rowsum, eM;
        coopMatReduceNV(rowmax, S, gl_CooperativeMatrixReduceRowNV, maxReduce);

        coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator> Mold = M;
        coopMatPerElementNV(M, rowmax, Max, Mold); // M = max(rowmax, Mold)
        coopMatPerElementNV(P, S - M, Exp);        // P = e^(S-M)
        coopMatPerElementNV(eM, Mold - M, Exp);    // per-row rescale factor

        // P in [0, 1] is exact-safe as an f16 operand; row sums via
        // P x ones so they ride the tensor cores too (f32 accumulate).
        coopmat<float16_t, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseA> P_A =
            coopmat<float16_t, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseA>(P);
        coopmat<float16_t, gl_ScopeWorkgroup, Bc, Bc, gl_MatrixUseB> One =
            coopmat<float16_t, gl_ScopeWorkgroup, Bc, Bc, gl_MatrixUseB>(1.0);
        rowsum = coopmat<float, gl_ScopeWorkgroup, Br, Bc, gl_MatrixUseAccumulator>(0.0);
        rowsum = coopMatMulAdd(P_A, One, rowsum);

        coopmat<float16_t, gl_ScopeWorkgroup, Bc, DH, gl_MatrixUseB> V;
#ifdef KV_F16
        coopMatLoadTensorNV(V, pc.v_ptr.v, v_base,
            sliceTensorLayoutNV(tlV, k0, Bc, 0, DH));
#else
        coopMatLoadTensorNV(V, pc.v_ptr.v, v_base,
            sliceTensorLayoutNV(tlV, k0, Bc, 0, DH), decodeKvF32);
#endif

        L = eM * L + rowsum;

        // Smear eM across DH columns, then rescale O in-place — the
        // fragment-resident per-row rescale coopmat1 could not do.
        coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> eMdiag;
        coopMatReduceNV(eMdiag, eM, gl_CooperativeMatrixReduceRowNV, smearReduce);
        O *= eMdiag;
        O = coopMatMulAdd(P_A, V, O);
    }

    // out = O / L, with the all-masked-row guard (kv_len == 0 leaves
    // L at zero) storing zeros — same contract as the SIMT kernel.
    coopmat<float, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> Ldiag;
    coopMatReduceNV(Ldiag, L, gl_CooperativeMatrixReduceRowNV, smearReduce);
    [[unroll]]
    for (int k = 0; k < Ldiag.length(); ++k) {
        Ldiag[k] = (Ldiag[k] == 0.0) ? 0.0 : (1.0 / Ldiag[k]);
    }
    O = Ldiag * O;

    // Bounds-checked store: rows past t_q are clamped by the layout.
    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> tlO =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    tlO = setTensorLayoutDimensionNV(tlO, pc.t_q, DH);
    tlO = setTensorLayoutStrideNV(tlO, DH, 1);
#ifdef IO_F16
    coopmat<float16_t, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator> O16 =
        coopmat<float16_t, gl_ScopeWorkgroup, Br, DH, gl_MatrixUseAccumulator>(O);
    coopMatStoreTensorNV(O16, pc.out_ptr.v, head * pc.t_q * DH,
        sliceTensorLayoutNV(tlO, q0, Br, 0, DH));
#else
    coopMatStoreTensorNV(O, pc.out_ptr.v, head * pc.t_q * DH,
        sliceTensorLayoutNV(tlO, q0, Br, 0, DH));
#endif
}
