// =====================================================================
//  matmul_epilogue_common.glsl — fused epilogue applied at C-store time.
//
//  Included by the BDA kernel bodies *after* their push-constant block
//  declaration: relies on `pc.bias_ptr`, `pc.bias_batch_stride`,
//  `pc.d_ptr`, `pc.beta` and the
//  `F32ReadOnly` / `F32V4ReadOnly` buffer_reference types.
//
//  All three knobs are specialization constants, so every branch below
//  folds at pipeline-creation time.  The default (0, 0, 0) epilogue
//  compiles to the identity and adds zero instructions to the plain
//  GEMM pipelines.
//
//  Semantics (applied after alpha-scale and ACCUMULATE):
//      v = alpha * (A@B) [+ C_old]
//      if EPI_BIAS:        v += bias[batch,g_col]     ([N] or [B,N])
//      v = act(v)                                     (per EPI_ACT)
//      if EPI_BINARY == 1: v += beta * D[c_idx]       (residual add)
//      if EPI_BINARY == 2: v *= D[c_idx]              (SwiGLU-style gate)
//      C[c_idx] = v
//  D has the same shape/layout as C (including batch stride).
// =====================================================================

layout(constant_id = 4) const uint EPI_BIAS   = 0u;  // 0=off, 1=+bias[N]
layout(constant_id = 5) const uint EPI_ACT    = 0u;  // 0=none 1=relu 2=silu 3=gelu(tanh)
layout(constant_id = 6) const uint EPI_BINARY = 0u;  // 0=none 1=+beta*D 2=*D

const bool EPI_ANY = (EPI_BIAS != 0u) || (EPI_ACT != 0u) || (EPI_BINARY != 0u);

float epi_activate(float x) {
    if (EPI_ACT == 1u) return max(x, 0.0);
    if (EPI_ACT == 2u) return x / (1.0 + exp(-x));
    if (EPI_ACT == 3u) {
        // tanh-approximated GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
        const float k0 = 0.7978845608028654;
        return 0.5 * x * (1.0 + tanh(k0 * fma(0.044715 * x * x, x, x)));
    }
    return x;
}

// Scalar path. `bias_batch_stride == 0` broadcasts one bias row.
float epi_apply(float v, uint c_idx, uint g_col, uint batch) {
    if (EPI_BIAS != 0u) {
        v += pc.bias_ptr.v[batch * pc.bias_batch_stride + g_col];
    }
    v = epi_activate(v);
    if (EPI_BINARY == 1u) v = fma(pc.beta, pc.d_ptr.v[c_idx], v);
    else if (EPI_BINARY == 2u) v *= pc.d_ptr.v[c_idx];
    return v;
}

// Vec4 path for the interior store: `c_idx4` = c_idx/4 and
// `g_col4` = g_col/4, both 16-byte aligned by the interior contract.
vec4 epi_apply4(vec4 v, uint c_idx4, uint g_col4, uint batch) {
    if (EPI_BIAS != 0u) {
        uint base = batch * pc.bias_batch_stride + g_col4 * 4u;
        v += vec4(
            pc.bias_ptr.v[base],
            pc.bias_ptr.v[base + 1u],
            pc.bias_ptr.v[base + 2u],
            pc.bias_ptr.v[base + 3u]
        );
    }
    if (EPI_ACT == 1u) {
        v = max(v, vec4(0.0));
    } else if (EPI_ACT == 2u) {
        v = v / (vec4(1.0) + exp(-v));
    } else if (EPI_ACT == 3u) {
        v.x = epi_activate(v.x);
        v.y = epi_activate(v.y);
        v.z = epi_activate(v.z);
        v.w = epi_activate(v.w);
    }
    if (EPI_BINARY == 1u) {
        F32V4ReadOnly d4 = F32V4ReadOnly(uint64_t(pc.d_ptr));
        v = fma(vec4(pc.beta), d4.v[c_idx4], v);
    } else if (EPI_BINARY == 2u) {
        F32V4ReadOnly d4 = F32V4ReadOnly(uint64_t(pc.d_ptr));
        v *= d4.v[c_idx4];
    }
    return v;
}
