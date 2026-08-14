/// Push constants for the matmul shader. Bit-for-bit identical to the
/// GLSL `PC` block.
///
/// `a_ptr`, `b_ptr`, `c_ptr` are GPU device addresses (Vulkan 1.2
/// `bufferDeviceAddress`). Kernels using descriptor bindings ignore
/// them; the `buffer_reference`-based variants dereference them via
/// `GL_EXT_buffer_reference` for direct LDG.128 access without the
/// descriptor indirection.
/// `flags` bit 1: the row-GEMV kernels fold an RMSNorm of the A row
/// into the product (`bias_ptr` carries the norm weight, `beta` eps).
/// Bit 0 is the legacy accumulate flag; bits 16..31 carry split-K
/// counts.
pub const MATMUL_FLAG_NORM_A: u32 = 1 << 1;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_c: u32,
    pub flags: u32,
    pub alpha: f32,
    pub a_ptr: u64,
    pub b_ptr: u64,
    pub c_ptr: u64,
    /// Epilogue residual / gate operand (same shape+layout as C), or 0.
    pub d_ptr: u64,
    /// Epilogue bias data, either broadcast `[N]` or batched `[B, N]`, or 0.
    pub bias_ptr: u64,
    /// Scale for the `+= beta * D` residual epilogue.
    pub beta: f32,
    /// Bias elements between batches; zero means broadcast one `[N]` row.
    pub bias_batch_stride: u32,
}

/// Per-call pipeline specialization. The eager fallbacks and lazily-built
/// exact variants let the shader fold out the corresponding branches.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct KernelVariant {
    /// `C += alpha*A@B` when true, `C = alpha*A@B` when false.
    pub accumulate: bool,
    /// Host knows alpha==1.0, so the shader skips the multiply.
    pub alpha_is_one: bool,
    /// Host knows M and N are multiples of the tile size, so the shader
    /// drops all m_full/n_full bounds checks.
    pub interior_only: bool,
    /// Host knows K is a multiple of the K tile, so the shader drops the
    /// tail-strip branch.
    pub k_multiple: bool,
}

impl KernelVariant {
    /// Number of logical specialization variants per kernel.
    pub const COUNT: usize = 16;
    /// Number of correctness-safe variants built eagerly per kernel.
    pub(crate) const FALLBACK_COUNT: usize = 4;

    #[inline]
    pub const fn index(self) -> usize {
        (self.accumulate as usize)
            | ((self.alpha_is_one as usize) << 1)
            | ((self.interior_only as usize) << 2)
            | ((self.k_multiple as usize) << 3)
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Self {
        Self {
            accumulate: (idx & 0b001) != 0,
            alpha_is_one: (idx & 0b010) != 0,
            interior_only: (idx & 0b100) != 0,
            k_multiple: (idx & 0b1000) != 0,
        }
    }

    /// Correctness-safe variant used while the aligned specialization is lazy.
    #[inline]
    pub(crate) const fn fallback(self) -> Self {
        Self {
            accumulate: self.accumulate,
            alpha_is_one: self.alpha_is_one,
            interior_only: false,
            k_multiple: false,
        }
    }
}

/// Specialization values for the fused-epilogue constants (IDs 4..6).
/// The eager per-kernel fallbacks use the zero epilogue; aligned and non-zero
/// epilogue combinations are compiled lazily and cached in `MatmulPipeline`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default)]
pub struct EpilogueKey {
    /// `constant_id = 4`: add a shared `[N]` or batched `[B, N]` bias.
    pub bias: bool,
    /// `constant_id = 5`: 0=none, 1=relu, 2=silu, 3=gelu(tanh).
    pub activation: u32,
    /// `constant_id = 6`: 0=none, 1=`+beta*D`, 2=`*D`.
    pub binary: u32,
}

impl EpilogueKey {
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.bias && self.activation == 0 && self.binary == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_variant_indices_are_bijective() {
        let mut seen = [false; KernelVariant::COUNT];
        for idx in 0..KernelVariant::COUNT {
            let variant = KernelVariant::from_index(idx);
            assert_eq!(variant.index(), idx, "round-trip broken for index {idx}");
            assert!(!seen[variant.index()], "duplicate index {idx}");
            seen[variant.index()] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn fallback_preserves_semantics_and_drops_alignment_assumptions() {
        for idx in 0..KernelVariant::COUNT {
            let variant = KernelVariant::from_index(idx);
            let fallback = variant.fallback();
            assert_eq!(fallback.accumulate, variant.accumulate);
            assert_eq!(fallback.alpha_is_one, variant.alpha_is_one);
            assert!(!fallback.interior_only);
            assert!(!fallback.k_multiple);
            assert!(fallback.index() < KernelVariant::FALLBACK_COUNT);
        }
    }
}
