use anyhow::{Result, bail};
use ash::vk;

/// Push constants for the matmul shader.  Bit-for-bit identical to the
/// GLSL `PC` block.
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
}

/// Default large-kernel output-tile dimensions.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 128;
pub const TILE_K: u32 = 32;
pub const SMALL_TILE_M: u32 = 64;
pub const SMALL_TILE_N: u32 = 64;
pub const SMALL_TILE_K: u32 = 32;
pub const M64N128_TILE_M: u32 = 64;
pub const M64N128_TILE_N: u32 = 128;
pub const M64N128_TILE_K: u32 = 32;
pub const M128N64_TILE_M: u32 = 128;
pub const M128N64_TILE_N: u32 = 64;
pub const M128N64_TILE_K: u32 = 32;
pub const M128N64K64_TILE_M: u32 = 128;
pub const M128N64K64_TILE_N: u32 = 64;
pub const M128N64K64_TILE_K: u32 = 64;
pub const M64N32_TILE_M: u32 = 64;
pub const M64N32_TILE_N: u32 = 32;
pub const M64N32_TILE_K: u32 = 32;
pub const K64_TILE_M: u32 = 64;
pub const K64_TILE_N: u32 = 64;
pub const K64_TILE_K: u32 = 64;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelSelection {
    Auto,
    Large,
    Small,
    M64N128,
    M128N64,
    M128N64K64,
    M64N32,
    K64,
}

impl KernelSelection {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "large" | "large_128" | "128" => Ok(Self::Large),
            "small" | "small_64" | "64" => Ok(Self::Small),
            "m64n128" | "64x128" | "wide" => Ok(Self::M64N128),
            "m128n64" | "128x64" | "tall" => Ok(Self::M128N64),
            "m128n64k64" | "128x64k64" => Ok(Self::M128N64K64),
            "m64n32" | "64x32" => Ok(Self::M64N32),
            "k64" | "small_k64" | "64k" => Ok(Self::K64),
            other => bail!(
                "invalid ML_KERNEL '{other}', expected auto, large, small, m64n128, m128n64, m128n64k64, m64n32, or k64"
            ),
        }
    }

    pub(super) fn from_env() -> Result<Self> {
        let value = std::env::var("ML_KERNEL").unwrap_or_else(|_| "auto".into());
        Self::parse(&value)
    }
}

/// Per-call pipeline specialization.  Selects one of the precompiled
/// variants of a kernel so the shader sees these as compile-time
/// constants and can fold out the corresponding branches.
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
    /// Number of distinct variants compiled per kernel.
    pub const COUNT: usize = 16;

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
}

pub struct MatmulKernel {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub shader_module: vk::ShaderModule,
    /// One pipeline per `KernelVariant`; indexed by `KernelVariant::index()`.
    pub variants: [vk::Pipeline; KernelVariant::COUNT],
}

impl MatmulKernel {
    #[inline]
    pub fn pipeline_for(&self, variant: KernelVariant) -> vk::Pipeline {
        self.variants[variant.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_selection() {
        assert_eq!(KernelSelection::parse("").unwrap(), KernelSelection::Auto);
        assert_eq!(
            KernelSelection::parse("auto").unwrap(),
            KernelSelection::Auto
        );
        assert_eq!(
            KernelSelection::parse("large_128").unwrap(),
            KernelSelection::Large
        );
        assert_eq!(
            KernelSelection::parse("64").unwrap(),
            KernelSelection::Small
        );
        assert_eq!(
            KernelSelection::parse("wide").unwrap(),
            KernelSelection::M64N128
        );
        assert_eq!(
            KernelSelection::parse("tall").unwrap(),
            KernelSelection::M128N64
        );
        assert_eq!(
            KernelSelection::parse("128x64k64").unwrap(),
            KernelSelection::M128N64K64
        );
        assert_eq!(
            KernelSelection::parse("64x32").unwrap(),
            KernelSelection::M64N32
        );
        assert_eq!(KernelSelection::parse("k64").unwrap(), KernelSelection::K64);
        assert!(KernelSelection::parse("wideish").is_err());
    }

    #[test]
    fn kernel_variant_index_round_trip() {
        for idx in 0..KernelVariant::COUNT {
            let v = KernelVariant::from_index(idx);
            assert_eq!(v.index(), idx, "round-trip broken for idx {idx}");
        }
    }

    #[test]
    fn kernel_variant_index_in_range() {
        let mut seen = [false; KernelVariant::COUNT];
        for accumulate in [false, true] {
            for alpha_is_one in [false, true] {
                for interior_only in [false, true] {
                    for k_multiple in [false, true] {
                        let v = KernelVariant {
                            accumulate,
                            alpha_is_one,
                            interior_only,
                            k_multiple,
                        };
                        let idx = v.index();
                        assert!(idx < KernelVariant::COUNT);
                        assert!(!seen[idx], "duplicate index {idx}");
                        seen[idx] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&b| b));
    }
}
