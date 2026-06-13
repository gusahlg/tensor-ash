use anyhow::{Result, bail};
use ash::vk;

/// Push constants for the matmul shader.  Bit-for-bit identical to the
/// GLSL `PC` block.
///
/// `a_ptr`, `b_ptr`, `c_ptr` are GPU device addresses (Vulkan 1.2
/// `bufferDeviceAddress`).  Kernels using descriptor bindings ignore
/// them; the `buffer_reference`-based variants dereference them via
/// `GL_EXT_buffer_reference` for direct LDG.128 access without the
/// descriptor indirection.
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
}

/// Static description of one matmul kernel: its display name, output-tile
/// dimensions, and embedded SPIR-V binary.  All concrete (non-Auto)
/// `KernelSelection` variants index into `KERNEL_SPECS` in declaration
/// order; see `KernelSelection::index`.
pub struct KernelSpec {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub spv: &'static [u8],
}

/// The full registry of compiled matmul kernels.  Order must match the
/// non-`Auto` order of `KernelSelection`.
pub const KERNEL_SPECS: &[KernelSpec] = &[
    KernelSpec {
        name: "large_128",
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv")),
    },
    KernelSpec {
        name: "small_64",
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small.spv")),
    },
    KernelSpec {
        name: "m64n128",
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n128.spv")),
    },
    KernelSpec {
        name: "m128n64",
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64.spv")),
    },
    KernelSpec {
        name: "m128n64k64",
        tile_m: 128,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64k64.spv")),
    },
    KernelSpec {
        name: "m64n32",
        tile_m: 64,
        tile_n: 32,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n32.spv")),
    },
    KernelSpec {
        name: "k64",
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_k64.spv")),
    },
    KernelSpec {
        name: "bk16_128x128",
        tile_m: 128,
        tile_n: 128,
        tile_k: 16,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_bk16.spv")),
    },
    KernelSpec {
        name: "v2_128x128_bk8",
        tile_m: 128,
        tile_n: 128,
        tile_k: 8,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_v2.spv")),
    },
    KernelSpec {
        name: "m64n128k64",
        tile_m: 64,
        tile_n: 128,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n128k64.spv")),
    },
    KernelSpec {
        name: "m128n128_t4",
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n128_t4.spv")),
    },
    KernelSpec {
        name: "m256n64",
        tile_m: 256,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m256n64.spv")),
    },
    KernelSpec {
        name: "v3_128x128_bk8_static",
        tile_m: 128,
        tile_n: 128,
        tile_k: 8,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_v3.spv")),
    },
    KernelSpec {
        name: "large_bda",
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_large_bda.spv")),
    },
    KernelSpec {
        name: "m128n64k64_bda",
        tile_m: 128,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64k64_bda.spv")),
    },
    KernelSpec {
        name: "k64_bda",
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_k64_bda.spv")),
    },
    KernelSpec {
        name: "small_bda",
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small_bda.spv")),
    },
    KernelSpec {
        name: "m64n32_bda",
        tile_m: 64,
        tile_n: 32,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n32_bda.spv")),
    },
    KernelSpec {
        name: "m128n64_bda",
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64_bda.spv")),
    },
    KernelSpec {
        name: "m64n128_bda",
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n128_bda.spv")),
    },
    KernelSpec {
        name: "large_bda_v4",
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_large_bda_v4.spv")),
    },
    KernelSpec {
        name: "m128n64k64_bda_v4",
        tile_m: 128,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64k64_bda_v4.spv")),
    },
    KernelSpec {
        name: "small_bda_v4",
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small_bda_v4.spv")),
    },
    KernelSpec {
        name: "k64_bda_v4",
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_k64_bda_v4.spv")),
    },
    KernelSpec {
        name: "m128n64_bda_v4",
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64_bda_v4.spv")),
    },
    KernelSpec {
        name: "m64n128_bda_v4",
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n128_bda_v4.spv")),
    },
    KernelSpec {
        name: "bk16_bda_v4",
        tile_m: 128,
        tile_n: 128,
        tile_k: 16,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_bk16_bda_v4.spv")),
    },
    KernelSpec {
        // Aligned-only variant of `large_bda_v4`: source-level removes
        // every bounds-check branch and the scalar fallback loaders.
        // Dispatcher MUST guarantee M%128==N%128==K%32==0 before
        // routing here.
        name: "large_bda_v4_aligned",
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_large_bda_v4_aligned.spv")),
    },
    KernelSpec {
        // Aligned-only variant of `m128n64k64_bda_v4`.
        // Dispatcher MUST guarantee M%128==N%64==K%64==0 before
        // routing here.
        name: "m128n64k64_bda_v4_aligned",
        tile_m: 128,
        tile_n: 64,
        tile_k: 64,
        spv: include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64k64_bda_v4_aligned.spv")),
    },
];

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
    Bk16,
    V2,
    M64N128K64,
    M128N128T4,
    M256N64,
    V3,
    LargeBda,
    M128N64K64Bda,
    K64Bda,
    SmallBda,
    M64N32Bda,
    M128N64Bda,
    M64N128Bda,
    LargeBdaV4,
    M128N64K64BdaV4,
    SmallBdaV4,
    K64BdaV4,
    M128N64BdaV4,
    M64N128BdaV4,
    Bk16BdaV4,
    LargeBdaV4Aligned,
    M128N64K64BdaV4Aligned,
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
            "bk16" | "128x128bk16" => Ok(Self::Bk16),
            "v2" | "128x128bk8" | "sota" => Ok(Self::V2),
            "m64n128k64" | "64x128k64" => Ok(Self::M64N128K64),
            "m128n128_t4" | "128x128_t4" => Ok(Self::M128N128T4),
            "m256n64" | "256x64" => Ok(Self::M256N64),
            "v3" | "128x128bk8static" => Ok(Self::V3),
            "large_bda" | "128x128_bda" => Ok(Self::LargeBda),
            "m128n64k64_bda" | "128x64k64_bda" => Ok(Self::M128N64K64Bda),
            "k64_bda" => Ok(Self::K64Bda),
            "small_bda" | "64_bda" => Ok(Self::SmallBda),
            "m64n32_bda" | "64x32_bda" => Ok(Self::M64N32Bda),
            "m128n64_bda" => Ok(Self::M128N64Bda),
            "m64n128_bda" => Ok(Self::M64N128Bda),
            "large_bda_v4" => Ok(Self::LargeBdaV4),
            "m128n64k64_bda_v4" => Ok(Self::M128N64K64BdaV4),
            "small_bda_v4" => Ok(Self::SmallBdaV4),
            "k64_bda_v4" => Ok(Self::K64BdaV4),
            "m128n64_bda_v4" => Ok(Self::M128N64BdaV4),
            "m64n128_bda_v4" => Ok(Self::M64N128BdaV4),
            "bk16_bda_v4" | "128x128bk16_bda_v4" => Ok(Self::Bk16BdaV4),
            "large_bda_v4_aligned" | "128x128_bda_v4_aligned" => Ok(Self::LargeBdaV4Aligned),
            "m128n64k64_bda_v4_aligned" => Ok(Self::M128N64K64BdaV4Aligned),
            other => bail!(
                "invalid ML_KERNEL '{other}', expected one of auto, large, small, m64n128, m128n64, m128n64k64, m64n32, k64, bk16, v2, m64n128k64, m128n128_t4, m256n64, v3, or any *_bda / *_bda_v4 variant"
            ),
        }
    }

    pub fn from_env() -> Result<Self> {
        let value = std::env::var("ML_KERNEL").unwrap_or_else(|_| "auto".into());
        Self::parse(&value)
    }

    /// Index into `KERNEL_SPECS` / `MatmulPipeline::kernels` for this
    /// selection.  `Auto` has no fixed kernel, so it returns `None`; the
    /// caller is responsible for resolving it to a concrete variant
    /// before looking up a kernel.
    #[inline]
    pub const fn index(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Large => Some(0),
            Self::Small => Some(1),
            Self::M64N128 => Some(2),
            Self::M128N64 => Some(3),
            Self::M128N64K64 => Some(4),
            Self::M64N32 => Some(5),
            Self::K64 => Some(6),
            Self::Bk16 => Some(7),
            Self::V2 => Some(8),
            Self::M64N128K64 => Some(9),
            Self::M128N128T4 => Some(10),
            Self::M256N64 => Some(11),
            Self::V3 => Some(12),
            Self::LargeBda => Some(13),
            Self::M128N64K64Bda => Some(14),
            Self::K64Bda => Some(15),
            Self::SmallBda => Some(16),
            Self::M64N32Bda => Some(17),
            Self::M128N64Bda => Some(18),
            Self::M64N128Bda => Some(19),
            Self::LargeBdaV4 => Some(20),
            Self::M128N64K64BdaV4 => Some(21),
            Self::SmallBdaV4 => Some(22),
            Self::K64BdaV4 => Some(23),
            Self::M128N64BdaV4 => Some(24),
            Self::M64N128BdaV4 => Some(25),
            Self::Bk16BdaV4 => Some(26),
            Self::LargeBdaV4Aligned => Some(27),
            Self::M128N64K64BdaV4Aligned => Some(28),
        }
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

    #[test]
    fn kernel_selection_index_covers_every_spec() {
        for selection in [
            KernelSelection::Large,
            KernelSelection::Small,
            KernelSelection::M64N128,
            KernelSelection::M128N64,
            KernelSelection::M128N64K64,
            KernelSelection::M64N32,
            KernelSelection::K64,
        ] {
            let idx = selection.index().expect("non-Auto selection has an index");
            assert!(idx < KERNEL_SPECS.len(), "index {idx} out of range");
        }
        assert_eq!(KernelSelection::Auto.index(), None);
    }
}
