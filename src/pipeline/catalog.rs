use anyhow::{Result, bail};

/// Static description of one matmul kernel.
///
/// `uses_descriptors` is `true` for kernels whose shader binds A/B/C to
/// SSBO bindings 0/1/2, and `false` for kernels that address A/B/C via
/// push-constant `buffer_reference` pointers.
pub struct KernelSpec {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub spv: &'static [u8],
    pub uses_descriptors: bool,
}

impl KernelSpec {
    /// Whether the shader reads B as f16 storage (the `f16w_*` family).
    #[inline]
    pub fn weights_f16(&self) -> bool {
        self.name.starts_with("f16w_")
    }
}

/// Defines kernel identity once and derives the public selection enum,
/// parser, stable registry order, and selection-to-registry mapping.
macro_rules! kernel_catalog {
    (
        $(
            $variant:ident => (
                $name:literal, $tile_m:literal, $tile_n:literal, $tile_k:literal,
                $shader:literal, $uses_descriptors:literal,
                [$($alias:literal),+ $(,)?]
            )
        ),+ $(,)?
    ) => {
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub enum KernelSelection {
            Auto,
            $($variant),+
        }

        /// The full registry of compiled matmul kernels. Its order is
        /// generated from the same declarations as `KernelSelection`.
        pub const KERNEL_SPECS: &[KernelSpec] = &[
            $(KernelSpec {
                name: $name,
                tile_m: $tile_m,
                tile_n: $tile_n,
                tile_k: $tile_k,
                spv: include_bytes!(concat!(env!("OUT_DIR"), "/", $shader)),
                uses_descriptors: $uses_descriptors,
            }),+
        ];

        impl KernelSelection {
            pub fn parse(value: &str) -> Result<Self> {
                match value.trim().to_ascii_lowercase().as_str() {
                    "" | "auto" => Ok(Self::Auto),
                    $($($alias)|+ => Ok(Self::$variant),)+
                    other => bail!(
                        "invalid ML_KERNEL '{other}', expected one of auto, large, small, m64n128, m128n64, m128n64k64, m64n32, k64, row_bda, col_bda, outer_bda, bk16, v2, m64n128k64, m128n128_t4, m256n64, v3, any *_bda / *_bda_v4 variant, or an f16w_* kernel (f16 weights)"
                    ),
                }
            }

            pub fn from_env() -> Result<Self> {
                let value = std::env::var("ML_KERNEL").unwrap_or_else(|_| "auto".into());
                Self::parse(&value)
            }

            /// Index into `KERNEL_SPECS` / `MatmulPipeline::kernels`.
            /// `Auto` has no fixed kernel and returns `None`.
            #[inline]
            pub const fn index(self) -> Option<usize> {
                match self {
                    Self::Auto => None,
                    concrete => Some(concrete as usize - 1),
                }
            }
        }

        #[cfg(test)]
        const CONCRETE_SELECTIONS: &[KernelSelection] = &[
            $(KernelSelection::$variant),+
        ];

        #[cfg(test)]
        const KERNEL_ALIASES: &[(&str, KernelSelection)] = &[
            ("", KernelSelection::Auto),
            ("auto", KernelSelection::Auto),
            $($(($alias, KernelSelection::$variant),)+)+
        ];
    };
}

kernel_catalog! {
    Large => ("large_128", 128, 128, 32, "matmul_f32.spv", true,
        ["large", "large_128", "128"]),
    Small => ("small_64", 64, 64, 32, "matmul_f32_small.spv", true,
        ["small", "small_64", "64"]),
    M64N128 => ("m64n128", 64, 128, 32, "matmul_f32_m64n128.spv", true,
        ["m64n128", "64x128", "wide"]),
    M128N64 => ("m128n64", 128, 64, 32, "matmul_f32_m128n64.spv", true,
        ["m128n64", "128x64", "tall"]),
    M128N64K64 => ("m128n64k64", 128, 64, 64, "matmul_f32_m128n64k64.spv", true,
        ["m128n64k64", "128x64k64"]),
    M64N32 => ("m64n32", 64, 32, 32, "matmul_f32_m64n32.spv", true,
        ["m64n32", "64x32"]),
    K64 => ("k64", 64, 64, 64, "matmul_f32_k64.spv", true,
        ["k64", "small_k64", "64k"]),
    Bk16 => ("bk16_128x128", 128, 128, 16, "matmul_f32_bk16.spv", true,
        ["bk16", "128x128bk16"]),
    V2 => ("v2_128x128_bk8", 128, 128, 8, "matmul_f32_v2.spv", true,
        ["v2", "128x128bk8", "sota"]),
    M64N128K64 => ("m64n128k64", 64, 128, 64, "matmul_f32_m64n128k64.spv", true,
        ["m64n128k64", "64x128k64"]),
    M128N128T4 => ("m128n128_t4", 128, 128, 32, "matmul_f32_m128n128_t4.spv", true,
        ["m128n128_t4", "128x128_t4"]),
    M256N64 => ("m256n64", 256, 64, 32, "matmul_f32_m256n64.spv", true,
        ["m256n64", "256x64"]),
    V3 => ("v3_128x128_bk8_static", 128, 128, 8, "matmul_f32_v3.spv", true,
        ["v3", "128x128bk8static"]),
    LargeBda => ("large_bda", 128, 128, 32, "matmul_f32_large_bda.spv", false,
        ["large_bda", "128x128_bda"]),
    M128N64K64Bda => ("m128n64k64_bda", 128, 64, 64, "matmul_f32_m128n64k64_bda.spv", false,
        ["m128n64k64_bda", "128x64k64_bda"]),
    K64Bda => ("k64_bda", 64, 64, 64, "matmul_f32_k64_bda.spv", false,
        ["k64_bda"]),
    SmallBda => ("small_bda", 64, 64, 32, "matmul_f32_small_bda.spv", false,
        ["small_bda", "64_bda"]),
    M64N32Bda => ("m64n32_bda", 64, 32, 32, "matmul_f32_m64n32_bda.spv", false,
        ["m64n32_bda", "64x32_bda"]),
    M128N64Bda => ("m128n64_bda", 128, 64, 32, "matmul_f32_m128n64_bda.spv", false,
        ["m128n64_bda"]),
    M64N128Bda => ("m64n128_bda", 64, 128, 32, "matmul_f32_m64n128_bda.spv", false,
        ["m64n128_bda"]),
    LargeBdaV4 => ("large_bda_v4", 128, 128, 32, "matmul_f32_large_bda_v4.spv", false,
        ["large_bda_v4"]),
    M128N64K64BdaV4 => ("m128n64k64_bda_v4", 128, 64, 64, "matmul_f32_m128n64k64_bda_v4.spv", false,
        ["m128n64k64_bda_v4"]),
    SmallBdaV4 => ("small_bda_v4", 64, 64, 32, "matmul_f32_small_bda_v4.spv", false,
        ["small_bda_v4"]),
    K64BdaV4 => ("k64_bda_v4", 64, 64, 64, "matmul_f32_k64_bda_v4.spv", false,
        ["k64_bda_v4"]),
    M128N64BdaV4 => ("m128n64_bda_v4", 128, 64, 32, "matmul_f32_m128n64_bda_v4.spv", false,
        ["m128n64_bda_v4"]),
    M64N128BdaV4 => ("m64n128_bda_v4", 64, 128, 32, "matmul_f32_m64n128_bda_v4.spv", false,
        ["m64n128_bda_v4"]),
    Bk16BdaV4 => ("bk16_bda_v4", 128, 128, 16, "matmul_f32_bk16_bda_v4.spv", false,
        ["bk16_bda_v4", "128x128bk16_bda_v4"]),
    M128N64K64BdaV4Tm8Tn8 => ("m128n64k64_bda_v4_tm8_tn8", 128, 64, 64,
        "matmul_f32_m128n64k64_bda_v4_tm8_tn8.spv", false, ["m128n64k64_bda_v4_tm8_tn8"]),
    M128N64K64BdaV4Tm16Tn4 => ("m128n64k64_bda_v4_tm16_tn4", 128, 64, 64,
        "matmul_f32_m128n64k64_bda_v4_tm16_tn4.spv", false, ["m128n64k64_bda_v4_tm16_tn4"]),
    K64BdaV4Tm8Tn4 => ("k64_bda_v4_tm8_tn4", 64, 64, 64,
        "matmul_f32_k64_bda_v4_tm8_tn4.spv", false, ["k64_bda_v4_tm8_tn4"]),
    K64BdaV4Tm4Tn8 => ("k64_bda_v4_tm4_tn8", 64, 64, 64,
        "matmul_f32_k64_bda_v4_tm4_tn8.spv", false, ["k64_bda_v4_tm4_tn8"]),
    K64BdaV4Tm8Tn8 => ("k64_bda_v4_tm8_tn8", 64, 64, 64,
        "matmul_f32_k64_bda_v4_tm8_tn8.spv", false, ["k64_bda_v4_tm8_tn8"]),
    // These source-stripped variants require exact tile alignment.
    LargeBdaV4Aligned => ("large_bda_v4_aligned", 128, 128, 32,
        "matmul_f32_large_bda_v4_aligned.spv", false,
        ["large_bda_v4_aligned", "128x128_bda_v4_aligned"]),
    M128N64K64BdaV4Aligned => ("m128n64k64_bda_v4_aligned", 128, 64, 64,
        "matmul_f32_m128n64k64_bda_v4_aligned.spv", false,
        ["m128n64k64_bda_v4_aligned"]),
    // Warp-sized row kernel for large batches of matrix-vector products.
    RowBda => ("row_bda", 1, 32, 1, "matmul_f32_row_bda.spv", false,
        ["row_bda", "row", "gemv_bda"]),
    // Cooperative K reduction for column vectors; general-shape correct so it
    // remains safe when selected explicitly for experiments.
    ColBda => ("col_bda", 2, 1, 1, "matmul_f32_col_bda.spv", false,
        ["col_bda", "col", "gemv_col_bda"]),
    // Register-only outer product for K=1 (rank-1 update); the tiny inner
    // loop keeps it general-shape correct when selected explicitly.
    OuterBda => ("outer_bda", 16, 128, 1, "matmul_f32_outer_bda.spv", false,
        ["outer_bda", "outer", "k1"]),
    // f16-storage-B ("f16w_" = half-precision weights) siblings of the
    // heuristic winners.  Same tiles and f32 accumulate; only global B
    // traffic halves.  Requires `VulkanContext::f16_storage_enabled`;
    // on devices without it these registry slots hold no pipelines.
    F16wLargeBdaV4 => ("f16w_large_bda_v4", 128, 128, 32,
        "matmul_f16w_large_bda_v4.spv", false, ["f16w_large_bda_v4", "f16w_large"]),
    F16wSmallBdaV4 => ("f16w_small_bda_v4", 64, 64, 32,
        "matmul_f16w_small_bda_v4.spv", false, ["f16w_small_bda_v4", "f16w_small"]),
    F16wK64BdaV4 => ("f16w_k64_bda_v4", 64, 64, 64,
        "matmul_f16w_k64_bda_v4.spv", false, ["f16w_k64_bda_v4", "f16w_k64"]),
    F16wM128N64K64BdaV4 => ("f16w_m128n64k64_bda_v4", 128, 64, 64,
        "matmul_f16w_m128n64k64_bda_v4.spv", false,
        ["f16w_m128n64k64_bda_v4", "f16w_m128n64k64"]),
    F16wRowBda => ("f16w_row_bda", 1, 32, 1, "matmul_f16w_row_bda.spv", false,
        ["f16w_row_bda", "f16w_row"]),
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    #[test]
    fn every_concrete_selection_maps_to_exactly_one_spec() {
        assert_eq!(CONCRETE_SELECTIONS.len(), KERNEL_SPECS.len());
        assert_eq!(KernelSelection::Auto.index(), None);

        let mut seen = vec![false; KERNEL_SPECS.len()];
        for (expected, selection) in CONCRETE_SELECTIONS.iter().copied().enumerate() {
            let index = selection.index().expect("concrete selection has an index");
            assert_eq!(index, expected, "registry order drift for {selection:?}");
            assert!(!seen[index], "duplicate registry index {index}");
            seen[index] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn every_alias_parses_case_insensitively_and_unambiguously() {
        let mut aliases = HashMap::new();
        for &(alias, expected) in KERNEL_ALIASES {
            let normalized = alias.trim().to_ascii_lowercase();
            assert_eq!(alias, normalized, "catalog aliases must be normalized");
            assert_eq!(KernelSelection::parse(alias).unwrap(), expected, "{alias}");
            assert_eq!(
                KernelSelection::parse(&format!("  {}  ", alias.to_ascii_uppercase())).unwrap(),
                expected,
                "{alias}"
            );
            assert!(
                aliases.insert(normalized, expected).is_none(),
                "duplicate alias {alias}"
            );
        }
        assert!(KernelSelection::parse("wideish").is_err());
    }

    #[test]
    fn registry_names_are_unique() {
        let mut names = HashSet::new();
        for spec in KERNEL_SPECS {
            assert!(
                names.insert(spec.name),
                "duplicate kernel name {}",
                spec.name
            );
        }
    }
}
