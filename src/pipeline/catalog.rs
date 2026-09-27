use std::collections::HashMap;

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
    /// Tile specialization `(constant_id, value)` pairs for kernels
    /// built from a spec-constant module (the `*_coopmat_g_*` generator
    /// configs, ids 16-23); empty for literal wrappers.
    pub spec: &'static [(u32, u32)],
}

impl KernelSpec {
    /// Whether the shader reads B as f16 storage (the `f16w_*` family).
    #[inline]
    pub fn weights_f16(&self) -> bool {
        self.name.starts_with("f16w_")
    }

    /// Whether the shader reads A and writes C as f16 storage (the
    /// `*_a16_*` f16-activations family).
    #[inline]
    pub fn a_f16(&self) -> bool {
        self.name.contains("_a16_")
    }

    /// Accumulator class: `F16` / `F16Promote` for the `*_acc16` /
    /// `*_acc16p` coopmat twins, `F32` for everything else.
    #[inline]
    pub fn accumulator(&self) -> super::CoopmatAcc {
        super::CoopmatAcc::of_kernel_name(self.name)
    }

    /// K elements per f16 partial of an `_acc16p` (promote) kernel:
    /// `PROMOTE_KT * tile_k`, read from a `_pk<KT>_` name tag
    /// (default PROMOTE_KT = 4, i.e. 128 K on the BK = 32 tiles).
    /// `None` for every other accumulator class.
    pub fn promote_chunk_k(&self) -> Option<u32> {
        if self.accumulator() != super::CoopmatAcc::F16Promote {
            return None;
        }
        let kt = self
            .name
            .split('_')
            .find_map(|part| part.strip_prefix("pk").and_then(|v| v.parse::<u32>().ok()))
            .unwrap_or(4);
        Some(kt * self.tile_k)
    }

    /// Total workgroup (shared) memory the shader declares, in bytes —
    /// the number drivers check against `maxComputeSharedMemorySize`
    /// (VUID-RuntimeSpirv-Workgroup-06530).  Read from the embedded
    /// SPIR-V so the budget gate stays exact for every kernel without
    /// maintaining a per-family size formula next to the GLSL.
    pub fn shared_memory_bytes(&self) -> u32 {
        workgroup_shared_bytes(self.spv, self.spec)
    }

    /// Whether this is a generator config (a spec-constant module plus
    /// a tile specialization) rather than a literal wrapper.
    #[inline]
    pub fn is_generated(&self) -> bool {
        !self.spec.is_empty()
    }
}

/// Sum the byte sizes of every `Workgroup`-storage-class variable in a
/// SPIR-V module.  Sizes use natural (scalar/vector/array) layout,
/// which matches how glslang emits shared declarations; no kernel puts
/// structs in shared memory, so struct padding is not modeled.
///
/// Spec-constant modules size their shared arrays with
/// `OpSpecConstantOp` expressions: `overrides` gives the specialization
/// `(constant_id, value)` pairs, every other spec constant takes its
/// default, and the integer ops glslang emits for array extents (IAdd,
/// ISub, IMul, UDiv, UMod, shifts) are folded.  A Workgroup variable
/// whose size cannot be evaluated panics with its id rather than being
/// silently under-counted (the budget gate depends on this number).
fn workgroup_shared_bytes(spv: &[u8], overrides: &[(u32, u32)]) -> u32 {
    const OP_DECORATE: u32 = 71;
    const DECORATION_SPEC_ID: u32 = 1;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_VECTOR: u32 = 23;
    const OP_TYPE_ARRAY: u32 = 28;
    const OP_TYPE_STRUCT: u32 = 30;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_CONSTANT: u32 = 43;
    const OP_SPEC_CONSTANT: u32 = 50;
    const OP_SPEC_CONSTANT_OP: u32 = 52;
    const OP_VARIABLE: u32 = 59;
    const STORAGE_CLASS_WORKGROUP: u32 = 4;
    // OpSpecConstantOp operations over 32-bit integers.
    const OP_IADD: u32 = 128;
    const OP_ISUB: u32 = 130;
    const OP_IMUL: u32 = 132;
    const OP_UDIV: u32 = 134;
    const OP_UMOD: u32 = 137;
    const OP_SHIFT_RIGHT_LOGICAL: u32 = 194;
    const OP_SHIFT_LEFT_LOGICAL: u32 = 196;

    let words: Vec<u32> = spv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    // build.rs embeds glslc output, which is always little-endian.
    assert_eq!(words[0], 0x0723_0203, "embedded SPIR-V has bad magic");

    let mut spec_ids: HashMap<u32, u32> = HashMap::new(); // result id -> SpecId
    let mut type_sizes: HashMap<u32, u32> = HashMap::new(); // type id -> bytes
    let mut constants: HashMap<u32, u32> = HashMap::new(); // constant id -> value
    let mut workgroup_pointees: HashMap<u32, u32> = HashMap::new(); // pointer type id -> pointee
    let mut total = 0u32;

    // Instructions follow the 5-word header: each is
    // (word_count << 16 | opcode) then operands.
    let mut pc = 5usize;
    while pc < words.len() {
        let word_count = (words[pc] >> 16) as usize;
        let opcode = words[pc] & 0xFFFF;
        assert!(word_count > 0, "malformed SPIR-V instruction");
        let operands = &words[pc + 1..pc + word_count];
        match opcode {
            // [target, decoration, literal...]
            OP_DECORATE if operands[1] == DECORATION_SPEC_ID => {
                spec_ids.insert(operands[0], operands[2]);
            }
            // [result, width-in-bits, ...]
            OP_TYPE_INT | OP_TYPE_FLOAT => {
                type_sizes.insert(operands[0], operands[1] / 8);
            }
            // [result, component type, component count]
            OP_TYPE_VECTOR => {
                let component = type_sizes[&operands[1]];
                type_sizes.insert(operands[0], component * operands[2]);
            }
            // [result, element type, length constant id]
            OP_TYPE_ARRAY => {
                if let (Some(&element), Some(&length)) =
                    (type_sizes.get(&operands[1]), constants.get(&operands[2]))
                {
                    type_sizes.insert(operands[0], element * length);
                }
            }
            // [result, member types...]; natural layout, no padding.
            OP_TYPE_STRUCT => {
                let sum = operands[1..]
                    .iter()
                    .map(|id| type_sizes.get(id).copied().unwrap_or(0))
                    .sum();
                type_sizes.insert(operands[0], sum);
            }
            // [result, storage class, pointee type]
            OP_TYPE_POINTER if operands[1] == STORAGE_CLASS_WORKGROUP => {
                workgroup_pointees.insert(operands[0], operands[2]);
            }
            // [result type, result, literal value...]
            OP_CONSTANT => {
                constants.insert(operands[1], operands[2]);
            }
            // [result type, result, default literal...]: the override
            // for its SpecId wins.
            OP_SPEC_CONSTANT => {
                let value = spec_ids
                    .get(&operands[1])
                    .and_then(|sid| overrides.iter().find(|&&(id, _)| id == *sid))
                    .map_or(operands[2], |&(_, value)| value);
                constants.insert(operands[1], value);
            }
            // [result type, result, opcode, operand ids...]
            OP_SPEC_CONSTANT_OP if word_count == 6 => {
                let (op, lhs, rhs) = (operands[2], operands[3], operands[4]);
                if let (Some(&x), Some(&y)) = (constants.get(&lhs), constants.get(&rhs)) {
                    let value = match op {
                        OP_IADD => Some(x.wrapping_add(y)),
                        OP_ISUB => Some(x.wrapping_sub(y)),
                        OP_IMUL => Some(x.wrapping_mul(y)),
                        OP_UDIV if y != 0 => Some(x / y),
                        OP_UMOD if y != 0 => Some(x % y),
                        OP_SHIFT_RIGHT_LOGICAL => Some(x.checked_shr(y).unwrap_or(0)),
                        OP_SHIFT_LEFT_LOGICAL => Some(x.checked_shl(y).unwrap_or(0)),
                        _ => None,
                    };
                    if let Some(value) = value {
                        constants.insert(operands[1], value);
                    }
                }
            }
            // [result (pointer) type, result, storage class, ...]
            OP_VARIABLE if operands[2] == STORAGE_CLASS_WORKGROUP => {
                let pointee = workgroup_pointees[&operands[0]];
                let size = type_sizes.get(&pointee).unwrap_or_else(|| {
                    panic!(
                        "workgroup variable %{} has a size the evaluator cannot fold \
                         (unsupported OpSpecConstantOp in its array extent)",
                        operands[1]
                    )
                });
                total += size;
            }
            _ => {}
        }
        pc += word_count;
    }
    total
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
                $(, spec [$(($sid:literal, $sval:literal)),* $(,)?])?
            )
        ),+ $(,)?
    ) => {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
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
                spec: &[$($(($sid, $sval)),*)?],
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

            /// The concrete selection at registry `index` (inverse of
            /// [`Self::index`]).
            #[inline]
            pub(crate) fn from_index(index: usize) -> Option<Self> {
                CONCRETE_SELECTIONS.get(index).copied()
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

        /// Every concrete selection in registry order
        /// (`CONCRETE_SELECTIONS[i].index() == Some(i)`).
        pub(crate) const CONCRETE_SELECTIONS: &[KernelSelection] = &[
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
    // Deep-K / narrow-N f16-weights GEMV: sixteen K-slice warps
    // instead of eight (ffn_down- and kv-projection-shaped decode
    // matmuls leave half the device idle at eight).
    F16wRowBdaK16 => ("f16w_row_bda_k16", 1, 32, 1, "matmul_f16w_row_bda_k16.spv", false,
        ["f16w_row_bda_k16", "f16w_row_k16"]),
    // VCOLS=2 sibling: each lane owns two adjacent output columns and
    // loads B as f16vec2, widening the per-warp B transaction (64B ->
    // 128B per k-step); tile_n = 64.  Wins ~3% on wide-N moderate-K
    // GEMVs (gate/up-shaped), where sixteen slices alone regress.
    // The 8-slice v2, the k16 v4, and the 8-slice v4 all measured
    // neutral-to-losing on every probed decode shape and were deleted.
    F16wRowBdaK16V2 => ("f16w_row_bda_k16_v2", 1, 64, 1,
        "matmul_f16w_row_bda_k16_v2.spv", false,
        ["f16w_row_bda_k16_v2", "f16w_row_k16_v2"]),
    // Packed-B decode GEMVs: same tiles as the k16 / k16_v2 pair, but
    // B is [N/tile_n][K][tile_n] so the K-walk is sequential (stride
    // tile_n instead of N) with the same inner-product order.  Selected
    // only when the op sets `packed_b`.
    F16wRowBdaK16Packed => ("f16w_row_bda_k16_packed", 1, 32, 1,
        "matmul_f16w_row_bda_k16_packed.spv", false,
        ["f16w_row_bda_k16_packed", "f16w_row_k16_packed"]),
    F16wRowBdaK16V2Packed => ("f16w_row_bda_k16_v2_packed", 1, 64, 1,
        "matmul_f16w_row_bda_k16_v2_packed.spv", false,
        ["f16w_row_bda_k16_v2_packed", "f16w_row_k16_v2_packed"]),
    // Tensor-core GEMM (KHR cooperative matrix): A f32 converted to
    // f16 at staging, B f16 storage, f32 accumulate.  Strictly aligned
    // (the suffix drives the host-side M/N/K % tile check) because
    // coopmat ops need subgroup-uniform control flow and this device
    // has no robust-buffer-access net for them.  Registry slot is
    // empty unless `VulkanContext::coopmat_enabled`.
    F16wCoopmat => ("f16w_coopmat_aligned", 128, 128, 32,
        "matmul_f16w_coopmat_aligned.spv", false,
        ["f16w_coopmat_aligned", "f16w_coopmat", "coopmat"]),
    // f16-activations sibling: A and C are f16 storage too (straight
    // uvec4 A staging, RNE-narrowed C store; f32 accumulate).  The
    // ONLY f16-A route — resolution enforces its tile alignment, so
    // routing is a fixed pick rather than a heuristic.  Same gating
    // as the plain coopmat kernel.
    F16wA16Coopmat => ("f16w_a16_coopmat_aligned", 128, 128, 32,
        "matmul_f16w_a16_coopmat_aligned.spv", false,
        ["f16w_a16_coopmat_aligned", "f16w_a16_coopmat", "a16_coopmat"]),
    // Wave-fill 64x64 coopmat: 4x the CTA count of 128x128 on M=512
    // prefill shapes (256 vs 64 tiles at 512x2048), so the last wave
    // does not leave 18/46 SMs idle.  Same KHR fragments, 128 threads.
    F16wCoopmatM64 => ("f16w_coopmat_m64n64", 64, 64, 32,
        "matmul_f16w_coopmat_m64n64.spv", false,
        ["f16w_coopmat_m64n64", "coopmat_m64"]),
    F16wA16CoopmatM64 => ("f16w_a16_coopmat_m64n64", 64, 64, 32,
        "matmul_f16w_a16_coopmat_m64n64.spv", false,
        ["f16w_a16_coopmat_m64n64", "a16_coopmat_m64"]),
    // Asymmetric wave-fill: 64x128 (wide) and 128x64 (tall).  Same
    // KHR fragments, 128 threads.  With a known SM count 128x64 is the
    // large-grid default (it matches or beats 128x128 on every measured
    // RTX 3070 shape); 64x128 (168 registers -> 3 CTAs/SM) is only
    // reachable through the SM-agnostic fallback or an explicit pick.
    F16wCoopmatM64N128 => ("f16w_coopmat_m64n128", 64, 128, 32,
        "matmul_f16w_coopmat_m64n128.spv", false,
        ["f16w_coopmat_m64n128", "coopmat_m64n128"]),
    F16wA16CoopmatM64N128 => ("f16w_a16_coopmat_m64n128", 64, 128, 32,
        "matmul_f16w_a16_coopmat_m64n128.spv", false,
        ["f16w_a16_coopmat_m64n128", "a16_coopmat_m64n128"]),
    F16wCoopmatM128N64 => ("f16w_coopmat_m128n64", 128, 64, 32,
        "matmul_f16w_coopmat_m128n64.spv", false,
        ["f16w_coopmat_m128n64", "coopmat_m128n64"]),
    F16wA16CoopmatM128N64 => ("f16w_a16_coopmat_m128n64", 128, 64, 32,
        "matmul_f16w_a16_coopmat_m128n64.spv", false,
        ["f16w_a16_coopmat_m128n64", "a16_coopmat_m128n64"]),
    // 64x64 in column-major (M-fastest) CTA order: consecutive CTAs
    // share a B column block, so a wide-N grid streams B from DRAM
    // ~once instead of once per row block.  Auto-routed by the
    // SM-aware rule when it cuts re-read traffic (routing.rs); on the
    // RTX 3070 it lifts a16 gate/up 512x5632x2048 from 29.1 to 35.2
    // TF/s.  The 128x64 / 128x128 M-fast twins, BK=64 (128x64, 64x64),
    // 128x64 on 256 threads, and 64x128 with 64x32 slices all lost on
    // both GPUs and were removed (benchmarks/campaign/coopmat-sm.md).
    F16wCoopmatM64Mfast => ("f16w_coopmat_m64n64_mfast", 64, 64, 32,
        "matmul_f16w_coopmat_m64n64_mfast.spv", false,
        ["f16w_coopmat_m64n64_mfast", "coopmat_m64_mfast"]),
    F16wA16CoopmatM64Mfast => ("f16w_a16_coopmat_m64n64_mfast", 64, 64, 32,
        "matmul_f16w_a16_coopmat_m64n64_mfast.spv", false,
        ["f16w_a16_coopmat_m64n64_mfast", "a16_coopmat_m64_mfast"]),
    // f16-ACCUMULATOR twins of the eight coopmat1 kernels above
    // (`_acc16`: HMMA.*.F16 end to end; `_acc16p`: f16 partials over
    // 128 K promoted into f32 accumulators).  A different numerics
    // class: never picked by the heuristic or the tuner, only through
    // the executor's `coopmat_acc` policy (ExecutorConfig /
    // ML_COOPMAT_ACC) or an explicit ML_KERNEL pin.  Slots are empty
    // unless `VulkanContext::coopmat_f16_acc_enabled`; their pipelines
    // are built on first use (no eager fallbacks).
    F16wCoopmatAcc16 => ("f16w_coopmat_acc16", 128, 128, 32,
        "matmul_f16w_coopmat_acc16.spv", false, ["f16w_coopmat_acc16", "coopmat_acc16"]),
    F16wCoopmatAcc16p => ("f16w_coopmat_acc16p", 128, 128, 32,
        "matmul_f16w_coopmat_acc16p.spv", false, ["f16w_coopmat_acc16p", "coopmat_acc16p"]),
    F16wCoopmatM64Acc16 => ("f16w_coopmat_m64n64_acc16", 64, 64, 32,
        "matmul_f16w_coopmat_m64n64_acc16.spv", false, ["f16w_coopmat_m64n64_acc16", "coopmat_m64_acc16"]),
    F16wCoopmatM64Acc16p => ("f16w_coopmat_m64n64_acc16p", 64, 64, 32,
        "matmul_f16w_coopmat_m64n64_acc16p.spv", false, ["f16w_coopmat_m64n64_acc16p", "coopmat_m64_acc16p"]),
    F16wCoopmatM64N128Acc16 => ("f16w_coopmat_m64n128_acc16", 64, 128, 32,
        "matmul_f16w_coopmat_m64n128_acc16.spv", false, ["f16w_coopmat_m64n128_acc16", "coopmat_m64n128_acc16"]),
    F16wCoopmatM64N128Acc16p => ("f16w_coopmat_m64n128_acc16p", 64, 128, 32,
        "matmul_f16w_coopmat_m64n128_acc16p.spv", false, ["f16w_coopmat_m64n128_acc16p", "coopmat_m64n128_acc16p"]),
    F16wCoopmatM128N64Acc16 => ("f16w_coopmat_m128n64_acc16", 128, 64, 32,
        "matmul_f16w_coopmat_m128n64_acc16.spv", false, ["f16w_coopmat_m128n64_acc16", "coopmat_m128n64_acc16"]),
    F16wCoopmatM128N64Acc16p => ("f16w_coopmat_m128n64_acc16p", 128, 64, 32,
        "matmul_f16w_coopmat_m128n64_acc16p.spv", false, ["f16w_coopmat_m128n64_acc16p", "coopmat_m128n64_acc16p"]),
    F16wA16CoopmatAcc16 => ("f16w_a16_coopmat_acc16", 128, 128, 32,
        "matmul_f16w_a16_coopmat_acc16.spv", false, ["f16w_a16_coopmat_acc16", "a16_coopmat_acc16"]),
    F16wA16CoopmatAcc16p => ("f16w_a16_coopmat_acc16p", 128, 128, 32,
        "matmul_f16w_a16_coopmat_acc16p.spv", false, ["f16w_a16_coopmat_acc16p", "a16_coopmat_acc16p"]),
    F16wA16CoopmatM64Acc16 => ("f16w_a16_coopmat_m64n64_acc16", 64, 64, 32,
        "matmul_f16w_a16_coopmat_m64n64_acc16.spv", false, ["f16w_a16_coopmat_m64n64_acc16", "a16_coopmat_m64_acc16"]),
    F16wA16CoopmatM64Acc16p => ("f16w_a16_coopmat_m64n64_acc16p", 64, 64, 32,
        "matmul_f16w_a16_coopmat_m64n64_acc16p.spv", false, ["f16w_a16_coopmat_m64n64_acc16p", "a16_coopmat_m64_acc16p"]),
    F16wA16CoopmatM64N128Acc16 => ("f16w_a16_coopmat_m64n128_acc16", 64, 128, 32,
        "matmul_f16w_a16_coopmat_m64n128_acc16.spv", false, ["f16w_a16_coopmat_m64n128_acc16", "a16_coopmat_m64n128_acc16"]),
    F16wA16CoopmatM64N128Acc16p => ("f16w_a16_coopmat_m64n128_acc16p", 64, 128, 32,
        "matmul_f16w_a16_coopmat_m64n128_acc16p.spv", false, ["f16w_a16_coopmat_m64n128_acc16p", "a16_coopmat_m64n128_acc16p"]),
    F16wA16CoopmatM128N64Acc16 => ("f16w_a16_coopmat_m128n64_acc16", 128, 64, 32,
        "matmul_f16w_a16_coopmat_m128n64_acc16.spv", false, ["f16w_a16_coopmat_m128n64_acc16", "a16_coopmat_m128n64_acc16"]),
    F16wA16CoopmatM128N64Acc16p => ("f16w_a16_coopmat_m128n64_acc16p", 128, 64, 32,
        "matmul_f16w_a16_coopmat_m128n64_acc16p.spv", false, ["f16w_a16_coopmat_m128n64_acc16p", "a16_coopmat_m128n64_acc16p"]),
    // Promote chunk-length sweep on the routed a16 prefill tile
    // (64x128): PROMOTE_KT = 2 (64 K) and 8 (256 K) beside the default
    // 4 (128 K) of `f16w_a16_coopmat_m64n128_acc16p`.  Pin-only.
    F16wA16CoopmatM64N128Pk2Acc16p => ("f16w_a16_coopmat_m64n128_pk2_acc16p", 64, 128, 32,
        "matmul_f16w_a16_coopmat_m64n128_pk2_acc16p.spv", false,
        ["f16w_a16_coopmat_m64n128_pk2_acc16p", "a16_coopmat_m64n128_pk2_acc16p"]),
    F16wA16CoopmatM64N128Pk8Acc16p => ("f16w_a16_coopmat_m64n128_pk8_acc16p", 64, 128, 32,
        "matmul_f16w_a16_coopmat_m64n128_pk8_acc16p.spv", false,
        ["f16w_a16_coopmat_m64n128_pk8_acc16p", "a16_coopmat_m64n128_pk8_acc16p"]),
    // Generator milestone M1: coopmat1 configs from ONE spec-constant
    // module per A storage (`matmul_f16w{,_a16}_coopmat_spec`, tile
    // params = spec ids 16-23: BM, BN, BK, WM, WN, THREADS, A_SKEW,
    // B_SKEW).  t1-t4 are twins of the four literal tiles (parity gate);
    // n1-n11 are new points (BK=64, 8-warp, skews, 32x64, ...).  Slots
    // stay empty unless ML_GEN=1 or ML_KERNEL pins one; pipelines are
    // built on first use.
    F16wCoopmatGT1 => ("f16w_coopmat_g_t1", 128, 128, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_t1", "coopmat_g_t1"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wCoopmatGT2 => ("f16w_coopmat_g_t2", 64, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_t2", "coopmat_g_t2"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGT3 => ("f16w_coopmat_g_t3", 64, 128, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_t3", "coopmat_g_t3"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGT4 => ("f16w_coopmat_g_t4", 128, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_t4", "coopmat_g_t4"],
        spec [(16, 128), (17, 64), (18, 32), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGN1 => ("f16w_coopmat_g_n1", 64, 64, 64, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n1", "coopmat_g_n1"],
        spec [(16, 64), (17, 64), (18, 64), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGN2 => ("f16w_coopmat_g_n2", 128, 64, 64, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n2", "coopmat_g_n2"],
        spec [(16, 128), (17, 64), (18, 64), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGN3 => ("f16w_coopmat_g_n3", 64, 128, 64, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n3", "coopmat_g_n3"],
        spec [(16, 64), (17, 128), (18, 64), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGN4 => ("f16w_coopmat_g_n4", 64, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n4", "coopmat_g_n4"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 16), (21, 256), (22, 1), (23, 1)]),
    F16wCoopmatGN5 => ("f16w_coopmat_g_n5", 128, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n5", "coopmat_g_n5"],
        spec [(16, 128), (17, 64), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wCoopmatGN6 => ("f16w_coopmat_g_n6", 64, 128, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n6", "coopmat_g_n6"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wCoopmatGN7 => ("f16w_coopmat_g_n7", 128, 128, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n7", "coopmat_g_n7"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wCoopmatGN8 => ("f16w_coopmat_g_n8", 128, 128, 64, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n8", "coopmat_g_n8"],
        spec [(16, 128), (17, 128), (18, 64), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wCoopmatGN9 => ("f16w_coopmat_g_n9", 64, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n9", "coopmat_g_n9"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 0), (23, 0)]),
    F16wCoopmatGN10 => ("f16w_coopmat_g_n10", 64, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n10", "coopmat_g_n10"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 2), (23, 2)]),
    F16wCoopmatGN11 => ("f16w_coopmat_g_n11", 32, 64, 32, "matmul_f16w_coopmat_spec.spv", false,
        ["f16w_coopmat_g_n11", "coopmat_g_n11"],
        spec [(16, 32), (17, 64), (18, 32), (19, 32), (20, 32), (21, 64), (22, 1), (23, 1)]),
    F16wA16CoopmatGT1 => ("f16w_a16_coopmat_g_t1", 128, 128, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_t1", "a16_coopmat_g_t1"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGT2 => ("f16w_a16_coopmat_g_t2", 64, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_t2", "a16_coopmat_g_t2"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT3 => ("f16w_a16_coopmat_g_t3", 64, 128, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_t3", "a16_coopmat_g_t3"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT4 => ("f16w_a16_coopmat_g_t4", 128, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_t4", "a16_coopmat_g_t4"],
        spec [(16, 128), (17, 64), (18, 32), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN1 => ("f16w_a16_coopmat_g_n1", 64, 64, 64, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n1", "a16_coopmat_g_n1"],
        spec [(16, 64), (17, 64), (18, 64), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN2 => ("f16w_a16_coopmat_g_n2", 128, 64, 64, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n2", "a16_coopmat_g_n2"],
        spec [(16, 128), (17, 64), (18, 64), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN3 => ("f16w_a16_coopmat_g_n3", 64, 128, 64, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n3", "a16_coopmat_g_n3"],
        spec [(16, 64), (17, 128), (18, 64), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN4 => ("f16w_a16_coopmat_g_n4", 64, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n4", "a16_coopmat_g_n4"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 16), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN5 => ("f16w_a16_coopmat_g_n5", 128, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n5", "a16_coopmat_g_n5"],
        spec [(16, 128), (17, 64), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN6 => ("f16w_a16_coopmat_g_n6", 64, 128, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n6", "a16_coopmat_g_n6"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN7 => ("f16w_a16_coopmat_g_n7", 128, 128, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n7", "a16_coopmat_g_n7"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN8 => ("f16w_a16_coopmat_g_n8", 128, 128, 64, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n8", "a16_coopmat_g_n8"],
        spec [(16, 128), (17, 128), (18, 64), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN9 => ("f16w_a16_coopmat_g_n9", 64, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n9", "a16_coopmat_g_n9"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 0), (23, 0)]),
    F16wA16CoopmatGN10 => ("f16w_a16_coopmat_g_n10", 64, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n10", "a16_coopmat_g_n10"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 2), (23, 2)]),
    F16wA16CoopmatGN11 => ("f16w_a16_coopmat_g_n11", 32, 64, 32, "matmul_f16w_a16_coopmat_spec.spv", false,
        ["f16w_a16_coopmat_g_n11", "a16_coopmat_g_n11"],
        spec [(16, 32), (17, 64), (18, 32), (19, 32), (20, 32), (21, 64), (22, 1), (23, 1)]),
    // Accumulator module axis: the same 15 configs on the f16-
    // accumulator a16 module (`_acc16` suffix: F16 numerics class, so
    // also gated on `coopmat_f16_acc_enabled`).
    F16wA16CoopmatGT1Acc16 => ("f16w_a16_coopmat_g_t1_acc16", 128, 128, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_t1_acc16", "a16_coopmat_g_t1_acc16"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGT2Acc16 => ("f16w_a16_coopmat_g_t2_acc16", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_t2_acc16", "a16_coopmat_g_t2_acc16"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT3Acc16 => ("f16w_a16_coopmat_g_t3_acc16", 64, 128, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_t3_acc16", "a16_coopmat_g_t3_acc16"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT4Acc16 => ("f16w_a16_coopmat_g_t4_acc16", 128, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_t4_acc16", "a16_coopmat_g_t4_acc16"],
        spec [(16, 128), (17, 64), (18, 32), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN1Acc16 => ("f16w_a16_coopmat_g_n1_acc16", 64, 64, 64, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n1_acc16", "a16_coopmat_g_n1_acc16"],
        spec [(16, 64), (17, 64), (18, 64), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN2Acc16 => ("f16w_a16_coopmat_g_n2_acc16", 128, 64, 64, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n2_acc16", "a16_coopmat_g_n2_acc16"],
        spec [(16, 128), (17, 64), (18, 64), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN3Acc16 => ("f16w_a16_coopmat_g_n3_acc16", 64, 128, 64, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n3_acc16", "a16_coopmat_g_n3_acc16"],
        spec [(16, 64), (17, 128), (18, 64), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN4Acc16 => ("f16w_a16_coopmat_g_n4_acc16", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n4_acc16", "a16_coopmat_g_n4_acc16"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 16), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN5Acc16 => ("f16w_a16_coopmat_g_n5_acc16", 128, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n5_acc16", "a16_coopmat_g_n5_acc16"],
        spec [(16, 128), (17, 64), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN6Acc16 => ("f16w_a16_coopmat_g_n6_acc16", 64, 128, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n6_acc16", "a16_coopmat_g_n6_acc16"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN7Acc16 => ("f16w_a16_coopmat_g_n7_acc16", 128, 128, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n7_acc16", "a16_coopmat_g_n7_acc16"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN8Acc16 => ("f16w_a16_coopmat_g_n8_acc16", 128, 128, 64, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n8_acc16", "a16_coopmat_g_n8_acc16"],
        spec [(16, 128), (17, 128), (18, 64), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGN9Acc16 => ("f16w_a16_coopmat_g_n9_acc16", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n9_acc16", "a16_coopmat_g_n9_acc16"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 0), (23, 0)]),
    F16wA16CoopmatGN10Acc16 => ("f16w_a16_coopmat_g_n10_acc16", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n10_acc16", "a16_coopmat_g_n10_acc16"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 2), (23, 2)]),
    F16wA16CoopmatGN11Acc16 => ("f16w_a16_coopmat_g_n11_acc16", 32, 64, 32, "matmul_f16w_a16_coopmat_spec_acc16.spv", false,
        ["f16w_a16_coopmat_g_n11_acc16", "a16_coopmat_g_n11_acc16"],
        spec [(16, 32), (17, 64), (18, 32), (19, 32), (20, 32), (21, 64), (22, 1), (23, 1)]),
    // K_UNROLL=8 body axis on the a16 module (`_ku8`): the S1 parity
    // rescue experiment for the twins plus the two best S2 configs.
    F16wA16CoopmatGT1Ku8 => ("f16w_a16_coopmat_g_t1_ku8", 128, 128, 32, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_t1_ku8", "a16_coopmat_g_t1_ku8"],
        spec [(16, 128), (17, 128), (18, 32), (19, 64), (20, 32), (21, 256), (22, 1), (23, 1)]),
    F16wA16CoopmatGT2Ku8 => ("f16w_a16_coopmat_g_t2_ku8", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_t2_ku8", "a16_coopmat_g_t2_ku8"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT3Ku8 => ("f16w_a16_coopmat_g_t3_ku8", 64, 128, 32, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_t3_ku8", "a16_coopmat_g_t3_ku8"],
        spec [(16, 64), (17, 128), (18, 32), (19, 32), (20, 64), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGT4Ku8 => ("f16w_a16_coopmat_g_t4_ku8", 128, 64, 32, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_t4_ku8", "a16_coopmat_g_t4_ku8"],
        spec [(16, 128), (17, 64), (18, 32), (19, 64), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN1Ku8 => ("f16w_a16_coopmat_g_n1_ku8", 64, 64, 64, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_n1_ku8", "a16_coopmat_g_n1_ku8"],
        spec [(16, 64), (17, 64), (18, 64), (19, 32), (20, 32), (21, 128), (22, 1), (23, 1)]),
    F16wA16CoopmatGN10Ku8 => ("f16w_a16_coopmat_g_n10_ku8", 64, 64, 32, "matmul_f16w_a16_coopmat_spec_ku8.spv", false,
        ["f16w_a16_coopmat_g_n10_ku8", "a16_coopmat_g_n10_ku8"],
        spec [(16, 64), (17, 64), (18, 32), (19, 32), (20, 32), (21, 128), (22, 2), (23, 2)]),
    // Tensor-core GEMM via NV_cooperative_matrix2 (workgroup-scope
    // matrices + tensor addressing) with the fused store epilogue the
    // coopmat1 kernel cannot implement (`coopMatPerElementNV` at
    // C-store time).  A f32 quantized to f16 in a load decode
    // callback, B f16 storage, f32 accumulate.  General-shape correct
    // via tensor-layout clamping (no `_aligned` suffix — that would
    // also disable `supports_epilogue`), but auto routing and tuning
    // keep it on coopmat-aligned shapes: batched tensor bases must be
    // 16-byte aligned, and measured plain-GEMM throughput trails
    // coopmat1 by 18-35% on GA104 (its value is the fused epilogue,
    // ~2x the old SIMT demote).  Registry slot is empty unless
    // `VulkanContext::coopmat2_enabled`.
    F16wCm2 => ("f16w_cm2", 128, 64, 64, "matmul_f16w_cm2.spv", false,
        ["f16w_cm2", "cm2", "cm2_gemm"]),
    // Transposed-A shared staging for the 128x128 f32 tile (k-major
    // AsT[k][m] as uvec4, XOR column swizzle): each k-step reads A with
    // TM/4 = 2 LDS.128 instead of 8 LDS.32; same 32,768 B budget as
    // `large_bda_v4`.  The auto route for f32 `Large` (routing.rs
    // `maybe_to_bda`): +1.4..+9.2% vs large_bda_v4 on the 4060 (12/12
    // shapes), +0.5..+6.7% on the 3070, never negative
    // (reports/night-kernels.md).  `large_bda_v4` stays selectable.
    LargeBdaV4t => ("large_bda_v4t", 128, 128, 32,
        "matmul_f32_large_bda_v4t.spv", false, ["large_bda_v4t"]),
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
    fn shared_memory_reflection_matches_declared_tiles() {
        let by_name = |name: &str| {
            KERNEL_SPECS
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no kernel named {name}"))
                .shared_memory_bytes()
        };
        // Hand-computed from the .glsl shared declarations:
        // As[BM][BK+1] f32 + Bs[BK][BN] f32 (or the byte-identical
        // uvec4 Bs[BK][BN/4] in the v4 family).
        assert_eq!(by_name("large_128"), 33_280); // 128*33*4 + 32*128*4
        assert_eq!(by_name("small_64"), 16_640); // 64*33*4 + 32*64*4
        assert_eq!(by_name("m128n64k64_bda_v4"), 49_664); // 128*65*4 + 64*64*4
        assert_eq!(by_name("m64n128k64"), 49_408); // 64*65*4 + 64*128*4
        assert_eq!(by_name("v2_128x128_bk8"), 17_408); // double-buffered
        assert_eq!(by_name("f16w_coopmat_aligned"), 18_944); // 128*5*16 + 32*17*16
        assert_eq!(by_name("f16w_coopmat_m64n64"), 9_728); // 64*5*16 + 32*9*16
        assert_eq!(by_name("f16w_coopmat_m64n128"), 13_824); // 64*5*16 + 32*17*16
        assert_eq!(by_name("f16w_coopmat_m128n64"), 14_848); // 128*5*16 + 32*9*16
        // v4t: AsT[BK][BM/4] uvec4 (no pad) + Bs[BK][BN/4] uvec4.
        assert_eq!(by_name("large_bda_v4t"), 32_768); // 32*32*16 + 32*32*16
        // The f16-accumulator twins declare the same staging tiles.
        for (base, bytes) in [
            ("coopmat", 18_944),
            ("coopmat_m64n64", 9_728),
            ("coopmat_m64n128", 13_824),
            ("coopmat_m128n64", 14_848),
        ] {
            for fam in ["f16w_", "f16w_a16_"] {
                for acc in ["_acc16", "_acc16p"] {
                    assert_eq!(
                        by_name(&format!("{fam}{base}{acc}")),
                        bytes,
                        "{fam}{base}{acc}"
                    );
                }
            }
        }
        assert_eq!(by_name("outer_bda"), 0); // register-only
    }

    #[test]
    fn shared_memory_overage_is_exactly_the_deep_k_tiles() {
        // Kernels above the 49,152 B limit that every desktop driver
        // reports for maxComputeSharedMemorySize.  These are gated off
        // at pipeline build wherever the driver enforces the limit
        // (see MatmulPipeline::new / VulkanContext::workgroup_shared_budget);
        // growing this set means another kernel became unroutable on
        // 48 KiB devices, so it must be a deliberate change.
        let over: Vec<&str> = KERNEL_SPECS
            .iter()
            .filter(|s| s.shared_memory_bytes() > 49_152)
            .map(|s| s.name)
            .collect();
        assert_eq!(
            over,
            [
                "m128n64k64",
                "m64n128k64",
                "m128n64k64_bda",
                "m128n64k64_bda_v4",
                "m128n64k64_bda_v4_tm8_tn8",
                "m128n64k64_bda_v4_tm16_tn4",
                "m128n64k64_bda_v4_aligned",
                "f16w_m128n64k64_bda_v4",
            ]
        );
        // The NVIDIA-proprietary allowance in workgroup_shared_budget
        // is capped at the largest shipped declaration; a new, bigger
        // tile must not silently ride the exception.
        let max = KERNEL_SPECS
            .iter()
            .map(|s| s.shared_memory_bytes())
            .max()
            .unwrap();
        assert_eq!(
            max,
            crate::context::VulkanContext::MAX_SHIPPED_WORKGROUP_BYTES
        );
    }

    /// Tile parameter of a generated config by spec id.
    fn spec_value(spec: &KernelSpec, id: u32) -> u32 {
        spec.spec
            .iter()
            .find(|&&(sid, _)| sid == id)
            .map(|&(_, v)| v)
            .unwrap_or_else(|| panic!("{} lacks spec id {id}", spec.name))
    }

    #[test]
    fn generated_configs_are_legal_and_consistent() {
        let generated: Vec<&KernelSpec> =
            KERNEL_SPECS.iter().filter(|s| s.is_generated()).collect();
        assert_eq!(
            generated.len(),
            51,
            "15 configs x {{a32, a16, a16-acc16}} + 6 ku8"
        );
        for spec in generated {
            let [bm, bn, bk, wm, wn, threads, skew_a, skew_b] =
                [16, 17, 18, 19, 20, 21, 22, 23].map(|id| spec_value(spec, id));
            assert!(spec.name.contains("_coopmat_g_"), "{}", spec.name);
            assert_eq!(
                [spec.tile_m, spec.tile_n, spec.tile_k],
                [bm, bn, bk],
                "{}",
                spec.name
            );
            assert!(
                wm % 16 == 0 && wn % 16 == 0 && bk % 16 == 0,
                "{}",
                spec.name
            );
            assert!(bm % wm == 0 && bn % wn == 0, "{}", spec.name);
            assert_eq!(threads, (bm / wm) * (bn / wn) * 32, "{}", spec.name);
            assert_eq!((bm * bk / 8) % threads, 0, "{} A staging", spec.name);
            assert_eq!((bk * bn / 8) % threads, 0, "{} B staging", spec.name);
            // The SPIR-V evaluation of the spec-sized shared arrays equals
            // the analytic size (R5).
            let analytic = bm * (bk / 8 + skew_a) * 16 + bk * (bn / 8 + skew_b) * 16;
            assert_eq!(spec.shared_memory_bytes(), analytic, "{}", spec.name);
        }
    }

    #[test]
    fn promote_chunk_k_reads_the_pk_tag() {
        let by = |name: &str| {
            KERNEL_SPECS
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("{name} missing"))
        };
        assert_eq!(
            by("f16w_a16_coopmat_m64n128_acc16p").promote_chunk_k(),
            Some(128)
        );
        assert_eq!(by("f16w_coopmat_acc16p").promote_chunk_k(), Some(128));
        assert_eq!(
            by("f16w_a16_coopmat_m64n128_pk2_acc16p").promote_chunk_k(),
            Some(64)
        );
        assert_eq!(
            by("f16w_a16_coopmat_m64n128_pk8_acc16p").promote_chunk_k(),
            Some(256)
        );
        assert_eq!(by("f16w_a16_coopmat_m64n128_acc16").promote_chunk_k(), None);
        assert_eq!(by("f16w_coopmat_aligned").promote_chunk_k(), None);
        // The chunk variants stage the same tiles as their base kernel.
        for name in [
            "f16w_a16_coopmat_m64n128_pk2_acc16p",
            "f16w_a16_coopmat_m64n128_pk8_acc16p",
        ] {
            assert_eq!(by(name).shared_memory_bytes(), 13_824, "{name}");
            assert!(by(name).a_f16(), "{name}");
        }
    }

    #[test]
    fn generated_twins_match_the_literal_tiles_shared_bytes() {
        let bytes = |name: &str| {
            KERNEL_SPECS
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("no kernel named {name}"))
                .shared_memory_bytes()
        };
        for fam in ["f16w_", "f16w_a16_"] {
            for (twin, literal, expect) in [
                ("t1", "coopmat_aligned", 18_944),
                ("t2", "coopmat_m64n64", 9_728),
                ("t3", "coopmat_m64n128", 13_824),
                ("t4", "coopmat_m128n64", 14_848),
            ] {
                assert_eq!(
                    bytes(&format!("{fam}coopmat_g_{twin}")),
                    expect,
                    "{fam}{twin}"
                );
                assert_eq!(bytes(&format!("{fam}{literal}")), expect, "{fam}{literal}");
            }
        }
        // Spec defaults (no overrides) evaluate to the 128x128 tile.
        let module = KERNEL_SPECS
            .iter()
            .find(|s| s.name == "f16w_a16_coopmat_g_t1")
            .unwrap()
            .spv;
        assert_eq!(workgroup_shared_bytes(module, &[]), 18_944);
    }

    #[test]
    fn from_index_inverts_index() {
        for (i, spec) in KERNEL_SPECS.iter().enumerate() {
            let selection = KernelSelection::from_index(i).expect("in range");
            assert_eq!(selection.index(), Some(i), "{}", spec.name);
        }
        assert_eq!(KernelSelection::from_index(KERNEL_SPECS.len()), None);
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
