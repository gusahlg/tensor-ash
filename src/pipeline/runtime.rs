use ash::vk;

use super::abi::KernelVariant;

/// A compiled kernel and the Vulkan objects it owns.
pub struct MatmulKernel {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub shader_module: vk::ShaderModule,
    /// A correctness-safe pipeline for each `KernelVariant`, indexed by
    /// `KernelVariant::index()`. Entries with the same accumulate/alpha flags
    /// initially alias one bounds-checked fallback; the dispatcher lazily
    /// builds exact interior/K specializations.
    pub variants: [vk::Pipeline; KernelVariant::COUNT],
    /// The pipeline layout the kernel's pipelines were built against,
    /// and the layout the dispatcher must pass to
    /// `vkCmdPushConstants` / `vkCmdBindDescriptorSets`. Descriptor
    /// kernels point at the matmul pipeline's descriptor-set-based
    /// layout; BDA kernels point at the push-constant-only BDA layout.
    pub pipeline_layout: vk::PipelineLayout,
    /// `true` if the shader reads A/B/C from SSBO bindings 0/1/2;
    /// `false` if it dereferences them through `buffer_reference`
    /// pointers in the push constants. Mirrors `KernelSpec::uses_descriptors`.
    pub uses_descriptors: bool,
}

impl MatmulKernel {
    #[inline]
    pub fn pipeline_for(&self, variant: KernelVariant) -> vk::Pipeline {
        self.variants[variant.index()]
    }

    /// Whether this kernel's shader body implements the fused-epilogue
    /// specialization constants (IDs 4..6). True for the BDA and
    /// BDA_V4 bodies; false for descriptor-bound kernels, the
    /// source-stripped `*_aligned` bodies, and every coopmat1 tile
    /// (those have no epilogue macros — the `_aligned` suffix is not
    /// on the 64x64 wave-fill names).
    #[inline]
    pub fn supports_epilogue(&self) -> bool {
        !self.uses_descriptors && !self.name.ends_with("_aligned") && !self.name.contains("coopmat")
    }

    /// Whether the shader reads B as f16 storage (the `f16w_*` family).
    /// A dispatch must pair such a kernel with an f16 B tensor and any
    /// other kernel with an f32 one; `record_one_matmul` enforces it.
    #[inline]
    pub fn weights_f16(&self) -> bool {
        self.name.starts_with("f16w_")
    }

    /// Whether the shader reads A and writes C as f16 storage (the
    /// `*_a16_*` f16-activations family).  Same dispatch-agreement
    /// contract as [`Self::weights_f16`].
    #[inline]
    pub fn a_f16(&self) -> bool {
        self.name.contains("_a16_")
    }

    /// Whether this kernel's shader body implements the fused normed-A
    /// GEMV (`flags` bit 1: RMSNorm of the A row folded before the
    /// product).  True for the row-GEMV family only.
    #[inline]
    pub fn supports_normed_a(&self) -> bool {
        self.name.contains("row_bda")
    }

    /// Whether this kernel's shader body implements the fused store
    /// epilogue (spec constants 7/8: RoPE-rotate and/or KV-cache
    /// scatter at C-store time).  f16-weights row-GEMV family only.
    #[inline]
    pub fn supports_store(&self) -> bool {
        self.weights_f16() && self.name.contains("row_bda")
    }
}
