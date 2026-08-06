use ash::vk;

use super::abi::KernelVariant;

/// A compiled kernel and the Vulkan objects it owns.
pub struct MatmulKernel {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub shader_module: vk::ShaderModule,
    /// One pipeline per `KernelVariant`; indexed by `KernelVariant::index()`.
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
    /// BDA_V4 bodies; false for descriptor-bound kernels and for the
    /// source-stripped `*_aligned` bodies.
    #[inline]
    pub fn supports_epilogue(&self) -> bool {
        !self.uses_descriptors && !self.name.ends_with("_aligned")
    }
}
