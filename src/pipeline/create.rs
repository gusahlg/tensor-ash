use std::sync::Arc;

use anyhow::{Result, anyhow};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

use super::types::{EpilogueKey, KernelVariant, MatmulKernel};

/// Build one pipeline for a (kernel, base-variant, epilogue) triple.
/// Used by the lazy epilogue-pipeline cache in `MatmulPipeline` — the
/// eager `create_kernel` path only builds the zero-epilogue variants.
pub(super) fn create_epilogue_pipeline(
    ctx: &Arc<VulkanContext>,
    kernel: &MatmulKernel,
    variant: KernelVariant,
    epilogue: EpilogueKey,
) -> Result<vk::Pipeline> {
    unsafe {
        let entry = std::ffi::CString::new("main").unwrap();
        let spec_entries: Vec<vk::SpecializationMapEntry> = (0u32..7)
            .map(|i| {
                vk::SpecializationMapEntry::default()
                    .constant_id(i)
                    .offset(i * 4)
                    .size(4)
            })
            .collect();
        let spec_data: [u32; 7] = [
            variant.accumulate as u32,
            variant.alpha_is_one as u32,
            variant.interior_only as u32,
            variant.k_multiple as u32,
            epilogue.bias as u32,
            epilogue.activation,
            epilogue.binary,
        ];
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(bytemuck::cast_slice(&spec_data));
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(kernel.shader_module)
            .name(&entry)
            .specialization_info(&spec_info);
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(kernel.pipeline_layout);

        match ctx.device.create_compute_pipelines(
            ctx.pipeline_cache,
            std::slice::from_ref(&create_info),
            None,
        ) {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((partial, err)) => {
                for p in partial {
                    if p != vk::Pipeline::null() {
                        ctx.device.destroy_pipeline(p, None);
                    }
                }
                Err(anyhow!(
                    "create_compute_pipelines (epilogue {epilogue:?} on {}): {err}",
                    kernel.name
                ))
            }
        }
    }
}

/// Destroy every Vulkan object owned by `kernel`.  Safe to call on a
/// partially-built kernel: null handles are skipped.
pub(super) unsafe fn destroy_kernel(ctx: &Arc<VulkanContext>, kernel: MatmulKernel) {
    unsafe {
        for p in kernel.variants {
            if p != vk::Pipeline::null() {
                ctx.device.destroy_pipeline(p, None);
            }
        }
        if kernel.shader_module != vk::ShaderModule::null() {
            ctx.device.destroy_shader_module(kernel.shader_module, None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn create_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    name: &'static str,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    spv: &[u8],
    uses_descriptors: bool,
) -> Result<MatmulKernel> {
    unsafe {
        assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
        let shader_guard =
            scopeguard::guard(shader_module, |m| ctx.device.destroy_shader_module(m, None));
        let entry = std::ffi::CString::new("main").unwrap();

        // One pipeline per specialization tuple, batched so the driver
        // can amortize and de-duplicate ISA compilation.
        let spec_entries = [
            vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(1)
                .offset(4)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(2)
                .offset(8)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(3)
                .offset(12)
                .size(4),
        ];

        let spec_data: Vec<[u32; 4]> = (0..KernelVariant::COUNT)
            .map(|i| {
                let v = KernelVariant::from_index(i);
                [
                    v.accumulate as u32,
                    v.alpha_is_one as u32,
                    v.interior_only as u32,
                    v.k_multiple as u32,
                ]
            })
            .collect();

        let spec_infos: Vec<vk::SpecializationInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::SpecializationInfo::default()
                    .map_entries(&spec_entries)
                    .data(bytemuck::cast_slice(&spec_data[i]))
            })
            .collect();

        let stages: Vec<vk::PipelineShaderStageCreateInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(&entry)
                    .specialization_info(&spec_infos[i])
            })
            .collect();

        let create_infos: Vec<vk::ComputePipelineCreateInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(stages[i])
                    .layout(pipeline_layout)
            })
            .collect();

        let pipelines =
            match ctx
                .device
                .create_compute_pipelines(ctx.pipeline_cache, &create_infos, None)
            {
                Ok(pipelines) => pipelines,
                Err((partial, err)) => {
                    for p in partial {
                        if p != vk::Pipeline::null() {
                            ctx.device.destroy_pipeline(p, None);
                        }
                    }
                    return Err(anyhow!("create_compute_pipelines {name}: {err}"));
                }
            };

        let mut variants = [vk::Pipeline::null(); KernelVariant::COUNT];
        for (i, p) in pipelines.iter().enumerate() {
            variants[i] = *p;
        }

        Ok(MatmulKernel {
            name,
            tile_m,
            tile_n,
            tile_k,
            shader_module: ScopeGuard::into_inner(shader_guard),
            variants,
            pipeline_layout,
            uses_descriptors,
        })
    }
}
