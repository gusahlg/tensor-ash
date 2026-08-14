use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

use super::{EpilogueKey, KernelSpec, KernelVariant, MatmulKernel};

/// Convert little-endian SPIR-V bytes to words and create the module.
pub(crate) fn create_shader_module(
    ctx: &VulkanContext,
    spv: &[u8],
    label: &str,
) -> Result<vk::ShaderModule> {
    assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        ctx.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .with_context(|| format!("create_shader_module ({label})"))
    }
}

/// `n` u32 specialization constants: id `i` at offset `i*4`, 4 bytes each.
fn spec_map_entries(n: u32) -> Vec<vk::SpecializationMapEntry> {
    (0..n)
        .map(|i| {
            vk::SpecializationMapEntry::default()
                .constant_id(i)
                .offset(i * 4)
                .size(4)
        })
        .collect()
}

/// Build one compute pipeline from an existing shader module through
/// the shared pipeline cache, specializing constant ids
/// `0..spec_data.len()` to `spec_data`.
pub(crate) fn create_specialized_pipeline(
    ctx: &VulkanContext,
    module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    spec_data: &[u32],
    label: &str,
) -> Result<vk::Pipeline> {
    unsafe {
        let entry = std::ffi::CString::new("main").unwrap();
        let spec_entries = spec_map_entries(spec_data.len() as u32);
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(bytemuck::cast_slice(spec_data));
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry)
            .specialization_info(&spec_info);
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

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
                Err(anyhow!("create_compute_pipelines ({label}): {err}"))
            }
        }
    }
}

/// SPIR-V bytes straight to a fully specialized compute pipeline.  On
/// pipeline-creation failure the freshly created module is destroyed
/// before returning; on success the caller owns both handles.
pub(crate) fn build_compute_pipeline(
    ctx: &VulkanContext,
    layout: vk::PipelineLayout,
    spec_data: &[u32],
    spv: &[u8],
    label: &str,
) -> Result<(vk::ShaderModule, vk::Pipeline)> {
    let module = create_shader_module(ctx, spv, label)?;
    let guard = scopeguard::guard(module, |m| unsafe {
        ctx.device.destroy_shader_module(m, None)
    });
    let pipeline = create_specialized_pipeline(ctx, module, layout, spec_data, label)?;
    Ok((ScopeGuard::into_inner(guard), pipeline))
}

/// Build one pipeline for a (kernel, base-variant, epilogue) triple.
/// Used by the lazy epilogue-pipeline cache in `MatmulPipeline` — the
/// eager `create_kernel` path only builds bounds-checked, zero-epilogue
/// fallbacks.
pub(super) fn create_epilogue_pipeline(
    ctx: &Arc<VulkanContext>,
    kernel: &MatmulKernel,
    variant: KernelVariant,
    epilogue: EpilogueKey,
) -> Result<vk::Pipeline> {
    let spec_count = if kernel.supports_epilogue() { 7 } else { 4 };
    let spec_data: [u32; 7] = [
        variant.accumulate as u32,
        variant.alpha_is_one as u32,
        variant.interior_only as u32,
        variant.k_multiple as u32,
        epilogue.bias as u32,
        epilogue.activation,
        epilogue.binary,
    ];
    create_specialized_pipeline(
        ctx,
        kernel.shader_module,
        kernel.pipeline_layout,
        &spec_data[..spec_count],
        &format!("epilogue {epilogue:?} on {}", kernel.name),
    )
}

/// Destroy every Vulkan object owned by `kernel`.  Safe to call on a
/// partially-built kernel: null handles are skipped.
pub(super) unsafe fn destroy_kernel(ctx: &Arc<VulkanContext>, kernel: MatmulKernel) {
    unsafe {
        for i in 0..KernelVariant::COUNT {
            let p = kernel.variants[i];
            if p != vk::Pipeline::null() && !kernel.variants[..i].contains(&p) {
                ctx.device.destroy_pipeline(p, None);
            }
        }
        if kernel.shader_module != vk::ShaderModule::null() {
            ctx.device.destroy_shader_module(kernel.shader_module, None);
        }
    }
}

pub(super) fn create_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    spec: &KernelSpec,
) -> Result<MatmulKernel> {
    let shader_module = create_shader_module(ctx, spec.spv, spec.name)?;
    unsafe {
        let shader_guard =
            scopeguard::guard(shader_module, |m| ctx.device.destroy_shader_module(m, None));
        let entry = std::ffi::CString::new("main").unwrap();

        // Only accumulate/alpha affect semantics.  Build the four fully
        // bounds-checked variants eagerly; aligned M/N/K specializations
        // are compiled on first use through `pipeline_for_epilogue`.
        let spec_entries = spec_map_entries(4);

        let spec_data: Vec<[u32; 4]> = (0..KernelVariant::FALLBACK_COUNT)
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

        let spec_infos: Vec<vk::SpecializationInfo> = (0..KernelVariant::FALLBACK_COUNT)
            .map(|i| {
                vk::SpecializationInfo::default()
                    .map_entries(&spec_entries)
                    .data(bytemuck::cast_slice(&spec_data[i]))
            })
            .collect();

        let stages: Vec<vk::PipelineShaderStageCreateInfo> = (0..KernelVariant::FALLBACK_COUNT)
            .map(|i| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(&entry)
                    .specialization_info(&spec_infos[i])
            })
            .collect();

        let create_infos: Vec<vk::ComputePipelineCreateInfo> = (0..KernelVariant::FALLBACK_COUNT)
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
                    return Err(anyhow!("create_compute_pipelines {}: {err}", spec.name));
                }
            };

        let mut variants = [vk::Pipeline::null(); KernelVariant::COUNT];
        for (i, variant) in variants.iter_mut().enumerate() {
            *variant = pipelines[KernelVariant::from_index(i).fallback().index()];
        }

        Ok(MatmulKernel {
            name: spec.name,
            tile_m: spec.tile_m,
            tile_n: spec.tile_n,
            tile_k: spec.tile_k,
            shader_module: ScopeGuard::into_inner(shader_guard),
            variants,
            pipeline_layout,
            uses_descriptors: spec.uses_descriptors,
        })
    }
}
