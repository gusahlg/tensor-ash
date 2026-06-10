use anyhow::{Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::ResolvedMatmul;
use crate::pipeline::{KernelVariant, MatmulPipeline};
use crate::tensor::Tensor;

use super::MatmulCall;

/// Point the pre-allocated descriptor sets at this submission's A/B/C
/// tensors.  `sets.len()` must equal `calls.len()`; the caller is
/// responsible for guaranteeing that the GPU has fenced out of the
/// previous use of these sets (we wait on the fence at the end of every
/// submit, so the next submit on the same slot is safe).
pub(super) fn update_matmul_descriptor_sets(
    ctx: &VulkanContext,
    sets: &[vk::DescriptorSet],
    calls: &[MatmulCall<'_>],
) {
    debug_assert_eq!(sets.len(), calls.len());

    if let ([set], [call]) = (sets, calls) {
        update_one_matmul_descriptor_set(ctx, *set, call);
        return;
    }

    // Build the buffer-info array up front so the &[WriteDescriptorSet]
    // we hand to Vulkan keeps stable references into it.
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(calls.len() * 3);
    for call in calls {
        buffer_infos.push(tensor_descriptor(call.a));
        buffer_infos.push(tensor_descriptor(call.b));
        buffer_infos.push(tensor_descriptor(call.c));
    }

    let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(calls.len() * 3);
    for (i, set) in sets.iter().copied().enumerate() {
        let base = i * 3;
        for binding in 0..3u32 {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[base + binding as usize])),
            );
        }
    }

    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }
}

fn update_one_matmul_descriptor_set(
    ctx: &VulkanContext,
    set: vk::DescriptorSet,
    call: &MatmulCall<'_>,
) {
    let buffer_infos = [
        tensor_descriptor(call.a),
        tensor_descriptor(call.b),
        tensor_descriptor(call.c),
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[2])),
    ];

    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }
}

pub(super) fn record_matmul_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    calls: &[MatmulCall<'_>],
    resolved: &[ResolvedMatmul],
) -> Result<()> {
    let max_groups = ctx.device_properties.limits.max_compute_work_group_count;
    let mut bound_pipeline = vk::Pipeline::null();
    for ((set, call), dims) in descriptor_sets
        .iter()
        .copied()
        .zip(calls.iter())
        .zip(resolved.iter())
    {
        let pc = dims.push_constants(call.alpha, call.accumulate);
        let kernel = pipeline.select_kernel(dims.batch, dims.m, dims.n, dims.k);

        // Pick the specialization variant whose constants match this call.
        // `interior_only` is safe whenever M and N are tile-aligned to the
        // selected kernel's tile size — the shader then skips all
        // m_full/n_full bounds checks.
        let variant = KernelVariant {
            accumulate: call.accumulate,
            alpha_is_one: call.alpha == 1.0,
            interior_only: dims.m.is_multiple_of(kernel.tile_m)
                && dims.n.is_multiple_of(kernel.tile_n),
            k_multiple: dims.k.is_multiple_of(kernel.tile_k),
        };
        let variant_pipeline = kernel.pipeline_for(variant);

        let gx = dims.n.div_ceil(kernel.tile_n);
        let gy = dims.m.div_ceil(kernel.tile_m);
        let gz = dims.batch;
        if gx > max_groups[0] || gy > max_groups[1] || gz > max_groups[2] {
            bail!(
                "matmul dispatch ({gx}, {gy}, {gz}) for M={}, N={}, K={}, batch={} \
                 exceeds device maxComputeWorkGroupCount ({}, {}, {})",
                dims.m,
                dims.n,
                dims.k,
                dims.batch,
                max_groups[0],
                max_groups[1],
                max_groups[2],
            );
        }

        unsafe {
            if bound_pipeline != variant_pipeline {
                ctx.device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, variant_pipeline);
                bound_pipeline = variant_pipeline;
            }
            ctx.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            ctx.device.cmd_push_constants(
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            ctx.device.cmd_dispatch(cb, gx, gy, gz);
        }
    }
    Ok(())
}

fn tensor_descriptor(tensor: &Tensor) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(tensor.raw_buffer())
        .offset(0)
        .range(tensor.size_bytes())
}
