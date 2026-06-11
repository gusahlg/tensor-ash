use anyhow::{Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::ResolvedMatmul;
use crate::pipeline::{KernelVariant, MatmulPipeline};
use crate::tensor::Tensor;

use super::MatmulCall;
use super::splitk::SplitKKernel;

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
        let (a_ptr, b_ptr, c_ptr) = if ctx.buffer_device_address_enabled {
            (
                ctx.buffer_device_address(call.a.raw_buffer()),
                ctx.buffer_device_address(call.b.raw_buffer()),
                ctx.buffer_device_address(call.c.raw_buffer()),
            )
        } else {
            (0, 0, 0)
        };
        let pc = dims.push_constants(call.alpha, call.accumulate, a_ptr, b_ptr, c_ptr);
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

/// Record a single split-K matmul into `cb`.  Zero-fills `C` first
/// (`vkCmdFillBuffer` with 0) so the atomic-add reduction starts from
/// a clean slate, then issues a buffer barrier and dispatches the
/// split-K compute pipeline.
///
/// The push constant block is the standard `MatmulPushConstants`; we
/// pack `num_k_splits` into the upper 16 bits of `flags`.  The lower
/// 16 bits stay reserved for the existing flag set (currently only the
/// `accumulate` bit, which split-K rejects in the executor anyway).
#[allow(clippy::too_many_arguments)]
pub(super) fn record_matmul_split_k_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    kernel: &SplitKKernel,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    call: &MatmulCall<'_>,
    resolved: &ResolvedMatmul,
    num_k_splits: u32,
    gx: u32,
    gy: u32,
    gz: u32,
) -> Result<()> {
    if descriptor_sets.is_empty() {
        bail!("record_matmul_split_k_commands: no descriptor set");
    }
    let set = descriptor_sets[0];

    let (a_ptr, b_ptr, c_ptr) = if ctx.buffer_device_address_enabled {
        (
            ctx.buffer_device_address(call.a.raw_buffer()),
            ctx.buffer_device_address(call.b.raw_buffer()),
            ctx.buffer_device_address(call.c.raw_buffer()),
        )
    } else {
        bail!("split-K kernel requires bufferDeviceAddress");
    };

    let pc = resolved.split_k_push_constants(call.alpha, a_ptr, b_ptr, c_ptr, num_k_splits);

    unsafe {
        // Step 1: zero-fill the entire C buffer.  The atomic-add
        // accumulator inside the kernel expects every cell to start at
        // 0.0 (binary all-zeros).  vkCmdFillBuffer writes the u32 0,
        // which is bit-identical to f32 0.0.
        ctx.device
            .cmd_fill_buffer(cb, call.c.raw_buffer(), 0, vk::WHOLE_SIZE, 0);

        // Step 2: barrier so the fill completes before the shader
        // reads/writes C.  TRANSFER -> COMPUTE_SHADER, and on the
        // memory side TRANSFER_WRITE -> SHADER_READ|SHADER_WRITE
        // (the kernel atomicCompSwap reads and writes the same cell).
        let buf_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(call.c.raw_buffer())
            .offset(0)
            .size(vk::WHOLE_SIZE);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&buf_barrier),
            &[],
        );

        // Step 3: bind pipeline + descriptor set + push constants and
        // dispatch (N/BN, M/BM, batch*num_k_splits).
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
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

    Ok(())
}
