mod descriptors;
mod graph;

use anyhow::{Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::{KernelVariant, MatmulPipeline};

use super::MatmulCall;
use super::splitk::SplitKKernel;

pub(super) use descriptors::{update_matmul_descriptor_sets, update_split_k_descriptor_set};
pub(super) use graph::{GraphSplitK2, record_matmul_graph_commands};

pub(super) fn record_matmul_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    resolved: &[ResolvedMatmul],
) -> Result<()> {
    let mut bound_pipeline = vk::Pipeline::null();
    for ((set, op), dims) in descriptor_sets
        .iter()
        .copied()
        .zip(ops.iter())
        .zip(resolved.iter())
    {
        let kernel = pipeline.select_kernel(dims.batch, dims.m, dims.n, dims.k);
        record_one_matmul(
            ctx,
            pipeline,
            cb,
            set,
            op,
            dims,
            kernel,
            &mut bound_pipeline,
        )?;
    }

    Ok(())
}

/// Record one dispatch with an explicitly-chosen kernel.  The regular
/// paths resolve the kernel through `MatmulPipeline::select_kernel`;
/// the auto-tuner forces each candidate in turn.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_one_matmul(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    set: vk::DescriptorSet,
    op: &MatmulOp<'_>,
    dims: &ResolvedMatmul,
    kernel: &crate::pipeline::MatmulKernel,
    bound_pipeline: &mut vk::Pipeline,
) -> Result<()> {
    let call = &op.call;
    let max_groups = ctx.device_properties.limits.max_compute_work_group_count;
    let (a_ptr, b_ptr, c_ptr) = if ctx.buffer_device_address_enabled {
        (
            ctx.buffer_device_address(call.a.raw_buffer()),
            ctx.buffer_device_address(call.b.raw_buffer()),
            ctx.buffer_device_address(call.c.raw_buffer()),
        )
    } else {
        (0, 0, 0)
    };
    let mut pc = dims.push_constants(call.alpha, call.accumulate, a_ptr, b_ptr, c_ptr);

    // Pick the specialization variant whose constants match this call.
    // `interior_only` is safe whenever M and N are tile-aligned to the
    // selected kernel's tile size — the shader then skips all
    // m_full/n_full bounds checks.
    let variant = KernelVariant {
        accumulate: call.accumulate,
        alpha_is_one: call.alpha == 1.0,
        interior_only: dims.m.is_multiple_of(kernel.tile_m) && dims.n.is_multiple_of(kernel.tile_n),
        k_multiple: dims.k.is_multiple_of(kernel.tile_k),
    };

    let epilogue = &op.epilogue;
    let variant_pipeline = if epilogue.is_none() {
        kernel.pipeline_for(variant)
    } else {
        if !ctx.buffer_device_address_enabled {
            bail!("fused epilogues require bufferDeviceAddress, which this device lacks");
        }
        if !kernel.supports_epilogue() {
            bail!(
                "kernel '{}' does not support fused epilogues (only the BDA / BDA_V4 \
                 kernel families do; unset ML_KERNEL or pick a *_bda / *_bda_v4 variant)",
                kernel.name
            );
        }
        if let Some(bias) = epilogue.bias {
            pc.bias_ptr = ctx.buffer_device_address(bias.raw_buffer());
            pc.bias_batch_stride = if bias.len() == dims.n as u64 {
                0
            } else {
                dims.n
            };
        }
        if let Some(d) = epilogue.d_tensor() {
            pc.d_ptr = ctx.buffer_device_address(d.raw_buffer());
        }
        pc.beta = epilogue.beta();
        pipeline.pipeline_for_epilogue(kernel, variant, epilogue.key())?
    };

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
        if *bound_pipeline != variant_pipeline {
            ctx.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, variant_pipeline);
            *bound_pipeline = variant_pipeline;
        }
        // BDA kernels carry their own push-constant-only pipeline
        // layout and never read SSBO bindings 0/1/2 — skip the
        // descriptor-set bind entirely.  Descriptor kernels go
        // through the matmul pipeline's descriptor-based layout
        // (which is what `kernel.pipeline_layout` resolves to for
        // them).
        if kernel.uses_descriptors {
            ctx.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
        }
        ctx.device.cmd_push_constants(
            cb,
            kernel.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
        );
        ctx.device.cmd_dispatch(cb, gx, gy, gz);
    }

    Ok(())
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
