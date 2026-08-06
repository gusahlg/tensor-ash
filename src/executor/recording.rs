use anyhow::{Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::{KernelVariant, MatmulPipeline};
use crate::tensor::Tensor;

use super::MatmulCall;
use super::splitk::SplitKKernel;
use super::splitk2::{SplitK2Dispatch, SplitK2Pipeline, record_split_k2_commands};

/// Split-K2 routing context for a graph recording: which ops route
/// through the two-stage path, and where each op's scratch region
/// lives inside the slot's scratch buffer.
pub(super) struct GraphSplitK2<'a> {
    pub(super) pipeline: &'a SplitK2Pipeline,
    /// Device address of the slot scratch buffer (16-byte aligned).
    pub(super) scratch_addr: u64,
    /// Per-op `(plan, byte offset into scratch)`; `None` = regular DP
    /// dispatch.
    pub(super) plans: &'a [Option<(SplitK2Dispatch, u64)>],
}

/// Point the pre-allocated descriptor sets at this submission's A/B/C
/// tensors.  `sets.len()` must equal `calls.len()`; the caller is
/// responsible for guaranteeing that the GPU has fenced out of the
/// previous use of these sets (we wait on the fence at the end of every
/// submit, so the next submit on the same slot is safe).
///
/// BDA kernels read A/B/C through `buffer_reference` pointers in the
/// push constants and never touch SSBO bindings 0/1/2.  For any call
/// whose resolved kernel is a BDA kernel we skip the descriptor write
/// entirely — the shader cannot observe the descriptor and the write
/// is pure CPU-side overhead.  A pure-BDA submission therefore issues
/// zero `vkUpdateDescriptorSets` work.
pub(super) fn update_matmul_descriptor_sets(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    resolved: &[ResolvedMatmul],
) {
    debug_assert_eq!(sets.len(), ops.len());
    debug_assert_eq!(sets.len(), resolved.len());

    // Build buffer-info + writes only for calls that actually need a
    // descriptor write.  Keep buffer_infos stable so the
    // &[WriteDescriptorSet] we hand to Vulkan keeps stable references.
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(ops.len() * 3);
    let mut needs_write: Vec<bool> = Vec::with_capacity(ops.len());
    for (op, dims) in ops.iter().zip(resolved.iter()) {
        let call = &op.call;
        let kernel = pipeline.select_kernel(dims.batch, dims.m, dims.n, dims.k);
        let need = kernel.uses_descriptors;
        needs_write.push(need);
        if need {
            buffer_infos.push(tensor_descriptor(call.a));
            buffer_infos.push(tensor_descriptor(call.b));
            buffer_infos.push(tensor_descriptor(call.c));
        }
    }

    if buffer_infos.is_empty() {
        // Pure-BDA submission: nothing to write.
        return;
    }

    let mut writes: Vec<vk::WriteDescriptorSet> =
        Vec::with_capacity(needs_write.iter().filter(|w| **w).count() * 3);
    let mut base = 0usize;
    for (i, set) in sets.iter().copied().enumerate() {
        if !needs_write[i] {
            continue;
        }
        for binding in 0..3u32 {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[base + binding as usize])),
            );
        }
        base += 3;
    }

    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }
}

/// Point a single descriptor set at `call`'s A/B/C tensors.  Used by
/// the split-K path, whose kernels still bind the descriptor set
/// (carried by the matmul pipeline layout that SplitKKernel was built
/// against) even though the shader itself addresses A/B/C through
/// BDA pointers in the push constants.  Kept as a dedicated helper so
/// the matmul-BDA fast path doesn't have to reason about split-K's
/// descriptor needs.
pub(super) fn update_split_k_descriptor_set(
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

/// Record dependent matmuls into one command buffer, inserting
/// compute→compute barriers only where a buffer hazard exists.
///
/// Hazards tracked per raw `vk::Buffer` handle:
///   * RAW — a call reads (A, B, or C-with-accumulate) a buffer some
///     earlier call in the batch wrote;
///   * WAW / WAR — a call writes a C buffer an earlier call wrote or
///     read.
///
/// A single global memory barrier flushes *all* prior writes, so after
/// emitting one the tracking sets reset; back-to-back independent calls
/// still record with zero barriers, exactly like `record_matmul_commands`.
pub(super) fn record_matmul_graph_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    resolved: &[ResolvedMatmul],
    splitk2: Option<&GraphSplitK2<'_>>,
) -> Result<()> {
    let mut bound_pipeline = vk::Pipeline::null();
    let mut written: Vec<vk::Buffer> = Vec::new();
    let mut read: Vec<vk::Buffer> = Vec::new();

    for (i, ((set, op), dims)) in descriptor_sets
        .iter()
        .copied()
        .zip(ops.iter())
        .zip(resolved.iter())
        .enumerate()
    {
        let call = &op.call;
        let a = call.a.raw_buffer();
        let b = call.b.raw_buffer();
        let c = call.c.raw_buffer();
        let bias = op.epilogue.bias.map(Tensor::raw_buffer);
        let d = op.epilogue.d_tensor().map(Tensor::raw_buffer);

        // Tuned split-K2 routing: record stage1 + internal barrier +
        // reduce inline.  The internal barrier is a global flush, so
        // only stage-1's own reads (A, B) need a pre-barrier; C
        // hazards are covered by the internal barrier, and every
        // write recorded before this op is visible after it.
        if let Some(g) = splitk2
            && let Some((plan, offset)) = g.plans.get(i).copied().flatten()
        {
            if written.contains(&a) || written.contains(&b) {
                record_compute_to_compute_barrier(ctx, cb);
            }
            record_split_k2_commands(
                ctx,
                g.pipeline,
                cb,
                call.alpha,
                dims,
                &plan,
                ctx.buffer_device_address(a),
                ctx.buffer_device_address(b),
                ctx.buffer_device_address(c),
                g.scratch_addr + offset,
            );
            written.clear();
            read.clear();
            read.push(a);
            read.push(b);
            written.push(c);
            // record_split_k2_commands bound its own pipelines.
            bound_pipeline = vk::Pipeline::null();
            continue;
        }

        let reads_written = written.contains(&a)
            || written.contains(&b)
            || (call.accumulate && written.contains(&c))
            || bias.is_some_and(|buf| written.contains(&buf))
            || d.is_some_and(|buf| written.contains(&buf));
        let write_hazard = written.contains(&c) || read.contains(&c);

        if reads_written || write_hazard {
            record_compute_to_compute_barrier(ctx, cb);
            written.clear();
            read.clear();
        }

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

        read.push(a);
        read.push(b);
        read.extend(bias);
        read.extend(d);
        written.push(c);
    }

    Ok(())
}

/// Full compute→compute execution + memory barrier.  One global
/// `vk::MemoryBarrier` (rather than per-buffer barriers) — on every
/// driver we target, buffer-granular compute barriers offer no extra
/// overlap for back-to-back dispatches, and the single global barrier
/// keeps recording cost flat.
fn record_compute_to_compute_barrier(ctx: &VulkanContext, cb: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
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
