//! Dependency-hazard tracking for matmul graphs.

use anyhow::Result;
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::MatmulPipeline;
use crate::tensor::Tensor;

use super::super::OpPlan;
use super::super::splitk2::{SplitK2Dispatch, SplitK2Pipeline, record_split_k2_commands};
use super::record_one_matmul;

/// Split-K2 routing context for a graph recording: which ops route
/// through the two-stage path, and where each op's scratch region
/// lives inside the slot's scratch buffer.
pub(in crate::executor) struct GraphSplitK2<'a> {
    pub(in crate::executor) pipeline: &'a SplitK2Pipeline,
    /// Device address of the slot scratch buffer (16-byte aligned).
    pub(in crate::executor) scratch_addr: u64,
    /// Per-op `(plan, byte offset into scratch)`; `None` = regular DP
    /// dispatch.
    pub(in crate::executor) plans: &'a [Option<(SplitK2Dispatch, u64)>],
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
#[allow(clippy::too_many_arguments)]
pub(in crate::executor) fn record_matmul_graph_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    resolved: &[ResolvedMatmul],
    plans: &[OpPlan],
    splitk2: Option<&GraphSplitK2<'_>>,
) -> Result<()> {
    let mut bound_pipeline = vk::Pipeline::null();
    let mut written: Vec<vk::Buffer> = Vec::new();
    let mut read: Vec<vk::Buffer> = Vec::new();

    for (i, (((set, op), dims), op_plan)) in descriptor_sets
        .iter()
        .copied()
        .zip(ops.iter())
        .zip(resolved.iter())
        .zip(plans.iter())
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
                call.a.device_address(),
                call.b.device_address(),
                call.c.device_address(),
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

        record_one_matmul(
            ctx,
            pipeline,
            cb,
            set,
            op,
            dims,
            pipeline.kernel_at(op_plan.kernel),
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
