mod descriptors;
mod graph;

use anyhow::{Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::{KernelVariant, MatmulPipeline};

use super::OpPlan;

pub(super) use descriptors::update_matmul_descriptor_sets;
pub(super) use graph::{GraphSplitK2, record_matmul_graph_commands};

/// Full compute→compute execution + memory barrier.  One global
/// `vk::MemoryBarrier` (rather than per-buffer barriers) — on every
/// driver we target, buffer-granular compute barriers offer no extra
/// overlap for back-to-back dispatches, and the single global barrier
/// keeps recording cost flat.
pub(super) fn record_compute_to_compute_barrier(ctx: &VulkanContext, cb: vk::CommandBuffer) {
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

fn ensure_kernel_alignment(kernel_name: &str, tile: [u32; 3], shape: [u32; 3]) -> Result<()> {
    let required = if kernel_name == "v3_128x128_bk8_static" {
        // V3 double-buffers exact pairs of BK=8 strips and has no edge/tail
        // path. One pair (K=16) is the smallest safe reduction.
        [tile[0], tile[1], 2 * tile[2]]
    } else {
        tile
    };
    let [tile_m, tile_n, tile_k] = required;
    let [m, n, k] = shape;
    if (kernel_name.ends_with("_aligned") || kernel_name == "v3_128x128_bk8_static")
        && (!m.is_multiple_of(tile_m) || !n.is_multiple_of(tile_n) || !k.is_multiple_of(tile_k))
    {
        bail!(
            "kernel '{kernel_name}' requires M/N/K aligned to ({tile_m}, {tile_n}, {tile_k}), \
             got ({m}, {n}, {k})"
        );
    }
    Ok(())
}

pub(super) fn record_matmul_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    resolved: &[ResolvedMatmul],
    plans: &[OpPlan],
) -> Result<()> {
    let mut bound_pipeline = vk::Pipeline::null();
    for (((set, op), dims), plan) in descriptor_sets
        .iter()
        .copied()
        .zip(ops.iter())
        .zip(resolved.iter())
        .zip(plans.iter())
    {
        record_one_matmul(
            ctx,
            pipeline,
            cb,
            set,
            op,
            dims,
            pipeline.kernel_at(plan.kernel),
            &mut bound_pipeline,
        )?;
    }

    Ok(())
}

/// Record one dispatch with an explicitly-chosen kernel.  The regular
/// paths resolve the kernel from the submission's per-op plans; the
/// auto-tuner forces each candidate in turn.
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
    ensure_kernel_alignment(
        kernel.name,
        [kernel.tile_m, kernel.tile_n, kernel.tile_k],
        [dims.m, dims.n, dims.k],
    )?;
    // Storage-type agreement: an f16w kernel would misread f32 bits
    // and vice versa.  Auto routes always agree; this guards explicit
    // ML_KERNEL selections.
    if kernel.weights_f16() != dims.b_f16 {
        bail!(
            "kernel '{}' expects {} B storage but B is {} (pick an {} kernel or unset ML_KERNEL)",
            kernel.name,
            if kernel.weights_f16() { "f16" } else { "f32" },
            if dims.b_f16 { "f16" } else { "f32" },
            if dims.b_f16 { "f16w_*" } else { "f32" },
        );
    }
    if dims.b_f16 && !ctx.buffer_device_address_enabled {
        bail!("f16-weight matmuls require bufferDeviceAddress, which this device lacks");
    }

    let call = &op.call;
    let max_groups = ctx.device_properties.limits.max_compute_work_group_count;
    let (a_ptr, b_ptr, c_ptr) = if ctx.buffer_device_address_enabled {
        (
            call.a.device_address(),
            call.b.device_address(),
            call.c.device_address(),
        )
    } else {
        (0, 0, 0)
    };
    let mut pc = dims.push_constants(call.alpha, call.accumulate, a_ptr, b_ptr, c_ptr);

    // Pick the specialization variant whose constants match this call.
    // `interior_only` is safe whenever M and N are tile-aligned to the
    // selected kernel's tile size — the shader then skips all
    // m_full/n_full bounds checks.  `tile_k == 1` divides every K and
    // those kernels (row/col/outer) never read K_MULTIPLE; forcing the
    // flag false lets their plain unaligned dispatches resolve to the
    // eager fallback pipeline instead of the lazy-cache mutex.
    let variant = KernelVariant {
        accumulate: call.accumulate,
        alpha_is_one: call.alpha == 1.0,
        interior_only: dims.m.is_multiple_of(kernel.tile_m) && dims.n.is_multiple_of(kernel.tile_n),
        k_multiple: kernel.tile_k > 1 && dims.k.is_multiple_of(kernel.tile_k),
    };

    let epilogue = &op.epilogue;
    if !epilogue.is_none() {
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
            pc.bias_ptr = bias.device_address();
            pc.bias_batch_stride = if bias.len() == dims.n as u64 {
                0
            } else {
                dims.n
            };
        }
        if let Some(d) = epilogue.d_tensor() {
            pc.d_ptr = d.device_address();
        }
        pc.beta = epilogue.beta();
    }
    if let Some((norm_weight, eps)) = op.normed_a {
        // Only the row-GEMV bodies implement the fused RMSNorm-of-A
        // preamble; every other route must fail loudly rather than
        // silently skip the normalization.
        if !kernel.supports_normed_a() {
            bail!(
                "kernel '{}' does not implement the fused normed-A GEMV \
                 (row_bda-family kernels only; normed-A requires an M=1 \
                 row-kernel route, got M={})",
                kernel.name,
                dims.m
            );
        }
        pc.flags |= crate::pipeline::MATMUL_FLAG_NORM_A;
        pc.bias_ptr = norm_weight.device_address();
        pc.beta = eps;
    }
    let mut epilogue_key = epilogue.key();
    if !op.store.is_none() {
        // Same loud-failure contract as normed-A: a store op must not
        // silently skip its rope/scatter on a route without it.
        if !kernel.supports_store() {
            bail!(
                "kernel '{}' does not implement the fused store epilogue \
                 (f16w row_bda-family kernels only; store requires an M=1 \
                 f16-weights row route, got M={})",
                kernel.name,
                dims.m
            );
        }
        let desc = op.store.desc();
        pc.store_pos_base = desc.pos_base;
        pc.store_pos_scale = desc.pos_scale;
        pc.store_stride_head = desc.stride_head;
        pc.store_stride_dim = desc.stride_dim;
        pc.store_head_dim = desc.head_dim;
        pc.store_pos_ptr = desc.pos_addr;
        if let Some(table) = op.store.table() {
            pc.store_table_ptr = table.device_address();
        }
        if let Some(dst) = op.store.dst() {
            pc.store_dst_ptr = dst.device_address();
            epilogue_key.store_f16 = dst.dtype() == crate::dtype::DType::F16;
        }
        epilogue_key.store_mode = op.store.mode();
    }
    let variant_pipeline = pipeline.pipeline_for_epilogue(kernel, variant, epilogue_key)?;

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

#[cfg(test)]
mod tests {
    use super::ensure_kernel_alignment;

    #[test]
    fn strict_aligned_kernels_require_every_tile_multiple() {
        let tile = [128, 128, 32];
        assert!(ensure_kernel_alignment("large_bda_v4_aligned", tile, tile).is_ok());
        for shape in [[127, 128, 32], [128, 127, 32], [128, 128, 31]] {
            let err = ensure_kernel_alignment("large_bda_v4_aligned", tile, shape)
                .unwrap_err()
                .to_string();
            assert!(err.contains("requires M/N/K aligned"));
        }
        assert!(ensure_kernel_alignment("large_bda_v4", tile, [1, 1, 1]).is_ok());
        let v3_tile = [128, 128, 8];
        assert!(ensure_kernel_alignment("v3_128x128_bk8_static", v3_tile, [128, 128, 16]).is_ok());
        for shape in [
            [127, 128, 16],
            [128, 127, 16],
            [128, 128, 8],
            [128, 128, 24],
        ] {
            assert!(ensure_kernel_alignment("v3_128x128_bk8_static", v3_tile, shape).is_err());
        }
    }
}
