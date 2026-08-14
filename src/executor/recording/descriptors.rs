//! Descriptor-set updates for descriptor-bound dispatches.

use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::MatmulOp;
use crate::pipeline::MatmulPipeline;
use crate::tensor::Tensor;

use super::super::OpPlan;

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
pub(in crate::executor) fn update_matmul_descriptor_sets(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    sets: &[vk::DescriptorSet],
    ops: &[MatmulOp<'_>],
    plans: &[OpPlan],
) {
    debug_assert_eq!(sets.len(), ops.len());
    debug_assert_eq!(sets.len(), plans.len());

    // BDA is the normal fast path on modern devices. Avoid two temporary
    // allocations per submission when no planned kernel observes a
    // descriptor set at all.
    let needs_write = |plan: &OpPlan| pipeline.kernel_at(plan.kernel).uses_descriptors;
    if !plans.iter().any(needs_write) {
        return;
    }

    // Build buffer-info + writes only for calls that actually need a
    // descriptor write.  Keep buffer_infos stable so the
    // &[WriteDescriptorSet] we hand to Vulkan keeps stable references.
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(ops.len() * 3);
    for (op, plan) in ops.iter().zip(plans.iter()) {
        if needs_write(plan) {
            buffer_infos.push(tensor_descriptor(op.call.a));
            buffer_infos.push(tensor_descriptor(op.call.b));
            buffer_infos.push(tensor_descriptor(op.call.c));
        }
    }

    let mut writes: Vec<vk::WriteDescriptorSet> = Vec::with_capacity(buffer_infos.len());
    let mut base = 0usize;
    for (set, plan) in sets.iter().copied().zip(plans.iter()) {
        if !needs_write(plan) {
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

fn tensor_descriptor(tensor: &Tensor) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(tensor.raw_buffer())
        .offset(0)
        .range(tensor.size_bytes())
}
