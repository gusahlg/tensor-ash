//! Descriptor-set updates for descriptor-bound dispatches.

use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::{MatmulOp, ResolvedMatmul};
use crate::pipeline::MatmulPipeline;
use crate::tensor::Tensor;

use super::super::MatmulCall;

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
pub(in crate::executor) fn update_split_k_descriptor_set(
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

fn tensor_descriptor(tensor: &Tensor) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(tensor.raw_buffer())
        .offset(0)
        .range(tensor.size_bytes())
}
