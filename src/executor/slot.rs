use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::buffer::Buffer;
use crate::context::VulkanContext;

pub(super) struct Slot {
    pub(super) cmd_pool: vk::CommandPool,
    pub(super) cmd: vk::CommandBuffer,
    pub(super) fence: vk::Fence,
    pub(super) descriptor_pool: vk::DescriptorPool,
    /// `max_calls_per_submit` descriptor sets pre-allocated at slot
    /// creation.  Reused across submissions by `vkUpdateDescriptorSets`
    /// — the pool is never reset, so the set handles stay valid for the
    /// life of the executor.
    pub(super) descriptor_sets: Vec<vk::DescriptorSet>,
    /// Two timestamps: start and end. Null when timestamps are unsupported.
    pub(super) query_pool: vk::QueryPool,
    pub(super) upload_staging: Option<Buffer>,
    pub(super) download_staging: Option<Buffer>,
    /// Device-local scratch for the two-stage split-K partials, grown
    /// on demand.  Slot-owned so concurrent submissions never share it.
    pub(super) splitk2_scratch: Option<Buffer>,
    /// Whether the slot was used at least once (cmd pool / fence need a reset).
    pub(super) used: bool,
}

impl Slot {
    pub(super) fn create(
        ctx: &Arc<VulkanContext>,
        max_sets: u32,
        set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        unsafe {
            // Each fallible step's handle is wrapped in a scopeguard, so any
            // early return frees what we've allocated up to that point.  The
            // happy path disarms the guards at the end.
            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.compute_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .context("create_command_pool")?;
            let cmd_pool_guard =
                scopeguard::guard(cmd_pool, |p| ctx.device.destroy_command_pool(p, None));

            let cmd = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .context("allocate_command_buffers")?[0];

            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .context("create_fence")?;
            let fence_guard = scopeguard::guard(fence, |f| ctx.device.destroy_fence(f, None));

            // Descriptor pool sized for `max_sets` matmul calls (3 STORAGE_BUFFER each).
            let descriptor_count = max_sets
                .checked_mul(3)
                .ok_or_else(|| anyhow::anyhow!("slot descriptor count overflows u32"))?;
            let pool_size = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(descriptor_count)];
            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&pool_size),
                    None,
                )
                .context("create_descriptor_pool")?;
            let descriptor_pool_guard = scopeguard::guard(descriptor_pool, |p| {
                ctx.device.destroy_descriptor_pool(p, None)
            });

            // Pre-allocate every descriptor set we'll ever need on this
            // slot.  vkUpdateDescriptorSets is the per-call hot path
            // instead of allocate+free, which keeps the cost of small
            // submits flat.  We deliberately omit FREE_DESCRIPTOR_SET on
            // the pool — sets live until the pool is destroyed.
            let layouts = vec![set_layout; max_sets as usize];
            let descriptor_sets = ctx
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )
                .context("allocate_descriptor_sets (slot prealloc)")?;

            let query_pool = if ctx.timestamps_supported {
                ctx.device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .context("create_query_pool")?
            } else {
                vk::QueryPool::null()
            };

            Ok(Self {
                cmd_pool: ScopeGuard::into_inner(cmd_pool_guard),
                cmd,
                fence: ScopeGuard::into_inner(fence_guard),
                descriptor_pool: ScopeGuard::into_inner(descriptor_pool_guard),
                descriptor_sets,
                query_pool,
                upload_staging: None,
                download_staging: None,
                splitk2_scratch: None,
                used: false,
            })
        }
    }
}
