//! Vulkan resource creation and destruction for the persistent runner.

use std::ffi::CString;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;

use super::{FALLBACK_SM_COUNT, PersistentMatmul, PersistentPushConstants, VARIANT_COUNT};

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_persistent_v4.spv"));

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpecData {
    accumulate: u32,
    alpha_is_one: u32,
    interior_only: u32,
    k_multiple: u32,
}

impl PersistentMatmul {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        if !ctx.buffer_device_address_enabled {
            bail!("persistent kernel requires Vulkan 1.2 bufferDeviceAddress");
        }
        let pc_size = size_of::<PersistentPushConstants>() as u32;
        let pc_limit = ctx.device_properties.limits.max_push_constants_size;
        if pc_size > pc_limit {
            bail!("persistent push constants require {pc_size} bytes; device limit is {pc_limit}");
        }

        unsafe {
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(pc_size)];
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_ranges),
                    None,
                )
                .context("create_pipeline_layout (persistent)")?;
            let layout_guard = scopeguard::guard(pipeline_layout, |layout| {
                ctx.device.destroy_pipeline_layout(layout, None)
            });

            if !SPV.len().is_multiple_of(4) {
                bail!("persistent shader SPIR-V byte length is not a multiple of four");
            }
            let words: Vec<_> = SPV
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let shader = ctx
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .context("create_shader_module (persistent)")?;
            let shader_guard = scopeguard::guard(shader, |module| {
                ctx.device.destroy_shader_module(module, None)
            });

            let entry = CString::new("main").unwrap();
            let spec_entries = [0, 4, 8, 12].map(|offset| {
                vk::SpecializationMapEntry::default()
                    .constant_id(offset / 4)
                    .offset(offset)
                    .size(4)
            });
            let spec_data: [SpecData; VARIANT_COUNT] = std::array::from_fn(|i| SpecData {
                accumulate: u32::from(i & 1 != 0),
                alpha_is_one: u32::from(i & 2 != 0),
                interior_only: u32::from(i & 4 != 0),
                k_multiple: u32::from(i & 8 != 0),
            });
            let spec_infos: [vk::SpecializationInfo; VARIANT_COUNT] = std::array::from_fn(|i| {
                vk::SpecializationInfo::default()
                    .map_entries(&spec_entries)
                    .data(bytemuck::bytes_of(&spec_data[i]))
            });
            let stages: [vk::PipelineShaderStageCreateInfo; VARIANT_COUNT] =
                std::array::from_fn(|i| {
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(&entry)
                        .specialization_info(&spec_infos[i])
                });
            let create_infos: [vk::ComputePipelineCreateInfo; VARIANT_COUNT] =
                std::array::from_fn(|i| {
                    vk::ComputePipelineCreateInfo::default()
                        .stage(stages[i])
                        .layout(pipeline_layout)
                });
            let pipeline_vec =
                match ctx
                    .device
                    .create_compute_pipelines(ctx.pipeline_cache, &create_infos, None)
                {
                    Ok(pipelines) => pipelines,
                    Err((partial, err)) => {
                        for pipeline in partial {
                            ctx.device.destroy_pipeline(pipeline, None);
                        }
                        bail!("create_compute_pipelines (persistent): {err}");
                    }
                };
            let pipelines: [vk::Pipeline; VARIANT_COUNT] = pipeline_vec
                .try_into()
                .map_err(|_| anyhow::anyhow!("persistent pipeline variant count mismatch"))?;
            let pipelines_guard = scopeguard::guard(pipelines, |variants| {
                for pipeline in variants {
                    ctx.device.destroy_pipeline(pipeline, None);
                }
            });

            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.compute_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .context("create_command_pool (persistent)")?;
            let cmd_pool_guard =
                scopeguard::guard(cmd_pool, |pool| ctx.device.destroy_command_pool(pool, None));
            let cmd = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .context("allocate_command_buffers (persistent)")?[0];
            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .context("create_fence (persistent)")?;
            let fence_guard =
                scopeguard::guard(fence, |fence| ctx.device.destroy_fence(fence, None));

            let query_pool = if ctx.timestamps_supported {
                ctx.device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .context("create_query_pool (persistent)")?
            } else {
                vk::QueryPool::null()
            };
            let query_guard = scopeguard::guard(query_pool, |pool| {
                if pool != vk::QueryPool::null() {
                    ctx.device.destroy_query_pool(pool, None);
                }
            });

            let counter_buffer = Buffer::new(
                ctx,
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                BufferLocation::Device,
            )
            .context("counter buffer allocation")?;
            let counter_addr = ctx.buffer_device_address(counter_buffer.raw);
            let timestamp_valid_bits = ctx
                .instance
                .get_physical_device_queue_family_properties(ctx.physical_device)
                [ctx.compute_family as usize]
                .timestamp_valid_bits;

            Ok(Self {
                ctx: Arc::clone(ctx),
                pipeline_layout: ScopeGuard::into_inner(layout_guard),
                shader: ScopeGuard::into_inner(shader_guard),
                pipelines: ScopeGuard::into_inner(pipelines_guard),
                sm_count: FALLBACK_SM_COUNT,
                timestamp_valid_bits,
                cmd_pool: ScopeGuard::into_inner(cmd_pool_guard),
                cmd,
                fence: ScopeGuard::into_inner(fence_guard),
                query_pool: ScopeGuard::into_inner(query_guard),
                counter_buffer,
                counter_addr,
                used: false,
            })
        }
    }
}

impl Drop for PersistentMatmul {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            if self.query_pool != vk::QueryPool::null() {
                self.ctx.device.destroy_query_pool(self.query_pool, None);
            }
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx.device.destroy_command_pool(self.cmd_pool, None);
            for pipeline in self.pipelines {
                self.ctx.device.destroy_pipeline(pipeline, None);
            }
            self.ctx.device.destroy_shader_module(self.shader, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
