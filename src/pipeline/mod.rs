//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

mod create;
mod selection;
mod types;

use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

use create::{create_kernel, destroy_kernel};
use selection::{auto_min_large_tiles_for, auto_selects_small_kernel};

pub use types::{
    KernelSelection, KernelVariant, MatmulKernel, MatmulPushConstants, SMALL_TILE_M, SMALL_TILE_N,
    TILE_M, TILE_N,
};

const SPV_MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
const SPV_MATMUL_F32_SMALL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small.spv"));

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub large: MatmulKernel,
    pub small: MatmulKernel,
    selection: KernelSelection,
    /// Minimum number of large-kernel tiles a problem must produce
    /// before the auto-selector prefers the large kernel.  Below this,
    /// the small kernel's 4x more workgroups give better device
    /// occupancy.  Derived from device kind at pipeline build time.
    auto_min_large_tiles: u64,
}

impl MatmulPipeline {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        Self::new_with_kernel_selection(ctx, KernelSelection::from_env()?)
    }

    pub fn new_with_kernel_selection(
        ctx: &Arc<VulkanContext>,
        selection: KernelSelection,
    ) -> Result<Self> {
        unsafe {
            let bindings = (0u32..3)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let set_layout = ctx
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .context("create_descriptor_set_layout")?;
            let set_layout_guard = scopeguard::guard(set_layout, |l| {
                ctx.device.destroy_descriptor_set_layout(l, None)
            });

            let pc_size = std::mem::size_of::<MatmulPushConstants>() as u32;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size)];
            let set_layouts = [set_layout];
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&pc_ranges),
                    None,
                )
                .context("create_pipeline_layout")?;
            let pipeline_layout_guard = scopeguard::guard(pipeline_layout, |l| {
                ctx.device.destroy_pipeline_layout(l, None)
            });

            let large = create_kernel(
                ctx,
                pipeline_layout,
                "large_128",
                TILE_M,
                TILE_N,
                SPV_MATMUL_F32,
            )
            .context("create large matmul kernel")?;
            let large_guard = scopeguard::guard(large, |k| destroy_kernel(ctx, k));

            let small = create_kernel(
                ctx,
                pipeline_layout,
                "small_64",
                SMALL_TILE_M,
                SMALL_TILE_N,
                SPV_MATMUL_F32_SMALL,
            )
            .context("create small matmul kernel")?;

            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout: ScopeGuard::into_inner(set_layout_guard),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                large: ScopeGuard::into_inner(large_guard),
                small,
                selection,
                auto_min_large_tiles: auto_min_large_tiles_for(ctx.device_kind()),
            })
        }
    }

    pub fn select_kernel(&self, m: u32, n: u32, k: u32) -> &MatmulKernel {
        match self.selection {
            KernelSelection::Large => &self.large,
            KernelSelection::Small => &self.small,
            KernelSelection::Auto
                if auto_selects_small_kernel(m, n, k, self.auto_min_large_tiles) =>
            {
                &self.small
            }
            KernelSelection::Auto => &self.large,
        }
    }
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [&self.large, &self.small] {
                for p in kernel.variants.iter().copied() {
                    if p != vk::Pipeline::null() {
                        self.ctx.device.destroy_pipeline(p, None);
                    }
                }
                self.ctx
                    .device
                    .destroy_shader_module(kernel.shader_module, None);
            }
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
