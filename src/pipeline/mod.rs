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
use selection::{auto_min_large_tiles_for, auto_select_kernel};

pub use types::{
    KERNEL_SPECS, KernelSelection, KernelSpec, KernelVariant, MatmulKernel, MatmulPushConstants,
};

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    /// One `MatmulKernel` per entry in `KERNEL_SPECS`, in the same order.
    /// Index with `KernelSelection::index()` (after resolving `Auto`).
    kernels: Vec<MatmulKernel>,
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

            // Build every kernel in the registry.  We wrap the accumulator
            // in a scopeguard so any already-built kernels are torn down
            // if a later one fails to build.
            let kernels_acc: Vec<MatmulKernel> = Vec::with_capacity(KERNEL_SPECS.len());
            let mut kernels_guard = scopeguard::guard(kernels_acc, |built| {
                for kernel in built {
                    destroy_kernel(ctx, kernel);
                }
            });
            for spec in KERNEL_SPECS {
                let kernel = create_kernel(
                    ctx,
                    pipeline_layout,
                    spec.name,
                    spec.tile_m,
                    spec.tile_n,
                    spec.tile_k,
                    spec.spv,
                )
                .with_context(|| format!("create {} matmul kernel", spec.name))?;
                kernels_guard.push(kernel);
            }
            // Disarm the cleanup: kernels now belong to the pipeline.
            let kernels = ScopeGuard::into_inner(kernels_guard);

            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout: ScopeGuard::into_inner(set_layout_guard),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                kernels,
                selection,
                auto_min_large_tiles: auto_min_large_tiles_for(ctx.device_kind()),
            })
        }
    }

    pub fn select_kernel(&self, batch: u32, m: u32, n: u32, k: u32) -> &MatmulKernel {
        let selection = match self.selection {
            KernelSelection::Auto => auto_select_kernel(batch, m, n, k, self.auto_min_large_tiles),
            explicit => explicit,
        };
        let idx = selection
            .index()
            .expect("auto selection must resolve to a concrete kernel");
        &self.kernels[idx]
    }
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in self.kernels.drain(..) {
                destroy_kernel(&self.ctx, kernel);
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
