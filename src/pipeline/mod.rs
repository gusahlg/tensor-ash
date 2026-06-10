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
    K64_TILE_K, K64_TILE_M, K64_TILE_N, KernelSelection, KernelVariant, M64N32_TILE_K,
    M64N32_TILE_M, M64N32_TILE_N, M64N128_TILE_K, M64N128_TILE_M, M64N128_TILE_N, M128N64_TILE_K,
    M128N64_TILE_M, M128N64_TILE_N, M128N64K64_TILE_K, M128N64K64_TILE_M, M128N64K64_TILE_N,
    MatmulKernel, MatmulPushConstants, SMALL_TILE_K, SMALL_TILE_M, SMALL_TILE_N, TILE_K, TILE_M,
    TILE_N,
};

const SPV_MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
const SPV_MATMUL_F32_SMALL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small.spv"));
const SPV_MATMUL_F32_M64N128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n128.spv"));
const SPV_MATMUL_F32_M128N64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64.spv"));
const SPV_MATMUL_F32_M128N64K64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m128n64k64.spv"));
const SPV_MATMUL_F32_M64N32: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_m64n32.spv"));
const SPV_MATMUL_F32_K64: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_k64.spv"));

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub large: MatmulKernel,
    pub small: MatmulKernel,
    m64n128: MatmulKernel,
    m128n64: MatmulKernel,
    m128n64k64: MatmulKernel,
    m64n32: MatmulKernel,
    k64: MatmulKernel,
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
                TILE_K,
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
                SMALL_TILE_K,
                SPV_MATMUL_F32_SMALL,
            )
            .context("create small matmul kernel")?;
            let small_guard = scopeguard::guard(small, |k| destroy_kernel(ctx, k));

            let m64n128 = create_kernel(
                ctx,
                pipeline_layout,
                "m64n128",
                M64N128_TILE_M,
                M64N128_TILE_N,
                M64N128_TILE_K,
                SPV_MATMUL_F32_M64N128,
            )
            .context("create m64n128 matmul kernel")?;
            let m64n128_guard = scopeguard::guard(m64n128, |k| destroy_kernel(ctx, k));

            let m128n64 = create_kernel(
                ctx,
                pipeline_layout,
                "m128n64",
                M128N64_TILE_M,
                M128N64_TILE_N,
                M128N64_TILE_K,
                SPV_MATMUL_F32_M128N64,
            )
            .context("create m128n64 matmul kernel")?;
            let m128n64_guard = scopeguard::guard(m128n64, |k| destroy_kernel(ctx, k));

            let m128n64k64 = create_kernel(
                ctx,
                pipeline_layout,
                "m128n64k64",
                M128N64K64_TILE_M,
                M128N64K64_TILE_N,
                M128N64K64_TILE_K,
                SPV_MATMUL_F32_M128N64K64,
            )
            .context("create m128n64k64 matmul kernel")?;
            let m128n64k64_guard = scopeguard::guard(m128n64k64, |k| destroy_kernel(ctx, k));

            let m64n32 = create_kernel(
                ctx,
                pipeline_layout,
                "m64n32",
                M64N32_TILE_M,
                M64N32_TILE_N,
                M64N32_TILE_K,
                SPV_MATMUL_F32_M64N32,
            )
            .context("create m64n32 matmul kernel")?;
            let m64n32_guard = scopeguard::guard(m64n32, |k| destroy_kernel(ctx, k));

            let k64 = create_kernel(
                ctx,
                pipeline_layout,
                "k64",
                K64_TILE_M,
                K64_TILE_N,
                K64_TILE_K,
                SPV_MATMUL_F32_K64,
            )
            .context("create k64 matmul kernel")?;
            let k64_guard = scopeguard::guard(k64, |k| destroy_kernel(ctx, k));

            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout: ScopeGuard::into_inner(set_layout_guard),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                large: ScopeGuard::into_inner(large_guard),
                small: ScopeGuard::into_inner(small_guard),
                m64n128: ScopeGuard::into_inner(m64n128_guard),
                m128n64: ScopeGuard::into_inner(m128n64_guard),
                m128n64k64: ScopeGuard::into_inner(m128n64k64_guard),
                m64n32: ScopeGuard::into_inner(m64n32_guard),
                k64: ScopeGuard::into_inner(k64_guard),
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
        match selection {
            KernelSelection::Auto => {
                unreachable!("auto selection must resolve to a concrete kernel")
            }
            KernelSelection::Large => &self.large,
            KernelSelection::Small => &self.small,
            KernelSelection::M64N128 => &self.m64n128,
            KernelSelection::M128N64 => &self.m128n64,
            KernelSelection::M128N64K64 => &self.m128n64k64,
            KernelSelection::M64N32 => &self.m64n32,
            KernelSelection::K64 => &self.k64,
        }
    }
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [
                &self.large,
                &self.small,
                &self.m64n128,
                &self.m128n64,
                &self.m128n64k64,
                &self.m64n32,
                &self.k64,
            ] {
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
