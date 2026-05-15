//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::context::VulkanContext;

/// Push constants for the matmul shader.  Bit-for-bit identical to the
/// GLSL `PC` block.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_c: u32,
    pub flags: u32, // bit 0: accumulate
    pub alpha: f32,
}

/// Default large-kernel output-tile dimensions.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 128;
pub const SMALL_TILE_M: u32 = 64;
pub const SMALL_TILE_N: u32 = 64;

const SPV_MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
const SPV_MATMUL_F32_SMALL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small.spv"));

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelSelection {
    Auto,
    Large,
    Small,
}

impl KernelSelection {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "large" | "large_128" | "128" => Ok(Self::Large),
            "small" | "small_64" | "64" => Ok(Self::Small),
            other => bail!("invalid ML_KERNEL '{other}', expected auto, large, or small"),
        }
    }

    fn from_env() -> Result<Self> {
        let value = std::env::var("ML_KERNEL").unwrap_or_else(|_| "auto".into());
        Self::parse(&value)
    }
}

pub struct MatmulKernel {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub shader_module: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
}

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub large: MatmulKernel,
    pub small: MatmulKernel,
    selection: KernelSelection,
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
            // ---- Descriptor set layout: A, B, C storage buffers ----
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

            // ---- Pipeline layout: 1 set + push constants ----
            let pc_size = std::mem::size_of::<MatmulPushConstants>() as u32;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size)];
            let set_layouts = [set_layout];
            let pipeline_layout = match ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&pc_ranges),
                None,
            ) {
                Ok(pipeline_layout) => pipeline_layout,
                Err(err) => {
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create_pipeline_layout");
                }
            };

            // ---- Compute pipeline ----
            let large = match create_kernel(
                ctx,
                pipeline_layout,
                "large_128",
                TILE_M,
                TILE_N,
                SPV_MATMUL_F32,
            ) {
                Ok(large) => large,
                Err(err) => {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create large matmul kernel");
                }
            };
            let small = match create_kernel(
                ctx,
                pipeline_layout,
                "small_64",
                SMALL_TILE_M,
                SMALL_TILE_N,
                SPV_MATMUL_F32_SMALL,
            ) {
                Ok(small) => small,
                Err(err) => {
                    ctx.device.destroy_pipeline(large.pipeline, None);
                    ctx.device.destroy_shader_module(large.shader_module, None);
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create small matmul kernel");
                }
            };

            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout,
                pipeline_layout,
                large,
                small,
                selection,
            })
        }
    }

    pub fn select_kernel(&self, m: u32, n: u32, k: u32) -> &MatmulKernel {
        match self.selection {
            KernelSelection::Large => &self.large,
            KernelSelection::Small => &self.small,
            KernelSelection::Auto if auto_selects_small_kernel(m, n, k) => &self.small,
            KernelSelection::Auto => &self.large,
        }
    }
}

fn auto_selects_small_kernel(m: u32, n: u32, k: u32) -> bool {
    m <= 1024 || n <= 1024 || k <= 128 || !m.is_multiple_of(TILE_M) || !n.is_multiple_of(TILE_N)
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.ctx.device.destroy_pipeline(self.large.pipeline, None);
            self.ctx
                .device
                .destroy_shader_module(self.large.shader_module, None);
            self.ctx.device.destroy_pipeline(self.small.pipeline, None);
            self.ctx
                .device
                .destroy_shader_module(self.small.shader_module, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn create_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    name: &'static str,
    tile_m: u32,
    tile_n: u32,
    spv: &[u8],
) -> Result<MatmulKernel> {
    unsafe {
        assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .context("create_shader_module")?;
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);
        let ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = ctx
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&ci), None)
            .map_err(|(_, err)| {
                ctx.device.destroy_shader_module(shader_module, None);
                anyhow::anyhow!("create_compute_pipelines {name}: {err}")
            })?[0];

        Ok(MatmulKernel {
            name,
            tile_m,
            tile_n,
            shader_module,
            pipeline,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_selection() {
        assert_eq!(KernelSelection::parse("").unwrap(), KernelSelection::Auto);
        assert_eq!(
            KernelSelection::parse("auto").unwrap(),
            KernelSelection::Auto
        );
        assert_eq!(
            KernelSelection::parse("large_128").unwrap(),
            KernelSelection::Large
        );
        assert_eq!(
            KernelSelection::parse("64").unwrap(),
            KernelSelection::Small
        );
        assert!(KernelSelection::parse("wide").is_err());
    }

    #[test]
    fn auto_selector_prefers_small_for_edge_or_small_shapes() {
        assert!(auto_selects_small_kernel(512, 512, 512));
        assert!(auto_selects_small_kernel(1024, 1024, 1024));
        assert!(auto_selects_small_kernel(1023, 2048, 1024));
        assert!(auto_selects_small_kernel(2048, 1025, 1024));
        assert!(auto_selects_small_kernel(2048, 2048, 64));
        assert!(!auto_selects_small_kernel(2048, 2048, 2048));
    }
}
