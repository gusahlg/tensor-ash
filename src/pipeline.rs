//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

use std::sync::Arc;

use anyhow::{Context, Result};
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

/// Output-tile dimensions, kept in sync with the shader.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 128;

const SPV_MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));

pub struct MatmulPipeline {
    ctx:                  Arc<VulkanContext>,
    pub shader_module:    vk::ShaderModule,
    pub set_layout:       vk::DescriptorSetLayout,
    pub pipeline_layout:  vk::PipelineLayout,
    pub pipeline:         vk::Pipeline,
}

impl MatmulPipeline {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> { unsafe {
        // ---- SPIR-V → ShaderModule ----
        assert!(SPV_MATMUL_F32.len() % 4 == 0, "SPIR-V size not 4-aligned");
        let words: Vec<u32> = SPV_MATMUL_F32
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_module = ctx.device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&words),
            None,
        ).context("create_shader_module")?;

        // ---- Descriptor set layout: A, B, C storage buffers ----
        let bindings = (0u32..3)
            .map(|i| vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect::<Vec<_>>();
        let set_layout = ctx.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        ).context("create_descriptor_set_layout")?;

        // ---- Pipeline layout: 1 set + push constants ----
        let pc_size = std::mem::size_of::<MatmulPushConstants>() as u32;
        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(pc_size)];
        let set_layouts = [set_layout];
        let pipeline_layout = ctx.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&pc_ranges),
            None,
        ).context("create_pipeline_layout")?;

        // ---- Compute pipeline ----
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);
        let ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = ctx.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&ci),
            None,
        ).map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {e}"))?[0];

        Ok(Self {
            ctx: Arc::clone(ctx), shader_module, set_layout, pipeline_layout, pipeline,
        })
    }}
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.ctx.device.destroy_shader_module(self.shader_module, None);
        }
    }
}
