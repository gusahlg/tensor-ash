//! Vulkan twin of the PTX `probe_affine` kernel, for ordering probes.

use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use super::PROBE_AFFINE_SPV;
use crate::context::VulkanContext;
use crate::pipeline::{build_compute_pipeline, create_pc_only_layout};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AffinePc {
    src: u64,
    dst: u64,
    n: u32,
    mul: u32,
    add: u32,
    _pad: u32,
}

/// `dst[j] = src[j]*mul + add`, walked in reverse (see
/// `shaders/ptx_probe_affine.comp`).
pub struct ProbeAffine {
    ctx: Arc<VulkanContext>,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

impl ProbeAffine {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let layout = create_pc_only_layout(ctx, std::mem::size_of::<AffinePc>() as u32)?;
        let built = build_compute_pipeline(ctx, layout, &[], PROBE_AFFINE_SPV, "ptx_probe_affine");
        let (module, pipeline) = match built {
            Ok(v) => v,
            Err(e) => {
                unsafe { ctx.device.destroy_pipeline_layout(layout, None) };
                return Err(e);
            }
        };
        Ok(Self {
            ctx: Arc::clone(ctx),
            layout,
            module,
            pipeline,
        })
    }

    /// # Safety
    /// `cb` is recording; `src`/`dst` are device addresses of >= `n` u32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cmd_dispatch(
        &self,
        cb: vk::CommandBuffer,
        src: u64,
        dst: u64,
        n: u32,
        mul: u32,
        add: u32,
        groups: u32,
    ) {
        let pc = AffinePc {
            src,
            dst,
            n,
            mul,
            add,
            _pad: 0,
        };
        let dev = &self.ctx.device;
        unsafe {
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            dev.cmd_dispatch(cb, groups, 1, 1);
        }
    }
}

impl Drop for ProbeAffine {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_shader_module(self.module, None);
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
