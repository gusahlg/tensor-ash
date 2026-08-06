//! Persistent-threads experimental kernel.
//!
//! This opt-in pipeline stays outside `MatmulPipeline`: it has a different
//! push-constant ABI and uses a global atomic counter to distribute output
//! tiles over a fixed number of resident workgroups.

mod execution;
mod resources;

use std::sync::Arc;

use ash::vk;

use crate::buffer::Buffer;
use crate::context::VulkanContext;

/// Output-tile dimensions of `matmul_f32_persistent_v4.comp`.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 64;
pub const TILE_K: u32 = 64;

pub(super) const WG_PER_SM: u32 = 2;
pub(super) const FALLBACK_SM_COUNT: u32 = 46;
pub(super) const VARIANT_COUNT: usize = 16;

/// Bit-for-bit identical to the shader's `PC` block.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PersistentPushConstants {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) batch_stride_a: u32,
    pub(super) batch_stride_b: u32,
    pub(super) batch_stride_c: u32,
    pub(super) flags: u32,
    pub(super) alpha: f32,
    pub(super) a_ptr: u64,
    pub(super) b_ptr: u64,
    pub(super) c_ptr: u64,
    pub(super) counter_ptr: u64,
    pub(super) grid_x: u32,
    pub(super) grid_y: u32,
    pub(super) num_tiles: u32,
    pub(super) _pad: u32,
}

/// Experimental persistent-thread matmul runner.
///
/// The runner owns one submission slot and therefore serializes its calls.
/// Create separate runners when independent host threads need to submit.
pub struct PersistentMatmul {
    pub(super) ctx: Arc<VulkanContext>,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) shader: vk::ShaderModule,
    pub(super) pipelines: [vk::Pipeline; VARIANT_COUNT],
    pub(super) sm_count: u32,
    pub(super) timestamp_valid_bits: u32,
    pub(super) cmd_pool: vk::CommandPool,
    pub(super) cmd: vk::CommandBuffer,
    pub(super) fence: vk::Fence,
    pub(super) query_pool: vk::QueryPool,
    pub(super) counter_buffer: Buffer,
    pub(super) counter_addr: u64,
    pub(super) used: bool,
}
