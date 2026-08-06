//! Stream-K experimental pipeline (CUTLASS-style hybrid DP + SK tail).
//!
//! For shapes where the tile count is close to but not a clean
//! multiple of the preferred grid size, the regular data-parallel
//! dispatch ends with a partial wave (e.g. 4096^3 with 32x32=1024
//! output tiles maps to 11 full waves of 92 WGs + a last wave of 12
//! WGs on an RTX 3070's 46 SMs).  Stream-K closes that gap by handing
//! the *bulk* tiles to a normal DP dispatch and routing the *tail*
//! tiles' work through a fixed-size persistent grid that atomic-adds
//! partial sums into C.
//!
//! Two pipelines, two dispatches per call:
//!
//!   1. **DP-flat kernel** — the standard 128x128 BDA_V4 aligned
//!      body, dispatched as `(n_tiles, m_tiles)` with an early-exit
//!      `if tile_id >= dp_tiles_total` so workgroups outside the bulk
//!      region drop out cheaply.
//!   2. **SK-tail kernel** — a persistent 1D grid of `g_sk`
//!      workgroups, each consuming a contiguous slice of the
//!      remaining `total_iters_sk` MAC iterations.  Tile owners use a
//!      plain store; partial contributors use `atomicAdd` (hardware
//!      via `VK_EXT_shader_atomic_float` → Ampere `RED.E.ADD.F32`).
//!
//! Splitting the two halves into separate SPIR-V binaries keeps the
//! DP path's hot loop byte-for-byte identical to BDA_V4 aligned (no
//! branch on a Stream-K mode bit in the inner loop).
//!
//! `StreamKSchedule::for_shape` computes the split and degenerates to
//! pure-DP (`g_sk == 0`) or pure-SK (`dp_tiles_total == 0`) as
//! needed.  The executor skips the corresponding dispatch in those
//! degenerate cases.
//!
//! Restrictions:
//!
//!   * Aligned shapes only: M%BM == 0, N%BN == 0, K%BK == 0.
//!   * batch == 1.
//!   * accumulate == false (host pre-zeros C; the kernel
//!     atomic-adds tail contributions, plain-stores DP and SK owners).
//!   * `VK_EXT_shader_atomic_float` with
//!     `shaderBufferFloat32AtomicAdd` must be enabled on the
//!     `VulkanContext`; pipeline creation returns an error otherwise
//!     and callers should fall back to `run_matmuls`.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::context::VulkanContext;

const SPIRV_TAIL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_streamk_128x128.spv"));
// DP-flat shader: reuse matmul_f32_large_bda_v4 directly.  Its PC
// block (56 bytes) is a prefix of StreamKPushConstants (88 bytes),
// so the host can push the full StreamK PC and the shader silently
// ignores the trailing Stream-K-specific fields per the Vulkan PC
// rules.  Using the base kernel — instead of a Stream-K-specific
// clone with a tile_id early-out — sidesteps glslang's structured-
// control-flow OpSwitch wrapper around main(), which on Ampere
// costs ~12% on the aligned hot path even with the early-out branch
// spec-const-elided.  This requires `dp_tiles_total` to be a clean
// multiple of `n_tiles` (see `StreamKSchedule::for_shape`).
const SPIRV_DP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_large_bda_v4.spv"));

/// Push constants for the hybrid Stream-K shaders. The prefix is the
/// regular BDA_V4 `PC` layout consumed by the DP kernel; the SK-tail shader
/// consumes the trailing schedule fields as well. Vulkan permits pushing the
/// full range when the bound DP layout declares only that prefix.
///
/// `g_sk`, `total_iters_sk`, `iters_per_wg_sk`, `rem_sk`, and
/// `iters_per_tile` describe the SK-tail iter-space. DP coverage is bounded
/// by its host-computed rectangular dispatch dimensions.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StreamKPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_c: u32,
    pub flags: u32,
    pub alpha: f32,
    pub a_ptr: u64,
    pub b_ptr: u64,
    pub c_ptr: u64,
    pub iters_per_tile: u32,
    pub iters_per_wg_sk: u32,
    pub rem_sk: u32,
    pub n_tiles: u32,
    pub dp_tiles_total: u32,
    pub g_sk: u32,
    pub total_iters_sk: u32,
    // Trailing pad so the 8-aligned struct (driven by the u64 BDA
    // fields) has no implicit padding; bytemuck::Pod requires zero
    // padding bytes.
    pub _pad: u32,
}

/// Static, host-side description of a Stream-K kernel half (DP-flat
/// or SK-tail).  Today we only ship the 128x128 BK=32 tile.
pub(super) struct StreamKKernel {
    pub(super) tile_m: u32,
    pub(super) tile_n: u32,
    pub(super) tile_k: u32,
    pub(super) shader_module: vk::ShaderModule,
    pub(super) pipeline: vk::Pipeline,
}

/// Hybrid Stream-K pipeline: separate DP-flat and SK-tail kernels
/// dispatched in sequence by `record_and_run_stream_k`.  The DP-flat
/// kernel is the regular `matmul_f32_large_bda_v4` SPIR-V (shared
/// with the auto path's main BDA_V4 dispatch); `StreamKSchedule`
/// guarantees the DP dispatch always lands on a clean 2D rectangle
/// of tiles, so no early-out branch is needed.  The SK-tail kernel
/// keeps the persistent-grid + atomicAdd machinery.
pub(super) struct StreamKPipeline {
    ctx: Arc<VulkanContext>,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) k128x128_dp: StreamKKernel,
    pub(super) k128x128_tail: StreamKKernel,
}

impl StreamKPipeline {
    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        if !ctx.buffer_device_address_enabled {
            bail!(
                "Stream-K kernel requires Vulkan 1.2 bufferDeviceAddress \
                 which is not enabled on this device"
            );
        }
        if !ctx.shader_buffer_float32_atomic_add_enabled {
            bail!(
                "Stream-K kernel requires VK_EXT_shader_atomic_float \
                 (shaderBufferFloat32AtomicAdd) which is not enabled \
                 on this device"
            );
        }
        unsafe {
            // Own pipeline_layout: no descriptor sets (all I/O via BDA
            // push constants), `pc_size = sizeof(StreamKPushConstants)`.
            let pc_size = std::mem::size_of::<StreamKPushConstants>() as u32;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size)];
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_ranges),
                    None,
                )
                .context("create_pipeline_layout (stream-K)")?;
            let pipeline_layout_guard = scopeguard::guard(pipeline_layout, |l| {
                ctx.device.destroy_pipeline_layout(l, None)
            });

            let k128x128_dp = build_kernel(ctx, pipeline_layout, 128, 128, 32, SPIRV_DP)
                .context("build Stream-K DP-flat 128x128 kernel")?;
            let k128x128_dp_guard = scopeguard::guard(k128x128_dp, |k| {
                if k.pipeline != vk::Pipeline::null() {
                    ctx.device.destroy_pipeline(k.pipeline, None);
                }
                if k.shader_module != vk::ShaderModule::null() {
                    ctx.device.destroy_shader_module(k.shader_module, None);
                }
            });
            let k128x128_tail = build_kernel(ctx, pipeline_layout, 128, 128, 32, SPIRV_TAIL)
                .context("build Stream-K tail 128x128 kernel")?;
            Ok(Self {
                ctx: Arc::clone(ctx),
                pipeline_layout: scopeguard::ScopeGuard::into_inner(pipeline_layout_guard),
                k128x128_dp: scopeguard::ScopeGuard::into_inner(k128x128_dp_guard),
                k128x128_tail,
            })
        }
    }

    /// Pick the DP-flat kernel for a given (M, N).  Today only one
    /// tile shape ships; future work will add 64x64 / 64x128 etc.
    pub(super) fn pick_dp(&self, _m: u32, _n: u32) -> &StreamKKernel {
        &self.k128x128_dp
    }

    /// Pick the SK-tail kernel for a given (M, N).
    pub(super) fn pick_tail(&self, _m: u32, _n: u32) -> &StreamKKernel {
        &self.k128x128_tail
    }
}

fn build_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    spv: &[u8],
) -> Result<StreamKKernel> {
    assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .context("create_shader_module (stream-k)")?;
        let entry = std::ffi::CString::new("main").unwrap();

        // Spec constants used by the DP-flat shader (matmul_f32_large_bda_v4):
        //   0 = ACCUMULATE    (false: kernel plain-stores)
        //   1 = ALPHA_IS_ONE  (true:  skip the alpha multiply)
        //   2 = INTERIOR_ONLY (true:  M%BM == 0, N%BN == 0)
        //   3 = K_MULTIPLE    (true:  K%BK == 0)
        // SK-tail shader (matmul_f32_streamk_128x128) only declares
        // IDs 0 and 1; entries for IDs not present in the shader are
        // ignored per Vulkan §11.4.1.
        let spec_entries = [
            vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(1)
                .offset(4)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(2)
                .offset(8)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(3)
                .offset(12)
                .size(4),
        ];
        let spec_data: [u32; 4] = [0, 1, 1, 1];
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_entries)
            .data(bytemuck::cast_slice(&spec_data));

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry)
            .specialization_info(&spec_info);

        let ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];

        let pipelines = match ctx
            .device
            .create_compute_pipelines(ctx.pipeline_cache, &ci, None)
        {
            Ok(p) => p,
            Err((partial, err)) => {
                for p in partial {
                    if p != vk::Pipeline::null() {
                        ctx.device.destroy_pipeline(p, None);
                    }
                }
                ctx.device.destroy_shader_module(shader_module, None);
                return Err(anyhow!(
                    "create_compute_pipelines stream-K {tile_m}x{tile_n}: {err}"
                ));
            }
        };

        Ok(StreamKKernel {
            tile_m,
            tile_n,
            tile_k,
            shader_module,
            pipeline: pipelines[0],
        })
    }
}

impl Drop for StreamKPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [&self.k128x128_dp, &self.k128x128_tail] {
                if kernel.pipeline != vk::Pipeline::null() {
                    self.ctx.device.destroy_pipeline(kernel.pipeline, None);
                }
                if kernel.shader_module != vk::ShaderModule::null() {
                    self.ctx
                        .device
                        .destroy_shader_module(kernel.shader_module, None);
                }
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.ctx
                    .device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}
