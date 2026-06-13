//! Stream-K experimental pipeline.
//!
//! Stream-K replaces the standard one-WG-per-output-tile dispatch with
//! a fixed grid of `G` persistent workgroups, each consuming a
//! contiguous slice of the linearized `m_tiles * n_tiles *
//! iters_per_tile` iteration space.  This eliminates the
//! wave-quantization tail on shapes where `(m_tiles * n_tiles) mod
//! num_sms` is small (e.g. 4096^3 with 32x32=1024 output tiles produces
//! a last-wave of 12/46 SMs busy under data-parallel scheduling;
//! Stream-K fills the whole GPU evenly).
//!
//! Restrictions (v1):
//!   * Aligned shapes only: M%BM == 0, N%BN == 0, K%BK == 0.
//!   * batch == 1.
//!   * accumulate == false (host pre-zeros C; kernel atomic-adds for
//!     partial-tile contributions, plain-stores for single owners).
//!
//! ## Atomic implementation
//!
//! Uses the hardware `atomicAdd(float, float)` from
//! `VK_EXT_shader_atomic_float` (`shaderBufferFloat32AtomicAdd`).
//! Maps to the Ampere `RED.E.ADD.F32` SASS instruction.  Pipeline
//! creation requires the extension to have been enabled on the
//! `VulkanContext`; the executor entry point returns an error
//! otherwise so callers can fall back to the regular DP path.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::context::VulkanContext;

const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_streamk_128x128.spv"));

/// Push constants for the hybrid Stream-K shader.  Bit-for-bit
/// identical to the GLSL `PC` block in matmul_streamk_kernel.glsl.
///
/// Mode A (DP) covers workgroups [0, dp_tiles_total); mode B (SK tail)
/// covers workgroups [dp_tiles_total, dp_tiles_total + g_sk).  The
/// remaining fields describe the SK-tail iter-space.
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

/// Static, host-side description of the Stream-K shader's tile.  Today
/// we only ship the 128x128 BK=32 variant; growing the pick later is a
/// trivial extension.
pub(super) struct StreamKKernel {
    pub(super) tile_m: u32,
    pub(super) tile_n: u32,
    pub(super) tile_k: u32,
    pub(super) shader_module: vk::ShaderModule,
    pub(super) pipeline: vk::Pipeline,
}

pub(super) struct StreamKPipeline {
    ctx: Arc<VulkanContext>,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) k128x128: StreamKKernel,
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

            let k128x128 = build_kernel(ctx, pipeline_layout, 128, 128, 32, SPIRV)
                .context("build Stream-K 128x128 kernel")?;
            Ok(Self {
                ctx: Arc::clone(ctx),
                pipeline_layout: scopeguard::ScopeGuard::into_inner(pipeline_layout_guard),
                k128x128,
            })
        }
    }

    pub(super) fn pick(&self, _m: u32, _n: u32) -> &StreamKKernel {
        &self.k128x128
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

        // The Stream-K shader only declares constant_ids 0 and 1
        // (ACCUMULATE, ALPHA_IS_ONE).  We still write 4 entries to keep
        // the dispatch convention uniform; entries for missing IDs are
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
            if self.k128x128.pipeline != vk::Pipeline::null() {
                self.ctx
                    .device
                    .destroy_pipeline(self.k128x128.pipeline, None);
            }
            if self.k128x128.shader_module != vk::ShaderModule::null() {
                self.ctx
                    .device
                    .destroy_shader_module(self.k128x128.shader_module, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.ctx
                    .device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}

/// Hybrid Stream-K schedule precomputed on host.  Stored as `u32` so
/// it goes straight into the push constant block.
///
/// The dispatch grid is a single 1D dispatch of `grid_total =
/// dp_tiles_total + g_sk` workgroups.  Workgroup IDs in
/// `[0, dp_tiles_total)` run in DP mode (one full tile each, plain
/// store epilogue, no atomics).  Workgroup IDs in `[dp_tiles_total,
/// dp_tiles_total + g_sk)` run in SK mode and split the tail iter
/// space `total_iters_sk = (T - dp_tiles_total) * iters_per_tile`
/// evenly across `g_sk` workgroups.
///
/// When `T <= preferred_g`, `dp_tiles_total = 0` and the schedule
/// degenerates to pure Stream-K (single persistent grid that owns the
/// whole iter space).  When `T % preferred_g == 0`, `g_sk = 0` and the
/// schedule degenerates to pure DP (one workgroup per tile).
#[derive(Copy, Clone, Debug)]
pub struct StreamKSchedule {
    pub iters_per_tile: u32,
    pub iters_per_wg_sk: u32,
    pub rem_sk: u32,
    pub n_tiles: u32,
    pub dp_tiles_total: u32,
    pub g_sk: u32,
    pub total_iters_sk: u32,
    /// Total workgroups in the 1D dispatch grid.
    pub grid_total: u32,
}

impl StreamKSchedule {
    /// Compute the hybrid schedule for a single-batch GEMM of dims
    /// (m, n, k) against tile dims (bm, bn, bk).  Caller must ensure
    /// shape is aligned to (BM, BN, BK).  `sm_count` drives the
    /// preferred persistent-grid size `G = sm_count * 2`.
    pub fn for_shape(m: u32, n: u32, k: u32, bm: u32, bn: u32, bk: u32, sm_count: u32) -> Self {
        let m_tiles = m / bm;
        let n_tiles = n / bn;
        let iters_per_tile = (k / bk).max(1);
        let total_tiles = m_tiles * n_tiles;
        let preferred_g = sm_count.max(1).saturating_mul(2);

        let (dp_tiles_total, tail_tiles) = if total_tiles < preferred_g {
            // Pure Stream-K: the shape doesn't have enough work for
            // the persistent grid to be a clean tiling.  Run all tiles
            // through the SK path.
            (0u32, total_tiles)
        } else {
            // Hybrid: clean waves go through DP, tail through SK.
            // When T % preferred_g == 0 this degenerates to pure DP
            // (tail_tiles = 0, g_sk = 0).
            let dp = (total_tiles / preferred_g) * preferred_g;
            (dp, total_tiles - dp)
        };

        let total_iters_sk = tail_tiles.saturating_mul(iters_per_tile);
        let g_sk = if total_iters_sk == 0 {
            0
        } else {
            // Use the full preferred persistent grid on the tail, even
            // if many workgroups end up sharing each tile.  The shader's
            // atomicAdd epilogue handles arbitrary fan-in correctly;
            // the two-tile clamp from v1 was overly conservative and
            // collapsed g_sk down to `tail_tiles` for square shapes,
            // erasing the wave-quantization savings.  Only floor at 1
            // to avoid a zero-division in the iters_per_wg_sk
            // computation.
            preferred_g.min(total_iters_sk.max(1)).max(1)
        };

        let (iters_per_wg_sk, rem_sk) = if g_sk == 0 {
            (0, 0)
        } else {
            let i = total_iters_sk / g_sk;
            let r = total_iters_sk - i * g_sk;
            (i, r)
        };

        let grid_total = dp_tiles_total + g_sk;

        Self {
            iters_per_tile,
            iters_per_wg_sk,
            rem_sk,
            n_tiles,
            dp_tiles_total,
            g_sk,
            total_iters_sk,
            grid_total,
        }
    }

    /// Returns true when the schedule has *no* Stream-K tail
    /// component.  In that case the dispatch is pure DP and callers
    /// can short-circuit to the regular `run_matmuls` path (or just
    /// dispatch this kernel; it's equivalent to DP for `g_sk == 0`).
    pub fn is_pure_dp(&self) -> bool {
        self.g_sk == 0
    }

    /// Returns true when the schedule has *no* DP-mode tiles — every
    /// workgroup is in the SK tail.  Used for the "tiny shape"
    /// regression where T <= preferred_g.
    pub fn is_pure_sk(&self) -> bool {
        self.dp_tiles_total == 0
    }
}

/// Default persistent-grid size for Stream-K (legacy / single-mode).
/// Target `sm_count * 2`, clamped down to preserve the two-tile
/// invariant (`iters_per_wg + 1 >= iters_per_tile`).
pub fn default_stream_k_grid(sm_count: u32, total_iters: u32, iters_per_tile: u32) -> u32 {
    let preferred = sm_count.max(1).saturating_mul(2);
    let max_g_two_tile = if iters_per_tile <= 1 {
        total_iters.max(1)
    } else {
        let raw = total_iters / iters_per_tile.saturating_sub(1).max(1);
        raw.max(1)
    };
    preferred.min(max_g_two_tile).min(total_iters.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_schedule_for_aligned_4096_cube() {
        // 4096^3 / 128x128/BK=32: m_tiles=32, n_tiles=32, T=1024.
        // G_pref=92.  dp_tiles_total = 11*92 = 1012.  tail = 12.
        // iters_per_tile=128; total_iters_sk = 12*128 = 1536.
        let s = StreamKSchedule::for_shape(4096, 4096, 4096, 128, 128, 32, 46);
        assert_eq!(s.iters_per_tile, 128);
        assert_eq!(s.n_tiles, 32);
        assert_eq!(s.dp_tiles_total, 1012);
        assert_eq!(s.g_sk, 92);
        assert_eq!(s.total_iters_sk, 12 * 128);
        assert_eq!(s.iters_per_wg_sk, 1536 / 92);
        assert_eq!(s.rem_sk, 1536 - (1536 / 92) * 92);
        assert_eq!(s.grid_total, 1012 + 92);
        // The two-tile clamp from v1 was removed: g_sk is sized to the
        // full preferred grid, so a tail tile may be touched by many
        // more than 2 WGs.  The shader's atomicAdd path handles this
        // correctly.
        assert!(!s.is_pure_dp());
        assert!(!s.is_pure_sk());
    }

    #[test]
    fn schedule_degenerates_to_pure_sk_for_small_shape() {
        // 512^3 / 128x128/BK=32: m_tiles=4, n_tiles=4, T=16,
        // iters_per_tile=16, total_iters=256.  T < G_pref=92 so all
        // tiles go through SK; g_sk is capped at total_iters_sk and
        // sized to spread work across the full preferred grid.
        let s = StreamKSchedule::for_shape(512, 512, 512, 128, 128, 32, 46);
        assert_eq!(s.dp_tiles_total, 0);
        assert_eq!(s.total_iters_sk, 256);
        // 92 < 256, so the preferred grid is honored.
        assert_eq!(s.g_sk, 92);
        assert!(s.is_pure_sk());
        assert!(!s.is_pure_dp());
    }

    #[test]
    fn schedule_rem_distributes_correctly_on_sk_tail() {
        let s = StreamKSchedule::for_shape(4096, 4096, 4096, 128, 128, 32, 46);
        assert_eq!(
            s.iters_per_wg_sk * s.g_sk + s.rem_sk,
            s.total_iters_sk
        );
        assert!(s.rem_sk < s.g_sk);
    }

    #[test]
    fn schedule_degenerates_to_pure_dp_when_tiles_clean_multiple_of_g() {
        // Construct a shape where T is exactly a multiple of G_pref=92.
        // T = 92 needs m_tiles*n_tiles = 92.  92 = 4*23, so use
        // M=128*4=512, N=128*23=2944, K=128.  iters_per_tile = 4.
        let s = StreamKSchedule::for_shape(512, 2944, 128, 128, 128, 32, 46);
        assert_eq!(s.dp_tiles_total, 92);
        assert_eq!(s.g_sk, 0);
        assert_eq!(s.total_iters_sk, 0);
        assert_eq!(s.grid_total, 92);
        assert!(s.is_pure_dp());
    }
}
