//! Two-stage split-K pipeline: partials to scratch + reduction kernel.
//!
//! The atomic split-K in `splitk.rs` pays `RED.E.ADD.F32` contention on
//! every output cell (measured 17-53% losses on K=256-1024 low-CTA
//! shapes).  This variant removes the atomics entirely:
//!
//!   * **Stage 1** (`matmul_splitk2_kernel.glsl`): one workgroup per
//!     (tile, batch, k_split) computes a partial C tile over its K
//!     window and plain-stores it into a scratch buffer of
//!     `num_splits` MxN planes per batch element.
//!   * **Stage 2** (`matmul_f32_splitk2_reduce.comp`): one thread per
//!     four output elements sums the planes into C.
//!
//! Costs one extra dispatch + `splits * M*N` scratch traffic; wins
//! when K is deep, M*N is small (so scratch is tiny and the DP grid
//! can't fill the device), and determinism matters — unlike the atomic
//! path, results are bit-stable across runs.
//!
//! Neither stage touches descriptor sets: both address buffers through
//! `buffer_reference` pointers in push constants.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::matmul::ResolvedMatmul;

const SPIRV_STAGE1_M128N128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_m128n128.spv"));
const SPIRV_STAGE1_M64N64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_m64n64.spv"));
const SPIRV_REDUCE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_reduce.spv"));

pub(super) fn stage1_dispatch_info(m: u32, n: u32) -> (&'static str, [u32; 3]) {
    if m >= 128 && n >= 128 {
        ("split_k2_m128n128", [128, 128, 32])
    } else {
        ("split_k2_m64n64", [64, 64, 32])
    }
}

/// Conservative untuned route for deep-K shapes whose data-parallel grid is
/// too small to fill the device. The two stage-1 tile families have different
/// occupancy targets; keep at least 64/128 K values per split so reduction
/// overhead never replaces useful inner-loop work.
pub(super) fn heuristic_splits(batch: u32, m: u32, n: u32, k: u32) -> Option<u32> {
    if batch == 0 || m <= 1 || n <= 1 || k < 1024 {
        return None;
    }
    let use_m128 = m >= 128 && n >= 128;
    let tile = if use_m128 { 128 } else { 64 };
    let tiles = u64::from(batch)
        .saturating_mul(u64::from(m.div_ceil(tile)))
        .saturating_mul(u64::from(n.div_ceil(tile)));
    let (target_wgs, min_k_per_split) = if use_m128 {
        let min_k = match tiles {
            1 => 1024,
            2..=4 => 2048,
            5..=16 => 4096,
            _ => return None,
        };
        if k < min_k {
            return None;
        }
        (if tiles >= 9 && k >= 8192 { 128u64 } else { 64 }, 128u32)
    } else {
        if tiles > 4 {
            return None;
        }
        (128u64, 64u32)
    };
    let wanted = target_wgs.div_ceil(tiles);
    let max_splits = if use_m128 && tiles == 1 {
        (k / min_k_per_split).max(16)
    } else {
        k / min_k_per_split
    };
    [64, 32, 16, 8, 4]
        .into_iter()
        .find(|&splits| u64::from(splits) <= wanted && splits <= max_splits)
}

/// Push constants for the reduce stage.  Bit-for-bit identical to the
/// GLSL `PC` block in `matmul_f32_splitk2_reduce.comp`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplitK2ReducePushConstants {
    pub total: u32,
    pub mn: u32,
    pub num_splits: u32,
    pub _pad: u32,
    pub scratch_ptr: u64,
    pub c_ptr: u64,
}

/// Threads per reduce workgroup (matches `local_size_x` in the shader).
pub(super) const REDUCE_WG_SIZE: u32 = 256;

/// Automatic routing stays below this per-submission scratch budget. Explicit
/// split-K2 calls remain free to request larger allocations.
pub(super) const AUTO_SCRATCH_CAP_BYTES: u64 = 256 << 20;

/// Resolve the automatic route without letting a heuristic overwrite a
/// measured DP winner (`Some(None)`).
pub(super) fn resolve_auto_splits(
    tuned: Option<Option<u32>>,
    heuristic: Option<u32>,
) -> Option<u32> {
    tuned.unwrap_or(heuristic)
}

/// Check every device/addressing limit used by an automatically selected
/// split-K2 route before allocating scratch or constructing a dispatch plan.
pub(super) fn auto_route_fits(
    max_groups: [u32; 3],
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    splits: u32,
) -> bool {
    if batch == 0 || m == 0 || n == 0 || !(2..=0xFFFF).contains(&splits) || splits > k {
        return false;
    }
    let (_, [tile_m, tile_n, _]) = stage1_dispatch_info(m, n);
    let Some(total) = u64::from(batch)
        .checked_mul(u64::from(m))
        .and_then(|v| v.checked_mul(u64::from(n)))
    else {
        return false;
    };
    let Some(scratch_elements) = total.checked_mul(u64::from(splits)) else {
        return false;
    };
    let Some(scratch_bytes) = scratch_elements.checked_mul(4) else {
        return false;
    };
    let Some(gz) = batch.checked_mul(splits) else {
        return false;
    };
    n.div_ceil(tile_n) <= max_groups[0]
        && m.div_ceil(tile_m) <= max_groups[1]
        && gz <= max_groups[2]
        && total.div_ceil(4 * u64::from(REDUCE_WG_SIZE)) <= u64::from(max_groups[0])
        && scratch_elements <= u64::from(u32::MAX) + 1
        && scratch_bytes <= AUTO_SCRATCH_CAP_BYTES
}

/// Reserve an aligned region inside the aggregate automatic graph budget.
pub(super) fn checked_auto_scratch_end(offset: u64, bytes: u64) -> Option<u64> {
    let end = offset.checked_add(bytes.checked_next_multiple_of(16)?)?;
    (end <= AUTO_SCRATCH_CAP_BYTES).then_some(end)
}

/// Validated dispatch geometry + scratch requirement for one split-K2
/// execution.  Computing this up front lets both the standalone entry
/// point and the graph recorder share the same checks.
#[derive(Copy, Clone, Debug)]
pub(super) struct SplitK2Dispatch {
    pub(super) splits: u32,
    pub(super) gx: u32,
    pub(super) gy: u32,
    pub(super) gz: u32,
    pub(super) mn: u32,
    pub(super) total: u32,
    pub(super) reduce_groups: u32,
    /// Stage-1 partial planes: `batch * splits * mn * 4` bytes.
    pub(super) scratch_bytes: u64,
    use_m128: bool,
}

impl SplitK2Dispatch {
    pub(super) fn plan(
        ctx: &VulkanContext,
        pipeline: &SplitK2Pipeline,
        resolved: &ResolvedMatmul,
        splits: u32,
    ) -> Result<Self> {
        if !(2..=0xFFFF).contains(&splits) {
            bail!("split-K2: splits={splits} out of range [2, 65535]");
        }
        validate_scratch_geometry(resolved.batch, resolved.m, resolved.n, resolved.k, splits)?;
        let kernel = pipeline.pick_stage1(resolved.m, resolved.n);
        let max_groups = ctx.device_properties.limits.max_compute_work_group_count;
        let gx = resolved.n.div_ceil(kernel.tile_n);
        let gy = resolved.m.div_ceil(kernel.tile_m);
        let gz = resolved
            .batch
            .checked_mul(splits)
            .ok_or_else(|| anyhow!("split-K2 dispatch grid Z overflows u32"))?;
        if gx > max_groups[0] || gy > max_groups[1] || gz > max_groups[2] {
            bail!(
                "split-K2 dispatch ({gx}, {gy}, {gz}) exceeds device \
                 maxComputeWorkGroupCount ({}, {}, {})",
                max_groups[0],
                max_groups[1],
                max_groups[2],
            );
        }
        let mn = resolved.m as u64 * resolved.n as u64;
        let total = mn
            .checked_mul(resolved.batch as u64)
            .ok_or_else(|| anyhow!("split-K2 output element count overflows"))?;
        if total > u32::MAX as u64 {
            bail!("split-K2: batch*M*N = {total} exceeds u32 addressing");
        }
        let scratch_elements = total * splits as u64;
        let scratch_bytes = scratch_elements
            .checked_mul(4)
            .ok_or_else(|| anyhow!("split-K2 scratch size overflows"))?;
        let reduce_groups = (total as u32).div_ceil(4 * REDUCE_WG_SIZE);
        if reduce_groups > max_groups[0] {
            bail!(
                "split-K2 reduce dispatch ({reduce_groups}) exceeds device \
                 maxComputeWorkGroupCount[0]"
            );
        }
        Ok(Self {
            splits,
            gx,
            gy,
            gz,
            mn: mn as u32,
            total: total as u32,
            reduce_groups,
            scratch_bytes,
            use_m128: resolved.m >= 128 && resolved.n >= 128,
        })
    }
}

fn validate_scratch_geometry(batch: u32, m: u32, n: u32, k: u32, splits: u32) -> Result<()> {
    if splits > k {
        bail!("split-K2: splits={splits} exceeds K={k}");
    }
    let elements = [batch, m, n, splits]
        .into_iter()
        .try_fold(1u64, |acc, value| acc.checked_mul(value as u64))
        .ok_or_else(|| anyhow!("split-K2 scratch size overflows"))?;
    // Stage 1 forms `split * mn + index` in uint. The element count may be
    // exactly 2^32 because the largest zero-based offset is u32::MAX.
    if elements > u32::MAX as u64 + 1 {
        bail!("split-K2: batch*M*N*splits = {elements} exceeds u32 scratch addressing");
    }
    Ok(())
}

/// Record both split-K2 stages (with the internal stage1→stage2
/// barrier) into `cb`.  `scratch_addr` must point at a 16-byte-aligned
/// device region of at least `plan.scratch_bytes` bytes.
///
/// Because the internal barrier is a *global* compute→compute
/// execution+memory barrier, all writes recorded before this call are
/// also visible to stage 2 and everything after it — callers tracking
/// hazards can treat this as a full flush point.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_split_k2_commands(
    ctx: &VulkanContext,
    pipeline: &SplitK2Pipeline,
    cb: vk::CommandBuffer,
    alpha: f32,
    resolved: &ResolvedMatmul,
    plan: &SplitK2Dispatch,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    scratch_addr: u64,
) {
    let stage1 = if plan.use_m128 {
        &pipeline.m128n128
    } else {
        &pipeline.m64n64
    };
    let stage1_pc = resolved.split_k_push_constants(alpha, a_ptr, b_ptr, scratch_addr, plan.splits);
    let reduce_pc = SplitK2ReducePushConstants {
        total: plan.total,
        mn: plan.mn,
        num_splits: plan.splits,
        _pad: 0,
        scratch_ptr: scratch_addr,
        c_ptr,
    };
    unsafe {
        let dev = &ctx.device;
        dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, stage1.pipeline);
        dev.cmd_push_constants(
            cb,
            pipeline.stage1_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&stage1_pc),
        );
        dev.cmd_dispatch(cb, plan.gx, plan.gy, plan.gz);

        // Scratch writes must land before the reducer reads.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );

        dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.reduce.pipeline);
        dev.cmd_push_constants(
            cb,
            pipeline.reduce_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&reduce_pc),
        );
        dev.cmd_dispatch(cb, plan.reduce_groups, 1, 1);
    }
}

pub(super) struct SplitK2Kernel {
    pub(super) tile_m: u32,
    pub(super) tile_n: u32,
    shader_module: vk::ShaderModule,
    pub(super) pipeline: vk::Pipeline,
}

pub(super) struct SplitK2Pipeline {
    ctx: Arc<VulkanContext>,
    /// Push-constant-only layout for stage 1 (standard
    /// `MatmulPushConstants`, `c_ptr` pointing at scratch).
    pub(super) stage1_layout: vk::PipelineLayout,
    /// Push-constant-only layout for the reducer.
    pub(super) reduce_layout: vk::PipelineLayout,
    pub(super) m128n128: SplitK2Kernel,
    pub(super) m64n64: SplitK2Kernel,
    pub(super) reduce: SplitK2Kernel,
}

impl SplitK2Pipeline {
    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        unsafe {
            let stage1_layout = create_pc_only_layout(
                ctx,
                std::mem::size_of::<crate::pipeline::MatmulPushConstants>() as u32,
            )
            .context("split-K2 stage-1 pipeline layout")?;
            let reduce_layout = create_pc_only_layout(
                ctx,
                std::mem::size_of::<SplitK2ReducePushConstants>() as u32,
            )
            .context("split-K2 reduce pipeline layout")?;

            let build = |tile_m, tile_n, spv| -> Result<SplitK2Kernel> {
                build_kernel(
                    ctx,
                    if tile_m == 0 {
                        reduce_layout
                    } else {
                        stage1_layout
                    },
                    tile_m,
                    tile_n,
                    spv,
                )
            };

            let m128n128 = build(128, 128, SPIRV_STAGE1_M128N128)
                .context("build split-K2 128x128 stage-1 kernel")?;
            let m64n64 = build(64, 64, SPIRV_STAGE1_M64N64)
                .context("build split-K2 64x64 stage-1 kernel")?;
            let reduce = build(0, 0, SPIRV_REDUCE).context("build split-K2 reduce kernel")?;

            Ok(Self {
                ctx: Arc::clone(ctx),
                stage1_layout,
                reduce_layout,
                m128n128,
                m64n64,
                reduce,
            })
        }
    }

    /// Stage-1 tile choice: 128x128 when both dims fill it, else 64x64.
    pub(super) fn pick_stage1(&self, m: u32, n: u32) -> &SplitK2Kernel {
        if m >= 128 && n >= 128 {
            &self.m128n128
        } else {
            &self.m64n64
        }
    }
}

unsafe fn create_pc_only_layout(
    ctx: &Arc<VulkanContext>,
    pc_size: u32,
) -> Result<vk::PipelineLayout> {
    let pc_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(pc_size)];
    unsafe {
        ctx.device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_ranges),
                None,
            )
            .context("create_pipeline_layout (split-K2)")
    }
}

fn build_kernel(
    ctx: &Arc<VulkanContext>,
    layout: vk::PipelineLayout,
    tile_m: u32,
    tile_n: u32,
    spv: &[u8],
) -> Result<SplitK2Kernel> {
    assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .context("create_shader_module (split-K2)")?;
        let entry = std::ffi::CString::new("main").unwrap();

        // Stage 1 reuses the standard spec-constant slots with neutral
        // values (alpha handled dynamically: ALPHA_IS_ONE=false keeps
        // the multiply, which folds to *1.0 anyway when alpha is 1).
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
        let spec_data: [u32; 4] = [0, 0, 0, 0];
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
            .layout(layout)];

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
                    "create_compute_pipelines split-K2 {tile_m}x{tile_n}: {err}"
                ));
            }
        };

        Ok(SplitK2Kernel {
            tile_m,
            tile_n,
            shader_module,
            pipeline: pipelines[0],
        })
    }
}

impl Drop for SplitK2Pipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [&self.m128n128, &self.m64n64, &self.reduce] {
                if kernel.pipeline != vk::Pipeline::null() {
                    self.ctx.device.destroy_pipeline(kernel.pipeline, None);
                }
                if kernel.shader_module != vk::ShaderModule::null() {
                    self.ctx
                        .device
                        .destroy_shader_module(kernel.shader_module, None);
                }
            }
            self.ctx
                .device
                .destroy_pipeline_layout(self.stage1_layout, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.reduce_layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AUTO_SCRATCH_CAP_BYTES, auto_route_fits, checked_auto_scratch_end, heuristic_splits,
        resolve_auto_splits, stage1_dispatch_info, validate_scratch_geometry,
    };

    #[test]
    fn scratch_geometry_rejects_empty_splits_and_uint_wrap() {
        assert!(validate_scratch_geometry(1, 64, 64, 64, 65).is_err());
        assert!(validate_scratch_geometry(1, 65_536, 65_536, 2, 2).is_err());
        assert!(validate_scratch_geometry(1, 32_768, 65_536, 2, 2).is_ok());
    }

    #[test]
    fn dispatch_info_matches_stage1_tile_choice() {
        assert_eq!(stage1_dispatch_info(128, 128).1, [128, 128, 32]);
        assert_eq!(stage1_dispatch_info(127, 128).1, [64, 64, 32]);
    }

    #[test]
    fn heuristic_targets_stage_occupancy_without_short_k_splits() {
        let cases = [
            ((1, 64, 64, 1024), Some(16)),
            ((1, 64, 64, 2048), Some(32)),
            ((1, 64, 128, 8192), Some(64)),
            ((4, 64, 64, 4096), Some(32)),
            ((1, 128, 128, 1024), Some(16)),
            ((1, 256, 256, 1024), None),
            ((1, 256, 256, 2048), Some(16)),
            ((1, 512, 512, 2048), None),
            ((1, 512, 512, 4096), Some(4)),
            ((1, 512, 512, 8192), Some(8)),
            ((1, 1024, 1024, 8192), None),
            ((1, 1, 4096, 8192), None),
            ((0, 64, 64, 1024), None),
        ];
        for ((batch, m, n, k), expected) in cases {
            assert_eq!(
                heuristic_splits(batch, m, n, k),
                expected,
                "{batch}x{m}x{n}x{k}"
            );
        }
    }

    #[test]
    fn measured_route_or_dp_winner_precedes_heuristic() {
        assert_eq!(resolve_auto_splits(Some(Some(8)), Some(64)), Some(8));
        assert_eq!(resolve_auto_splits(Some(None), Some(64)), None);
        assert_eq!(resolve_auto_splits(None, Some(64)), Some(64));
        assert_eq!(resolve_auto_splits(None, None), None);
    }

    #[test]
    fn auto_route_checks_grid_addressing_and_scratch_limits() {
        let ample = [u32::MAX; 3];
        assert!(auto_route_fits(ample, 1, 64, 64, 1024, 16));
        assert!(!auto_route_fits(ample, 0, 64, 64, 1024, 16));
        assert!(!auto_route_fits(ample, 1, 64, 64, 1024, 1));
        assert!(!auto_route_fits(ample, 1, 64, 64, 65_536, 65_536));
        assert!(!auto_route_fits(ample, 1, 64, 64, 8, 16));
        assert!(!auto_route_fits([1, 1, 15], 1, 64, 64, 1024, 16));
        assert!(!auto_route_fits([1, 1, u32::MAX], 1, 65, 64, 1024, 16));
        assert!(!auto_route_fits(
            [1, u32::MAX, u32::MAX],
            1,
            64,
            65,
            1024,
            16
        ));
        assert!(!auto_route_fits(
            [1, u32::MAX, u32::MAX],
            1,
            1025,
            64,
            1024,
            16
        ));
        assert!(!auto_route_fits(
            ample,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            2
        ));
        // 4 * 1024 * 1024 * 64 == the 256 MiB automatic budget.
        assert!(auto_route_fits(ample, 1, 1024, 1024, 4096, 64));
        assert!(!auto_route_fits(ample, 1, 1024, 1025, 4096, 64));
    }

    #[test]
    fn graph_scratch_reservation_is_aligned_checked_and_capped() {
        assert_eq!(checked_auto_scratch_end(0, 1), Some(16));
        assert_eq!(
            checked_auto_scratch_end(AUTO_SCRATCH_CAP_BYTES - 16, 16),
            Some(AUTO_SCRATCH_CAP_BYTES)
        );
        assert_eq!(checked_auto_scratch_end(AUTO_SCRATCH_CAP_BYTES, 1), None);
        assert_eq!(checked_auto_scratch_end(u64::MAX - 7, 8), None);
        assert_eq!(checked_auto_scratch_end(0, u64::MAX), None);
    }
}
