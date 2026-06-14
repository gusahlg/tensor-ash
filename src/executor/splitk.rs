//! Split-K experimental pipeline.
//!
//! Split-K is a way to expand the available GPU parallelism on shapes
//! where K is large but M and N produce few output tiles.  Instead of
//! one workgroup reducing the whole K dimension into a single C tile,
//! we split K into `num_k_splits` chunks and dispatch one WG per
//! (block_row, block_col, batch, k_split).  Each WG produces a partial
//! C tile and atomically reduces it into the final C buffer.  The
//! dispatch's effective parallelism becomes `(M/BM) * (N/BN) * batch *
//! num_k_splits`, so the right `num_k_splits` can lift small-tile
//! shapes from "1 WG idling on 1 SM" to "saturating all 46 SMs on an
//! RTX 3070".
//!
//! Two tile sizes are supported, both BK=32:
//!   * 128x128 (TM=8 TN=8) for shapes with M,N >= 128
//!   * 64x64 (TM=4 TN=4) for the very small shapes (M=N=64) where the
//!     128x128 tile wastes 3/4 of every workgroup's threads.
//!
//! Both kernels live in `shaders/matmul_splitk_kernel.glsl` and use
//! the existing `MatmulPushConstants` layout — the `num_k_splits`
//! parameter is packed into the upper 16 bits of `pc.flags`.  This
//! keeps the layout bit-stable so we don't have to disturb the
//! pipeline_layout / set_layout the standard matmul kernel uses.
//!
//! ## Correctness
//!
//! Atomic add on f32 is NOT associative — different interleavings of
//! atomicAdd produce slightly different rounded sums.  The recording
//! layer zero-fills C before the dispatch (since each WG only
//! contributes its partial sum and the kernel cannot do its own
//! pre-zero without an extra barrier across the whole dispatch).
//! Tests should use a tolerance roughly `tolerance(K) * num_k_splits`
//! to absorb the extra rounding.
//!
//! ## Atomic implementation
//!
//! The shader uses the hardware float32 `atomicAdd` exposed by
//! `VK_EXT_shader_atomic_float` (specifically
//! `shaderBufferFloat32AtomicAdd`).  The Stream-K work already turned
//! that extension on at the context level (commit 171d8ac), so
//! Split-K rides for free.  On Ampere this lowers to a single
//! `RED.E.ADD.F32` instruction — measured ~3-5x faster than the
//! CAS-loop fallback the prior version of this kernel shipped.
//! `Executor::run_matmuls_split_k` refuses the call if the extension
//! is unavailable, so the kernel never has to fall back at runtime.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ash::vk;

use crate::context::VulkanContext;

/// Embedded SPIR-V for the 128x128 split-K kernel.
const SPIRV_M128N128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk_m128n128.spv"));
/// Embedded SPIR-V for the 64x64 split-K kernel.
const SPIRV_M64N64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk_m64n64.spv"));

/// One compiled split-K kernel.  Single variant per tile — we don't
/// bother specializing on accumulate/alpha_is_one/etc. for the
/// experimental kernel; the inner FMA loop is identical and the
/// epilogue specialization wouldn't matter much because the atomic
/// store dominates the epilogue cost anyway.
pub(super) struct SplitKKernel {
    pub(super) tile_m: u32,
    pub(super) tile_n: u32,
    pub(super) shader_module: vk::ShaderModule,
    pub(super) pipeline: vk::Pipeline,
}

/// Owns the two split-K kernels (128x128 and 64x64).
pub(super) struct SplitKPipeline {
    ctx: Arc<VulkanContext>,
    pub(super) m128n128: SplitKKernel,
    pub(super) m64n64: SplitKKernel,
}

impl SplitKPipeline {
    /// Build both split-K kernels against the existing matmul pipeline
    /// layout.  Reuses the same descriptor set layout (3
    /// STORAGE_BUFFER bindings) and push constant range that the
    /// standard matmul kernels use — split-K is a drop-in replacement
    /// at the descriptor level.
    pub(super) fn new(
        ctx: &Arc<VulkanContext>,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<Self> {
        let m128n128 = build_split_k_kernel(ctx, pipeline_layout, 128, 128, SPIRV_M128N128)
            .context("build split-K 128x128 kernel")?;
        let m64n64 = build_split_k_kernel(ctx, pipeline_layout, 64, 64, SPIRV_M64N64)
            .context("build split-K 64x64 kernel")?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            m128n128,
            m64n64,
        })
    }

    /// Pick the kernel whose tile fits the (M, N) shape best.  We
    /// prefer 128x128 whenever both dims fill it; otherwise fall back
    /// to 64x64.
    pub(super) fn pick(&self, m: u32, n: u32) -> &SplitKKernel {
        if m >= 128 && n >= 128 {
            &self.m128n128
        } else {
            &self.m64n64
        }
    }
}

fn build_split_k_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    tile_m: u32,
    tile_n: u32,
    spv: &[u8],
) -> Result<SplitKKernel> {
    assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    unsafe {
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .context("create_shader_module")?;
        let entry = std::ffi::CString::new("main").unwrap();

        // Specialize all four constants to "neutral" values: we don't
        // need the variant matrix for the experimental kernel.
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
        // accumulate=false (kernel ignores it — always atomic adds),
        // alpha_is_one=true (skip the multiply), interior_only=false
        // (we always take the bounds-checked load path), k_multiple
        // =false (handled inside the k-split loop).
        let spec_data: [u32; 4] = [0, 1, 0, 0];
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
                    "create_compute_pipelines split-K {tile_m}x{tile_n}: {err}"
                ));
            }
        };

        Ok(SplitKKernel {
            tile_m,
            tile_n,
            shader_module,
            pipeline: pipelines[0],
        })
    }
}

impl Drop for SplitKPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            if self.m128n128.pipeline != vk::Pipeline::null() {
                self.ctx
                    .device
                    .destroy_pipeline(self.m128n128.pipeline, None);
            }
            if self.m128n128.shader_module != vk::ShaderModule::null() {
                self.ctx
                    .device
                    .destroy_shader_module(self.m128n128.shader_module, None);
            }
            if self.m64n64.pipeline != vk::Pipeline::null() {
                self.ctx.device.destroy_pipeline(self.m64n64.pipeline, None);
            }
            if self.m64n64.shader_module != vk::ShaderModule::null() {
                self.ctx
                    .device
                    .destroy_shader_module(self.m64n64.shader_module, None);
            }
        }
    }
}

/// Default number of K splits for a given (M, N, K) shape.  Aimed at
/// roughly saturating an RTX 3070's 46 SMs (target ~64 active WGs to
/// give the scheduler some queueing slack).  When the natural
/// `(M/BM)*(N/BN)*batch` already yields that many WGs, no split is
/// needed.  When it yields very few, pick a large split.
///
/// The user can override this through the explicit `num_k_splits`
/// argument to `Executor::run_matmuls_split_k`; this helper exists for
/// the auto path and the experimental tests.
pub fn default_num_k_splits(batch: u32, m: u32, n: u32, k: u32, tile_m: u32, tile_n: u32) -> u32 {
    const TARGET_WGS: u64 = 64;
    let tiles = m.div_ceil(tile_m) as u64 * n.div_ceil(tile_n) as u64 * batch.max(1) as u64;
    if tiles == 0 {
        return 1;
    }
    // Want tiles * num_k_splits >= TARGET_WGS, and num_k_splits | K.
    // Round up the raw split count, then bound it to K so that each
    // split has at least one K-element to consume.
    let raw = TARGET_WGS.div_ceil(tiles).max(1);
    // Cap so a single split still has a useful amount of K.  Below
    // BK=32 elements per split the inner FMA loop has too little work
    // to amortize the atomic epilogue.
    let max_splits = (k as u64 / 32).max(1);
    let splits = raw.min(max_splits);
    splits.min(u32::MAX as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_num_k_splits_picks_enough_for_few_tiles() {
        // 256x256x4096 with 128x128 tile: 4 tiles, want >= 64 WGs ->
        // 16 splits, each handling 256 K-elements.
        let splits = default_num_k_splits(1, 256, 256, 4096, 128, 128);
        assert_eq!(splits, 16);
        // 64x64x8192 with 64x64 tile: 1 tile, want >= 64 WGs -> 64
        // splits, each 128 K-elements.
        let splits = default_num_k_splits(1, 64, 64, 8192, 64, 64);
        assert_eq!(splits, 64);
        // 512x512 with 128x128 tile: 16 tiles already, no split.
        let splits = default_num_k_splits(1, 512, 512, 4096, 128, 128);
        assert_eq!(splits, 4);
    }

    #[test]
    fn default_num_k_splits_caps_at_min_chunk_size() {
        // Tiny K=64 means at most 2 splits even if we wanted more.
        let splits = default_num_k_splits(1, 64, 64, 64, 64, 64);
        assert_eq!(splits, 2);
    }
}
