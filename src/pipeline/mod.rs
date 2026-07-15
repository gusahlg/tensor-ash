//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

mod create;
mod selection;
mod tuning;
mod types;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use parking_lot::{Mutex, RwLock};
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

use create::{create_epilogue_pipeline, create_kernel, destroy_kernel};
use selection::{auto_min_large_tiles_for, auto_select_kernel};
use tuning::{load_tuned, save_tuned, shader_registry_hash};

pub use tuning::{TuneEntry, TuneKey};
pub use types::{
    EpilogueKey, KERNEL_SPECS, KernelSelection, KernelSpec, KernelVariant, MatmulKernel,
    MatmulPushConstants,
};

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    /// Pipeline layout for the descriptor-based matmul kernels (3
    /// STORAGE_BUFFER bindings + push constants).
    pub pipeline_layout: vk::PipelineLayout,
    /// Pipeline layout for the BDA matmul kernels (push constants
    /// only — A/B/C live as `buffer_reference` pointers inside the
    /// push constants, so no descriptor set is required).  The BDA
    /// dispatch path skips `vkUpdateDescriptorSets` and
    /// `vkCmdBindDescriptorSets` entirely.
    pub pipeline_layout_bda: vk::PipelineLayout,
    /// One `MatmulKernel` per entry in `KERNEL_SPECS`, in the same order.
    /// Index with `KernelSelection::index()` (after resolving `Auto`).
    kernels: Vec<MatmulKernel>,
    /// Lazily-built pipelines for non-zero epilogue specializations,
    /// keyed by (kernel name, base-variant index, epilogue key).  The
    /// eager `variants` arrays only cover the zero epilogue; real
    /// workloads use a handful of epilogue combos, so building them on
    /// first use (against the persistent pipeline cache) keeps startup
    /// flat.
    epilogue_pipelines: Mutex<HashMap<(&'static str, usize, EpilogueKey), vk::Pipeline>>,
    selection: KernelSelection,
    /// Minimum number of large-kernel tiles a problem must produce
    /// before the auto-selector prefers the large kernel.  Below this,
    /// the small kernel's 4x more workgroups give better device
    /// occupancy.  Derived from device kind at pipeline build time.
    auto_min_large_tiles: u64,
    /// Measured per-shape winners (see `tuning.rs`).  Consulted before
    /// the static heuristic whenever `selection` is `Auto`; explicit
    /// `ML_KERNEL=` picks bypass it entirely.  Loaded from the
    /// persistent store at build time; the executor's tuner inserts new
    /// winners at runtime.
    tuned: RwLock<HashMap<TuneKey, TuneEntry>>,
    /// Hash binding persisted winners to this exact shader build.
    shader_hash: u64,
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

            // BDA layout: push-constant-only, no descriptor set.  The
            // BDA shaders address A/B/C through `buffer_reference`
            // pointers in the push constants and never bind any
            // descriptor set, so the dispatcher can skip
            // `vkCmdBindDescriptorSets` and `vkUpdateDescriptorSets`
            // entirely for these kernels.
            let pipeline_layout_bda = ctx
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_ranges),
                    None,
                )
                .context("create_pipeline_layout (BDA)")?;
            let pipeline_layout_bda_guard = scopeguard::guard(pipeline_layout_bda, |l| {
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
                let layout = if spec.uses_descriptors {
                    pipeline_layout
                } else {
                    pipeline_layout_bda
                };
                let kernel = create_kernel(
                    ctx,
                    layout,
                    spec.name,
                    spec.tile_m,
                    spec.tile_n,
                    spec.tile_k,
                    spec.spv,
                    spec.uses_descriptors,
                )
                .with_context(|| format!("create {} matmul kernel", spec.name))?;
                kernels_guard.push(kernel);
            }
            // Disarm the cleanup: kernels now belong to the pipeline.
            let kernels = ScopeGuard::into_inner(kernels_guard);

            let shader_hash = shader_registry_hash();
            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout: ScopeGuard::into_inner(set_layout_guard),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                pipeline_layout_bda: ScopeGuard::into_inner(pipeline_layout_bda_guard),
                kernels,
                epilogue_pipelines: Mutex::new(HashMap::new()),
                selection,
                auto_min_large_tiles: auto_min_large_tiles_for(ctx.device_kind()),
                tuned: RwLock::new(load_tuned(ctx, shader_hash)),
                shader_hash,
            })
        }
    }

    pub fn select_kernel(&self, batch: u32, m: u32, n: u32, k: u32) -> &MatmulKernel {
        &self.kernels[self.select_kernel_index(batch, m, n, k)]
    }

    /// Index into `KERNEL_SPECS` for this problem.  Resolution order:
    /// explicit `ML_KERNEL=` selection > measured tuned winner >
    /// static shape heuristic.
    pub fn select_kernel_index(&self, batch: u32, m: u32, n: u32, k: u32) -> usize {
        match self.selection {
            KernelSelection::Auto => {
                if let Some(entry) = self.tuned.read().get(&TuneKey { batch, m, n, k }) {
                    return entry.kernel;
                }
                self.heuristic_kernel_index(batch, m, n, k)
            }
            explicit => explicit
                .index()
                .expect("explicit selection has a concrete kernel"),
        }
    }

    /// Measured split-K2 routing for this shape: `Some(splits)` when
    /// the two-stage split-K beat every DP kernel during tuning.  Only
    /// meaningful for single plain calls; batched / accumulate /
    /// epilogue dispatches use the DP winner instead.
    pub fn tuned_splitk2(&self, batch: u32, m: u32, n: u32, k: u32) -> Option<u32> {
        if !self.is_auto() {
            return None;
        }
        self.tuned
            .read()
            .get(&TuneKey { batch, m, n, k })
            .and_then(|entry| entry.splitk2_splits)
    }

    /// The static shape heuristic's pick, ignoring any tuned winner.
    /// Serves as the tuner's prior and as the fallback for untuned
    /// shapes.
    pub fn heuristic_kernel_index(&self, batch: u32, m: u32, n: u32, k: u32) -> usize {
        let tile = auto_select_kernel(batch, m, n, k, self.auto_min_large_tiles);
        let selection = if self.ctx.buffer_device_address_enabled {
            maybe_to_bda(tile)
        } else {
            tile
        };
        // No aligned-variant promotion: empirically the source-
        // stripped `*_bda_v4_aligned` kernels measure 2-5%
        // slower than the spec-const-folded `*_bda_v4` kernels
        // (validated on 768^3, 1024^3, 2048^3, 4096^3).  They
        // remain selectable via `ML_KERNEL=large_bda_v4_aligned`
        // for explicit experimentation.
        selection
            .index()
            .expect("auto selection must resolve to a concrete kernel")
    }

    /// True when the auto selector is active (no explicit `ML_KERNEL`)
    /// — the only mode in which measured tuning applies.
    #[inline]
    pub fn is_auto(&self) -> bool {
        self.selection == KernelSelection::Auto
    }

    /// Whether `shape` already has a measured winner (or tuning is
    /// moot because selection is explicit).
    pub fn is_tuned(&self, key: TuneKey) -> bool {
        !self.is_auto() || self.tuned.read().contains_key(&key)
    }

    /// Kernel indices the tuner should measure: every BDA-family
    /// kernel that handles arbitrary shapes (bounds-checked bodies
    /// only — the strict `*_aligned` variants are excluded).  Requires
    /// `bufferDeviceAddress`; on devices without it the tuner is
    /// disabled and the heuristic stands.
    pub fn tune_candidate_indices(&self) -> Vec<usize> {
        if !self.ctx.buffer_device_address_enabled {
            return Vec::new();
        }
        self.kernels
            .iter()
            .enumerate()
            .filter(|(_, kernel)| !kernel.uses_descriptors && !kernel.name.ends_with("_aligned"))
            .map(|(idx, _)| idx)
            .collect()
    }

    #[inline]
    pub fn kernel_at(&self, idx: usize) -> &MatmulKernel {
        &self.kernels[idx]
    }

    /// Record a measured winner and persist the store.
    pub fn record_tuned(&self, key: TuneKey, entry: TuneEntry) {
        let snapshot = {
            let mut tuned = self.tuned.write();
            tuned.insert(key, entry);
            tuned.clone()
        };
        save_tuned(&self.ctx, self.shader_hash, &snapshot);
    }

    /// Pipeline for a (kernel, base-variant, epilogue) triple.  The
    /// zero epilogue resolves to the eagerly-built variant table; any
    /// other combination is compiled on first use and cached for the
    /// lifetime of the pipeline.  Compilation goes through the
    /// persistent `VkPipelineCache`, so repeat processes skip the
    /// ISA compile.
    pub fn pipeline_for_epilogue(
        &self,
        kernel: &MatmulKernel,
        variant: KernelVariant,
        epilogue: EpilogueKey,
    ) -> Result<vk::Pipeline> {
        if epilogue.is_none() {
            return Ok(kernel.pipeline_for(variant));
        }
        let key = (kernel.name, variant.index(), epilogue);
        let mut cache = self.epilogue_pipelines.lock();
        if let Some(&p) = cache.get(&key) {
            return Ok(p);
        }
        let pipeline = create_epilogue_pipeline(&self.ctx, kernel, variant, epilogue)?;
        cache.insert(key, pipeline);
        Ok(pipeline)
    }
}

/// Promote a tile choice from `auto_select_kernel` to its
/// buffer_reference (LDG.128) sibling when the device supports
/// `bufferDeviceAddress`.  The BDA variants beat the descriptor-based
/// ones by roughly 5-15% on every shape we benchmarked, so when the
/// feature is present the auto-selector should always pick the BDA
/// path.  Explicit selections (`ML_KERNEL=large`, etc.) are honored
/// verbatim so per-kernel tuning still works.
/// The V4 (uvec4 shared) BDA variants beat the plain BDA variants by
/// another 5-15% on every TN>=4 tile we measured.  For the TN=2
/// `m64n32` kernel the V4 path isn't available (LDS.128 over a 2-col
/// stride is non-sensical), so we fall back to the plain BDA sibling.
fn maybe_to_bda(tile: KernelSelection) -> KernelSelection {
    match tile {
        KernelSelection::Large => KernelSelection::LargeBdaV4,
        KernelSelection::Small => KernelSelection::SmallBdaV4,
        KernelSelection::M128N64K64 => KernelSelection::M128N64K64BdaV4,
        // K64 promotes to the TM=8 TN=4 register-tile variant: empirically
        // wins +6-7% on every K64-routed shape (medium_384,
        // skinny_1024x128x512, wide_128x1024x512) vs the TM=4 TN=4 default
        // by trading half the active threads for double the M-side
        // register strip per thread (verified with 10-round interleaved
        // A/B on RTX 3070).  The plain V4 sibling stays selectable via
        // ML_KERNEL=k64_bda_v4 for back-comparison.
        KernelSelection::K64 => KernelSelection::K64BdaV4Tm8Tn4,
        KernelSelection::M128N64 => KernelSelection::M128N64BdaV4,
        KernelSelection::M64N128 => KernelSelection::M64N128BdaV4,
        // TN=2 has no V4 path; use the plain BDA fallback.
        KernelSelection::M64N32 => KernelSelection::M64N32Bda,
        other => other,
    }
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for (_, pipeline) in self.epilogue_pipelines.lock().drain() {
                if pipeline != vk::Pipeline::null() {
                    self.ctx.device.destroy_pipeline(pipeline, None);
                }
            }
            for kernel in self.kernels.drain(..) {
                destroy_kernel(&self.ctx, kernel);
            }
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout_bda, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod bda_tests {
    use super::*;

    #[test]
    fn bda_promotion_covers_every_auto_target() {
        // The auto-selector's possible returns should each promote to
        // their BDA_V4 sibling (or BDA for TN=2 / unchanged when no
        // BDA path exists).  Listing them explicitly here keeps the
        // rule from silently drifting when new kernels land.
        assert_eq!(
            maybe_to_bda(KernelSelection::Large),
            KernelSelection::LargeBdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::Small),
            KernelSelection::SmallBdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M128N64K64),
            KernelSelection::M128N64K64BdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::K64),
            KernelSelection::K64BdaV4Tm8Tn4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M128N64),
            KernelSelection::M128N64BdaV4
        );
        assert_eq!(
            maybe_to_bda(KernelSelection::M64N128),
            KernelSelection::M64N128BdaV4
        );
        // TN=2 stays on the plain BDA path (no V4 sibling).
        assert_eq!(
            maybe_to_bda(KernelSelection::M64N32),
            KernelSelection::M64N32Bda
        );
        // No BDA sibling at all — pass through.
        assert_eq!(maybe_to_bda(KernelSelection::Bk16), KernelSelection::Bk16);
        assert_eq!(maybe_to_bda(KernelSelection::V2), KernelSelection::V2);
    }
}
