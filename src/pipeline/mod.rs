//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

mod abi;
mod catalog;
mod create;
mod runtime;
mod selection;
mod tuning;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;
use parking_lot::{Mutex, RwLock};
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

use create::{create_epilogue_pipeline, create_kernel, destroy_kernel};
use selection::{auto_min_large_tiles_for, auto_select_kernel};
use tuning::{load_tuned, save_tuned, shader_registry_hash};

pub use abi::{EpilogueKey, KernelVariant, MatmulPushConstants};
pub use catalog::{KERNEL_SPECS, KernelSelection, KernelSpec};
pub(crate) use create::build_compute_pipeline;
pub use runtime::MatmulKernel;
pub(crate) use tuning::{TuneEntry, TuneKey};

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
    /// One slot per entry in `KERNEL_SPECS`, in the same order. Index
    /// with `KernelSelection::index()` (after resolving `Auto`).
    /// `None` marks a kernel gated off by missing device features
    /// (today: the `f16w_*` family without f16 storage support);
    /// routing, tuning, and explicit-selection validation all skip
    /// empty slots, so `kernel_at` on a routed index cannot fail.
    kernels: Vec<Option<MatmulKernel>>,
    /// Lazily-built aligned and epilogue specializations, keyed by (kernel
    /// name, base-variant index, epilogue key). The eager `variants` arrays
    /// hold four bounds-checked fallbacks; real workloads use a small subset
    /// of the other combinations, so first-use creation keeps startup flat.
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
    /// Serializes tuning-store snapshots so concurrent first-use tuning cannot
    /// persist an older map after a newer one.
    tune_save: Mutex<()>,
    /// Hash binding persisted winners to this exact shader build.
    shader_hash: u64,
}

impl MatmulPipeline {
    pub(crate) fn belongs_to(&self, ctx: &Arc<VulkanContext>) -> bool {
        Arc::ptr_eq(&self.ctx, ctx)
    }

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
            let kernels_acc: Vec<Option<MatmulKernel>> = Vec::with_capacity(KERNEL_SPECS.len());
            let mut kernels_guard = scopeguard::guard(kernels_acc, |built| {
                for kernel in built.into_iter().flatten() {
                    destroy_kernel(ctx, kernel);
                }
            });
            for spec in KERNEL_SPECS {
                // f16-storage kernels declare SPIR-V capabilities the
                // device may not have; leave their registry slots empty
                // instead of failing the whole pipeline. Routing and
                // validation never select an empty slot.
                let coopmat_gated = spec.name.contains("coopmat") && !ctx.coopmat_enabled;
                if (spec.weights_f16() && !ctx.f16_storage_enabled) || coopmat_gated {
                    kernels_guard.push(None);
                    continue;
                }
                let layout = if spec.uses_descriptors {
                    pipeline_layout
                } else {
                    pipeline_layout_bda
                };
                let kernel = create_kernel(ctx, layout, spec)
                    .with_context(|| format!("create {} matmul kernel", spec.name))?;
                kernels_guard.push(Some(kernel));
            }
            if let Some(index) = selection.index()
                && kernels_guard[index].is_none()
            {
                bail!(
                    "ML_KERNEL selects '{}', which requires f16 storage support \
                     (shaderFloat16 + storageBuffer16BitAccess) this device lacks",
                    KERNEL_SPECS[index].name
                );
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
                tune_save: Mutex::new(()),
                shader_hash,
            })
        }
    }

    pub fn select_kernel(&self, batch: u32, m: u32, n: u32, k: u32) -> &MatmulKernel {
        self.kernel_at(self.route(batch, m, n, k, false).0)
    }

    /// Atomically snapshot the complete route for this shape in one
    /// tuned-map read.  The kernel index resolves as: explicit
    /// `ML_KERNEL=` selection, else measured tuned winner, else static
    /// shape heuristic.  The second field is the tuned split-K2
    /// verdict: `None` for an untuned shape, `Some(None)` when a
    /// measured DP winner vetoes the split heuristic, and
    /// `Some(Some(s))` when split-K2 measured faster than every DP
    /// kernel.
    pub(crate) fn route(
        &self,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        b_f16: bool,
    ) -> (usize, Option<Option<u32>>) {
        match self.selection {
            KernelSelection::Auto => {
                let key = TuneKey {
                    batch,
                    m,
                    n,
                    k,
                    b_f16,
                };
                if let Some(entry) = self.tuned.read().get(&key) {
                    return (entry.kernel, Some(entry.splitk2_splits));
                }
                (self.heuristic_kernel_index(batch, m, n, k, b_f16), None)
            }
            explicit => (
                explicit
                    .index()
                    .expect("explicit selection has a concrete kernel"),
                None,
            ),
        }
    }

    /// The static shape heuristic's pick, ignoring any tuned winner.
    /// Serves as the tuner's prior and as the fallback for untuned
    /// shapes.
    pub(crate) fn heuristic_kernel_index(
        &self,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        b_f16: bool,
    ) -> usize {
        if self.ctx.buffer_device_address_enabled {
            let selection = if m == 1 {
                Some(if b_f16 {
                    KernelSelection::F16wRowBda
                } else {
                    KernelSelection::RowBda
                })
            } else if n == 1 && !b_f16 {
                Some(KernelSelection::ColBda)
            } else if k == 1 && !b_f16 {
                Some(KernelSelection::OuterBda)
            } else {
                None
            };
            if let Some(selection) = selection {
                let spec = &KERNEL_SPECS[selection.index().expect("GEMV kernel is concrete")];
                let max_groups = self
                    .ctx
                    .device_properties
                    .limits
                    .max_compute_work_group_count;
                if n.div_ceil(spec.tile_n) <= max_groups[0]
                    && m.div_ceil(spec.tile_m) <= max_groups[1]
                    && batch <= max_groups[2]
                {
                    return selection.index().expect("GEMV kernel is concrete");
                }
            }
        }
        let tile = auto_select_kernel(batch, m, n, k, self.auto_min_large_tiles);
        let selection = if b_f16 {
            // Tensor cores first when the shape allows them: strictly
            // aligned and big enough that the 128x128 tiles fill the
            // device (small aligned shapes stay on the SIMT f16w
            // kernels, which handle low occupancy better).
            if self.ctx.coopmat_enabled
                && m.is_multiple_of(128)
                && n.is_multiple_of(128)
                && k.is_multiple_of(32)
                && m >= 256
                && n >= 256
            {
                KernelSelection::F16wCoopmat
            } else {
                // f16-storage B is validated as BDA-capable upstream;
                // map the tile class onto the nearest f16w sibling.
                to_f16w(tile)
            }
        } else if self.ctx.buffer_device_address_enabled {
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
    pub(crate) fn is_tuned(&self, key: TuneKey) -> bool {
        !self.is_auto() || self.tuned.read().contains_key(&key)
    }

    /// Kernel indices the tuner should measure: every BDA-family
    /// kernel that handles arbitrary shapes (bounds-checked bodies
    /// only — the strict `*_aligned` variants are excluded).  Requires
    /// `bufferDeviceAddress`; on devices without it the tuner is
    /// disabled and the heuristic stands.
    pub(crate) fn tune_candidate_indices(&self, m: u32, n: u32, k: u32, b_f16: bool) -> Vec<usize> {
        if !self.ctx.buffer_device_address_enabled {
            return Vec::new();
        }
        self.kernels
            .iter()
            .enumerate()
            .filter_map(|(idx, kernel)| kernel.as_ref().map(|kernel| (idx, kernel)))
            .filter(|(_, kernel)| {
                let coopmat_fits = kernel.name.contains("coopmat")
                    && m.is_multiple_of(128)
                    && n.is_multiple_of(128)
                    && k.is_multiple_of(32);
                !kernel.uses_descriptors
                    && (!kernel.name.ends_with("_aligned") || coopmat_fits)
                    && kernel.weights_f16() == b_f16
                    && (m == 1 || !kernel.name.ends_with("row_bda"))
                    && (n == 1 || kernel.name != "col_bda")
                    && (k == 1 || kernel.name != "outer_bda")
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    #[inline]
    pub(crate) fn kernel_at(&self, idx: usize) -> &MatmulKernel {
        self.kernels[idx]
            .as_ref()
            .expect("routed kernel is gated off on this device")
    }

    /// Record a measured winner and persist the store.
    pub(crate) fn record_tuned(&self, key: TuneKey, entry: TuneEntry) {
        let _save = self.tune_save.lock();
        let snapshot = {
            let mut tuned = self.tuned.write();
            tuned.insert(key, entry);
            tuned.clone()
        };
        save_tuned(&self.ctx, self.shader_hash, &snapshot);
    }

    /// Pipeline for a (kernel, base-variant, epilogue) triple. Fully
    /// bounds-checked zero-epilogue variants resolve to the eager fallback;
    /// every other combination is compiled on first use and cached for the
    /// pipeline lifetime. Creation goes through the persistent
    /// `VkPipelineCache`, so repeat processes skip the ISA compile.
    pub(crate) fn pipeline_for_epilogue(
        &self,
        kernel: &MatmulKernel,
        variant: KernelVariant,
        epilogue: EpilogueKey,
    ) -> Result<vk::Pipeline> {
        if epilogue.is_none() && variant == variant.fallback() {
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
/// Map an auto-selected tile class onto its f16-storage-B sibling.
/// Coarser than the f32 family on purpose: three tile classes cover
/// the heuristic winners, and the measured tuner refines per shape
/// within the f16w candidate set.
fn to_f16w(tile: KernelSelection) -> KernelSelection {
    match tile {
        KernelSelection::Large | KernelSelection::M64N128 => KernelSelection::F16wLargeBdaV4,
        KernelSelection::M128N64 | KernelSelection::M128N64K64 => {
            KernelSelection::F16wM128N64K64BdaV4
        }
        KernelSelection::K64 => KernelSelection::F16wK64BdaV4,
        _ => KernelSelection::F16wSmallBdaV4,
    }
}

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
            for kernel in self.kernels.drain(..).flatten() {
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
