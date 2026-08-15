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

pub use abi::{EpilogueKey, KernelVariant, MATMUL_FLAG_NORM_A, MatmulPushConstants};
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
    /// `None` marks a kernel gated off on this device: the `f16w_*`
    /// family without f16 storage support, `coopmat` without the
    /// feature, or a shared-memory tile over the device's
    /// [`VulkanContext::workgroup_shared_budget`]; routing, tuning,
    /// and explicit-selection validation all skip empty slots, so
    /// `kernel_at` on a routed index cannot fail.
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
            let shared_budget = ctx.workgroup_shared_budget();
            for spec in KERNEL_SPECS {
                // f16-storage kernels declare SPIR-V capabilities the
                // device may not have; leave their registry slots empty
                // instead of failing the whole pipeline. Routing and
                // validation never select an empty slot.
                let coopmat_gated = spec.name.contains("coopmat") && !ctx.coopmat_enabled;
                // The NV_cooperative_matrix2 GEMM ships SPIR-V 1.6 and
                // workgroup-scope matrices; its slot stays empty
                // without the full coopmat2 feature gate.
                let cm2_gated = spec.name.contains("cm2") && !ctx.coopmat2_enabled;
                // Likewise for tiles whose shared declarations exceed
                // the device budget: strict drivers reject or lose the
                // device on them (see workgroup_shared_budget), so
                // their slots stay empty and routing demotes to an
                // in-budget sibling (`in_budget_index`).
                let shared_gated = spec.shared_memory_bytes() > shared_budget;
                if (spec.weights_f16() && !ctx.f16_storage_enabled)
                    || coopmat_gated
                    || cm2_gated
                    || shared_gated
                {
                    if shared_gated {
                        log::info!(
                            "tensor-ash: kernel '{}' gated off: {} B workgroup memory > {} B device budget",
                            spec.name,
                            spec.shared_memory_bytes(),
                            shared_budget
                        );
                    }
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
                let spec = &KERNEL_SPECS[index];
                let reason = if spec.name.contains("cm2") && !ctx.coopmat2_enabled {
                    "requires NV cooperative matrix 2 support this device lacks".to_string()
                } else if spec.name.contains("coopmat") && !ctx.coopmat_enabled {
                    "requires cooperative-matrix support this device lacks".to_string()
                } else if spec.weights_f16() && !ctx.f16_storage_enabled {
                    "requires f16 storage support (shaderFloat16 + \
                     storageBuffer16BitAccess) this device lacks"
                        .to_string()
                } else {
                    format!(
                        "declares {} B of workgroup memory, over this device's {} B budget \
                         (maxComputeSharedMemorySize)",
                        spec.shared_memory_bytes(),
                        shared_budget
                    )
                };
                bail!("ML_KERNEL selects '{}', which {reason}", spec.name);
            }
            // Disarm the cleanup: kernels now belong to the pipeline.
            let kernels = ScopeGuard::into_inner(kernels_guard);

            let shader_hash = shader_registry_hash();
            // A persisted winner recorded before its slot became gated
            // (e.g. a store written by a pre-gate build on this device)
            // must not route dispatches into an empty slot.
            let mut tuned = load_tuned(ctx, shader_hash);
            tuned.retain(|_, entry| kernels[entry.kernel].is_some());
            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout: ScopeGuard::into_inner(set_layout_guard),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                pipeline_layout_bda: ScopeGuard::into_inner(pipeline_layout_bda_guard),
                kernels,
                epilogue_pipelines: Mutex::new(HashMap::new()),
                selection,
                auto_min_large_tiles: auto_min_large_tiles_for(ctx.device_kind()),
                tuned: RwLock::new(tuned),
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
                    // Sixteen K-slice warps win when K is deep, N is
                    // narrow, or the shape is square-ish (measured:
                    // 2048^2 -23%, K=5632/N=2048 -22%, N=256 helps;
                    // wide-N moderate-K like N=5632/K=2048 regresses
                    // -8% and stays on eight).
                    if k >= 4096 || n <= 512 || (k >= 2048 && n <= 2048) {
                        KernelSelection::F16wRowBdaK16
                    } else {
                        // Wide-N moderate-K: sixteen slices alone lose
                        // ~5% to the eight-slice kernel here, but with
                        // two packed columns per lane (f16vec2 loads)
                        // they win ~3% instead (K=2048/N=5632: 58 ->
                        // 56 µs); VCOLS=2 is measured-neutral on the
                        // k16-routed shapes and -12% on N=256, so the
                        // narrow/deep branch stays on plain k16.
                        KernelSelection::F16wRowBdaK16V2
                    }
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
        self.in_budget_index(selection)
    }

    /// Resolve `selection` to a registry index, demoting through
    /// in-budget siblings when the shared-memory gate emptied its slot
    /// (`workgroup_shared_budget`).  Chains terminate at the 64x64
    /// small tile (16,640 B), which fits every device this library
    /// realistically initializes on; a budget below that panics with
    /// the kernel name rather than dispatching into an empty slot.
    fn in_budget_index(&self, selection: KernelSelection) -> usize {
        use KernelSelection as K;
        let mut selection = selection;
        loop {
            let index = selection.index().expect("selection is concrete");
            if self.kernels[index].is_some() {
                return index;
            }
            selection = match selection {
                // The BK=64 deep-K tiles (49,664 B) exceed the
                // universal 49,152 B budget; the BK=32 sibling keeps
                // the same 128x64 output tile, so dispatch geometry
                // and edge handling are unchanged.
                K::M128N64K64 => K::M128N64,
                K::M128N64K64BdaV4 => K::M128N64BdaV4,
                // No f16w m128n64 sibling exists; use the family's
                // general 64x64 fallback (to_f16w's default arm).
                K::F16wM128N64K64BdaV4 => K::F16wSmallBdaV4,
                // Sub-floor budgets (software Vulkan reports
                // 32,768 B): the >= 32 KiB tiles fall to their
                // family's 64x64 tile.
                K::Large | K::K64 | K::M128N64 | K::M64N128 => K::Small,
                K::LargeBdaV4 | K::K64BdaV4Tm8Tn4 | K::M128N64BdaV4 | K::M64N128BdaV4 => {
                    K::SmallBdaV4
                }
                K::F16wLargeBdaV4 | K::F16wK64BdaV4 => K::F16wSmallBdaV4,
                other => panic!(
                    "kernel '{}' is gated off on this device and has no in-budget fallback",
                    KERNEL_SPECS[other.index().expect("selection is concrete")].name
                ),
            };
        }
    }

    /// Epilogue-capable kernel for this shape: the normal heuristic
    /// pick when it fuses, else the CM2 tensor-core GEMM when the
    /// device and shape allow it, else the SIMT BDA_V4 sibling of the
    /// tile class.  Used to demote routes (heuristic coopmat, or a
    /// tuned winner recorded for the plain shape) that land on kernels
    /// without the fused-epilogue specialization.
    pub(crate) fn epilogue_fallback_index(
        &self,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        b_f16: bool,
    ) -> usize {
        let index = self.heuristic_kernel_index(batch, m, n, k, b_f16);
        if self.kernel_at(index).supports_epilogue() {
            return index;
        }
        // NV_cooperative_matrix2 GEMM: keeps the tensor cores AND
        // fuses the epilogue via its `coopMatPerElementNV` store
        // callback, so coopmat-eligible fused ops no longer demote to
        // the SIMT family.  Constraint mirrors the coopmat1 heuristic
        // gate exactly (M, N % 128 == 0, K % 32 == 0, M, N >= 256):
        // the kernel body is general-shape correct via tensor-layout
        // clamping, but batched tensor base addresses must stay
        // 16-byte aligned (guaranteed by the alignment), and small
        // shapes are better served by the low-occupancy-friendly SIMT
        // epilogue kernels.  Only this fused path reroutes — plain
        // shapes keep the measured coopmat1 winner (CM2 plain GEMM is
        // expected parity, not a win).
        if b_f16
            && m.is_multiple_of(128)
            && n.is_multiple_of(128)
            && k.is_multiple_of(32)
            && m >= 256
            && n >= 256
        {
            let cm2 = KernelSelection::F16wCm2
                .index()
                .expect("F16wCm2 is concrete");
            // Slot presence implies `coopmat2_enabled`.
            if self.kernels[cm2].is_some() {
                return cm2;
            }
        }
        let tile = auto_select_kernel(batch, m, n, k, self.auto_min_large_tiles);
        let selection = if b_f16 {
            to_f16w(tile)
        } else {
            maybe_to_bda(tile)
        };
        self.in_budget_index(selection)
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
                // Tensor-core kernels only measure on coopmat-aligned
                // shapes: coopmat1 is strictly `_aligned`, and the CM2
                // kernel — though general-shape correct via clamping —
                // needs 16-byte-aligned batched tensor bases, which the
                // alignment guarantees.
                let tensor_core = kernel.name.contains("coopmat") || kernel.name.contains("cm2");
                let coopmat_fits = tensor_core
                    && m.is_multiple_of(128)
                    && n.is_multiple_of(128)
                    && k.is_multiple_of(32);
                !kernel.uses_descriptors
                    && (!kernel.name.ends_with("_aligned") || coopmat_fits)
                    && (!tensor_core || coopmat_fits)
                    && kernel.weights_f16() == b_f16
                    // The a16 kernel reads f16 A; the tuner's ops are
                    // always f32-A, and f16-A routes are fixed picks.
                    && !kernel.a_f16()
                    && (m == 1 || !kernel.name.contains("row_bda"))
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
