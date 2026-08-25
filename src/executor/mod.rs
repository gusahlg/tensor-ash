//! Thread-safe GEMM dispatcher.
//!
//! Architecture
//! ============
//! An `Executor` owns a small pool of *slots*.  A slot bundles every
//! resource that is per-submission (and therefore can't be shared
//! across in-flight submissions): a command pool, a command buffer, a
//! fence, a descriptor pool, and a timestamp query pool.
//!
//! `run_matmuls` checks out one slot under a mutex, releases the mutex
//! while it does the actual recording + submit + wait, then returns
//! the slot.  Multiple host threads can therefore record concurrently
//! up to `n_slots`, after which they block at checkout.
//!
//! The queue submission is serialized by a separate mutex (Vulkan
//! requires external synchronization on a single VkQueue).
//!
//! Why not multi-queue?  On consumer GPUs, multiple compute queues from
//! the same family time-multiplex on the hardware, so for the GEMM
//! workload here they provide ~0 throughput benefit but lots of
//! synchronization complexity.  The pool-of-slots design instead
//! captures the real benefit (CPU-side recording parallelism, CPU/GPU
//! overlap of consecutive submissions).

mod dispatch;
mod elementwise;
mod graph_ops;
mod prepared;
mod recording;
mod reduction;
mod slot;
mod splitk2;
mod submission;
mod transfer;
mod tuning;
mod validation;

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use ash::vk;
use parking_lot::{Condvar, Mutex};

use crate::context::{DeviceKind, VulkanContext};
use crate::pipeline::MatmulPipeline;

use slot::Slot;
use splitk2::SplitK2Pipeline;

pub use crate::matmul::{MatmulCall, RunStats};
pub use elementwise::{
    ATTN_DECODE_MAX_CHUNKS, AttnDecodeDesc, BinaryOp, CopyDesc, FlashAttentionDesc,
    GEMV_CHAIN_MAX_JOBS, PrefillQkvPackDesc, RopeDesc, RopeScatterDesc, SoftmaxMask,
};
pub use graph_ops::{ExecOp, HostU32Buffer, PosBuffer, TokenIdBuffer};
pub use prepared::PreparedOps;

/// Read-only description of the route selected for a plain matmul shape.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchInfo {
    pub kernel: &'static str,
    pub tile: [u32; 3],
    pub split_k2_splits: Option<u32>,
}

impl DispatchInfo {
    /// Describe an explicit two-stage split-K dispatch. This mirrors
    /// [`Executor::run_matmuls_split_k2`] for benchmark diagnostics.
    pub fn split_k2(m: u32, n: u32, splits: u32) -> Self {
        let (kernel, tile) = splitk2::stage1_dispatch_info(m, n);
        Self {
            kernel,
            tile,
            split_k2_splits: Some(splits),
        }
    }
}

/// Explicit executor policy. Use [`Executor::new_with_config`] when a library
/// caller must not depend on process-wide `ML_TUNE` state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorConfig {
    pub n_slots: usize,
    pub max_calls_per_submit: u32,
    pub tune: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            n_slots: 2,
            max_calls_per_submit: 64,
            tune: false,
        }
    }
}

pub struct Executor {
    ctx: Arc<VulkanContext>,
    pipeline: Arc<MatmulPipeline>,
    /// Available-slot queue with a condvar for blocking checkout.
    slots: Mutex<VecDeque<Slot>>,
    slot_avail: Condvar,
    /// Maximum descriptor sets we'll allocate per submission (= max
    /// matmul calls in one `run_matmuls`).
    max_calls_per_submit: u32,
    /// Two-stage split-K pipeline (scratch partials + reduce), lazily
    /// built on first use of `run_matmuls_split_k2`.
    split_k2: OnceLock<SplitK2Pipeline>,
    /// Non-GEMM model ops (softmax/norm/RoPE/copy), lazily built on
    /// first use of any `run_*` elementwise entry point.
    elementwise: OnceLock<elementwise::ElementwisePipeline>,
    /// `ML_TUNE=1`: measure every eligible kernel the first time a new
    /// shape is submitted, record the winner in the pipeline's
    /// persistent tuning store, and use it from then on.  Off by
    /// default — persisted winners from previous tuned runs still
    /// apply either way.
    tune_enabled: bool,
}

/// Complete validated route for one op shape, resolved once per
/// submission and handed to every consumer (descriptor updates, command
/// recording, diagnostics).  Snapshotting up front keeps a concurrent
/// first-use tune from splitting one submission across two registry
/// states.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpPlan {
    /// Data-parallel kernel index into `KERNEL_SPECS`; used whenever
    /// `splitk2` is `None`.
    pub(crate) kernel: usize,
    /// Validated two-stage split-K route (splits >= 2).  Only set for
    /// call forms that may take it — single plain non-accumulating
    /// ops, or graph ops (whose barriers make the internal one free).
    pub(crate) splitk2: Option<u32>,
}

impl Executor {
    /// Report the kernel and optional tuned split-K2 route a plain,
    /// non-accumulating matmul (f32 B) would use. Useful for benchmark
    /// diagnostics.
    pub fn dispatch_info(&self, batch: u32, m: u32, n: u32, k: u32) -> DispatchInfo {
        self.dispatch_info_for(batch, m, n, k, false)
    }

    /// Like [`Self::dispatch_info`] with the B storage type explicit:
    /// `b_f16 = true` reports the route for an f16-weights matmul.
    pub fn dispatch_info_for(
        &self,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        b_f16: bool,
    ) -> DispatchInfo {
        let plan = self.plan_shape(batch, m, n, k, b_f16, true);
        let (kernel_name, tile) = match plan.splitk2 {
            Some(_) => splitk2::stage1_dispatch_info(m, n),
            None => {
                let kernel = self.pipeline.kernel_at(plan.kernel);
                (kernel.name, [kernel.tile_m, kernel.tile_n, kernel.tile_k])
            }
        };
        DispatchInfo {
            kernel: kernel_name,
            tile,
            split_k2_splits: plan.splitk2,
        }
    }

    /// Resolve the complete route for one shape.  `splitk2_eligible`
    /// captures the call form (single plain non-accumulating op, or a
    /// graph op); the measured or conservative untuned split-K2 route is
    /// validated against device addressing, grid, and scratch limits, so
    /// a `Some` split count here is dispatchable as-is.
    pub(super) fn plan_shape(
        &self,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        b_f16: bool,
        splitk2_eligible: bool,
    ) -> OpPlan {
        let (kernel, tuned) = self.pipeline.route(batch, m, n, k, b_f16);
        // Split-K2's stage-1 kernels read f32 B only, so f16-weight
        // calls always take the data-parallel route.
        let splitk2 = (splitk2_eligible
            && !b_f16
            && self.pipeline.is_auto()
            && self.ctx.buffer_device_address_enabled)
            .then(|| {
                let heuristic = (tuned.is_none()
                    && self.ctx.device_kind() == DeviceKind::DiscreteGpu)
                    .then(|| splitk2::heuristic_splits(batch, m, n, k))
                    .flatten();
                let splits = splitk2::resolve_auto_splits(tuned, heuristic)?;
                let max = self
                    .ctx
                    .device_properties
                    .limits
                    .max_compute_work_group_count;
                splitk2::auto_route_fits(max, batch, m, n, k, splits).then_some(splits)
            })
            .flatten();
        OpPlan { kernel, splitk2 }
    }

    /// Resolve the complete route for one resolved matmul, including
    /// the f16-activations short-circuit: f16-A ops pick one of the
    /// two aligned a16 coopmat tiles (128x128 or wave-fill 64x64), so
    /// they bypass tuning, the shape heuristic, and split-K entirely
    /// — but only on devices with the cooperative-matrix + f16-storage
    /// features those kernels need.
    pub(super) fn plan_matmul(
        &self,
        dims: &crate::matmul::ResolvedMatmul,
        splitk2_eligible: bool,
    ) -> anyhow::Result<OpPlan> {
        if dims.packed_b {
            // Fail-closed through the budget gate: an empty packed
            // slot panics rather than falling back to a row-major
            // sibling that would misread B.
            let selection = crate::pipeline::f16w_row_packed_selection(dims.k, dims.n);
            return Ok(OpPlan {
                kernel: self.pipeline.in_budget_index(selection),
                splitk2: None,
            });
        }
        if dims.a_f16 {
            if !(self.ctx.coopmat_enabled && self.ctx.f16_storage_enabled) {
                anyhow::bail!(
                    "matmul with f16 A storage requires cooperative-matrix and f16-storage \
                     support, which this device lacks"
                );
            }
            let selection = if crate::pipeline::coopmat_prefers_m64(dims.batch, dims.m, dims.n) {
                crate::pipeline::KernelSelection::F16wA16CoopmatM64
            } else {
                crate::pipeline::KernelSelection::F16wA16Coopmat
            };
            return Ok(OpPlan {
                kernel: selection
                    .index()
                    .expect("a16 coopmat selection is concrete"),
                splitk2: None,
            });
        }
        Ok(self.plan_shape(
            dims.batch,
            dims.m,
            dims.n,
            dims.k,
            dims.b_f16,
            splitk2_eligible,
        ))
    }

    /// Reroute an op whose planned kernel cannot honor its fusions:
    /// fused-epilogue ops off kernels without the epilogue constants
    /// (heuristic coopmat, or a tuned winner measured for the plain
    /// shape) onto the epilogue-capable CM2 tensor-core GEMM when the
    /// device and shape allow it (see `epilogue_fallback_index`) or
    /// else the SIMT sibling, and normed-A ops
    /// off any non-row route (e.g. a tuned M=1 winner) back onto the
    /// row-GEMV heuristic pick.  No-op for plain ops and for routes
    /// that already fuse.
    pub(super) fn demote_for_op(
        &self,
        op: &crate::matmul::MatmulOp<'_>,
        dims: &crate::matmul::ResolvedMatmul,
        plan: OpPlan,
    ) -> OpPlan {
        // Packed B is a different layout; falling back to any unpacked
        // sibling would silently misread weights.  Record-time already
        // rejects a packed/unpacked mismatch.
        if dims.packed_b {
            return plan;
        }
        // Only auto routes demote.  An explicit ML_KERNEL selection of
        // a non-fusing kernel keeps its documented loud failure at
        // record time rather than being silently overridden.
        let mut plan = plan;
        if !op.epilogue.is_none()
            && !self.pipeline.kernel_at(plan.kernel).supports_epilogue()
            && self.pipeline.is_auto()
        {
            plan = OpPlan {
                kernel: self
                    .pipeline
                    .epilogue_fallback_index(dims.batch, dims.m, dims.n, dims.k, dims.b_f16),
                splitk2: None,
            };
        }
        // Normed-A and store ops need a row route that implements
        // their fusions (a tuned M=1 winner may not).
        let fusions_fit = |kernel: &crate::pipeline::MatmulKernel| {
            (op.normed_a.is_none() || kernel.supports_normed_a())
                && (op.store.is_none() || kernel.supports_store())
        };
        if !fusions_fit(self.pipeline.kernel_at(plan.kernel)) && self.pipeline.is_auto() {
            // For M=1 the heuristic always picks a row kernel; other
            // shapes keep their plan and fail loudly at record time.
            let index = self
                .pipeline
                .heuristic_kernel_index(dims.batch, dims.m, dims.n, dims.k, dims.b_f16);
            if fusions_fit(self.pipeline.kernel_at(index)) {
                plan = OpPlan {
                    kernel: index,
                    splitk2: None,
                };
            }
        }
        plan
    }

    /// `n_slots` = how many submissions can be in flight at once. 2 is
    /// the sweet spot: one being recorded by the host while the other
    /// runs on the GPU.  Higher values benefit hosts that submit
    /// concurrently from many threads.
    pub fn new(
        ctx: Arc<VulkanContext>,
        pipeline: Arc<MatmulPipeline>,
        n_slots: usize,
        max_calls_per_submit: u32,
    ) -> Result<Self> {
        let tune = std::env::var("ML_TUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self::new_with_config(
            ctx,
            pipeline,
            ExecutorConfig {
                n_slots,
                max_calls_per_submit,
                tune,
            },
        )
    }

    pub fn new_with_config(
        ctx: Arc<VulkanContext>,
        pipeline: Arc<MatmulPipeline>,
        config: ExecutorConfig,
    ) -> Result<Self> {
        if !pipeline.belongs_to(&ctx) {
            anyhow::bail!("Executor::new: pipeline belongs to a different VulkanContext");
        }
        let n_slots = config.n_slots.max(1);
        let max_calls_per_submit = config.max_calls_per_submit.max(1);
        let mut slots = VecDeque::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push_back(Slot::create(
                &ctx,
                max_calls_per_submit,
                pipeline.set_layout,
            )?);
        }
        Ok(Self {
            ctx,
            pipeline,
            slots: Mutex::new(slots),
            slot_avail: Condvar::new(),
            max_calls_per_submit,
            split_k2: OnceLock::new(),
            elementwise: OnceLock::new(),
            tune_enabled: config.tune,
        })
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        let slots = std::mem::take(&mut *self.slots.lock());
        for s in slots {
            unsafe {
                if s.query_pool != vk::QueryPool::null() {
                    self.ctx.device.destroy_query_pool(s.query_pool, None);
                }
                self.ctx
                    .device
                    .destroy_descriptor_pool(s.descriptor_pool, None);
                self.ctx.device.destroy_fence(s.fence, None);
                self.ctx.device.destroy_command_pool(s.cmd_pool, None);
            }
        }
    }
}
