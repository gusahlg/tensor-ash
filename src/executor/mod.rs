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
mod recording;
mod reduction;
mod slot;
mod splitk;
mod splitk2;
mod streamk;
mod streamk_exec;
mod streamk_schedule;
mod submission;
mod transfer;
mod tuning;
mod validation;

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use ash::vk;
use parking_lot::{Condvar, Mutex};

use crate::context::VulkanContext;
use crate::pipeline::MatmulPipeline;

use slot::Slot;
use splitk::SplitKPipeline;
pub use splitk::default_num_k_splits;
use splitk2::SplitK2Pipeline;
pub use splitk2::SplitK2ReducePushConstants;
use streamk::StreamKPipeline;
pub use streamk::StreamKPushConstants;
pub use streamk_schedule::{StreamKSchedule, stream_k_should_fire};

pub use crate::matmul::{MatmulCall, RunStats};

/// Fallback SM count for Stream-K's persistent-grid sizing on
/// devices where we don't have a runtime probe yet.  46 matches the
/// RTX 3070 we develop on; the value only drives the preferred grid
/// width `g_pref = sm_count * 2`, so being slightly off is harmless.
/// `src/persistent.rs::FALLBACK_SM_COUNT` carries the same constant
/// for the persistent kernel.
const STREAMK_FALLBACK_SM_COUNT: u32 = 46;

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
    /// Experimental split-K pipeline, lazily built on first use of
    /// `run_matmuls_split_k`.  See `executor/splitk.rs` for the design
    /// rationale.
    split_k: OnceLock<SplitKPipeline>,
    /// Two-stage split-K pipeline (scratch partials + reduce), lazily
    /// built on first use of `run_matmuls_split_k2`.
    split_k2: OnceLock<SplitK2Pipeline>,
    /// Experimental Stream-K pipeline, lazily built on first use of
    /// `run_matmuls_stream_k`.  See `executor/streamk.rs`.
    stream_k: OnceLock<StreamKPipeline>,
    /// `ML_TUNE=1`: measure every eligible kernel the first time a new
    /// shape is submitted, record the winner in the pipeline's
    /// persistent tuning store, and use it from then on.  Off by
    /// default — persisted winners from previous tuned runs still
    /// apply either way.
    tune_enabled: bool,
}

impl Executor {
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
            split_k: OnceLock::new(),
            split_k2: OnceLock::new(),
            stream_k: OnceLock::new(),
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
