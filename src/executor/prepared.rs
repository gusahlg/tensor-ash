//! Record-once, replay-many submission.
//!
//! `run_matmuls` re-validates, re-plans, and re-records every call on
//! every invocation, then blocks on the fence — for the smallest GEMMs
//! that synchronous host round trip (~0.02 ms) costs more than the GPU
//! work.  A [`PreparedOps`] hoists everything shape- and buffer-stable
//! to prepare time: the ops are validated, routed, and recorded into a
//! dedicated resubmittable command buffer exactly once, so each replay
//! costs one `vkQueueSubmit` (plus a fence wait when the caller needs
//! the result).  `submit`/`wait` are split so two prepared objects can
//! ping-pong: while one runs on the GPU the host already submitted the
//! other.
//!
//! The borrow of the executor and of every tensor is held for the
//! prepared object's lifetime, and both [`PreparedOps::run`] and `Drop`
//! wait out any in-flight submission, so the recorded device addresses
//! cannot dangle under normal use.  Push constants bake those addresses
//! into the command buffer, which is exactly why the object is shape-
//! and buffer-fixed.  The one escape hatch is leaking the object while
//! a submission is in flight, which skips `Drop`; that is why
//! [`PreparedOps::submit`] is `unsafe` (see its safety contract).
//!
//! Scope: BDA kernels only (descriptor-set lifetimes stay out of the
//! picture; on devices without `bufferDeviceAddress`, preparation
//! fails).  Routing uses the data-parallel plan — the two-stage
//! split-K's slot-owned scratch is per-submission by design and is not
//! captured in a replayable buffer.

use std::marker::PhantomData;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::matmul::{MatmulCall, MatmulOp, ResolvedMatmulBatch, RunStats, total_flops};

use super::recording::{record_matmul_commands, record_matmul_graph_commands};
use super::submission::{read_gpu_time_ns, record_timestamped, submit_one, wait_fence_spin};
use super::{Executor, OpPlan};

/// A validated, pre-recorded batch of matmul ops bound to fixed
/// tensors.  Create with [`Executor::prepare_matmuls`],
/// [`Executor::prepare_ops`], or [`Executor::prepare_op_graph`]; replay
/// with [`run`](Self::run) or the [`submit`](Self::submit) /
/// [`wait`](Self::wait) pair.
pub struct PreparedOps<'e, 't> {
    exec: &'e Executor,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Two timestamps around the recorded work; null when the device
    /// lacks timestamp support.
    query_pool: vk::QueryPool,
    in_flight: bool,
    n_calls: usize,
    total_flops: u64,
    /// The recorded push constants point at these tensors' memory.
    _tensors: PhantomData<&'t ()>,
}

impl Executor {
    /// [`prepare_ops`](Self::prepare_ops) for plain calls.
    pub fn prepare_matmuls<'e, 't>(
        &'e self,
        calls: &[MatmulCall<'t>],
    ) -> Result<PreparedOps<'e, 't>> {
        let ops: Vec<MatmulOp<'t>> = calls.iter().copied().map(MatmulOp::new).collect();
        self.prepare_impl(&ops, false)
    }

    /// Validate, route, and record `ops` once for repeated replay.
    /// The ops must be independent (no barriers are recorded), exactly
    /// like [`run_ops`](Self::run_ops).
    pub fn prepare_ops<'e, 't>(&'e self, ops: &[MatmulOp<'t>]) -> Result<PreparedOps<'e, 't>> {
        self.prepare_impl(ops, false)
    }

    /// [`prepare_ops`](Self::prepare_ops) for dependent chains:
    /// hazard barriers are recorded exactly like
    /// [`run_op_graph`](Self::run_op_graph).
    pub fn prepare_op_graph<'e, 't>(&'e self, ops: &[MatmulOp<'t>]) -> Result<PreparedOps<'e, 't>> {
        self.prepare_impl(ops, true)
    }

    fn prepare_impl<'e, 't>(
        &'e self,
        ops: &[MatmulOp<'t>],
        with_dependency_barriers: bool,
    ) -> Result<PreparedOps<'e, 't>> {
        if ops.is_empty() {
            bail!("prepare_ops: empty op batch");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("prepared submission requires bufferDeviceAddress, which this device lacks");
        }
        for op in ops {
            self.validate_op_context(op)?;
        }
        let resolved = ResolvedMatmulBatch::from_ops(ops)?;
        let resolved = resolved.as_slice();
        let flops = total_flops(resolved)?;

        // Data-parallel routes only: split-K2's scratch is slot-owned
        // and per-submission, so it cannot live in a replayable buffer.
        let plans: Vec<OpPlan> = resolved
            .iter()
            .map(|dims| self.plan_shape(dims.batch, dims.m, dims.n, dims.k, dims.b_f16, false))
            .collect();
        if let Some(plan) = plans
            .iter()
            .find(|plan| self.pipeline.kernel_at(plan.kernel).uses_descriptors)
        {
            bail!(
                "prepared submission supports only BDA kernels; '{}' binds descriptor sets \
                 (unset ML_KERNEL or pick a *_bda / *_bda_v4 variant)",
                self.pipeline.kernel_at(plan.kernel).name
            );
        }

        let dev = &self.ctx.device;
        unsafe {
            let cmd_pool = dev
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.ctx.compute_family),
                    None,
                )
                .context("create_command_pool (prepared)")?;
            let pool_guard = scopeguard::guard(cmd_pool, |p| dev.destroy_command_pool(p, None));

            let cmd = dev
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .context("allocate_command_buffers (prepared)")?[0];

            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .context("create_fence (prepared)")?;
            let fence_guard = scopeguard::guard(fence, |f| dev.destroy_fence(f, None));

            let query_pool = if self.ctx.timestamps_supported {
                dev.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .context("create_query_pool (prepared)")?
            } else {
                vk::QueryPool::null()
            };
            let query_guard = scopeguard::guard(query_pool, |q| {
                if q != vk::QueryPool::null() {
                    dev.destroy_query_pool(q, None);
                }
            });

            // Record once, without ONE_TIME_SUBMIT, so the buffer can
            // be resubmitted indefinitely.  The BDA kernels never bind
            // descriptor sets, so null handles satisfy the recorder.
            dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .context("begin_command_buffer (prepared)")?;
            let null_sets = vec![vk::DescriptorSet::null(); ops.len()];
            record_timestamped(dev, cmd, query_pool, || {
                if with_dependency_barriers {
                    record_matmul_graph_commands(
                        &self.ctx,
                        &self.pipeline,
                        cmd,
                        &null_sets,
                        ops,
                        resolved,
                        &plans,
                        None,
                    )
                } else {
                    record_matmul_commands(
                        &self.ctx,
                        &self.pipeline,
                        cmd,
                        &null_sets,
                        ops,
                        resolved,
                        &plans,
                    )
                }
            })?;
            dev.end_command_buffer(cmd)
                .context("end_command_buffer (prepared)")?;

            Ok(PreparedOps {
                exec: self,
                cmd_pool: scopeguard::ScopeGuard::into_inner(pool_guard),
                cmd,
                fence: scopeguard::ScopeGuard::into_inner(fence_guard),
                query_pool: scopeguard::ScopeGuard::into_inner(query_guard),
                in_flight: false,
                n_calls: ops.len(),
                total_flops: flops,
                _tensors: PhantomData,
            })
        }
    }
}

impl PreparedOps<'_, '_> {
    /// Submit and block until GPU completion, like
    /// [`Executor::run_matmuls`] without the per-call re-validation and
    /// re-recording.
    pub fn run(&mut self) -> Result<RunStats> {
        // Sound despite `submit` being unsafe: control does not return
        // to the caller until `wait` has fenced the work out, so the
        // leak hazard below cannot arise.
        unsafe { self.submit()? };
        self.wait()
    }

    /// Submit without waiting.  The caller must [`wait`](Self::wait)
    /// before submitting this object again; overlap is achieved by
    /// ping-ponging two prepared objects.  Writes to this object's
    /// output tensors race with any other in-flight submission that
    /// touches them — replay dependent work as one prepared graph.
    ///
    /// # Safety
    ///
    /// This returns with GPU work in flight against device addresses
    /// baked into the recorded command buffer.  Normally the borrows
    /// plus [`Drop`] (which waits on the fence) keep that sound, but
    /// leaking the object (`std::mem::forget`, `Box::leak`, an `Rc`
    /// cycle) skips `Drop` and ends the borrows while the GPU is still
    /// running.  The caller must not leak this object between `submit`
    /// and [`wait`](Self::wait); the borrowed tensors and executor must
    /// stay alive until the submission completes.
    pub unsafe fn submit(&mut self) -> Result<()> {
        if self.in_flight {
            bail!("PreparedOps::submit: already in flight; call wait() first");
        }
        unsafe {
            submit_one(
                &self.exec.ctx,
                self.cmd,
                self.fence,
                "queue_submit (prepared)",
            )?
        };
        self.in_flight = true;
        Ok(())
    }

    /// Block until the in-flight replay completes and report its stats.
    pub fn wait(&mut self) -> Result<RunStats> {
        if !self.in_flight {
            bail!("PreparedOps::wait: nothing submitted");
        }
        let ctx = &self.exec.ctx;
        wait_fence_spin(&ctx.device, self.fence)?;
        unsafe {
            ctx.device
                .reset_fences(&[self.fence])
                .context("reset_fences (prepared)")?;
        }
        self.in_flight = false;
        let gpu_time_ns = if self.query_pool == vk::QueryPool::null() {
            None
        } else {
            Some(read_gpu_time_ns(
                ctx,
                self.query_pool,
                "get_query_pool_results (prepared)",
            )?)
        };
        Ok(RunStats {
            gpu_time_ns,
            n_calls: self.n_calls,
            total_flops: self.total_flops,
        })
    }
}

impl Drop for PreparedOps<'_, '_> {
    fn drop(&mut self) {
        let ctx = &self.exec.ctx;
        unsafe {
            if self.in_flight {
                let _ = ctx.device.wait_for_fences(&[self.fence], true, u64::MAX);
            }
            if self.query_pool != vk::QueryPool::null() {
                ctx.device.destroy_query_pool(self.query_pool, None);
            }
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}
