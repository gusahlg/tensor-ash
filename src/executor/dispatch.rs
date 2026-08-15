//! Standard and dependency-aware matmul dispatch.

use anyhow::{Result, bail};

use crate::matmul::{
    MatmulCall, MatmulOp, ResolvedMatmul, ResolvedMatmulBatch, RunStats, total_flops,
};
use crate::pipeline::TuneKey;

use super::recording::{
    record_matmul_commands, record_matmul_graph_commands, update_matmul_descriptor_sets,
};
use super::splitk2::SplitK2Pipeline;
use super::{Executor, OpPlan, Slot, recording, splitk2};

impl Executor {
    /// Run a batch of matmul calls.  Blocks until GPU completion.
    ///
    /// The calls in `calls` are dispatched back-to-back in a single
    /// command buffer with **no pipeline barriers between them**.  That
    /// is intentional — the common usage is many *independent* matmuls
    /// (different C tensors per call), and barriers would serialize
    /// them on the GPU for no benefit.
    ///
    /// If you need to chain calls — i.e. the second call reads a `C`
    /// that the first call writes — use [`run_matmul_graph`], which
    /// inserts the required barriers automatically and still submits
    /// once.
    ///
    /// [`run_matmul_graph`]: Self::run_matmul_graph
    pub fn run_matmuls(&self, calls: &[MatmulCall<'_>]) -> Result<RunStats> {
        self.run_calls_impl(calls, false)
    }

    /// Run a *dependent* chain of matmul calls in one submission.
    ///
    /// Unlike [`run_matmuls`] (which records back-to-back with no
    /// barriers and therefore requires the calls to be independent),
    /// this entry point tracks which buffers each call reads and
    /// writes and inserts a compute→compute barrier exactly where a
    /// later call depends on an earlier call's output.  The whole
    /// chain is submitted once and waits on one fence, so a
    /// multi-layer network step costs one host round-trip instead of
    /// one per dependency level.
    ///
    /// Independent calls between barriers still overlap on the GPU
    /// exactly as they do in `run_matmuls`.
    ///
    /// [`run_matmuls`]: Self::run_matmuls
    pub fn run_matmul_graph(&self, calls: &[MatmulCall<'_>]) -> Result<RunStats> {
        self.run_calls_impl(calls, true)
    }

    /// [`run_matmuls`] for ops with fused epilogues (bias, activation,
    /// residual add / gating mul applied in-register at store time).
    /// No inter-op barriers — the ops must be independent.
    ///
    /// [`run_matmuls`]: Self::run_matmuls
    pub fn run_ops(&self, ops: &[MatmulOp<'_>]) -> Result<RunStats> {
        self.run_ops_impl(ops, false)
    }

    /// [`run_matmul_graph`] for ops with fused epilogues: dependent
    /// chains in one submission, hazards (including epilogue bias / D
    /// operands) resolved with automatic barriers.
    ///
    /// [`run_matmul_graph`]: Self::run_matmul_graph
    pub fn run_op_graph(&self, ops: &[MatmulOp<'_>]) -> Result<RunStats> {
        self.run_ops_impl(ops, true)
    }

    fn run_calls_impl(
        &self,
        calls: &[MatmulCall<'_>],
        with_dependency_barriers: bool,
    ) -> Result<RunStats> {
        match calls {
            [] => self.run_ops_impl(&[], with_dependency_barriers),
            [call] => self.run_ops_impl(&[MatmulOp::new(*call)], with_dependency_barriers),
            _ => {
                let ops: Vec<MatmulOp<'_>> = calls.iter().copied().map(MatmulOp::new).collect();
                self.run_ops_impl(&ops, with_dependency_barriers)
            }
        }
    }

    fn run_ops_impl(
        &self,
        ops: &[MatmulOp<'_>],
        with_dependency_barriers: bool,
    ) -> Result<RunStats> {
        if ops.is_empty() {
            return Ok(RunStats {
                gpu_time_ns: None,
                n_calls: 0,
                total_flops: 0,
            });
        }
        if ops.len() > self.max_calls_per_submit as usize {
            bail!(
                "run_ops: {} ops > max_calls_per_submit {}",
                ops.len(),
                self.max_calls_per_submit
            );
        }

        for op in ops {
            self.validate_op_context(op)?;
        }

        let resolved = ResolvedMatmulBatch::from_ops(ops)?;
        let resolved = resolved.as_slice();
        let total_flops = total_flops(resolved)?;

        // Online tuning: the first time a new shape arrives as a plain
        // single non-accumulating op, measure all candidates against
        // it (each candidate fully overwrites C, so the caller's
        // result stays correct) and record the winner.  Every later
        // call — including batched, accumulating, and epilogue ops on
        // the same shape — picks the winner up via `select_kernel`.
        if self.tune_enabled
            && let [op] = ops
            && !op.call.accumulate
            && op.epilogue.is_none()
            && op.normed_a.is_none()
            && op.store.is_none()
            && !resolved[0].a_f16
        {
            let dims = &resolved[0];
            let key = TuneKey {
                batch: dims.batch,
                m: dims.m,
                n: dims.n,
                k: dims.k,
                b_f16: dims.b_f16,
            };
            if !self.pipeline.is_tuned(key)
                && self.ctx.timestamps_supported
                && let Err(err) = self.tune_op(op, dims)
            {
                log::warn!(
                    "tensor-ash: tuning failed for B={} {}x{}x{}: {err}",
                    dims.batch,
                    dims.m,
                    dims.n,
                    dims.k
                );
            }
        }

        // Resolve every op's complete route once.  The split-K2 leg is
        // only eligible for plain non-accumulating ops that are either
        // alone in the submission or part of a graph (where the
        // two-stage path's internal barrier costs nothing extra).
        let plans: Vec<OpPlan> = ops
            .iter()
            .zip(resolved.iter())
            .map(|(op, dims)| {
                let eligible = !op.call.accumulate
                    && op.epilogue.is_none()
                    && op.normed_a.is_none()
                    && op.store.is_none()
                    && (with_dependency_barriers || ops.len() == 1);
                let plan = self.plan_matmul(dims, eligible)?;
                Ok(self.demote_for_op(op, dims, plan))
            })
            .collect::<Result<_>>()?;

        // Measured reduction routing: a lone plain op whose route says
        // two-stage split-K goes through the dedicated entry point.
        if !with_dependency_barriers
            && let [plan] = plans.as_slice()
            && let Some(splits) = plan.splitk2
        {
            return self.run_matmuls_split_k2(ops[0].call, splits);
        }

        let mut slot = self.checkout_slot();
        let result =
            self.record_and_run(&mut slot, ops, resolved, &plans, with_dependency_barriers);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: ops.len(),
            total_flops,
        })
    }

    fn record_and_run(
        &self,
        slot: &mut Slot,
        ops: &[MatmulOp<'_>],
        resolved: &[ResolvedMatmul],
        plans: &[OpPlan],
        with_dependency_barriers: bool,
    ) -> Result<Option<u64>> {
        debug_assert_eq!(ops.len(), resolved.len());
        debug_assert_eq!(ops.len(), plans.len());
        let descriptor_set_count = ops.len();

        // Graph submissions route individual ops through the two-stage
        // split-K when their plan says so.  Plan scratch offsets up
        // front so all routed ops share one slot-local buffer.
        let mut graph_plans: Vec<Option<(splitk2::SplitK2Dispatch, u64)>> = Vec::new();
        let mut graph_scratch_addr = 0u64;
        let mut graph_sk2: Option<&SplitK2Pipeline> = None;
        if with_dependency_barriers && self.ctx.buffer_device_address_enabled {
            let mut offset = 0u64;
            let mut sk2_plans = vec![None; ops.len()];
            for (i, (plan, dims)) in plans.iter().zip(resolved.iter()).enumerate() {
                let Some(splits) = plan.splitk2 else {
                    continue;
                };
                let pipe = match graph_sk2 {
                    Some(p) => p,
                    None => {
                        let p = self.split_k2_pipeline()?;
                        graph_sk2 = Some(p);
                        p
                    }
                };
                if let Ok(sk2) = splitk2::SplitK2Dispatch::plan(&self.ctx, pipe, dims, splits) {
                    let Some(end) = splitk2::checked_auto_scratch_end(offset, sk2.scratch_bytes)
                    else {
                        continue;
                    };
                    sk2_plans[i] = Some((sk2, offset));
                    offset = end;
                }
            }
            if offset > 0 {
                graph_scratch_addr = self.ensure_splitk2_scratch(slot, offset)?;
                graph_plans = sk2_plans;
            }
        }
        let graph_splitk2 = match (graph_sk2, graph_plans.is_empty()) {
            (Some(pipeline), false) => Some(recording::GraphSplitK2 {
                pipeline,
                scratch_addr: graph_scratch_addr,
                plans: &graph_plans,
            }),
            _ => None,
        };

        unsafe {
            // Point the slot's pre-allocated descriptor sets at this
            // submission's tensors.  By now the previous submission on
            // this slot has fenced out (see submit_recorded), so the
            // sets are safe to update.  `update_matmul_descriptor_sets`
            // skips writes for any call whose resolved kernel is a BDA
            // kernel (those shaders ignore the descriptor set), so a
            // pure-BDA submission incurs zero `vkUpdateDescriptorSets`
            // overhead here.
            update_matmul_descriptor_sets(
                &self.ctx,
                &self.pipeline,
                &slot.descriptor_sets[..ops.len()],
                ops,
                plans,
            );

            self.submit_timed(slot, "get_query_pool_results", |_dev, cb, slot| {
                let descriptor_sets = &slot.descriptor_sets[..descriptor_set_count];
                if with_dependency_barriers {
                    record_matmul_graph_commands(
                        &self.ctx,
                        &self.pipeline,
                        cb,
                        descriptor_sets,
                        ops,
                        resolved,
                        plans,
                        graph_splitk2.as_ref(),
                    )?;
                } else {
                    record_matmul_commands(
                        &self.ctx,
                        &self.pipeline,
                        cb,
                        descriptor_sets,
                        ops,
                        resolved,
                        plans,
                    )?;
                }
                Ok(())
            })
        }
    }
}
