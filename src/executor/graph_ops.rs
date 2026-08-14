//! Mixed-op single-submission graphs: a dependent chain of matmuls and
//! model ops recorded into ONE command buffer with compute barriers
//! between consecutive ops, submitted and awaited once.
//!
//! This is the decode-latency machinery: a llama-style token step is
//! ~350 small dispatches, and paying a submit + fence wait per op costs
//! ~10 µs each.  One submission amortizes that to a single wait; the
//! ops still serialize on the GPU exactly as the per-op path would.

use anyhow::{Result, bail};
use ash::vk;

use crate::matmul::{MatmulOp, ResolvedMatmul, RunStats};
use crate::tensor::Tensor;

use super::elementwise::ElementwiseDispatch;
use super::recording::{record_compute_to_compute_barrier, record_one_matmul};
use super::{CopyDesc, Executor, OpPlan, RopeDesc, SoftmaxMask};

/// One step of a mixed-op graph.  All variants execute in submission
/// order with a full compute barrier between consecutive steps, so a
/// step may freely read any earlier step's output.
pub enum ExecOp<'t> {
    /// A matmul (optionally with a fused epilogue).  BDA routes only —
    /// descriptor-bound kernels cannot be recorded without per-op sets.
    Matmul(MatmulOp<'t>),
    RmsNorm {
        input: &'t Tensor,
        weight: &'t Tensor,
        output: &'t Tensor,
        eps: f32,
    },
    LayerNorm {
        input: &'t Tensor,
        weight: &'t Tensor,
        bias: &'t Tensor,
        output: &'t Tensor,
        eps: f32,
    },
    SoftmaxRows {
        input: &'t Tensor,
        output: &'t Tensor,
        scale: f32,
        mask: SoftmaxMask,
    },
    Rope {
        input: &'t Tensor,
        table: &'t Tensor,
        output: &'t Tensor,
        desc: RopeDesc,
    },
    CopyStrided {
        src: &'t Tensor,
        dst: &'t Tensor,
        desc: CopyDesc,
    },
}

enum Planned<'t, 'p> {
    Matmul {
        op: &'p MatmulOp<'t>,
        dims: ResolvedMatmul,
        plan: OpPlan,
    },
    Elementwise(ElementwiseDispatch),
}

/// Buffers an op touches, for hazard-aware barrier insertion.
#[derive(Default)]
struct Access {
    reads: Vec<vk::Buffer>,
    writes: Vec<vk::Buffer>,
}

impl Access {
    fn read(mut self, tensor: &Tensor) -> Self {
        self.reads.push(tensor.raw_buffer());
        self
    }
    fn write(mut self, tensor: &Tensor) -> Self {
        self.writes.push(tensor.raw_buffer());
        self
    }
}

impl Executor {
    /// Run a dependent chain of mixed ops as one submission.  Every op
    /// is fully validated and planned before anything is recorded, so
    /// failures cannot leave partial work on the queue.
    pub fn run_exec_ops(&self, ops: &[ExecOp<'_>]) -> Result<RunStats> {
        if ops.is_empty() {
            bail!("run_exec_ops: empty op list");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("run_exec_ops: requires bufferDeviceAddress");
        }

        let mut planned = Vec::with_capacity(ops.len());
        let mut accesses = Vec::with_capacity(ops.len());
        let mut total_flops = 0u64;
        for op in ops {
            accesses.push(match op {
                ExecOp::Matmul(op) => {
                    let mut access = Access::default()
                        .read(op.call.a)
                        .read(op.call.b)
                        .write(op.call.c);
                    if op.call.accumulate {
                        access = access.read(op.call.c);
                    }
                    if let Some(bias) = op.epilogue.bias {
                        access = access.read(bias);
                    }
                    if let Some(d) = op.epilogue.d_tensor() {
                        access = access.read(d);
                    }
                    access
                }
                ExecOp::RmsNorm {
                    input,
                    weight,
                    output,
                    ..
                } => Access::default().read(input).read(weight).write(output),
                ExecOp::LayerNorm {
                    input,
                    weight,
                    bias,
                    output,
                    ..
                } => Access::default()
                    .read(input)
                    .read(weight)
                    .read(bias)
                    .write(output),
                ExecOp::SoftmaxRows { input, output, .. } => {
                    Access::default().read(input).write(output)
                }
                ExecOp::Rope {
                    input,
                    table,
                    output,
                    ..
                } => Access::default().read(input).read(table).write(output),
                ExecOp::CopyStrided { src, dst, .. } => Access::default().read(src).write(dst),
            });
            planned.push(match op {
                ExecOp::Matmul(op) => {
                    self.validate_op_context(op)?;
                    let dims = ResolvedMatmul::from_op(op)?;
                    let plan = self.demote_for_epilogue(
                        &op.epilogue,
                        &dims,
                        self.plan_shape(dims.batch, dims.m, dims.n, dims.k, dims.b_f16, false),
                    );
                    let kernel = self.pipeline.kernel_at(plan.kernel);
                    if kernel.uses_descriptors {
                        bail!(
                            "run_exec_ops: kernel '{}' binds descriptor sets and cannot \
                             be recorded in a mixed graph (unset ML_KERNEL or pick a \
                             BDA kernel)",
                            kernel.name
                        );
                    }
                    total_flops = total_flops.saturating_add(dims.total_flops);
                    Planned::Matmul { op, dims, plan }
                }
                ExecOp::RmsNorm {
                    input,
                    weight,
                    output,
                    eps,
                } => {
                    let pipeline = self
                        .norm_common("run_rms_norm", input, weight, None, output)?
                        .rmsnorm
                        .pipeline;
                    Planned::Elementwise(
                        self.plan_norm(pipeline, input, weight, None, output, *eps)?,
                    )
                }
                ExecOp::LayerNorm {
                    input,
                    weight,
                    bias,
                    output,
                    eps,
                } => {
                    let pipeline = self
                        .norm_common("run_layer_norm", input, weight, Some(bias), output)?
                        .layernorm
                        .pipeline;
                    Planned::Elementwise(self.plan_norm(
                        pipeline,
                        input,
                        weight,
                        Some(bias),
                        output,
                        *eps,
                    )?)
                }
                ExecOp::SoftmaxRows {
                    input,
                    output,
                    scale,
                    mask,
                } => Planned::Elementwise(self.plan_softmax_rows(input, output, *scale, *mask)?),
                ExecOp::Rope {
                    input,
                    table,
                    output,
                    desc,
                } => Planned::Elementwise(self.plan_rope(input, table, output, *desc)?),
                ExecOp::CopyStrided { src, dst, desc } => {
                    Planned::Elementwise(self.plan_copy_strided(src, dst, *desc)?)
                }
            });
        }

        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (exec graph)",
                |_dev, cb, _slot| {
                    let mut bound = vk::Pipeline::null();
                    // Hazard tracking: barrier only when this op reads
                    // or overwrites something written since the last
                    // barrier (RAW/WAW), or writes something read since
                    // (WAR).  Independent neighbours overlap on the
                    // GPU, which both removes ~7 us of drain per
                    // avoided barrier and lets tiny dispatches fill
                    // idle SMs.
                    let mut pending_writes: Vec<vk::Buffer> = Vec::new();
                    let mut pending_reads: Vec<vk::Buffer> = Vec::new();
                    for (index, step) in planned.iter().enumerate() {
                        let access = &accesses[index];
                        let hazard = access
                            .reads
                            .iter()
                            .chain(&access.writes)
                            .any(|b| pending_writes.contains(b))
                            || access.writes.iter().any(|b| pending_reads.contains(b));
                        if index > 0 && hazard {
                            record_compute_to_compute_barrier(&self.ctx, cb);
                            pending_writes.clear();
                            pending_reads.clear();
                        }
                        pending_reads.extend(&access.reads);
                        pending_writes.extend(&access.writes);
                        match step {
                            Planned::Matmul { op, dims, plan } => {
                                record_one_matmul(
                                    &self.ctx,
                                    &self.pipeline,
                                    cb,
                                    vk::DescriptorSet::null(),
                                    op,
                                    dims,
                                    self.pipeline.kernel_at(plan.kernel),
                                    &mut bound,
                                )?;
                            }
                            Planned::Elementwise(dispatch) => {
                                self.record_elementwise(cb, dispatch);
                                // The elementwise bind invalidates the
                                // matmul bind-tracking.
                                bound = vk::Pipeline::null();
                            }
                        }
                    }
                    Ok(())
                },
            )
        }?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: ops.len(),
            total_flops,
        })
    }
}
