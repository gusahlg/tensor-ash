//! Mixed-op single-submission graphs: a dependent chain of matmuls and
//! model ops recorded into ONE command buffer with compute barriers
//! between consecutive ops, submitted and awaited once.
//!
//! This is the decode-latency machinery: a llama-style token step is
//! ~350 small dispatches, and paying a submit + fence wait per op costs
//! ~10 µs each.  One submission amortizes that to a single wait; the
//! ops still serialize on the GPU exactly as the per-op path would.

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::matmul::{MatmulOp, ResolvedMatmul, RunStats};
use crate::tensor::Tensor;

use super::elementwise::ElementwiseDispatch;
use super::prepared::PreparedOps;
use super::recording::{record_compute_to_compute_barrier, record_one_matmul};
use super::{
    AttnDecodeDesc, BinaryOp, CopyDesc, Executor, OpPlan, RopeDesc, RopeScatterDesc, SoftmaxMask,
};

/// A 4-byte host-visible, device-readable position cell for replayable
/// decode graphs (see [`Executor::prepare_exec_ops`]).  The ops that
/// depend on the token position take its [`device_address`]
/// (`RopeDesc::pos_addr`, `CopyDesc::pos_addr`,
/// `AttnDecodeDesc::pos_addr`); the recorded shaders read the current
/// value each execution, so the host just [`set`]s the new position
/// between replays instead of re-recording push constants.
///
/// Host writes made before a `vkQueueSubmit` are visible to that
/// submission (the submit performs the domain operation; `set` flushes
/// non-coherent memory), so no barrier is needed — only the usual
/// "don't write while a replay is in flight" discipline, which
/// [`PreparedOps::run`]'s fence wait already enforces.
///
/// [`device_address`]: Self::device_address
/// [`set`]: Self::set
pub struct PosBuffer {
    buffer: Buffer,
}

impl PosBuffer {
    /// Write a new position for the next submission.
    pub fn set(&self, value: u32) -> Result<()> {
        self.buffer.write_pod_slice(&[value])
    }

    /// GPU pointer for the shaders' position indirection; store it in
    /// the descs' `pos_addr` fields.  The buffer must outlive every
    /// execution of any op that captured this address.
    pub fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }
}

impl Executor {
    /// Allocate a [`PosBuffer`] on this executor's device.
    pub fn create_pos_buffer(&self) -> Result<PosBuffer> {
        if !self.ctx.buffer_device_address_enabled {
            bail!("create_pos_buffer: requires bufferDeviceAddress");
        }
        let buffer = Buffer::new(
            &self.ctx,
            std::mem::size_of::<u32>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferLocation::Host,
        )
        .context("create_pos_buffer")?;
        Ok(PosBuffer { buffer })
    }
}

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
    /// Fused RoPE + strided scatter (see
    /// [`Executor::run_rope_scatter`]): rotate and write straight into
    /// a strided destination (KV-cache append) in one dispatch.
    RopeScatter {
        input: &'t Tensor,
        table: &'t Tensor,
        dst: &'t Tensor,
        desc: RopeDesc,
        scatter: RopeScatterDesc,
    },
    CopyStrided {
        src: &'t Tensor,
        dst: &'t Tensor,
        desc: CopyDesc,
    },
    Binary {
        a: &'t Tensor,
        b: &'t Tensor,
        out: &'t Tensor,
        op: BinaryOp,
    },
    /// Fused split-K decode attention (see
    /// [`Executor::run_attn_decode`]).  Plans as TWO dispatches
    /// (stage 1 writes per-chunk partials to `scratch`, stage 2 merges
    /// them into `out`) whose access sets force the hazard barrier
    /// between them.
    AttnDecode {
        q: &'t Tensor,
        kt: &'t Tensor,
        v: &'t Tensor,
        scratch: &'t Tensor,
        out: &'t Tensor,
        desc: AttnDecodeDesc,
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
        let (planned, accesses, total_flops) = self.plan_exec_ops(ops)?;

        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (exec graph)",
                |_dev, cb, _slot| self.record_exec_ops(cb, &planned, &accesses),
            )
        }?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: ops.len(),
            total_flops,
        })
    }

    /// [`run_exec_ops`](Self::run_exec_ops), recorded ONCE for repeated
    /// replay — the decode-loop machinery: the whole token's op chain
    /// becomes one resubmittable command buffer, so each token costs a
    /// single `vkQueueSubmit` with zero re-validation, re-planning, or
    /// re-recording on the host.
    ///
    /// Everything the recording bakes in (device addresses, push
    /// constants, grids) is fixed at prepare time; token-position
    /// dependence is expressed through `pos`: descs whose `pos_addr`
    /// field carries `pos.device_address()` read the position cell at
    /// execution time, so `pos.set(p)` before each
    /// [`run`](PreparedOps::run) retargets the replay.  Every
    /// `pos_addr` used by `ops` must match `pos` (validated here).
    pub fn prepare_exec_ops<'e, 't>(
        &'e self,
        ops: &[ExecOp<'t>],
        pos: Option<&'t PosBuffer>,
    ) -> Result<PreparedOps<'e, 't>> {
        if ops.is_empty() {
            bail!("prepare_exec_ops: empty op list");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("prepare_exec_ops: requires bufferDeviceAddress");
        }
        let pos_addr = pos.map_or(0, PosBuffer::device_address);
        for op in ops {
            let used = match op {
                ExecOp::Rope { desc, .. } | ExecOp::RopeScatter { desc, .. } => desc.pos_addr,
                ExecOp::CopyStrided { desc, .. } => desc.pos_addr,
                ExecOp::AttnDecode { desc, .. } => desc.pos_addr,
                _ => 0,
            };
            if used != 0 && used != pos_addr {
                bail!(
                    "prepare_exec_ops: op references pos_addr {used:#x} but the prepared \
                     position buffer is {pos_addr:#x} (pass the same PosBuffer)"
                );
            }
        }
        let (planned, accesses, total_flops) = self.plan_exec_ops(ops)?;
        self.prepare_recorded(ops.len(), total_flops, |cb| {
            self.record_exec_ops(cb, &planned, &accesses)
        })
    }

    /// Validate and plan every op of a mixed graph; nothing touches the
    /// queue, so failures cannot leave partial work behind.
    fn plan_exec_ops<'t, 'p>(
        &self,
        ops: &'p [ExecOp<'t>],
    ) -> Result<(Vec<Planned<'t, 'p>>, Vec<Access>, u64)> {
        let mut planned = Vec::with_capacity(ops.len() + 1);
        let mut accesses = Vec::with_capacity(ops.len() + 1);
        let mut total_flops = 0u64;
        for op in ops {
            // Multi-dispatch ops push one (planned, access) pair per
            // dispatch so the hazard tracker sees their internal
            // dependency.
            if let ExecOp::AttnDecode {
                q,
                kt,
                v,
                scratch,
                out,
                desc,
            } = op
            {
                let (stage1, combine) = self.plan_attn_decode(q, kt, v, scratch, out, *desc)?;
                accesses.push(Access::default().read(q).read(kt).read(v).write(scratch));
                planned.push(Planned::Elementwise(stage1));
                accesses.push(Access::default().read(scratch).write(out));
                planned.push(Planned::Elementwise(combine));
                continue;
            }
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
                    if let Some((norm_weight, _)) = op.normed_a {
                        access = access.read(norm_weight);
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
                ExecOp::RopeScatter {
                    input, table, dst, ..
                } => Access::default().read(input).read(table).write(dst),
                ExecOp::CopyStrided { src, dst, .. } => Access::default().read(src).write(dst),
                ExecOp::Binary { a, b, out, .. } => Access::default().read(a).read(b).write(out),
                ExecOp::AttnDecode { .. } => unreachable!("handled above"),
            });
            planned.push(match op {
                ExecOp::Matmul(op) => {
                    self.validate_op_context(op)?;
                    let dims = ResolvedMatmul::from_op(op)?;
                    let plan = self.demote_for_op(
                        op,
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
                ExecOp::RopeScatter {
                    input,
                    table,
                    dst,
                    desc,
                    scatter,
                } => Planned::Elementwise(
                    self.plan_rope_scatter(input, table, dst, *desc, *scatter)?,
                ),
                ExecOp::CopyStrided { src, dst, desc } => {
                    Planned::Elementwise(self.plan_copy_strided(src, dst, *desc)?)
                }
                ExecOp::Binary { a, b, out, op } => {
                    Planned::Elementwise(self.plan_binary(a, b, out, *op)?)
                }
                ExecOp::AttnDecode { .. } => unreachable!("handled above"),
            });
        }
        Ok((planned, accesses, total_flops))
    }

    /// Record a planned mixed graph into `cb` with hazard-aware
    /// barriers.  Shared verbatim by the immediate
    /// ([`run_exec_ops`](Self::run_exec_ops)) and prepared
    /// ([`prepare_exec_ops`](Self::prepare_exec_ops)) paths.
    fn record_exec_ops(
        &self,
        cb: vk::CommandBuffer,
        planned: &[Planned<'_, '_>],
        accesses: &[Access],
    ) -> Result<()> {
        let mut bound = vk::Pipeline::null();
        // Hazard tracking: barrier only when this op reads or
        // overwrites something written since the last barrier
        // (RAW/WAW), or writes something read since (WAR).
        // Independent neighbours overlap on the GPU, which both
        // removes ~7 us of drain per avoided barrier and lets tiny
        // dispatches fill idle SMs.
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
                    // The elementwise bind invalidates the matmul
                    // bind-tracking.
                    bound = vk::Pipeline::null();
                }
            }
        }
        Ok(())
    }
}
