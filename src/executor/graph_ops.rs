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

use super::elementwise::{ElementwiseDispatch, GEMV_CHAIN_MAX_JOBS};
use super::prepared::PreparedOps;
use super::recording::{record_compute_to_compute_barrier, record_one_matmul};
use super::{
    AttnDecodeDesc, BinaryOp, CopyDesc, Executor, FlashAttentionDesc, OpPlan, PrefillQkvPackDesc,
    RopeDesc, RopeScatterDesc, SoftmaxMask,
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

/// A 4-byte u32 cell that is BOTH device-writable and host-readable —
/// the [`PosBuffer`] machinery pointed the other way.  The GPU-argmax
/// decode loop writes the chosen token id here
/// ([`ExecOp::Argmax`] / [`Executor::run_argmax`]), a chained
/// [`ExecOp::EmbedGather`] reads it back on-device for the next token's
/// embedding, and the host [`read`]s ONE u32 after the fence instead of
/// downloading the logits.
///
/// Device writes become visible to the host once the submission's fence
/// wait returns (the fence signal is the domain operation; `read`
/// invalidates non-coherent memory), so the usual "wait before you
/// read" discipline — which [`super::PreparedOps::run`]'s fence already
/// enforces — is the only requirement.
///
/// [`read`]: Self::read
pub struct HostU32Buffer {
    buffer: Buffer,
}

impl HostU32Buffer {
    /// Read the current value (call only after the writing submission's
    /// fence wait has returned).
    pub fn read(&self) -> Result<u32> {
        let mut value = [0u32];
        self.buffer.read_pod_slice(&mut value)?;
        Ok(value[0])
    }

    /// Host-side write, for seeding the cell before a submission that
    /// only reads it (e.g. a standalone embed-gather).
    pub fn set(&self, value: u32) -> Result<()> {
        self.buffer.write_pod_slice(&[value])
    }

    /// GPU pointer for the shaders' indirection.  The buffer must
    /// outlive every execution of any op that captured this address.
    pub fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

fn create_u32_cell(exec: &Executor, label: &str) -> Result<Buffer> {
    if !exec.ctx.buffer_device_address_enabled {
        bail!("{label}: requires bufferDeviceAddress");
    }
    Buffer::new(
        &exec.ctx,
        std::mem::size_of::<u32>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        BufferLocation::Host,
    )
    .context(label.to_string())
}

impl Executor {
    /// Allocate a [`PosBuffer`] on this executor's device.
    pub fn create_pos_buffer(&self) -> Result<PosBuffer> {
        Ok(PosBuffer {
            buffer: create_u32_cell(self, "create_pos_buffer")?,
        })
    }

    /// Allocate a [`HostU32Buffer`] on this executor's device.
    pub fn create_host_u32_buffer(&self) -> Result<HostU32Buffer> {
        Ok(HostU32Buffer {
            buffer: create_u32_cell(self, "create_host_u32_buffer")?,
        })
    }

    /// Host-visible u32 list (prefill token ids).  `write` is a
    /// coherent/flushed CPU store — no staging submit — so a 512-token
    /// prompt is 2 KiB instead of a 4 MiB embedding upload.
    pub fn create_token_id_buffer(&self, cap: u32) -> Result<TokenIdBuffer> {
        if cap == 0 {
            bail!("create_token_id_buffer: cap must be non-zero");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("create_token_id_buffer: requires bufferDeviceAddress");
        }
        let bytes = match (cap as u64).checked_mul(std::mem::size_of::<u32>() as u64) {
            Some(bytes) => bytes,
            None => bail!("create_token_id_buffer: cap overflows"),
        };
        Ok(TokenIdBuffer {
            buffer: Buffer::new(
                &self.ctx,
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                BufferLocation::Host,
            )
            .context("create_token_id_buffer")?,
            cap,
        })
    }
}

/// Host-visible list of token ids for [`ExecOp::EmbedGatherRows`].
pub struct TokenIdBuffer {
    buffer: Buffer,
    cap: u32,
}

impl TokenIdBuffer {
    /// Overwrite the first `ids.len()` slots.  The gather shader reads
    /// `n_tokens` entries, so the caller must pass the same count.
    pub fn write(&self, ids: &[u32]) -> Result<()> {
        if ids.len() as u32 > self.cap {
            bail!(
                "TokenIdBuffer::write: {} ids exceeds cap {}",
                ids.len(),
                self.cap
            );
        }
        self.buffer.write_pod_slice(ids)
    }

    pub fn device_address(&self) -> u64 {
        self.buffer.device_address()
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// One step of a mixed-op graph.  All variants execute in submission
/// order with a full compute barrier between consecutive steps, so a
/// step may freely read any earlier step's output.
pub enum ExecOp<'t> {
    /// A matmul (optionally with a fused epilogue).  BDA routes only —
    /// descriptor-bound kernels cannot be recorded without per-op sets.
    Matmul(MatmulOp<'t>),
    /// RMSNorm over the last dimension (see [`Executor::run_rms_norm`]).
    /// In-place (`input == output`) is safe.
    RmsNorm {
        input: &'t Tensor,
        weight: &'t Tensor,
        output: &'t Tensor,
        eps: f32,
    },
    /// LayerNorm over the last dimension (see
    /// [`Executor::run_layer_norm`]).  In-place is safe.
    LayerNorm {
        input: &'t Tensor,
        weight: &'t Tensor,
        bias: &'t Tensor,
        output: &'t Tensor,
        eps: f32,
    },
    /// Masked row softmax (see [`Executor::run_softmax_rows`]).
    /// In-place is safe.
    SoftmaxRows {
        input: &'t Tensor,
        output: &'t Tensor,
        scale: f32,
        mask: SoftmaxMask,
    },
    /// Rotary position embedding (see [`Executor::run_rope`]).
    /// In-place is safe.
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
    /// Strided 3D copy (see [`Executor::run_copy_strided`]); `src` and
    /// `dst` must be different tensors.
    CopyStrided {
        src: &'t Tensor,
        dst: &'t Tensor,
        desc: CopyDesc,
    },
    /// Binary elementwise combine (see [`Executor::run_binary`]);
    /// `out` may alias either input.
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
    /// Fused causal prefill attention (see
    /// [`Executor::run_flash_attention`]): one dispatch, CM2-first
    /// routing exactly like the standalone call.
    FlashAttn {
        q: &'t Tensor,
        kt: &'t Tensor,
        v: &'t Tensor,
        out: &'t Tensor,
        desc: FlashAttentionDesc,
    },
    /// Fused wide-prefill QKV pack (see
    /// [`Executor::run_prefill_qkv_pack`]): RoPE Q to head-major
    /// `[H, T, dh]`, RoPE-scatter K into the Kt cache, copy V into
    /// the V cache.
    PrefillQkvPack {
        qkv: &'t Tensor,
        table: &'t Tensor,
        q: &'t Tensor,
        kt: &'t Tensor,
        v: &'t Tensor,
        desc: PrefillQkvPackDesc,
    },
    /// Greedy argmax (see [`Executor::run_argmax`]): writes the index
    /// of `input`'s largest element into the host-readable `result`
    /// cell (ties keep the largest index, matching Rust's
    /// `max_by(total_cmp)`).
    Argmax {
        input: &'t Tensor,
        result: &'t HostU32Buffer,
    },
    /// Embedding-row gather (see [`Executor::run_embed_gather`]): the
    /// u32 `token` cell is read on-device, so chaining this after an
    /// [`Argmax`](Self::Argmax) closes the decode loop entirely on the
    /// GPU.
    EmbedGather {
        token: &'t HostU32Buffer,
        table: &'t Tensor,
        out: &'t Tensor,
    },
    /// Multi-row embedding gather: `tokens[0..n]` indexes rows of
    /// `table` into `out` shaped `[n, embd]` (f16 or f32).  Prefill
    /// uses this so the 512-token prompt is a 2 KiB id write plus a
    /// gather, matching CUDA's on-device embedding table.
    EmbedGatherRows {
        tokens: &'t TokenIdBuffer,
        table: &'t Tensor,
        out: &'t Tensor,
    },
    /// Persistent GEMV chain: up to [`GEMV_CHAIN_MAX_JOBS`] M=1
    /// f16-weight row GEMVs in ONE dispatch. Dependent groups wait on
    /// an in-kernel device-scope quorum; independent neighbours share
    /// a flattened tile space and stay concurrent.  Requires
    /// `ML_DEVICE_SCOPE=1`.  llama-ash decode stays on packed row
    /// GEMVs by default — enabling the chain globally scrambled CM2
    /// flash numerics.
    GemvChain {
        jobs: Box<[MatmulOp<'t>; GEMV_CHAIN_MAX_JOBS]>,
        n: u32,
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
    fn read_cell(mut self, cell: &HostU32Buffer) -> Self {
        self.reads.push(cell.buffer().raw_buffer());
        self
    }
    fn read_raw(mut self, buffer: vk::Buffer) -> Self {
        self.reads.push(buffer);
        self
    }
    fn write_cell(mut self, cell: &HostU32Buffer) -> Self {
        self.writes.push(cell.buffer().raw_buffer());
        self
    }

    /// RAW/WAW/WAR test against the work recorded since the last
    /// barrier — the single hazard definition shared by recording
    /// ([`Executor::run_exec_ops`]) and the plan-only census
    /// ([`Executor::exec_ops_barrier_count`]).
    fn hazard_with(&self, pending_writes: &[vk::Buffer], pending_reads: &[vk::Buffer]) -> bool {
        self.reads
            .iter()
            .chain(&self.writes)
            .any(|b| pending_writes.contains(b))
            || self.writes.iter().any(|b| pending_reads.contains(b))
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
                ExecOp::Matmul(op) => op.store.desc().pos_addr,
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
        let retain: Vec<std::sync::Arc<crate::buffer::Buffer>> = planned
            .iter()
            .filter_map(|step| match step {
                Planned::Elementwise(dispatch) => dispatch.retain.clone(),
                Planned::Matmul { .. } => None,
            })
            .collect();
        self.prepare_recorded_retain(ops.len(), total_flops, retain, |cb| {
            self.record_exec_ops(cb, &planned, &accesses)
        })
    }

    /// Plan a mixed graph WITHOUT executing it and report its dispatch
    /// census: `(dispatches, barriers)` — the op count after
    /// multi-dispatch expansion (`AttnDecode` plans as two) and the
    /// number of compute barriers the hazard tracker would record
    /// between them.  Nothing touches the queue.
    ///
    /// Diagnostics for performance accounting: a full compute barrier
    /// drains the GPU (~7.7 µs measured on GA104), so pricing a graph
    /// requires knowing how many the hazard tracker will emit.  The
    /// perf-thesis harness (`ml_bench thesis`) uses this to separate
    /// barrier drain from kernel time.
    pub fn exec_ops_barrier_count(&self, ops: &[ExecOp<'_>]) -> Result<(usize, usize)> {
        if ops.is_empty() {
            bail!("exec_ops_barrier_count: empty op list");
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("exec_ops_barrier_count: requires bufferDeviceAddress");
        }
        let (planned, accesses, _total_flops) = self.plan_exec_ops(ops)?;
        let mut pending_writes: Vec<vk::Buffer> = Vec::new();
        let mut pending_reads: Vec<vk::Buffer> = Vec::new();
        let mut barriers = 0usize;
        for (index, access) in accesses.iter().enumerate() {
            if index > 0 && access.hazard_with(&pending_writes, &pending_reads) {
                barriers += 1;
                pending_writes.clear();
                pending_reads.clear();
            }
            pending_reads.extend(&access.reads);
            pending_writes.extend(&access.writes);
        }
        Ok((planned.len(), barriers))
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
            // Each arm produces matching (Planned, Access) entries at
            // the same index; multi-dispatch ops push one pair per
            // dispatch so the hazard tracker sees their internal
            // dependency.
            let (step, access) = match op {
                ExecOp::Matmul(op) => {
                    self.validate_op_context(op)?;
                    let dims = ResolvedMatmul::from_op(op)?;
                    let plan = self.demote_for_op(op, &dims, self.plan_matmul(&dims, false)?);
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
                    let mut access = Access::default().read(op.call.a).read(op.call.b);
                    // The scatter store modes redirect the C store into
                    // the cache tensor; C itself is never touched, so
                    // the hazard tracker must see the cache write (the
                    // attention read of it barriers correctly) and not
                    // a phantom C write.
                    match op.store.dst() {
                        Some(dst) => access = access.write(dst),
                        None => access = access.write(op.call.c),
                    }
                    if let Some(table) = op.store.table() {
                        access = access.read(table);
                    }
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
                    (Planned::Matmul { op, dims, plan }, access)
                }
                ExecOp::RmsNorm {
                    input,
                    weight,
                    output,
                    eps,
                } => (
                    Planned::Elementwise(self.plan_norm(
                        "run_rms_norm",
                        input,
                        weight,
                        None,
                        output,
                        *eps,
                    )?),
                    Access::default().read(input).read(weight).write(output),
                ),
                ExecOp::LayerNorm {
                    input,
                    weight,
                    bias,
                    output,
                    eps,
                } => (
                    Planned::Elementwise(self.plan_norm(
                        "run_layer_norm",
                        input,
                        weight,
                        Some(bias),
                        output,
                        *eps,
                    )?),
                    Access::default()
                        .read(input)
                        .read(weight)
                        .read(bias)
                        .write(output),
                ),
                ExecOp::SoftmaxRows {
                    input,
                    output,
                    scale,
                    mask,
                } => (
                    Planned::Elementwise(self.plan_softmax_rows(input, output, *scale, *mask)?),
                    Access::default().read(input).write(output),
                ),
                ExecOp::Rope {
                    input,
                    table,
                    output,
                    desc,
                } => (
                    Planned::Elementwise(self.plan_rope(input, table, output, *desc)?),
                    Access::default().read(input).read(table).write(output),
                ),
                ExecOp::RopeScatter {
                    input,
                    table,
                    dst,
                    desc,
                    scatter,
                } => (
                    Planned::Elementwise(
                        self.plan_rope_scatter(input, table, dst, *desc, *scatter)?,
                    ),
                    Access::default().read(input).read(table).write(dst),
                ),
                ExecOp::CopyStrided { src, dst, desc } => (
                    Planned::Elementwise(self.plan_copy_strided(src, dst, *desc)?),
                    Access::default().read(src).write(dst),
                ),
                ExecOp::Binary { a, b, out, op } => (
                    Planned::Elementwise(self.plan_binary(a, b, out, *op)?),
                    Access::default().read(a).read(b).write(out),
                ),
                ExecOp::FlashAttn {
                    q,
                    kt,
                    v,
                    out,
                    desc,
                } => (
                    Planned::Elementwise(self.plan_flash_attention(q, kt, v, out, *desc, false)?),
                    Access::default().read(q).read(kt).read(v).write(out),
                ),
                ExecOp::PrefillQkvPack {
                    qkv,
                    table,
                    q,
                    kt,
                    v,
                    desc,
                } => (
                    Planned::Elementwise(self.plan_prefill_qkv_pack(qkv, table, q, kt, v, *desc)?),
                    Access::default()
                        .read(qkv)
                        .read(table)
                        .write(q)
                        .write(kt)
                        .write(v),
                ),
                ExecOp::Argmax { input, result } => (
                    Planned::Elementwise(self.plan_argmax(input, result)?),
                    Access::default().read(input).write_cell(result),
                ),
                ExecOp::EmbedGather { token, table, out } => (
                    Planned::Elementwise(self.plan_embed_gather(
                        token.device_address(),
                        token.buffer(),
                        table,
                        out,
                    )?),
                    Access::default().read_cell(token).read(table).write(out),
                ),
                ExecOp::EmbedGatherRows { tokens, table, out } => (
                    Planned::Elementwise(self.plan_embed_gather(
                        tokens.device_address(),
                        tokens.buffer(),
                        table,
                        out,
                    )?),
                    Access::default()
                        .read_raw(tokens.buffer().raw_buffer())
                        .read(table)
                        .write(out),
                ),
                ExecOp::AttnDecode {
                    q,
                    kt,
                    v,
                    scratch,
                    out,
                    desc,
                } => {
                    let (stage1, combine) = self.plan_attn_decode(q, kt, v, scratch, out, *desc)?;
                    planned.push(Planned::Elementwise(stage1));
                    accesses.push(Access::default().read(q).read(kt).read(v).write(scratch));
                    (
                        Planned::Elementwise(combine),
                        Access::default().read(scratch).write(out),
                    )
                }
                ExecOp::GemvChain { jobs, n } => {
                    let n = *n as usize;
                    if n == 0 || n > GEMV_CHAIN_MAX_JOBS {
                        bail!("run_exec_ops: GemvChain n={n} out of range");
                    }
                    let slice = &jobs[..n];
                    let dispatch = self.plan_gemv_chain(slice)?;
                    let mut access = Access::default();
                    let mut flops = 0u64;
                    for op in slice {
                        let dims = ResolvedMatmul::from_op(op)?;
                        flops = flops.saturating_add(dims.total_flops);
                        access = access.read(op.call.a).read(op.call.b).write(op.call.c);
                        if let Some(d) = op.epilogue.d_tensor() {
                            access = access.read(d);
                        }
                        if let Some((w, _)) = op.normed_a {
                            access = access.read(w);
                        }
                    }
                    total_flops = total_flops.saturating_add(flops);
                    (Planned::Elementwise(dispatch), access)
                }
            };
            planned.push(step);
            accesses.push(access);
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
            if index > 0 && access.hazard_with(&pending_writes, &pending_reads) {
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
