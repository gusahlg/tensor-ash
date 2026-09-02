//! Persistent GEMV-chain (opt-in, `ML_DEVICE_SCOPE=1`).

use std::sync::Arc;

use crate::buffer::{Buffer, BufferLocation};
use crate::dtype::DType;
use crate::matmul::{Activation, EpilogueBinary, MatmulOp, ResolvedMatmul, RunStats};
use crate::tensor::Tensor;
use anyhow::{Context, Result, bail};
use ash::vk;

use super::pc::{ELEMENTWISE_PC_BYTES, GemvChainPc, GemvJob};
use super::{ElementwiseDispatch, Executor, GEMV_CHAIN_MAX_JOBS, GEMV_CHAIN_MAX_WG};

impl Executor {
    /// Plan a persistent GEMV chain (see [`super::ExecOp::GemvChain`]).
    pub(in crate::executor) fn plan_gemv_chain(
        &self,
        jobs: &[MatmulOp<'_>],
    ) -> Result<ElementwiseDispatch> {
        if !self.ctx.memory_model_device_scope_enabled {
            bail!(
                "gemv_chain: device lacks vulkanMemoryModelDeviceScope \
                 (required for the quorum barrier)"
            );
        }
        if jobs.is_empty() {
            bail!("gemv_chain: empty job list");
        }
        if jobs.len() > GEMV_CHAIN_MAX_JOBS {
            bail!(
                "gemv_chain: {} jobs exceeds max {GEMV_CHAIN_MAX_JOBS}",
                jobs.len()
            );
        }
        let pipeline = self
            .elementwise()?
            .gemv_chain
            .as_ref()
            .map(|k| k.pipeline)
            .ok_or_else(|| anyhow::anyhow!("gemv_chain: kernel was not built"))?;

        let mut packed = Vec::with_capacity(jobs.len());
        for (i, op) in jobs.iter().enumerate() {
            packed.push(self.pack_gemv_job(op, i)?);
        }
        // SYNC_AFTER on job i means "barrier after the group that
        // *ends* at i".  Independent neighbours (no RAW/WAW/WAR)
        // share a group so they occupy one flattened tile space.
        for i in 0..packed.len().saturating_sub(1) {
            if gemv_jobs_hazard(&jobs[..=i], &jobs[i + 1]) {
                packed[i].flags |= GEMV_FLAG_SYNC_AFTER;
            }
        }

        // Header: arrived[2] + phase + pad, then 80-byte jobs.  Device-local
        // so the quorum atomics stay on the GPU; a host-visible sync
        // cell would bounce every arrival over PCIe and lose the
        // 7.7 µs pipeline-barrier win.
        let header = [0u32; 4];
        let nbytes = std::mem::size_of_val(&header) + packed.len() * std::mem::size_of::<GemvJob>();
        let buf = Buffer::new(
            &self.ctx,
            nbytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferLocation::Device,
        )
        .context("gemv_chain job buffer")?;
        let job_bytes = bytemuck::cast_slice::<GemvJob, u8>(&packed);
        let mut all = vec![0u8; nbytes];
        all[..16].copy_from_slice(bytemuck::bytes_of(&header));
        all[16..].copy_from_slice(job_bytes);
        self.upload_bytes_to_buffer(&buf, &all)?;

        let max_tiles = packed
            .iter()
            .map(|j| {
                let vcols = if j.vcols == 2 { 2 } else { 1 };
                j.n.div_ceil(32 * vcols)
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let n_wg = max_tiles.min(GEMV_CHAIN_MAX_WG);
        let pc = GemvChainPc {
            jobs_ptr: buf.device_address() + 16,
            sync_ptr: buf.device_address(),
            n_jobs: packed.len() as u32,
            n_wg,
        };
        let bytes = bytemuck::bytes_of(&pc);
        let mut push = [0u8; ELEMENTWISE_PC_BYTES];
        push[..bytes.len()].copy_from_slice(bytes);
        Ok(ElementwiseDispatch {
            pipeline,
            layout: self.elementwise()?.layout,
            push,
            push_len: bytes.len(),
            groups: (n_wg, 1),
            retain: Some(Arc::new(buf)),
        })
    }

    fn pack_gemv_job(&self, op: &MatmulOp<'_>, index: usize) -> Result<GemvJob> {
        self.validate_op_context(op)?;
        if !matches!(op.store, crate::matmul::MatmulStore::None) {
            bail!("gemv_chain[{index}]: fused store epilogues are not supported");
        }
        if op.call.accumulate {
            bail!("gemv_chain[{index}]: accumulate is not supported");
        }
        if op.call.a.dtype() != DType::F32 || op.call.c.dtype() != DType::F32 {
            bail!("gemv_chain[{index}]: A and C must be f32 storage");
        }
        if op.call.b.dtype() != DType::F16 {
            bail!("gemv_chain[{index}]: B must be f16 storage");
        }
        if op.packed_b {
            bail!("gemv_chain[{index}]: packed-B layout is not supported");
        }
        if op.epilogue.bias.is_some() {
            bail!("gemv_chain[{index}]: bias epilogue is not supported");
        }
        let dims = ResolvedMatmul::from_op(op)?;
        if dims.batch != 1 || dims.m != 1 {
            bail!(
                "gemv_chain[{index}]: requires batch=1 M=1 (got B={} M={})",
                dims.batch,
                dims.m
            );
        }
        if dims.a_f16 {
            bail!("gemv_chain[{index}]: f16 activations are not supported");
        }
        let (epi_bin, epi_beta) = match op.epilogue.binary {
            EpilogueBinary::None => (0u32, 0.0f32),
            EpilogueBinary::AddScaled { beta, .. } => (1, beta),
            EpilogueBinary::Mul { .. } => (2, 0.0),
        };
        let epi_act = match op.epilogue.activation {
            Activation::None => 0u32,
            Activation::Silu => 2,
            other => bail!("gemv_chain[{index}]: unsupported activation {other:?}"),
        };
        if op.normed_a.is_some() && epi_bin == 1 {
            bail!("gemv_chain[{index}]: NORM_A cannot combine with AddScaled");
        }
        let mut flags = 0u32;
        if op.normed_a.is_some() {
            flags |= GEMV_FLAG_NORM_A;
        }
        flags |= (epi_bin & 3) << 8;
        flags |= (epi_act & 3) << 16;
        let beta = if let Some((_, eps)) = op.normed_a {
            eps
        } else {
            epi_beta
        };
        let vcols = gemv_chain_vcols(dims.k, dims.n);
        let d_ptr = op.epilogue.d_tensor().map_or(0, Tensor::device_address);
        let bias_ptr = op.normed_a.map(|(w, _)| w.device_address()).unwrap_or(0);
        Ok(GemvJob {
            n: dims.n,
            k: dims.k,
            flags,
            vcols,
            alpha: op.call.alpha,
            beta,
            pad0: 0,
            pad1: 0,
            a_ptr: op.call.a.device_address(),
            b_ptr: op.call.b.device_address(),
            c_ptr: op.call.c.device_address(),
            d_ptr,
            bias_ptr,
            pad2: 0,
        })
    }

    /// Run a GEMV chain as its own submission (tests / microbench).
    pub fn run_gemv_chain(&self, jobs: &[MatmulOp<'_>]) -> Result<RunStats> {
        let dispatch = self.plan_gemv_chain(jobs)?;
        let total_flops = jobs
            .iter()
            .map(|op| ResolvedMatmul::from_op(op).map(|d| d.total_flops))
            .sum::<Result<u64>>()?;
        let stats = self.submit_one_elementwise(dispatch)?;
        Ok(RunStats {
            gpu_time_ns: stats.gpu_time_ns,
            n_calls: jobs.len(),
            total_flops,
        })
    }
}

fn gemv_chain_vcols(k: u32, n: u32) -> u32 {
    if k >= 4096 || n <= 512 || (k >= 2048 && n <= 2048) {
        1
    } else {
        2
    }
}

const GEMV_FLAG_NORM_A: u32 = 1;
const GEMV_FLAG_SYNC_AFTER: u32 = 1 << 24;

fn gemv_jobs_hazard(prior: &[MatmulOp<'_>], next: &MatmulOp<'_>) -> bool {
    let next_reads = gemv_job_reads(next);
    let next_writes = gemv_job_writes(next);
    for prev in prior {
        let writes = gemv_job_writes(prev);
        let reads = gemv_job_reads(prev);
        if next_reads.iter().any(|b| writes.contains(b))
            || next_writes
                .iter()
                .any(|b| writes.contains(b) || reads.contains(b))
        {
            return true;
        }
    }
    false
}

fn gemv_job_reads(op: &MatmulOp<'_>) -> Vec<vk::Buffer> {
    let mut v = vec![op.call.a.raw_buffer(), op.call.b.raw_buffer()];
    if let Some(d) = op.epilogue.d_tensor() {
        v.push(d.raw_buffer());
    }
    if let Some((w, _)) = op.normed_a {
        v.push(w.raw_buffer());
    }
    if op.call.accumulate {
        v.push(op.call.c.raw_buffer());
    }
    v
}

fn gemv_job_writes(op: &MatmulOp<'_>) -> Vec<vk::Buffer> {
    vec![op.call.c.raw_buffer()]
}
