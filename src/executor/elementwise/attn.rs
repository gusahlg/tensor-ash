//! Fused split-K decode attention.

use anyhow::{Result, bail};

use crate::dtype::DType;
use crate::matmul::RunStats;
use crate::tensor::Tensor;

use super::pc::{AttnCombinePc, AttnDecodePc};
use super::{
    ATTN_DECODE_MAX_CHUNKS, AttnDecodeDesc, ElementwiseDispatch, Executor, Op,
    attn_decode_num_chunks,
};

impl Executor {
    /// Fused split-K decode attention: `out = softmax(q @ K^T * scale)
    /// @ V` for ONE query row per head, reading only the `kv_len`
    /// valid cache prefix.  Two dispatches in one submission: stage 1
    /// writes per-chunk online-softmax partials to `scratch`
    /// (`[kv_heads, num_chunks, group, dh+2]` f32, at least
    /// `kv_heads * ATTN_DECODE_MAX_CHUNKS * group * (dh+2)` elements),
    /// stage 2 merges them exactly.  Layouts match the composed path:
    /// `q`/`out` are `[kv_heads, group, dh]` (or any contiguous
    /// reshape ending in `dh`), `kt` is `[H_kv, dh, T_max]`, `v` is
    /// `[H_kv, T_max, dh]`, f32 or f16 caches (matching).  `dh` must
    /// be 64 (the compiled variant) and `group <= 8`.
    pub fn run_attn_decode(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        scratch: &Tensor,
        out: &Tensor,
        desc: AttnDecodeDesc,
    ) -> Result<RunStats> {
        let (stage1, combine) = self.plan_attn_decode(q, kt, v, scratch, out, desc)?;
        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (attn_decode)",
                |_dev, cb, _slot| {
                    self.record_elementwise(cb, &stage1);
                    crate::executor::recording::record_compute_to_compute_barrier(&self.ctx, cb);
                    self.record_elementwise(cb, &combine);
                    Ok(())
                },
            )
        }?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 2,
            total_flops: 0,
        })
    }

    /// Validate and plan both decode-attention dispatches.  The caller
    /// must place a compute barrier between them (stage 2 reads the
    /// scratch stage 1 writes).
    pub(in crate::executor) fn plan_attn_decode(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        scratch: &Tensor,
        out: &Tensor,
        desc: AttnDecodeDesc,
    ) -> Result<(ElementwiseDispatch, ElementwiseDispatch)> {
        self.ensure_f32(q, "run_attn_decode", "q")?;
        self.validate_tensor_context(kt, "kt")?;
        self.validate_tensor_context(v, "v")?;
        self.ensure_f32(scratch, "run_attn_decode", "scratch")?;
        self.ensure_f32(out, "run_attn_decode", "out")?;
        if kt.dtype() != v.dtype() {
            bail!(
                "run_attn_decode: kt ({}) and v ({}) must share a storage type",
                kt.dtype().name(),
                v.dtype().name()
            );
        }
        let kv_f16 = kt.dtype() == DType::F16;
        let [kv_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_attn_decode: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_attn_decode: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        let dh = kt_dh;
        if v_dh != dh || v_t != t_max || v_heads != kv_heads || kv_heads == 0 {
            bail!(
                "run_attn_decode: inconsistent caches kt {:?}, v {:?}",
                kt.shape(),
                v.shape()
            );
        }
        if dh != 64 {
            bail!("run_attn_decode: head dimension {dh} unsupported (compiled variant: 64)");
        }
        let heads_elems = kv_heads as u64 * dh as u64;
        if q.is_empty() || !q.len().is_multiple_of(heads_elems) {
            bail!(
                "run_attn_decode: q length {} must be kv_heads*group*dh (kv_heads {kv_heads}, dh {dh})",
                q.len()
            );
        }
        let group = u32::try_from(q.len() / heads_elems)
            .map_err(|_| anyhow::anyhow!("run_attn_decode: group exceeds u32"))?;
        if group == 0 || group > 8 {
            bail!("run_attn_decode: GQA group {group} unsupported (1..=8)");
        }
        if out.len() != q.len() {
            bail!(
                "run_attn_decode: out length {} must equal q length {}",
                out.len(),
                q.len()
            );
        }
        if desc.kv_len == 0 || desc.kv_len > t_max {
            bail!(
                "run_attn_decode: kv_len {} out of range 1..={t_max}",
                desc.kv_len
            );
        }
        // Position-driven dispatches cannot size the grid per token, so
        // they always use the fixed MAX_CHUNKS decomposition; chunks
        // past the effective kv_len write neutral partials.
        let num_chunks = if desc.pos_addr != 0 {
            ATTN_DECODE_MAX_CHUNKS
        } else {
            attn_decode_num_chunks(desc.kv_len)
        };
        let needed = kv_heads as u64 * num_chunks as u64 * group as u64 * (dh as u64 + 2);
        if scratch.len() < needed {
            bail!(
                "run_attn_decode: scratch length {} < required {needed} \
                 (kv_heads {kv_heads} * chunks {num_chunks} * group {group} * (dh+2))",
                scratch.len()
            );
        }
        let pipes = self.elementwise()?;
        let stage1_kernel = if kv_f16 {
            Op::AttnDecodeKv16Dh64
        } else {
            Op::AttnDecodeDh64
        };
        let stage1 = self.plan_elementwise(
            pipes.pipeline(stage1_kernel),
            &AttnDecodePc {
                kv_len: desc.kv_len,
                num_chunks,
                group,
                t_max,
                scale: desc.scale,
                _pad0: 0,
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                scratch_ptr: scratch.device_address(),
                pos_ptr: desc.pos_addr,
            },
            num_chunks,
            kv_heads,
        )?;
        let combine = self.plan_elementwise(
            pipes.pipeline(Op::AttnDecodeCombine),
            &AttnCombinePc {
                num_chunks,
                group,
                dh,
                _pad0: 0,
                scratch_ptr: scratch.device_address(),
                out_ptr: out.device_address(),
            },
            kv_heads * group,
            1,
        )?;
        Ok((stage1, combine))
    }
}
