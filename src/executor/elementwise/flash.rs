//! Fused flash-attention and wide-prefill QKV pack.

use anyhow::{Result, bail};

use crate::dtype::DType;
use crate::matmul::RunStats;
use crate::tensor::Tensor;

use super::pc::{FlashPc, PrefillQkvPackPc};
use super::{Cm2Op, ElementwiseDispatch, Executor, FlashAttentionDesc, Op, PrefillQkvPackDesc, WG};

impl Executor {
    /// Fused causal prefill attention (FlashAttention pattern):
    /// `out = softmax_causal(q @ K^T * scale) @ V` in one dispatch with
    /// online softmax — score tiles never touch global memory, and
    /// tiles above the causal frontier are skipped.  Layouts match the
    /// composed path: `q`/`out` are `[H, T_q, dh]`, `kt` is
    /// `[H_kv, dh, T_max]`, `v` is `[H_kv, T_max, dh]`; GQA works when
    /// `H` is a multiple of `H_kv`.  `dh` must be 64 or 128 (the
    /// compiled head-dimension variants).
    ///
    /// On devices where [`VulkanContext::coopmat2_enabled`] is set the
    /// dispatch routes to the tensor-core `NV_cooperative_matrix2`
    /// kernels (f16 operands, f32 accumulate — expect f16-level
    /// rounding vs the f32 SIMT path); otherwise the SIMT kernels run.
    pub fn run_flash_attention(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
    ) -> Result<RunStats> {
        self.run_flash_attention_impl(q, kt, v, out, desc, false)
    }

    /// [`Self::run_flash_attention`] pinned to the SIMT kernels even
    /// when the `NV_cooperative_matrix2` path is available.  A/B hook
    /// for the GPU validation tests; not part of the stable API.
    #[doc(hidden)]
    pub fn run_flash_attention_simt(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
    ) -> Result<RunStats> {
        self.run_flash_attention_impl(q, kt, v, out, desc, true)
    }

    fn run_flash_attention_impl(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
        force_simt: bool,
    ) -> Result<RunStats> {
        let dispatch = self.plan_flash_attention(q, kt, v, out, desc, force_simt)?;
        self.submit_one_elementwise(dispatch)
    }

    /// Validate and plan one flash-attention dispatch (CM2-first
    /// routing, exactly like [`Self::run_flash_attention`]).
    pub(in crate::executor) fn plan_flash_attention(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
        force_simt: bool,
    ) -> Result<ElementwiseDispatch> {
        // q/out may both be f16 (f16 activations): the CM2 kv16 io16
        // kernel loads halves directly and narrows the output store.
        let io_f16 = self.io_dtype_f16("run_flash_attention", q, out)?;
        self.validate_tensor_context(kt, "kt")?;
        self.validate_tensor_context(v, "v")?;
        if kt.dtype() != v.dtype() {
            bail!(
                "run_flash_attention: kt ({}) and v ({}) must share a storage type",
                kt.dtype().name(),
                v.dtype().name()
            );
        }
        let kv_f16 = kt.dtype() == DType::F16;
        let [kv_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_flash_attention: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_flash_attention: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        if v_t != t_max || v_heads != kv_heads || v_dh != kt_dh {
            bail!(
                "run_flash_attention: inconsistent cache shapes kt {:?}, v {:?}",
                kt.shape(),
                v.shape()
            );
        }
        let dh = kt_dh;
        let (heads, t_q, q_head_stride, q_row_stride) = if let Some(heads) = desc.token_major_heads
        {
            if heads == 0 {
                bail!("run_flash_attention: token_major_heads must be nonzero");
            }
            let t_q = q.shape().first().copied().unwrap_or(0);
            let q_elems = q.len();
            let want = t_q as u64 * heads as u64 * dh as u64;
            if t_q == 0 || q_elems != want {
                bail!(
                    "run_flash_attention: token-major q {:?} does not hold [T={t_q}, H={heads}, dh={dh}]",
                    q.shape()
                );
            }
            // [T, H, dh] or rank-2 [T, H*dh]: head stride is dh, row stride is H*dh.
            (heads, t_q, dh, heads * dh)
        } else {
            let [heads, t_q, q_dh] = *q.shape() else {
                bail!(
                    "run_flash_attention: q must be [H, T_q, dh], got {:?}",
                    q.shape()
                );
            };
            if q_dh != dh {
                bail!(
                    "run_flash_attention: q dh {q_dh} != cache dh {dh} (q {:?}, kt {:?})",
                    q.shape(),
                    kt.shape()
                );
            }
            (heads, t_q, t_q * dh, dh)
        };
        let (o_head_stride, o_row_stride) = if let Some(oh) = desc.out_token_major_heads {
            if oh == 0 {
                bail!("run_flash_attention: out_token_major_heads must be nonzero");
            }
            if oh != heads {
                bail!("run_flash_attention: out_token_major_heads {oh} != q heads {heads}");
            }
            let want = t_q as u64 * oh as u64 * dh as u64;
            if out.len() != want {
                bail!(
                    "run_flash_attention: token-major out {:?} does not hold [T={t_q}, H={oh}, dh={dh}]",
                    out.shape()
                );
            }
            (dh, oh * dh)
        } else if desc.token_major_heads.is_some() {
            if out.len() != q.len() {
                bail!(
                    "run_flash_attention: out shape {:?} must match token-major q {:?}",
                    out.shape(),
                    q.shape()
                );
            }
            (q_head_stride, q_row_stride)
        } else if out.shape() != q.shape() || out.len() != q.len() {
            bail!(
                "run_flash_attention: out shape {:?} must equal q shape {:?}",
                out.shape(),
                q.shape()
            );
        } else {
            (q_head_stride, q_row_stride)
        };
        if v_dh != dh {
            bail!(
                "run_flash_attention: inconsistent shapes q {:?}, kt {:?}, v {:?}",
                q.shape(),
                kt.shape(),
                v.shape()
            );
        }
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            bail!("run_flash_attention: H={heads} must be a multiple of H_kv={kv_heads}");
        }
        if desc.kv_len > t_max {
            bail!(
                "run_flash_attention: kv_len {} exceeds cache T_max {t_max}",
                desc.kv_len
            );
        }
        let pipes = self.elementwise()?;
        // Query rows per workgroup: the cm2 kernels tile Br=64 rows
        // across 128 threads; the SIMT kernels take 128 rows.
        let (pipeline, rows_per_wg) = if io_f16 {
            // No SIMT sibling reads f16 q — the compiled variant is
            // the CM2 kv16 io16 dh64 kernel only.
            if kv_f16 && dh == 64 && !force_simt {
                match pipes.cm2_pipeline(Cm2Op::FlashKv16Io16Dh64) {
                    Some(pipeline) => (pipeline, 64),
                    None => bail!(
                        "run_flash_attention: f16 q/out require NV_cooperative_matrix2 \
                         support, which this device lacks"
                    ),
                }
            } else {
                bail!(
                    "run_flash_attention: f16 q/out are compiled for f16 KV caches with \
                     head dimension 64 on the CM2 route only (kv {}, dh {dh})",
                    kt.dtype().name()
                )
            }
        } else {
            let (simt_kernel, cm2_kernel) = match (dh, kv_f16) {
                (64, false) => (Op::FlashDh64, Cm2Op::FlashDh64),
                (128, false) => (Op::FlashDh128, Cm2Op::FlashDh128),
                (64, true) => (Op::FlashKv16Dh64, Cm2Op::FlashKv16Dh64),
                (128, true) => (Op::FlashKv16Dh128, Cm2Op::FlashKv16Dh128),
                (other, _) => bail!(
                    "run_flash_attention: head dimension {other} unsupported (compiled variants: 64, 128)"
                ),
            };
            let cm2 = if force_simt {
                None
            } else {
                pipes.cm2_pipeline(cm2_kernel)
            };
            match cm2 {
                Some(pipeline) => (pipeline, 64),
                None => (pipes.pipeline(simt_kernel), 128),
            }
        };
        self.plan_elementwise(
            pipeline,
            &FlashPc {
                t_q,
                t_max,
                kv_len: desc.kv_len,
                pos_base: desc.pos_base,
                group_size: heads / kv_heads,
                scale: desc.scale,
                q_head_stride,
                q_row_stride,
                o_head_stride,
                o_row_stride,
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                out_ptr: out.device_address(),
            },
            t_q.div_ceil(rows_per_wg),
            heads,
        )
    }

    /// Fused wide-prefill QKV pack: RoPE-rotate Q into head-major
    /// `[H, T, dh]`, RoPE-scatter K into the Kt cache, and copy V into
    /// the V cache, all from one concatenated `[T, n_qkv]` row.  f16
    /// activations and f16 caches only — the T≥128 a16 prefill path.
    pub fn run_prefill_qkv_pack(
        &self,
        qkv: &Tensor,
        table: &Tensor,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        desc: PrefillQkvPackDesc,
    ) -> Result<RunStats> {
        let dispatch = self.plan_prefill_qkv_pack(qkv, table, q, kt, v, desc)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_prefill_qkv_pack(
        &self,
        qkv: &Tensor,
        table: &Tensor,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        desc: PrefillQkvPackDesc,
    ) -> Result<ElementwiseDispatch> {
        for (tensor, name) in [(qkv, "qkv"), (q, "q"), (kt, "kt"), (v, "v")] {
            self.validate_tensor_context(tensor, name)?;
            if tensor.dtype() != DType::F16 {
                bail!(
                    "run_prefill_qkv_pack: {name} must be f16 storage, got {}",
                    tensor.dtype().name()
                );
            }
        }
        self.ensure_f32(table, "run_prefill_qkv_pack", "table")?;
        if desc.heads == 0
            || desc.kv_heads == 0
            || desc.head_dim == 0
            || desc.rot_dim == 0
            || !desc.rot_dim.is_multiple_of(2)
            || desc.rot_dim > desc.head_dim
        {
            bail!(
                "run_prefill_qkv_pack: invalid heads/kv_heads/dh/rot_dim \
                 ({}/{}/{}/{})",
                desc.heads,
                desc.kv_heads,
                desc.head_dim,
                desc.rot_dim
            );
        }
        let [tokens, qkv_stride] = *qkv.shape() else {
            bail!(
                "run_prefill_qkv_pack: qkv must be [T, n_qkv], got {:?}",
                qkv.shape()
            );
        };
        let embd = desc.heads * desc.head_dim;
        let kv_dim = desc.kv_heads * desc.head_dim;
        let want_stride = embd + 2 * kv_dim;
        if qkv_stride != want_stride {
            bail!("run_prefill_qkv_pack: qkv N {qkv_stride} != H*dh + 2*H_kv*dh ({want_stride})");
        }
        if *q.shape() != [desc.heads, tokens, desc.head_dim] {
            bail!(
                "run_prefill_qkv_pack: q must be head-major [H={}, T={tokens}, dh={}], got {:?}",
                desc.heads,
                desc.head_dim,
                q.shape()
            );
        }
        let [kt_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_prefill_qkv_pack: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_prefill_qkv_pack: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        if kt_heads != desc.kv_heads
            || v_heads != desc.kv_heads
            || kt_dh != desc.head_dim
            || v_dh != desc.head_dim
            || v_t != t_max
        {
            bail!(
                "run_prefill_qkv_pack: cache shapes kt {:?} v {:?} vs H_kv={} dh={}",
                kt.shape(),
                v.shape(),
                desc.kv_heads,
                desc.head_dim
            );
        }
        if desc.pos_base.saturating_add(tokens) > t_max {
            bail!(
                "run_prefill_qkv_pack: pos_base {} + T {tokens} exceeds T_max {t_max}",
                desc.pos_base
            );
        }
        let needed = (desc.pos_base as u64 + tokens as u64) * desc.rot_dim as u64;
        if table.len() < needed {
            bail!(
                "run_prefill_qkv_pack: table length {} < required {needed} \
                 (pos_base {} + {tokens} tokens, rot_dim {})",
                table.len(),
                desc.pos_base,
                desc.rot_dim
            );
        }
        if !desc.head_dim.is_multiple_of(4) {
            bail!(
                "run_prefill_qkv_pack: head_dim {} must be a multiple of 4",
                desc.head_dim
            );
        }
        let q_pairs = tokens * desc.heads * (desc.rot_dim / 2);
        let k_pairs = tokens * desc.kv_heads * (desc.rot_dim / 2);
        let v_vec4s = tokens * desc.kv_heads * (desc.head_dim / 4);
        let threads = q_pairs.saturating_add(k_pairs).saturating_add(v_vec4s);
        let pipeline = self.elementwise()?.pipeline(Op::PrefillQkvPack);
        self.plan_elementwise(
            pipeline,
            &PrefillQkvPackPc {
                tokens,
                heads: desc.heads,
                kv_heads: desc.kv_heads,
                head_dim: desc.head_dim,
                rot_dim: desc.rot_dim,
                pos_base: desc.pos_base,
                qkv_stride,
                t_max,
                k_offset: embd,
                v_offset: embd + kv_dim,
                qkv_ptr: qkv.device_address(),
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                table_ptr: table.device_address(),
            },
            threads.div_ceil(WG),
            1,
        )
    }
}
