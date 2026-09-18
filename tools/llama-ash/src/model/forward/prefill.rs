//! Prefill graph construction and execution.

use anyhow::{Context, Result, ensure};
use tensor_ash::{
    CopyDesc, DType, ExecOp, FlashAttentionDesc, MatmulOp, PrefillQkvPackDesc, PreparedOps,
    RopeDesc,
};

use super::super::{GraphStats, Model, PrefillPrepared, Scratch, mm};

impl Model {
    /// Prefill `tokens` starting from the current position (flash
    /// attention path).  Returns the greedy next token and its logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(u32, Vec<f32>)> {
        self.prefill_inner(tokens, true)
    }

    /// [`prefill`](Self::prefill) without the logits download — the
    /// bench timed path only needs the greedy id.
    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<u32> {
        Ok(self.prefill_inner(tokens, false)?.0)
    }

    fn prefill_inner(&mut self, tokens: &[u32], download_logits: bool) -> Result<(u32, Vec<f32>)> {
        let t = u32::try_from(tokens.len()).context("prompt too long")?;
        ensure!(t > 0, "prefill needs at least one token");
        let pos_base = self.pos;
        ensure!(pos_base + t <= self.cfg.t_max, "KV cache overflow");
        if self.prefill_scratch.as_ref().is_none_or(|s| s.t != t) {
            // Addresses in a cached recording dangle if scratch moves.
            self.prefill_prepared = None;
            // f16 activations need the plain-matmul MLP branch
            // (t >= 128) and a16 M alignment (t % 64 == 0); other
            // widths keep the f32 scratch and match main exactly.
            let act_f16 = self.prefill_act_f16 && t >= 128 && t.is_multiple_of(64);
            self.prefill_scratch = Some(Scratch::new(&self.ctx, &self.cfg, t, false, act_f16)?);
        }
        self.token_ids.write(tokens)?;
        let stats = if pos_base == 0 && self.prefill_prepared.as_ref().is_some_and(|p| p.t == t) {
            self.prefill_prepared.as_mut().unwrap().prepared.run()?
        } else {
            let s = self.prefill_scratch.take().unwrap();
            let recorded = (|| {
                let ops = self.prefill_ops(&s, t, pos_base);
                if pos_base == 0 {
                    let mut prepared = self.exec.prepare_exec_ops(&ops, None)?;
                    let stats = prepared.run()?;
                    drop(ops);
                    self.prefill_prepared = Some(PrefillPrepared {
                        t,
                        // SAFETY: baked addresses belong to `self`'s
                        // scratch / weights / caches.  Cleared when
                        // scratch is recreated and in `Drop`.
                        prepared: unsafe {
                            std::mem::transmute::<PreparedOps<'_, '_>, PreparedOps<'static, 'static>>(
                                prepared,
                            )
                        },
                    });
                    Ok(stats)
                } else {
                    self.exec.run_exec_ops(&ops)
                }
            })();
            self.prefill_scratch = Some(s);
            recorded?
        };
        self.note("prefill_total", &stats);
        let best = self.token_buf.read()?;
        let logits = if download_logits {
            let s = self.prefill_scratch.as_ref().unwrap();
            let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
            self.exec.download(&s.logits, &mut logits)?;
            logits
        } else {
            Vec::new()
        };
        self.pos = pos_base + t;
        Ok((best, logits))
    }

    /// Build the prefill's full op chain against `s` (flash attention
    /// path): per layer the attention RMSNorm, one concatenated QKV
    /// GEMM, then either the fused QKV pack + token-major-O flash
    /// (f16 T≥128) or the composed split/RoPE/append/permute path;
    /// then the MLP block and the LM-head tail.  Scratch reuse across
    /// layers (xn, qkv, q_heads, attn_heads) and the x ping-pong
    /// serialize through `run_exec_ops`' hazard tracker.
    fn prefill_ops<'a>(&'a self, s: &'a Scratch, t: u32, pos_base: u32) -> Vec<ExecOp<'a>> {
        let (heads, kv, dh, t_max) = (
            self.cfg.heads,
            self.cfg.kv_heads,
            self.cfg.dh,
            self.cfg.t_max,
        );
        let rope = |heads| RopeDesc {
            heads,
            head_dim: dh,
            rot_dim: dh,
            pos_base,
            ..Default::default()
        };
        let mut ops: Vec<ExecOp<'a>> = Vec::with_capacity(self.layers.len() * 16 + 5);
        ops.push(ExecOp::EmbedGatherRows {
            tokens: &self.token_ids,
            table: &self.embd_gpu,
            out: &s.x_a,
        });
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        for layer in &self.layers {
            ops.push(ExecOp::RmsNorm {
                input: x_in,
                weight: &layer.attn_norm,
                output: &s.xn,
                eps: self.cfg.rms_eps,
            });
            // One concatenated QKV GEMM for every prefill width so we
            // keep a single weight copy.  f16 T≥128 then packs RoPE +
            // KV-append + head-major Q in one dispatch; everything
            // else splits the row and uses the composed rope/copy
            // path (T<128 still permutes for flash).
            let qkv = s.qkv.as_ref().expect("prefill scratch has qkv");
            let embd = self.cfg.embd;
            let kv_dim = kv * dh;
            let n_qkv = embd + 2 * kv_dim;
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.w_qkv, qkv))));
            if t >= 128 && s.q.dtype() == DType::F16 && layer.kt_cache.dtype() == DType::F16 {
                ops.push(ExecOp::PrefillQkvPack {
                    qkv,
                    table: &self.rope_table,
                    q: &s.q_heads,
                    kt: &layer.kt_cache,
                    v: &layer.v_cache,
                    desc: PrefillQkvPackDesc {
                        heads,
                        kv_heads: kv,
                        head_dim: dh,
                        rot_dim: dh,
                        pos_base,
                    },
                });
                ops.push(ExecOp::FlashAttn {
                    q: &s.q_heads,
                    kt: &layer.kt_cache,
                    v: &layer.v_cache,
                    out: &s.attn_flat,
                    desc: FlashAttentionDesc {
                        kv_len: pos_base + t,
                        pos_base,
                        scale: 1.0 / (dh as f32).sqrt(),
                        token_major_heads: None,
                        out_token_major_heads: Some(heads),
                    },
                });
            } else {
                ops.push(ExecOp::CopyStrided {
                    src: qkv,
                    dst: &s.q,
                    desc: CopyDesc {
                        extent: [embd, t, 1],
                        src_offset: 0,
                        src_strides: [1, n_qkv, 0],
                        dst_offset: 0,
                        dst_strides: [1, embd, 0],
                        ..Default::default()
                    },
                });
                ops.push(ExecOp::CopyStrided {
                    src: qkv,
                    dst: &s.k,
                    desc: CopyDesc {
                        extent: [kv_dim, t, 1],
                        src_offset: embd,
                        src_strides: [1, n_qkv, 0],
                        dst_offset: 0,
                        dst_strides: [1, kv_dim, 0],
                        ..Default::default()
                    },
                });
                ops.push(ExecOp::CopyStrided {
                    src: qkv,
                    dst: &s.v,
                    desc: CopyDesc {
                        extent: [kv_dim, t, 1],
                        src_offset: embd + kv_dim,
                        src_strides: [1, n_qkv, 0],
                        dst_offset: 0,
                        dst_strides: [1, kv_dim, 0],
                        ..Default::default()
                    },
                });
                ops.push(ExecOp::Rope {
                    input: &s.q,
                    table: &self.rope_table,
                    output: &s.q,
                    desc: rope(heads),
                });
                ops.push(ExecOp::Rope {
                    input: &s.k,
                    table: &self.rope_table,
                    output: &s.k,
                    desc: rope(kv),
                });
                ops.push(ExecOp::CopyStrided {
                    src: &s.k,
                    dst: &layer.kt_cache,
                    desc: CopyDesc {
                        extent: [dh, kv, t],
                        src_offset: 0,
                        src_strides: [1, dh, kv * dh],
                        dst_offset: pos_base,
                        dst_strides: [t_max, dh * t_max, 1],
                        ..Default::default()
                    },
                });
                ops.push(ExecOp::CopyStrided {
                    src: &s.v,
                    dst: &layer.v_cache,
                    desc: CopyDesc {
                        extent: [dh, kv, t],
                        src_offset: 0,
                        src_strides: [1, dh, kv * dh],
                        dst_offset: pos_base * dh,
                        dst_strides: [1, t_max * dh, dh],
                        ..Default::default()
                    },
                });
                if t >= 128 {
                    ops.push(ExecOp::FlashAttn {
                        q: &s.q,
                        kt: &layer.kt_cache,
                        v: &layer.v_cache,
                        out: &s.attn_flat,
                        desc: FlashAttentionDesc {
                            kv_len: pos_base + t,
                            pos_base,
                            scale: 1.0 / (dh as f32).sqrt(),
                            token_major_heads: Some(heads),
                            out_token_major_heads: None,
                        },
                    });
                } else {
                    ops.push(ExecOp::CopyStrided {
                        src: &s.q,
                        dst: &s.q_heads,
                        desc: CopyDesc {
                            extent: [dh, t, heads],
                            src_offset: 0,
                            src_strides: [1, heads * dh, dh],
                            dst_offset: 0,
                            dst_strides: [1, dh, t * dh],
                            ..Default::default()
                        },
                    });
                    ops.push(ExecOp::FlashAttn {
                        q: &s.q_heads,
                        kt: &layer.kt_cache,
                        v: &layer.v_cache,
                        out: &s.attn_heads,
                        desc: FlashAttentionDesc {
                            kv_len: pos_base + t,
                            pos_base,
                            scale: 1.0 / (dh as f32).sqrt(),
                            token_major_heads: None,
                            out_token_major_heads: None,
                        },
                    });
                    ops.push(ExecOp::CopyStrided {
                        src: &s.attn_heads,
                        dst: &s.attn_flat,
                        desc: CopyDesc {
                            extent: [dh, t, heads],
                            src_offset: 0,
                            src_strides: [1, dh, t * dh],
                            dst_offset: 0,
                            dst_strides: [1, heads * dh, dh],
                            ..Default::default()
                        },
                    });
                }
            }
            self.push_mlp_ops(&mut ops, layer, s, x_in, x_out);
            std::mem::swap(&mut x_in, &mut x_out);
        }
        self.push_lm_head_ops(&mut ops, s, x_in);
        ops.push(ExecOp::Argmax {
            input: &s.logits,
            result: &self.token_buf,
        });
        ops
    }

    /// Plan-only census of the prefill graph at width `t` (see
    /// [`GraphStats`]): the same op chain [`prefill`](Self::prefill)
    /// would submit at the current position, counted instead of
    /// executed.  Nothing touches the queue or the KV caches.
    pub fn prefill_graph_stats(&mut self, t: u32) -> Result<GraphStats> {
        ensure!(t > 0, "prefill graph needs at least one token");
        ensure!(self.pos + t <= self.cfg.t_max, "KV cache overflow");
        if self.prefill_scratch.as_ref().is_none_or(|s| s.t != t) {
            // Addresses in a cached recording dangle if scratch moves.
            self.prefill_prepared = None;
            // Mirror prefill()'s f16-activation eligibility so the
            // census counts the graph prefill would actually submit.
            let act_f16 = self.prefill_act_f16 && t >= 128 && t.is_multiple_of(64);
            self.prefill_scratch = Some(Scratch::new(&self.ctx, &self.cfg, t, false, act_f16)?);
        }
        let s = self.prefill_scratch.take().unwrap();
        let ops = self.prefill_ops(&s, t, self.pos);
        let counted = self.exec.exec_ops_barrier_count(&ops);
        drop(ops);
        self.prefill_scratch = Some(s);
        let (dispatches, barriers) = counted?;
        Ok(GraphStats {
            dispatches,
            barriers,
        })
    }
}
