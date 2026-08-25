//! Decode graph construction and execution.

use anyhow::{Result, ensure};
use tensor_ash::{
    AttnDecodeDesc, CopyDesc, ExecOp, FlashAttentionDesc, MatmulOp, PreparedOps, SoftmaxMask,
};

use super::super::{DecodeMode, GraphStats, Model, Scratch, argmax, mm};

impl Model {
    /// One greedy decode step for `token` at the current position.
    /// Returns the next token and its logits.
    pub fn decode(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        // A single prepared step would pay the recording without any
        // replays to amortize it; batch decoding goes through
        // `decode_many`, so one-off steps take the graph path.
        if matches!(self.decode_mode, DecodeMode::Graph | DecodeMode::Prepared) {
            return self.decode_graph(token);
        }
        self.decode_perop(token)
    }

    /// The whole token as ONE submission: every op of every layer is
    /// recorded into a single command buffer (`run_exec_ops`), paying
    /// one submit + fence wait instead of ~350.
    fn decode_graph(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        let pos = self.pos;
        ensure!(pos < self.cfg.t_max, "KV cache overflow");
        let host_x = self.embed(&[token])?;
        let s = &self.decode_scratch;
        self.exec.upload(&host_x, &s.x_a)?;
        let ops = self.decode_ops(s, pos, 0);
        let stats = self.exec.run_exec_ops(&ops)?;
        self.note("graph_total", &stats);
        drop(ops);

        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let argmax = argmax(&logits)?;
        self.pos = pos + 1;
        Ok((argmax, logits))
    }

    /// Prepared decode: record the token's op chain ONCE, then replay
    /// it for `n` greedy steps, bumping only the device-side position
    /// buffer between submits — no host-side re-validation, planning,
    /// or recording per token.  Falls back to per-token [`decode`]
    /// (graph/perop/flash) when the mode or the fused attention
    /// requirement says so.  Returns the `n` generated tokens.
    ///
    /// [`decode`]: Self::decode
    pub fn decode_many(&mut self, mut token: u32, n: u32) -> Result<Vec<u32>> {
        if self.decode_mode != DecodeMode::Prepared || !self.fused_attn {
            let mut generated = Vec::with_capacity(n as usize);
            for _ in 0..n {
                (token, _) = self.decode(token)?;
                generated.push(token);
            }
            return Ok(generated);
        }
        let start_pos = self.pos;
        ensure!(start_pos + n <= self.cfg.t_max, "KV cache overflow");
        let mut generated = Vec::with_capacity(n as usize);
        if self.decode_prepared.is_none() {
            let s = &self.decode_scratch;
            // Pos-relative descs + the position buffer: one recording
            // serves every token position.  The graph tail closes the
            // token loop on the GPU: argmax writes the next token id to
            // the host-readable cell, and the embedding gather writes
            // that token's row into `x_a` — the very tensor the graph
            // reads first — so the next replay feeds itself.
            let mut ops = self.decode_ops(s, 0, self.pos_buf.device_address());
            ops.extend(self.decode_close_ops(s));
            let prepared = self.exec.prepare_exec_ops(&ops, Some(&self.pos_buf))?;
            drop(ops);
            self.decode_prepared = Some(unsafe {
                std::mem::transmute::<PreparedOps<'_, '_>, PreparedOps<'static, 'static>>(prepared)
            });
        }
        {
            let s = &self.decode_scratch;
            // Seed the first replay with the prefill's (CPU-argmaxed)
            // token; from then on the graph's own gather refills x_a.
            let host_x = self.embed(&[token])?;
            self.exec.upload(&host_x, &s.x_a)?;
            for i in 0..n {
                // The position write is host-visible and becomes
                // visible to the device at submit; run()'s fence wait
                // orders the token-cell read after the GPU argmax.
                self.pos_buf.set(start_pos + i)?;
                self.decode_prepared.as_mut().unwrap().run_silent()?;
                token = self.token_buf.read()?;
                generated.push(token);
            }
        }
        self.pos = start_pos + n;
        Ok(generated)
    }

    /// One-submit unroll of `n` decode steps with positions baked in.
    /// Used by `llama_ash bench` so tg128 is not 128 queue submits.
    /// Returns only the last generated id (the bench only needs that).
    pub fn decode_unrolled(&mut self, mut token: u32, n: u32) -> Result<Vec<u32>> {
        if self.decode_mode != DecodeMode::Prepared || !self.fused_attn || n == 0 {
            return self.decode_many(token, n);
        }
        let start_pos = self.pos;
        ensure!(start_pos + n <= self.cfg.t_max, "KV cache overflow");
        let reuse = self
            .decode_unrolled
            .as_ref()
            .is_some_and(|(p, count, _)| *p == start_pos && *count == n);
        if !reuse {
            self.decode_unrolled = None;
            let s = &self.decode_scratch;
            let mut ops = Vec::with_capacity((self.layers.len() * 16 + 3) * n as usize);
            for i in 0..n {
                ops.extend(self.decode_ops(s, start_pos + i, 0));
                ops.extend(self.decode_close_ops(s));
            }
            let prepared = self.exec.prepare_exec_ops(&ops, None)?;
            drop(ops);
            self.decode_unrolled = Some((start_pos, n, unsafe {
                std::mem::transmute::<PreparedOps<'_, '_>, PreparedOps<'static, 'static>>(prepared)
            }));
        }
        {
            let s = &self.decode_scratch;
            let host_x = self.embed(&[token])?;
            self.exec.upload(&host_x, &s.x_a)?;
            let stats = self.decode_unrolled.as_mut().unwrap().2.run()?;
            self.note("prepared_total", &stats);
            token = self.token_buf.read()?;
        }
        self.pos = start_pos + n;
        Ok(vec![token])
    }

    /// Plan-only census of one decode step's graph (see
    /// [`GraphStats`]).  Prepared mode counts the exact replayed chain
    /// (pos-relative ops plus the GPU argmax + embed-gather tail);
    /// every other mode counts the re-recorded graph at the current
    /// position, which is also the dispatch chain the per-op path
    /// submits one by one.
    pub fn decode_graph_stats(&self) -> Result<GraphStats> {
        let s = &self.decode_scratch;
        let (dispatches, barriers) = if self.decode_mode == DecodeMode::Prepared && self.fused_attn
        {
            let mut ops = self.decode_ops(s, 0, self.pos_buf.device_address());
            ops.extend(self.decode_close_ops(s));
            self.exec.exec_ops_barrier_count(&ops)?
        } else {
            let pos = self.pos.min(self.cfg.t_max.saturating_sub(1));
            self.exec
                .exec_ops_barrier_count(&self.decode_ops(s, pos, 0))?
        };
        Ok(GraphStats {
            dispatches,
            barriers,
        })
    }

    /// Build the decode step's full op chain against `s`.
    ///
    /// `pos_addr == 0`: the literal `pos` is baked into the push
    /// constants (the re-recorded graph path).  Non-zero `pos_addr`
    /// (the [`PosBuffer`] address): the position-dependent values are
    /// recorded in pos-relative form — RoPE bases 0, KV-append offsets
    /// 0 with a per-position scale, attention `kv_len` 1 with the
    /// fixed chunk grid — and the shaders add the buffered position at
    /// execution time, which is what makes the recording replayable.
    fn decode_ops<'a>(&'a self, s: &'a Scratch, pos: u32, pos_addr: u64) -> Vec<ExecOp<'a>> {
        let dh = self.cfg.dh;
        assert!(
            pos_addr == 0 || self.fused_attn,
            "pos-relative decode needs the fused attention (composed softmax has no pos read)"
        );
        // Position-dependent bases: zero when the position buffer
        // supplies the offset at execution time.
        let base = if pos_addr == 0 { pos } else { 0 };

        // Contiguous reshapes are free: the GQA-batched attention
        // views of q and of the attention output share memory with
        // their flat forms, eliminating two copy dispatches per layer.
        let q_gqa = s.q_gqa.as_ref().unwrap();
        let attn_gqa = s.attn_gqa.as_ref().unwrap();
        let mut ops: Vec<ExecOp<'a>> = Vec::with_capacity(self.layers.len() * 16 + 3);
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        for layer in &self.layers {
            let scores = s.scores.as_ref().unwrap();
            // The attention RMSNorm is folded into the q/k/v row GEMVs
            // (each recomputes the K-length reduction, which is trivia
            // next to the weight traffic) — the standalone norm op and
            // its xn round-trip are gone.  The RoPE and KV appends fold
            // into the same GEMVs' store epilogues: q ropes as it
            // stores, k ropes and scatters straight into the Kt cache,
            // v scatters into its cache — the separate rope /
            // rope-scatter / copy dispatches (and the k-append's
            // hazard barrier) are gone with them.
            for op in self.qkv_decode_ops(layer, s, x_in, pos, pos_addr) {
                ops.push(ExecOp::Matmul(op));
            }
            if self.fused_attn {
                // 2 dispatches instead of 3, no scores round-trip, and
                // only the valid cache prefix is read.
                ops.push(ExecOp::AttnDecode {
                    q: q_gqa,
                    kt: &layer.kt_cache,
                    v: &layer.v_cache,
                    scratch: s.attn_partials.as_ref().unwrap(),
                    out: attn_gqa,
                    desc: AttnDecodeDesc {
                        // Effective kv_len is base + 1 + buffered pos.
                        kv_len: base + 1,
                        scale: 1.0 / (dh as f32).sqrt(),
                        pos_addr,
                    },
                });
            } else {
                ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                    q_gqa,
                    &layer.kt_cache,
                    scores,
                ))));
                ops.push(ExecOp::SoftmaxRows {
                    input: scores,
                    output: scores,
                    scale: 1.0 / (dh as f32).sqrt(),
                    mask: SoftmaxMask::Prefix { valid: pos + 1 },
                });
                ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                    scores,
                    &layer.v_cache,
                    attn_gqa,
                ))));
            }
            // The FFN RMSNorm folds into the up and gate GEMVs the same
            // way (gate's Silu+Mul epilogue rides along: llama has no
            // bias, so the bias slot is free for the norm weight).
            self.push_mlp_ops(&mut ops, layer, s, x_in, x_out);
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Final norm + LM head (decode scratch is t=1: last row = x).
        self.push_lm_head_ops(&mut ops, s, x_in);
        ops
    }

    /// One greedy decode step, per-op submission path (composed
    /// attention exactly like decoder.rs, or flash when selected).
    fn decode_perop(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        let pos = self.pos;
        ensure!(pos < self.cfg.t_max, "KV cache overflow");
        let host_x = self.embed(&[token])?;
        let (h, dh) = (self.cfg.embd, self.cfg.dh);
        let s = &self.decode_scratch;
        self.exec.upload(&host_x, &s.x_a)?;
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        let straight = |len: u32| CopyDesc {
            extent: [len, 1, 1],
            src_offset: 0,
            src_strides: [1, 0, 0],
            dst_offset: 0,
            dst_strides: [1, 0, 0],
            ..Default::default()
        };
        for layer in &self.layers {
            self.qkv_fused_decode(layer, s, x_in, pos)?;
            // Reshape q [1, embd] -> [kv_heads, group, dh] (same memory
            // order): one batched matmul covers the GQA groups.
            if self.decode_mode == DecodeMode::Flash {
                // Fused single-row attention; GQA is native.
                let q_flash = s.q_flash.as_ref().unwrap();
                let attn_flash = s.attn_flash.as_ref().unwrap();
                self.exec.run_copy_strided(&s.q, q_flash, straight(h))?;
                self.exec.run_flash_attention(
                    q_flash,
                    &layer.kt_cache,
                    &layer.v_cache,
                    attn_flash,
                    FlashAttentionDesc {
                        kv_len: pos + 1,
                        pos_base: pos,
                        scale: 1.0 / (dh as f32).sqrt(),
                        token_major_heads: None,
                        out_token_major_heads: None,
                    },
                )?;
                self.exec
                    .run_copy_strided(attn_flash, &s.attn_flat, straight(h))?;
            } else if self.fused_attn {
                // Fused split-K decode attention.  The GQA views of q
                // and of the attention output are contiguous reshapes,
                // so no copies bracket the op.
                let (kv, group) = (self.cfg.kv_heads, self.cfg.heads / self.cfg.kv_heads);
                let q_gqa = s.q.alias_with_shape(&[kv, group, dh])?;
                let attn_gqa = s.attn_flat.alias_with_shape(&[kv, group, dh])?;
                let stats = self.exec.run_attn_decode(
                    &q_gqa,
                    &layer.kt_cache,
                    &layer.v_cache,
                    s.attn_partials.as_ref().unwrap(),
                    &attn_gqa,
                    AttnDecodeDesc {
                        kv_len: pos + 1,
                        scale: 1.0 / (dh as f32).sqrt(),
                        ..Default::default()
                    },
                )?;
                self.note("attn_decode", &stats);
            } else {
                // Reshape q [1, embd] -> [kv_heads, group, dh] (same
                // memory order): one batched matmul covers the GQA
                // groups.
                let stats = self.exec.run_copy_strided(&s.q, &s.q_heads, straight(h))?;
                self.note("reshape_copy", &stats);
                let scores = s.scores.as_ref().unwrap();
                let stats = self
                    .exec
                    .run_matmuls(&[mm(&s.q_heads, &layer.kt_cache, scores)])?;
                self.note("attn_scores_mm", &stats);
                let stats = self.exec.run_softmax_rows(
                    scores,
                    scores,
                    1.0 / (dh as f32).sqrt(),
                    SoftmaxMask::Prefix { valid: pos + 1 },
                )?;
                self.note("attn_softmax", &stats);
                let stats = self
                    .exec
                    .run_matmuls(&[mm(scores, &layer.v_cache, &s.attn_heads)])?;
                self.note("attn_pv_mm", &stats);
                let stats = self
                    .exec
                    .run_copy_strided(&s.attn_heads, &s.attn_flat, straight(h))?;
                self.note("reshape_copy", &stats);
            }
            self.mlp_block_into(layer, s, x_in, x_out)?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        let result = self.lm_head_last_row(s, x_in)?;
        self.pos = pos + 1;
        Ok(result)
    }
}
