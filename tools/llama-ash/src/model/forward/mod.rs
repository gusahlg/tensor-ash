//! Shared prefill/decode graph builders.

use anyhow::{Result, ensure};
use tensor_ash::{
    Activation, BinaryOp, CopyDesc, EpilogueBinary, ExecOp, MatmulOp, MatmulStoreDesc, Tensor,
};

use super::{Layer, Model, Scratch, argmax, epi, mm};

mod decode;
mod prefill;

impl Model {
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let embd = self.cfg.embd as usize;
        let mut x = Vec::with_capacity(tokens.len() * embd);
        for &tok in tokens {
            ensure!(
                tok < self.cfg.vocab,
                "token id {tok} >= vocab {}",
                self.cfg.vocab
            );
            let row = tok as usize * embd;
            x.extend_from_slice(&self.embd_cpu[row..row + embd]);
        }
        Ok(x)
    }

    /// Record one op-class GPU time into the `LLAMA_ASH_BREAKDOWN`
    /// diagnostics table.
    fn note(&self, class: &'static str, stats: &tensor_ash::RunStats) {
        if let Some(ns) = stats.gpu_time_ns {
            self.breakdown.borrow_mut().push((class, ns));
        }
    }

    /// Decode-only fused QKV: packed row GEMVs with the same store
    /// epilogues the graph path uses (RMSNorm folded in; q ropes as it
    /// stores; k rope-scatters into Kt; v scatters into V).  Three
    /// dispatches, no extra rope/copy.
    fn qkv_fused_decode(&self, layer: &Layer, s: &Scratch, x_in: &Tensor, pos: u32) -> Result<()> {
        let ops = self.qkv_decode_ops(layer, s, x_in, pos, 0);
        let stats = self.exec.run_ops(&ops)?;
        self.note("qkv_matmul", &stats);
        Ok(())
    }

    /// The three packed QKV projections for one decode step.  `pos_addr`
    /// 0 bakes `pos`; a nonzero cell is the prepared-graph indirection.
    fn qkv_decode_ops<'a>(
        &'a self,
        layer: &'a Layer,
        s: &'a Scratch,
        x_in: &'a Tensor,
        pos: u32,
        pos_addr: u64,
    ) -> [MatmulOp<'a>; 3] {
        let (dh, t_max) = (self.cfg.dh, self.cfg.t_max);
        let eps = self.cfg.rms_eps;
        let base = if pos_addr == 0 { pos } else { 0 };
        let store = |pos_scale, stride_head, stride_dim| MatmulStoreDesc {
            head_dim: dh,
            pos_base: base,
            pos_scale,
            stride_head,
            stride_dim,
            pos_addr,
        };
        [
            MatmulOp::new(mm(x_in, &layer.wq_p, &s.q))
                .with_packed_b()
                .with_normed_a(&layer.attn_norm, eps)
                .with_store_rope(&self.rope_table, store(0, 0, 0)),
            MatmulOp::new(mm(x_in, &layer.wk_p, &s.k))
                .with_packed_b()
                .with_normed_a(&layer.attn_norm, eps)
                .with_store_rope_scatter(
                    &self.rope_table,
                    &layer.kt_cache,
                    store(1, dh * t_max, t_max),
                ),
            MatmulOp::new(mm(x_in, &layer.wv_p, &s.v))
                .with_packed_b()
                .with_normed_a(&layer.attn_norm, eps)
                .with_store_scatter(&layer.v_cache, store(dh, t_max * dh, 1)),
        ]
    }

    /// Final norm + LM head on the last row of `x`; returns argmax and
    /// logits.
    fn lm_head_last_row(&self, s: &Scratch, x: &Tensor) -> Result<(u32, Vec<f32>)> {
        let stats = if s.t == 1 {
            self.exec
                .run_ops(&[MatmulOp::new(mm(x, &self.lm_head_p, &s.logits))
                    .with_packed_b()
                    .with_normed_a(&self.output_norm, self.cfg.rms_eps)])?
        } else {
            let h = self.cfg.embd;
            self.exec.run_copy_strided(
                x,
                &s.last,
                CopyDesc {
                    extent: [h, 1, 1],
                    src_offset: (s.t - 1) * h,
                    src_strides: [1, 0, 0],
                    dst_offset: 0,
                    dst_strides: [1, 0, 0],
                    ..Default::default()
                },
            )?;
            self.exec
                .run_rms_norm(&s.last, &self.output_norm, &s.last_n, self.cfg.rms_eps)?;
            self.exec
                .run_matmuls(&[mm(&s.last_n, &self.lm_head, &s.logits)])?
        };
        self.note("lm_head_mm", &stats);
        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let best = argmax(&logits)?;
        Ok((best, logits))
    }

    /// Emit the MLP block (o-projection + residual, FFN norm, gated
    /// up/down projections) for one layer, `x_in -> x_out`.
    ///
    /// For coopmat-eligible row counts (T >= 128) the three
    /// residual/gated projections run as PLAIN matmuls (keeping the
    /// tensor-core route, which cannot fuse epilogues) plus standalone
    /// binary combines — the epilogue's saved bandwidth pass is far
    /// cheaper than demoting the whole GEMM to the SIMT family.  At
    /// T == 1 (decode) the FFN RMSNorm additionally folds into the up
    /// and gate row GEMVs (gate keeps its Silu+Mul epilogue — llama
    /// has no bias, so the bias slot carries the norm weight).
    fn push_mlp_ops<'a>(
        &'a self,
        ops: &mut Vec<ExecOp<'a>>,
        layer: &'a Layer,
        s: &'a Scratch,
        x_in: &'a Tensor,
        x_out: &'a Tensor,
    ) {
        let eps = self.cfg.rms_eps;
        if s.t >= 128 {
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                &s.attn_flat,
                &layer.wo,
                &s.o,
            ))));
            ops.push(ExecOp::Binary {
                a: &s.o,
                b: x_in,
                out: &s.o,
                op: BinaryOp::AddScaled { beta: 1.0 },
            });
            ops.push(ExecOp::RmsNorm {
                input: &s.o,
                weight: &layer.ffn_norm,
                output: &s.on,
                eps,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.on, &layer.w_up, &s.up))));
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                &s.on,
                &layer.w_gate,
                &s.gate,
            ))));
            ops.push(ExecOp::Binary {
                a: &s.gate,
                b: &s.up,
                out: &s.gate,
                op: BinaryOp::SiluMul,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                &s.gate,
                &layer.w_down,
                x_out,
            ))));
            ops.push(ExecOp::Binary {
                a: x_out,
                b: &s.o,
                out: x_out,
                op: BinaryOp::AddScaled { beta: 1.0 },
            });
            return;
        }
        ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
            mm(&s.attn_flat, &layer.wo, &s.o),
            epi(
                Activation::None,
                EpilogueBinary::AddScaled { d: x_in, beta: 1.0 },
            ),
        )));
        if s.t == 1 {
            ops.push(ExecOp::Matmul(
                MatmulOp::new(mm(&s.o, &layer.w_up_p, &s.up))
                    .with_packed_b()
                    .with_normed_a(&layer.ffn_norm, eps),
            ));
            ops.push(ExecOp::Matmul(
                MatmulOp::with_epilogue(
                    mm(&s.o, &layer.w_gate_p, &s.gate),
                    epi(Activation::Silu, EpilogueBinary::Mul { d: &s.up }),
                )
                .with_packed_b()
                .with_normed_a(&layer.ffn_norm, eps),
            ));
        } else {
            ops.push(ExecOp::RmsNorm {
                input: &s.o,
                weight: &layer.ffn_norm,
                output: &s.on,
                eps,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.on, &layer.w_up, &s.up))));
            ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
                mm(&s.on, &layer.w_gate, &s.gate),
                epi(Activation::Silu, EpilogueBinary::Mul { d: &s.up }),
            )));
        }
        ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
            mm(&s.gate, &layer.w_down, x_out),
            epi(
                Activation::None,
                EpilogueBinary::AddScaled { d: &s.o, beta: 1.0 },
            ),
        )));
    }

    /// Emit the final-norm + LM-head tail on the last row of `x`
    /// (logits land in `s.logits`).
    fn push_lm_head_ops<'a>(&'a self, ops: &mut Vec<ExecOp<'a>>, s: &'a Scratch, x: &'a Tensor) {
        // Decode is a single row: fold the final RMSNorm into the
        // LM-head GEMV (same normed-A slot the FFN projections use)
        // and skip the last-row copy.
        if s.t == 1 {
            ops.push(ExecOp::Matmul(
                MatmulOp::new(mm(x, &self.lm_head_p, &s.logits))
                    .with_packed_b()
                    .with_normed_a(&self.output_norm, self.cfg.rms_eps),
            ));
            return;
        }
        let h = self.cfg.embd;
        ops.push(ExecOp::CopyStrided {
            src: x,
            dst: &s.last,
            desc: CopyDesc {
                extent: [h, 1, 1],
                src_offset: (s.t - 1) * h,
                src_strides: [1, 0, 0],
                dst_offset: 0,
                dst_strides: [1, 0, 0],
                ..Default::default()
            },
        });
        ops.push(ExecOp::RmsNorm {
            input: &s.last,
            weight: &self.output_norm,
            output: &s.last_n,
            eps: self.cfg.rms_eps,
        });
        ops.push(ExecOp::Matmul(MatmulOp::new(mm(
            &s.last_n,
            &self.lm_head,
            &s.logits,
        ))));
    }

    /// Per-op decode MLP block, `x_in -> x_out` (the graph paths emit
    /// the same chain through [`push_mlp_ops`](Self::push_mlp_ops)):
    /// o-projection with the residual add fused, the FFN RMSNorm folded
    /// into the up and gate row GEMVs (gate keeps its Silu+Mul
    /// epilogue — llama has no bias, so the bias slot carries the norm
    /// weight), and the down projection with the residual add fused.
    fn mlp_block_into(
        &self,
        layer: &Layer,
        s: &Scratch,
        x_in: &Tensor,
        x_out: &Tensor,
    ) -> Result<()> {
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            mm(&s.attn_flat, &layer.wo, &s.o),
            epi(
                Activation::None,
                EpilogueBinary::AddScaled { d: x_in, beta: 1.0 },
            ),
        )])?;
        self.note("o_proj_mm", &stats);
        let stats = self
            .exec
            .run_ops(&[MatmulOp::new(mm(&s.o, &layer.w_up_p, &s.up))
                .with_packed_b()
                .with_normed_a(&layer.ffn_norm, self.cfg.rms_eps)])?;
        self.note("ffn_up_mm", &stats);
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            mm(&s.o, &layer.w_gate_p, &s.gate),
            epi(Activation::Silu, EpilogueBinary::Mul { d: &s.up }),
        )
        .with_packed_b()
        .with_normed_a(&layer.ffn_norm, self.cfg.rms_eps)])?;
        self.note("ffn_gate_mm", &stats);
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            mm(&s.gate, &layer.w_down, x_out),
            epi(
                Activation::None,
                EpilogueBinary::AddScaled { d: &s.o, beta: 1.0 },
            ),
        )])?;
        self.note("ffn_down_mm", &stats);
        Ok(())
    }

    /// GPU argmax + embed-gather tail that closes a prepared/unrolled
    /// decode loop on-device (next token id into `x_a`).
    fn decode_close_ops<'a>(&'a self, s: &'a Scratch) -> [ExecOp<'a>; 2] {
        [
            ExecOp::Argmax {
                input: &s.logits,
                result: &self.token_buf,
            },
            ExecOp::EmbedGather {
                token: &self.token_buf,
                table: &self.embd_gpu,
                out: &s.x_a,
            },
        ]
    }
}
