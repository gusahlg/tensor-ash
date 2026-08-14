//! Llama-family model built on tensor-ash ops: GGUF weight loading
//! (with the GGML->tensor-ash transpose), KV caches, and the
//! prefill / decode forward passes.
//!
//! The per-layer decode step mirrors tests/correctness/decoder.rs;
//! prefill swaps the composed attention for run_flash_attention.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use tensor_ash::{
    Activation, CopyDesc, Epilogue, EpilogueBinary, ExecOp, Executor, FlashAttentionDesc,
    MatmulCall, MatmulOp, RopeDesc, SoftmaxMask, Tensor, VulkanContext,
};

use crate::gguf::{GGML_TYPE_F32, GgufFile};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layers: u32,
    pub embd: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub dh: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub rms_eps: f32,
    pub rope_base: f32,
    /// KV-cache capacity in tokens.
    pub t_max: u32,
}

struct Layer {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// Kt cache f32 [kv_heads, dh, t_max].
    kt_cache: Tensor,
    /// V cache f32 [kv_heads, t_max, dh].
    v_cache: Tensor,
}

/// Scratch activations for one forward width `t`, reused across layers.
struct Scratch {
    t: u32,
    x_a: Tensor,
    x_b: Tensor,
    xn: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// Prefill: q permuted to [heads, t, dh].  Decode: q reshaped to
    /// [kv_heads, group, dh].
    q_heads: Tensor,
    /// Same shape story as `q_heads`, holds the attention output.
    attn_heads: Tensor,
    /// Decode only: [kv_heads, group, t_max] score rows.
    scores: Option<Tensor>,
    /// Decode only: same memory order as `q_heads`/`attn_heads` but
    /// shaped [heads, 1, dh] for the flash-decode variant.
    q_flash: Option<Tensor>,
    attn_flash: Option<Tensor>,
    attn_flat: Tensor,
    o: Tensor,
    on: Tensor,
    up: Tensor,
    gate: Tensor,
    last: Tensor,
    last_n: Tensor,
    logits: Tensor,
}

impl Scratch {
    fn new(ctx: &Arc<VulkanContext>, cfg: &Config, t: u32) -> Result<Self> {
        let (h, f, kv) = (cfg.embd, cfg.ffn, cfg.kv_heads * cfg.dh);
        let dev = |shape: &[u32]| Tensor::uninit_device(ctx, shape);
        let decode = t == 1;
        let (q_heads, attn_heads, scores, q_flash, attn_flash) = if decode {
            let group = cfg.heads / cfg.kv_heads;
            (
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                Some(dev(&[cfg.kv_heads, group, cfg.t_max])?),
                Some(dev(&[cfg.heads, 1, cfg.dh])?),
                Some(dev(&[cfg.heads, 1, cfg.dh])?),
            )
        } else {
            (
                dev(&[cfg.heads, t, cfg.dh])?,
                dev(&[cfg.heads, t, cfg.dh])?,
                None,
                None,
                None,
            )
        };
        Ok(Self {
            t,
            x_a: dev(&[t, h])?,
            x_b: dev(&[t, h])?,
            xn: dev(&[t, h])?,
            q: dev(&[t, h])?,
            k: dev(&[t, kv])?,
            v: dev(&[t, kv])?,
            q_heads,
            attn_heads,
            scores,
            q_flash,
            attn_flash,
            attn_flat: dev(&[t, h])?,
            o: dev(&[t, h])?,
            on: dev(&[t, h])?,
            up: dev(&[t, f])?,
            gate: dev(&[t, f])?,
            last: dev(&[1, h])?,
            last_n: dev(&[1, h])?,
            logits: dev(&[1, cfg.vocab])?,
        })
    }
}

pub struct Model {
    ctx: Arc<VulkanContext>,
    exec: Arc<Executor>,
    pub cfg: Config,
    layers: Vec<Layer>,
    /// Token embeddings on the CPU, row-major [vocab][embd] f32.
    embd_cpu: Vec<f32>,
    output_norm: Tensor,
    lm_head: Tensor,
    rope_table: Tensor,
    /// Tokens currently in the KV caches.
    pub pos: u32,
    decode_scratch: Scratch,
    prefill_scratch: Option<Scratch>,
    /// Decode strategy.  `LLAMA_ASH_DECODE=graph|perop|flash`
    /// overrides the default (graph: the whole token as one
    /// submission).
    decode_mode: DecodeMode,
    /// Per-op-class GPU nanoseconds for the current perop decode step
    /// (diagnostics; filled when `LLAMA_ASH_BREAKDOWN=1`).
    pub breakdown: std::cell::RefCell<Vec<(&'static str, u64)>>,
}

#[derive(Copy, Clone, PartialEq)]
enum DecodeMode {
    /// One `run_exec_ops` submission per token (default).
    Graph,
    /// The original per-op path: one submit + wait per dispatch.
    PerOp,
    /// Per-op with the fused flash kernel for attention.
    Flash,
}

/// GGML row-major [n_out][n_in] -> tensor-ash [n_in][n_out].
fn transpose(src: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    assert_eq!(src.len(), n_out * n_in);
    let mut dst = vec![0.0_f32; src.len()];
    for r in 0..n_out {
        for c in 0..n_in {
            dst[c * n_out + r] = src[r * n_in + c];
        }
    }
    dst
}

impl Model {
    pub fn load(
        ctx: &Arc<VulkanContext>,
        exec: &Arc<Executor>,
        path: &Path,
        t_max: u32,
    ) -> Result<Self> {
        let start = Instant::now();
        let mut gguf = GgufFile::open(path)?;

        let arch = match gguf.metadata.get("general.architecture") {
            Some(crate::gguf::Value::Str(s)) => s.clone(),
            _ => bail!("GGUF missing general.architecture"),
        };
        ensure!(
            arch == "llama",
            "unsupported architecture {arch} (want llama)"
        );

        let n_layers = gguf.require_u64("llama.block_count")? as u32;
        let embd = gguf.require_u64("llama.embedding_length")? as u32;
        let heads = gguf.require_u64("llama.attention.head_count")? as u32;
        let kv_heads = gguf
            .require_u64("llama.attention.head_count_kv")
            .unwrap_or(heads as u64) as u32;
        let ffn = gguf.require_u64("llama.feed_forward_length")? as u32;
        let ctx_len = gguf.require_u64("llama.context_length")? as u32;
        let rms_eps = gguf.f64_or("llama.attention.layer_norm_rms_epsilon", 1e-5) as f32;
        let rope_base = gguf.f64_or("llama.rope.freq_base", 10000.0) as f32;
        ensure!(heads > 0 && embd.is_multiple_of(heads), "bad head geometry");
        ensure!(
            heads.is_multiple_of(kv_heads),
            "heads must divide by kv_heads"
        );
        let dh = embd / heads;
        ensure!(
            dh == 64 || dh == 128,
            "flash attention needs dh 64/128, got {dh}"
        );
        let t_max = t_max.min(ctx_len);

        // Vocab size from the embedding tensor: ne = [embd, vocab].
        let embd_info = gguf.info("token_embd.weight")?.clone();
        ensure!(
            embd_info.ne.len() == 2 && embd_info.ne[0] == embd as u64,
            "token_embd.weight ne {:?} does not match embd {embd}",
            embd_info.ne
        );
        let vocab = embd_info.ne[1] as u32;

        let cfg = Config {
            n_layers,
            embd,
            heads,
            kv_heads,
            dh,
            ffn,
            vocab,
            rms_eps,
            rope_base,
            t_max,
        };
        log::info!(
            "config: layers={} embd={} heads={} kv_heads={} dh={} ffn={} vocab={} \
             eps={:e} rope_base={} t_max={}",
            cfg.n_layers,
            cfg.embd,
            cfg.heads,
            cfg.kv_heads,
            cfg.dh,
            cfg.ffn,
            cfg.vocab,
            cfg.rms_eps,
            cfg.rope_base,
            cfg.t_max
        );

        // Loads a 2D linear weight (y = W.x, ggml ne = [n_in, n_out]),
        // transposes to [n_in][n_out], and uploads as f16.
        let load_linear = |gguf: &mut GgufFile, name: &str, n_in: u32, n_out: u32| {
            let info = gguf.info(name)?.clone();
            ensure!(
                info.ne == [n_in as u64, n_out as u64],
                "{name}: ne {:?} != expected [{n_in}, {n_out}]",
                info.ne
            );
            let host = gguf.read_f32(name)?;
            let t = transpose(&host, n_out as usize, n_in as usize);
            let tensor = Tensor::uninit_device_f16(ctx, &[n_in, n_out])?;
            exec.upload(&t, &tensor)?;
            anyhow::Ok((tensor, t))
        };
        let load_norm = |gguf: &mut GgufFile, name: &str| {
            let info = gguf.info(name)?.clone();
            ensure!(
                info.ne == [embd as u64] && info.ggml_type == GGML_TYPE_F32,
                "{name}: expected f32 [{embd}], got ne {:?} type {}",
                info.ne,
                info.ggml_type
            );
            let host = gguf.read_f32(name)?;
            let tensor = Tensor::uninit_device(ctx, &[embd])?;
            exec.upload(&host, &tensor)?;
            anyhow::Ok(tensor)
        };

        let kv_dim = kv_heads * dh;
        let mut layers = Vec::with_capacity(n_layers as usize);
        let zeros_kt = vec![0.0_f32; (kv_heads * dh * t_max) as usize];
        let mut checksum_logged = false;
        for i in 0..n_layers {
            let p = |suffix: &str| format!("blk.{i}.{suffix}");
            let (wq, wq_host) = load_linear(&mut gguf, &p("attn_q.weight"), embd, embd)?;
            if !checksum_logged {
                // Sanity: checksum of transposed row 0 of blk.0.attn_q.
                let row_sum: f64 = wq_host[..embd as usize].iter().map(|&v| v as f64).sum();
                log::info!("blk.0.attn_q.weight transposed row0 sum = {row_sum:.6}");
                checksum_logged = true;
            }
            let (wk, _) = load_linear(&mut gguf, &p("attn_k.weight"), embd, kv_dim)?;
            let (wv, _) = load_linear(&mut gguf, &p("attn_v.weight"), embd, kv_dim)?;
            let (wo, _) = load_linear(&mut gguf, &p("attn_output.weight"), embd, embd)?;
            let (w_gate, _) = load_linear(&mut gguf, &p("ffn_gate.weight"), embd, ffn)?;
            let (w_up, _) = load_linear(&mut gguf, &p("ffn_up.weight"), embd, ffn)?;
            let (w_down, _) = load_linear(&mut gguf, &p("ffn_down.weight"), ffn, embd)?;
            let attn_norm = load_norm(&mut gguf, &p("attn_norm.weight"))?;
            let ffn_norm = load_norm(&mut gguf, &p("ffn_norm.weight"))?;
            let kt_cache = Tensor::uninit_device(ctx, &[kv_heads, dh, t_max])?;
            let v_cache = Tensor::uninit_device(ctx, &[kv_heads, t_max, dh])?;
            exec.upload(&zeros_kt, &kt_cache)?;
            exec.upload(&zeros_kt, &v_cache)?;
            layers.push(Layer {
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                attn_norm,
                ffn_norm,
                kt_cache,
                v_cache,
            });
        }

        let embd_cpu = gguf.read_f32("token_embd.weight")?;
        let output_norm = load_norm(&mut gguf, "output_norm.weight")?;
        // Some GGUFs tie the LM head to the embeddings.
        let lm_head = if gguf.tensors.contains_key("output.weight") {
            load_linear(&mut gguf, "output.weight", embd, vocab)?.0
        } else {
            log::info!("output.weight absent; using tied token_embd.weight");
            let t = transpose(&embd_cpu, vocab as usize, embd as usize);
            let tensor = Tensor::uninit_device_f16(ctx, &[embd, vocab])?;
            exec.upload(&t, &tensor)?;
            tensor
        };

        // Rope table [t_max, dh/2, 2]: (cos, sin) of
        // pos / rope_base^(2i/dh), llama NORM style (adjacent pairs).
        let half = (dh / 2) as usize;
        let mut table = vec![0.0_f32; t_max as usize * half * 2];
        for pos in 0..t_max as usize {
            for pair in 0..half {
                let theta = pos as f64 / (rope_base as f64).powf(2.0 * pair as f64 / dh as f64);
                table[(pos * half + pair) * 2] = theta.cos() as f32;
                table[(pos * half + pair) * 2 + 1] = theta.sin() as f32;
            }
        }
        let rope_table = Tensor::uninit_device(ctx, &[t_max, dh / 2, 2])?;
        exec.upload(&table, &rope_table)?;

        let decode_scratch = Scratch::new(ctx, &cfg, 1)?;
        log::info!("model loaded in {:.2}s", start.elapsed().as_secs_f64());
        Ok(Self {
            ctx: ctx.clone(),
            exec: exec.clone(),
            cfg,
            layers,
            embd_cpu,
            output_norm,
            lm_head,
            rope_table,
            pos: 0,
            decode_scratch,
            prefill_scratch: None,
            breakdown: std::cell::RefCell::new(Vec::new()),
            decode_mode: match std::env::var("LLAMA_ASH_DECODE").as_deref() {
                Ok("flash") => DecodeMode::Flash,
                Ok("composed") | Ok("perop") => DecodeMode::PerOp,
                Ok("graph") | Err(_) => DecodeMode::Graph,
                Ok(other) => bail!("LLAMA_ASH_DECODE must be graph, perop, or flash, got {other}"),
            },
        })
    }

    /// Re-zeroes the KV caches and rewinds to position 0.
    pub fn reset(&mut self) -> Result<()> {
        let zeros = vec![0.0_f32; (self.cfg.kv_heads * self.cfg.dh * self.cfg.t_max) as usize];
        for layer in &self.layers {
            self.exec.upload(&zeros, &layer.kt_cache)?;
            self.exec.upload(&zeros, &layer.v_cache)?;
        }
        self.pos = 0;
        Ok(())
    }

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

    /// Appends roped k/v rows for `t` tokens starting at `pos_base`
    /// into the layer caches, one strided copy per cache.
    fn append_kv(&self, layer: &Layer, s: &Scratch, t: u32, pos_base: u32) -> Result<()> {
        let (kv, dh, t_max) = (self.cfg.kv_heads, self.cfg.dh, self.cfg.t_max);
        // k [t][h][d] -> Kt[h][d][pos_base + z]
        let stats = self.exec.run_copy_strided(
            &s.k,
            &layer.kt_cache,
            CopyDesc {
                extent: [dh, kv, t],
                src_offset: 0,
                src_strides: [1, dh, kv * dh],
                dst_offset: pos_base,
                dst_strides: [t_max, dh * t_max, 1],
            },
        )?;
        self.note("kv_append", &stats);
        // v [t][h][d] -> V[h][pos_base + z][d]
        let stats = self.exec.run_copy_strided(
            &s.v,
            &layer.v_cache,
            CopyDesc {
                extent: [dh, kv, t],
                src_offset: 0,
                src_strides: [1, dh, kv * dh],
                dst_offset: pos_base * dh,
                dst_strides: [1, t_max * dh, dh],
            },
        )?;
        self.note("kv_append", &stats);
        Ok(())
    }

    /// QKV projections + RoPE for `t` tokens at `pos_base`; reads
    /// `s.xn`, fills `s.q`, `s.k`, `s.v` (roped).
    fn note(&self, class: &'static str, stats: &tensor_ash::RunStats) {
        if let Some(ns) = stats.gpu_time_ns {
            self.breakdown.borrow_mut().push((class, ns));
        }
    }

    fn qkv(&self, layer: &Layer, s: &Scratch, pos_base: u32) -> Result<()> {
        let call = |b, c| MatmulCall {
            a: &s.xn,
            b,
            c,
            alpha: 1.0,
            accumulate: false,
        };
        let stats = self.exec.run_matmuls(&[
            call(&layer.wq, &s.q),
            call(&layer.wk, &s.k),
            call(&layer.wv, &s.v),
        ])?;
        self.note("qkv_matmul", &stats);
        let rope = |heads| RopeDesc {
            heads,
            head_dim: self.cfg.dh,
            rot_dim: self.cfg.dh,
            pos_base,
        };
        let stats = self
            .exec
            .run_rope(&s.q, &self.rope_table, &s.q, rope(self.cfg.heads))?;
        self.note("rope", &stats);
        let stats = self
            .exec
            .run_rope(&s.k, &self.rope_table, &s.k, rope(self.cfg.kv_heads))?;
        self.note("rope", &stats);
        Ok(())
    }

    /// Final norm + LM head on the last row of `x`; returns argmax and
    /// logits.
    fn lm_head_last_row(&self, s: &Scratch, x: &Tensor) -> Result<(u32, Vec<f32>)> {
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
            },
        )?;
        self.exec
            .run_rms_norm(&s.last, &self.output_norm, &s.last_n, self.cfg.rms_eps)?;
        let stats = self.exec.run_matmuls(&[MatmulCall {
            a: &s.last_n,
            b: &self.lm_head,
            c: &s.logits,
            alpha: 1.0,
            accumulate: false,
        }])?;
        self.note("lm_head_mm", &stats);
        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .context("empty logits")?;
        Ok((argmax, logits))
    }

    /// Prefill `tokens` starting from the current position (flash
    /// attention path).  Returns the greedy next token and its logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(u32, Vec<f32>)> {
        let t = u32::try_from(tokens.len()).context("prompt too long")?;
        ensure!(t > 0, "prefill needs at least one token");
        let pos_base = self.pos;
        ensure!(pos_base + t <= self.cfg.t_max, "KV cache overflow");
        if self.prefill_scratch.as_ref().is_none_or(|s| s.t != t) {
            self.prefill_scratch = Some(Scratch::new(&self.ctx, &self.cfg, t)?);
        }
        let host_x = self.embed(tokens)?;
        let s = self.prefill_scratch.take().unwrap();
        let result = self.prefill_with(&s, &host_x, t, pos_base);
        self.prefill_scratch = Some(s);
        let (argmax, logits) = result?;
        self.pos = pos_base + t;
        Ok((argmax, logits))
    }

    fn prefill_with(
        &self,
        s: &Scratch,
        host_x: &[f32],
        t: u32,
        pos_base: u32,
    ) -> Result<(u32, Vec<f32>)> {
        let (heads, dh) = (self.cfg.heads, self.cfg.dh);
        self.exec.upload(host_x, &s.x_a)?;
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        for layer in &self.layers {
            self.exec
                .run_rms_norm(x_in, &layer.attn_norm, &s.xn, self.cfg.rms_eps)?;
            self.qkv(layer, s, pos_base)?;
            self.append_kv(layer, s, t, pos_base)?;
            // q [t][h][d] -> q_heads [h][t][d] for flash.
            self.exec.run_copy_strided(
                &s.q,
                &s.q_heads,
                CopyDesc {
                    extent: [dh, t, heads],
                    src_offset: 0,
                    src_strides: [1, heads * dh, dh],
                    dst_offset: 0,
                    dst_strides: [1, dh, t * dh],
                },
            )?;
            self.exec.run_flash_attention(
                &s.q_heads,
                &layer.kt_cache,
                &layer.v_cache,
                &s.attn_heads,
                FlashAttentionDesc {
                    kv_len: pos_base + t,
                    pos_base,
                    scale: 1.0 / (dh as f32).sqrt(),
                },
            )?;
            // attn [h][t][d] -> attn_flat [t][h][d].
            self.exec.run_copy_strided(
                &s.attn_heads,
                &s.attn_flat,
                CopyDesc {
                    extent: [dh, t, heads],
                    src_offset: 0,
                    src_strides: [1, dh, t * dh],
                    dst_offset: 0,
                    dst_strides: [1, heads * dh, dh],
                },
            )?;
            self.mlp_block_into(layer, s, x_in, x_out)?;
            std::mem::swap(&mut x_in, &mut x_out);
        }
        self.lm_head_last_row(s, x_in)
    }

    /// `mlp_block` with an explicit output (prefill ping-pongs x).
    fn mlp_block_into(
        &self,
        layer: &Layer,
        s: &Scratch,
        x_in: &Tensor,
        x_out: &Tensor,
    ) -> Result<()> {
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            MatmulCall {
                a: &s.attn_flat,
                b: &layer.wo,
                c: &s.o,
                alpha: 1.0,
                accumulate: false,
            },
            Epilogue {
                bias: None,
                activation: Activation::None,
                binary: EpilogueBinary::AddScaled { d: x_in, beta: 1.0 },
            },
        )])?;
        self.note("o_proj_mm", &stats);
        let stats = self
            .exec
            .run_rms_norm(&s.o, &layer.ffn_norm, &s.on, self.cfg.rms_eps)?;
        self.note("rms_norm", &stats);
        let stats = self.exec.run_matmuls(&[MatmulCall {
            a: &s.on,
            b: &layer.w_up,
            c: &s.up,
            alpha: 1.0,
            accumulate: false,
        }])?;
        self.note("ffn_up_mm", &stats);
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            MatmulCall {
                a: &s.on,
                b: &layer.w_gate,
                c: &s.gate,
                alpha: 1.0,
                accumulate: false,
            },
            Epilogue {
                bias: None,
                activation: Activation::Silu,
                binary: EpilogueBinary::Mul { d: &s.up },
            },
        )])?;
        self.note("ffn_gate_mm", &stats);
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            MatmulCall {
                a: &s.gate,
                b: &layer.w_down,
                c: x_out,
                alpha: 1.0,
                accumulate: false,
            },
            Epilogue {
                bias: None,
                activation: Activation::None,
                binary: EpilogueBinary::AddScaled { d: &s.o, beta: 1.0 },
            },
        )])?;
        self.note("ffn_down_mm", &stats);
        Ok(())
    }

    /// One greedy decode step for `token` at the current position.
    /// Returns the next token and its logits.
    pub fn decode(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        if self.decode_mode == DecodeMode::Graph {
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
        let (h, dh, kv, t_max) = (
            self.cfg.embd,
            self.cfg.dh,
            self.cfg.kv_heads,
            self.cfg.t_max,
        );
        let eps = self.cfg.rms_eps;
        let s = &self.decode_scratch;
        self.exec.upload(&host_x, &s.x_a)?;

        fn mm<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
            MatmulCall {
                a,
                b,
                c,
                alpha: 1.0,
                accumulate: false,
            }
        }
        let straight = CopyDesc {
            extent: [h, 1, 1],
            src_offset: 0,
            src_strides: [1, 0, 0],
            dst_offset: 0,
            dst_strides: [1, 0, 0],
        };
        let rope = |heads| RopeDesc {
            heads,
            head_dim: dh,
            rot_dim: dh,
            pos_base: pos,
        };

        // Contiguous reshapes are free: the GQA-batched attention
        // views of q and of the attention output share memory with
        // their flat forms, eliminating two copy dispatches per layer.
        let group = self.cfg.heads / kv;
        let q_gqa = s.q.alias_with_shape(&[kv, group, dh])?;
        let attn_gqa = s.attn_flat.alias_with_shape(&[kv, group, dh])?;
        let mut ops: Vec<ExecOp<'_>> = Vec::with_capacity(self.layers.len() * 16 + 3);
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        for layer in &self.layers {
            let scores = s.scores.as_ref().unwrap();
            ops.push(ExecOp::RmsNorm {
                input: x_in,
                weight: &layer.attn_norm,
                output: &s.xn,
                eps,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wq, &s.q))));
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wk, &s.k))));
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wv, &s.v))));
            ops.push(ExecOp::Rope {
                input: &s.q,
                table: &self.rope_table,
                output: &s.q,
                desc: rope(self.cfg.heads),
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
                    extent: [dh, kv, 1],
                    src_offset: 0,
                    src_strides: [1, dh, kv * dh],
                    dst_offset: pos,
                    dst_strides: [t_max, dh * t_max, 1],
                },
            });
            ops.push(ExecOp::CopyStrided {
                src: &s.v,
                dst: &layer.v_cache,
                desc: CopyDesc {
                    extent: [dh, kv, 1],
                    src_offset: 0,
                    src_strides: [1, dh, kv * dh],
                    dst_offset: pos * dh,
                    dst_strides: [1, t_max * dh, dh],
                },
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(
                &q_gqa,
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
                &attn_gqa,
            ))));
            ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
                mm(&s.attn_flat, &layer.wo, &s.o),
                Epilogue {
                    bias: None,
                    activation: Activation::None,
                    binary: EpilogueBinary::AddScaled { d: x_in, beta: 1.0 },
                },
            )));
            ops.push(ExecOp::RmsNorm {
                input: &s.o,
                weight: &layer.ffn_norm,
                output: &s.on,
                eps,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.on, &layer.w_up, &s.up))));
            ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
                mm(&s.on, &layer.w_gate, &s.gate),
                Epilogue {
                    bias: None,
                    activation: Activation::Silu,
                    binary: EpilogueBinary::Mul { d: &s.up },
                },
            )));
            ops.push(ExecOp::Matmul(MatmulOp::with_epilogue(
                mm(&s.gate, &layer.w_down, x_out),
                Epilogue {
                    bias: None,
                    activation: Activation::None,
                    binary: EpilogueBinary::AddScaled { d: &s.o, beta: 1.0 },
                },
            )));
            std::mem::swap(&mut x_in, &mut x_out);
        }
        // Final norm + LM head (decode scratch is t=1: last row = x).
        ops.push(ExecOp::CopyStrided {
            src: x_in,
            dst: &s.last,
            desc: straight,
        });
        ops.push(ExecOp::RmsNorm {
            input: &s.last,
            weight: &self.output_norm,
            output: &s.last_n,
            eps,
        });
        ops.push(ExecOp::Matmul(MatmulOp::new(mm(
            &s.last_n,
            &self.lm_head,
            &s.logits,
        ))));
        let stats = self.exec.run_exec_ops(&ops)?;
        self.note("graph_total", &stats);
        drop(ops);

        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .context("empty logits")?;
        self.pos = pos + 1;
        Ok((argmax, logits))
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
        };
        for layer in &self.layers {
            let stats = self
                .exec
                .run_rms_norm(x_in, &layer.attn_norm, &s.xn, self.cfg.rms_eps)?;
            self.note("rms_norm", &stats);
            self.qkv(layer, s, pos)?;
            self.append_kv(layer, s, 1, pos)?;
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
                    },
                )?;
                self.exec
                    .run_copy_strided(attn_flash, &s.attn_flat, straight(h))?;
            } else {
                // Reshape q [1, embd] -> [kv_heads, group, dh] (same
                // memory order): one batched matmul covers the GQA
                // groups.
                let stats = self.exec.run_copy_strided(&s.q, &s.q_heads, straight(h))?;
                self.note("reshape_copy", &stats);
                let scores = s.scores.as_ref().unwrap();
                let stats = self.exec.run_matmuls(&[MatmulCall {
                    a: &s.q_heads,
                    b: &layer.kt_cache,
                    c: scores,
                    alpha: 1.0,
                    accumulate: false,
                }])?;
                self.note("attn_scores_mm", &stats);
                let stats = self.exec.run_softmax_rows(
                    scores,
                    scores,
                    1.0 / (dh as f32).sqrt(),
                    SoftmaxMask::Prefix { valid: pos + 1 },
                )?;
                self.note("attn_softmax", &stats);
                let stats = self.exec.run_matmuls(&[MatmulCall {
                    a: scores,
                    b: &layer.v_cache,
                    c: &s.attn_heads,
                    alpha: 1.0,
                    accumulate: false,
                }])?;
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
