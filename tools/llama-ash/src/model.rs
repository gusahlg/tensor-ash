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
    ATTN_DECODE_MAX_CHUNKS, Activation, AttnDecodeDesc, BinaryOp, CopyDesc, Epilogue,
    EpilogueBinary, ExecOp, Executor, FlashAttentionDesc, HostU32Buffer, MatmulCall, MatmulOp,
    MatmulStoreDesc, PosBuffer, RopeDesc, RopeScatterDesc, SoftmaxMask, Tensor, VulkanContext,
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
    /// Kt cache [kv_heads, dh, t_max], f16 by default
    /// (`LLAMA_ASH_KV=f32` opts out).
    kt_cache: Tensor,
    /// V cache [kv_heads, t_max, dh], same storage as `kt_cache`.
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
    /// Decode only: split-K partials for the fused attention op,
    /// [kv_heads * MAX_CHUNKS * group * (dh+2)] f32.
    attn_partials: Option<Tensor>,
    /// Decode only: same memory order as `q_heads`/`attn_heads` but
    /// shaped [heads, 1, dh] for the flash-decode variant.
    q_flash: Option<Tensor>,
    attn_flash: Option<Tensor>,
    /// Decode only: free contiguous reshapes of `q` / `attn_flat` to
    /// [kv_heads, group, dh] for the fused GQA attention op.
    q_gqa: Option<Tensor>,
    attn_gqa: Option<Tensor>,
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
        let q = dev(&[t, h])?;
        let attn_flat = dev(&[t, h])?;
        let (q_gqa, attn_gqa) = if decode {
            let group = cfg.heads / cfg.kv_heads;
            (
                Some(q.alias_with_shape(&[cfg.kv_heads, group, cfg.dh])?),
                Some(attn_flat.alias_with_shape(&[cfg.kv_heads, group, cfg.dh])?),
            )
        } else {
            (None, None)
        };
        let (q_heads, attn_heads, scores, attn_partials, q_flash, attn_flash) = if decode {
            let group = cfg.heads / cfg.kv_heads;
            let partials_len = cfg.kv_heads * ATTN_DECODE_MAX_CHUNKS * group * (cfg.dh + 2);
            (
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                Some(dev(&[cfg.kv_heads, group, cfg.t_max])?),
                Some(dev(&[partials_len])?),
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
                None,
            )
        };
        Ok(Self {
            t,
            x_a: dev(&[t, h])?,
            x_b: dev(&[t, h])?,
            xn: dev(&[t, h])?,
            q,
            k: dev(&[t, kv])?,
            v: dev(&[t, kv])?,
            q_heads,
            attn_heads,
            scores,
            attn_partials,
            q_flash,
            attn_flash,
            q_gqa,
            attn_gqa,
            attn_flat,
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
    /// Token embeddings on the CPU, row-major [vocab][embd] f32
    /// (prefill embeds T rows host-side).
    embd_cpu: Vec<f32>,
    /// The same table on the DEVICE ([vocab, embd], in the GGUF's
    /// storage precision) for the decode loop's on-GPU row gather.
    embd_gpu: Tensor,
    output_norm: Tensor,
    lm_head: Tensor,
    rope_table: Tensor,
    /// Tokens currently in the KV caches.
    pub pos: u32,
    decode_scratch: Scratch,
    prefill_scratch: Option<Scratch>,
    /// The 4-byte device-readable position cell the prepared decode
    /// graph reads; the host bumps it between replays.
    pos_buf: PosBuffer,
    /// The 4-byte host-readable token cell the prepared graph's GPU
    /// argmax writes; per token the host reads this ONE u32 instead of
    /// downloading the 32000-float logits.
    token_buf: HostU32Buffer,
    /// Decode strategy.  `LLAMA_ASH_DECODE=prepared|graph|perop|flash`
    /// overrides the default (prepared: record the token's command
    /// buffer once, replay it per token via the position buffer).
    decode_mode: DecodeMode,
    /// Fused split-K decode attention (default when dh == 64).
    /// `LLAMA_ASH_ATTN=composed` restores the scores/softmax/PV trio.
    fused_attn: bool,
    /// Per-op-class GPU nanoseconds for the current perop decode step
    /// (diagnostics; filled when `LLAMA_ASH_BREAKDOWN=1`).
    pub breakdown: std::cell::RefCell<Vec<(&'static str, u64)>>,
}

#[derive(Copy, Clone, PartialEq)]
enum DecodeMode {
    /// Record the whole token ONCE, replay per token (default): the
    /// position-dependent values are read from a device-side position
    /// buffer, so a decode step costs one `vkQueueSubmit` with zero
    /// host-side re-recording.  Requires the fused decode attention.
    Prepared,
    /// One `run_exec_ops` submission per token, re-recorded each call.
    Graph,
    /// The original per-op path: one submit + wait per dispatch.
    PerOp,
    /// Per-op with the prefill flash kernel for attention.  Kept as a
    /// measured comparison switch only — it loses to the fused split-K
    /// decode attention (`AttnDecode`) at every probed context length
    /// (see benchmarks/experiment-branch.md).
    Flash,
}

/// Greedy sampling: index of the largest logit.
fn argmax(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .context("empty logits")
}

/// The plain matmul call every projection uses: `c = a @ b`.
fn mm<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
    MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }
}

/// Bias-free fused epilogue (llama has no linear biases).
fn epi<'t>(activation: Activation, binary: EpilogueBinary<'t>) -> Epilogue<'t> {
    Epilogue {
        bias: None,
        activation,
        binary,
    }
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
        // f16 caches halve attention-side cache traffic and KV memory;
        // the composed decode matmuls pick up the f16w routes
        // automatically, and prefill uses the kv16 flash variants.
        // LLAMA_ASH_KV=f32 restores full precision.
        let kv_f32 = std::env::var("LLAMA_ASH_KV").as_deref() == Ok("f32");
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
            let (kt_cache, v_cache) = if kv_f32 {
                (
                    Tensor::uninit_device(ctx, &[kv_heads, dh, t_max])?,
                    Tensor::uninit_device(ctx, &[kv_heads, t_max, dh])?,
                )
            } else {
                (
                    Tensor::uninit_device_f16(ctx, &[kv_heads, dh, t_max])?,
                    Tensor::uninit_device_f16(ctx, &[kv_heads, t_max, dh])?,
                )
            };
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
        // Device copy of the embedding table for the decode loop's
        // on-GPU row gather, in the GGUF's own storage precision so the
        // gathered row is bit-identical to the CPU embed (an f16 GGUF
        // round-trips f16 -> f32 -> f16 exactly; an f32 GGUF stays f32).
        let embd_gpu = if embd_info.ggml_type == GGML_TYPE_F32 {
            Tensor::uninit_device(ctx, &[vocab, embd])?
        } else {
            Tensor::uninit_device_f16(ctx, &[vocab, embd])?
        };
        exec.upload(&embd_cpu, &embd_gpu)?;
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
        let pos_buf = exec.create_pos_buffer()?;
        let token_buf = exec.create_host_u32_buffer()?;
        log::info!("model loaded in {:.2}s", start.elapsed().as_secs_f64());
        Ok(Self {
            ctx: ctx.clone(),
            exec: exec.clone(),
            cfg,
            layers,
            embd_cpu,
            embd_gpu,
            output_norm,
            lm_head,
            rope_table,
            pos: 0,
            decode_scratch,
            prefill_scratch: None,
            pos_buf,
            token_buf,
            breakdown: std::cell::RefCell::new(Vec::new()),
            decode_mode: match std::env::var("LLAMA_ASH_DECODE").as_deref() {
                Ok("flash") => DecodeMode::Flash,
                Ok("composed") | Ok("perop") => DecodeMode::PerOp,
                Ok("graph") => DecodeMode::Graph,
                Ok("prepared") | Err(_) => DecodeMode::Prepared,
                Ok(other) => {
                    bail!("LLAMA_ASH_DECODE must be prepared, graph, perop, or flash, got {other}")
                }
            },
            // The fused split-K decode-attention op ships a dh64
            // kernel only; dh128 models keep the composed trio.
            fused_attn: match std::env::var("LLAMA_ASH_ATTN").as_deref() {
                Ok("composed") => false,
                Ok("fused") | Err(_) => {
                    if dh != 64 {
                        log::info!("fused decode attention needs dh 64 (have {dh}); composed path");
                    }
                    dh == 64
                }
                Ok(other) => bail!("LLAMA_ASH_ATTN must be fused or composed, got {other}"),
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

    /// Record one op-class GPU time into the `LLAMA_ASH_BREAKDOWN`
    /// diagnostics table.
    fn note(&self, class: &'static str, stats: &tensor_ash::RunStats) {
        if let Some(ns) = stats.gpu_time_ns {
            self.breakdown.borrow_mut().push((class, ns));
        }
    }

    /// Decode-only fused QKV: the attention RMSNorm folds into the
    /// three row GEMVs (reading `x_in` directly, no `xn` round-trip),
    /// q ropes in place, k ropes straight into the Kt cache, and the
    /// V row appends — six dispatches instead of nine.
    fn qkv_fused_decode(&self, layer: &Layer, s: &Scratch, x_in: &Tensor, pos: u32) -> Result<()> {
        let (kv, dh, t_max) = (self.cfg.kv_heads, self.cfg.dh, self.cfg.t_max);
        let normed =
            |b, c| MatmulOp::new(mm(x_in, b, c)).with_normed_a(&layer.attn_norm, self.cfg.rms_eps);
        let stats = self.exec.run_ops(&[
            normed(&layer.wq, &s.q),
            normed(&layer.wk, &s.k),
            normed(&layer.wv, &s.v),
        ])?;
        self.note("qkv_matmul", &stats);
        let rope = |heads| RopeDesc {
            heads,
            head_dim: dh,
            rot_dim: dh,
            pos_base: pos,
            ..Default::default()
        };
        let stats = self
            .exec
            .run_rope(&s.q, &self.rope_table, &s.q, rope(self.cfg.heads))?;
        self.note("rope", &stats);
        let stats = self.exec.run_rope_scatter(
            &s.k,
            &self.rope_table,
            &layer.kt_cache,
            rope(kv),
            RopeScatterDesc {
                dst_offset: pos,
                dst_strides: [1, dh * t_max, t_max],
                ..Default::default()
            },
        )?;
        self.note("rope_scatter", &stats);
        let stats = self.exec.run_copy_strided(
            &s.v,
            &layer.v_cache,
            CopyDesc {
                extent: [dh, kv, 1],
                src_offset: 0,
                src_strides: [1, dh, kv * dh],
                dst_offset: pos * dh,
                dst_strides: [1, t_max * dh, dh],
                ..Default::default()
            },
        )?;
        self.note("kv_append", &stats);
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
                ..Default::default()
            },
        )?;
        self.exec
            .run_rms_norm(&s.last, &self.output_norm, &s.last_n, self.cfg.rms_eps)?;
        let stats = self
            .exec
            .run_matmuls(&[mm(&s.last_n, &self.lm_head, &s.logits)])?;
        self.note("lm_head_mm", &stats);
        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let best = argmax(&logits)?;
        Ok((best, logits))
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

    /// The whole prefill as ONE submission: every op of every layer
    /// plus the LM-head tail is recorded into a single command buffer
    /// (`run_exec_ops`), paying one submit + fence wait instead of
    /// ~300 — the same treatment [`decode_graph`](Self::decode_graph)
    /// gives the token step.
    fn prefill_with(
        &self,
        s: &Scratch,
        host_x: &[f32],
        t: u32,
        pos_base: u32,
    ) -> Result<(u32, Vec<f32>)> {
        self.exec.upload(host_x, &s.x_a)?;
        let ops = self.prefill_ops(s, t, pos_base);
        let stats = self.exec.run_exec_ops(&ops)?;
        self.note("prefill_total", &stats);
        drop(ops);
        let mut logits = vec![0.0_f32; self.cfg.vocab as usize];
        self.exec.download(&s.logits, &mut logits)?;
        let best = argmax(&logits)?;
        Ok((best, logits))
    }

    /// Build the prefill's full op chain against `s` (flash attention
    /// path): per layer the attention RMSNorm, q/k/v projections, RoPE,
    /// KV-cache appends, head-permute copies bracketing the flash
    /// attention, and the MLP block; then the LM-head tail.  Scratch
    /// reuse across layers (xn, q/k/v, q_heads, attn_heads) and the
    /// x ping-pong serialize through `run_exec_ops`' hazard tracker.
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
        let mut ops: Vec<ExecOp<'a>> = Vec::with_capacity(self.layers.len() * 16 + 3);
        let (mut x_in, mut x_out) = (&s.x_a, &s.x_b);
        for layer in &self.layers {
            ops.push(ExecOp::RmsNorm {
                input: x_in,
                weight: &layer.attn_norm,
                output: &s.xn,
                eps: self.cfg.rms_eps,
            });
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wq, &s.q))));
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wk, &s.k))));
            ops.push(ExecOp::Matmul(MatmulOp::new(mm(&s.xn, &layer.wv, &s.v))));
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
            // KV append: k [t][h][d] -> Kt[h][d][pos_base + z] ...
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
            // ... and v [t][h][d] -> V[h][pos_base + z][d].
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
            // q [t][h][d] -> q_heads [h][t][d] for flash.
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
                },
            });
            // attn [h][t][d] -> attn_flat [t][h][d].
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
            self.push_mlp_ops(&mut ops, layer, s, x_in, x_out);
            std::mem::swap(&mut x_in, &mut x_out);
        }
        self.push_lm_head_ops(&mut ops, s, x_in);
        ops
    }

    /// Emit the MLP block (o-projection + residual, FFN norm, gated
    /// up/down projections) for one layer, `x_in -> x_out`.
    ///
    /// For coopmat-eligible row counts (T >= 256) the three
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
        if s.t >= 256 {
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
                MatmulOp::new(mm(&s.o, &layer.w_up, &s.up)).with_normed_a(&layer.ffn_norm, eps),
            ));
            ops.push(ExecOp::Matmul(
                MatmulOp::with_epilogue(
                    mm(&s.o, &layer.w_gate, &s.gate),
                    epi(Activation::Silu, EpilogueBinary::Mul { d: &s.up }),
                )
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
            .run_ops(&[MatmulOp::new(mm(&s.o, &layer.w_up, &s.up))
                .with_normed_a(&layer.ffn_norm, self.cfg.rms_eps)])?;
        self.note("ffn_up_mm", &stats);
        let stats = self.exec.run_ops(&[MatmulOp::with_epilogue(
            mm(&s.o, &layer.w_gate, &s.gate),
            epi(Activation::Silu, EpilogueBinary::Mul { d: &s.up }),
        )
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
        {
            let s = &self.decode_scratch;
            // Pos-relative descs + the position buffer: one recording
            // serves every token position.  The graph tail closes the
            // token loop on the GPU: argmax writes the next token id to
            // the host-readable cell, and the embedding gather writes
            // that token's row into `x_a` — the very tensor the graph
            // reads first — so the next replay feeds itself.
            let mut ops = self.decode_ops(s, 0, self.pos_buf.device_address());
            ops.push(ExecOp::Argmax {
                input: &s.logits,
                result: &self.token_buf,
            });
            ops.push(ExecOp::EmbedGather {
                token: &self.token_buf,
                table: &self.embd_gpu,
                out: &s.x_a,
            });
            let mut prepared = self.exec.prepare_exec_ops(&ops, Some(&self.pos_buf))?;
            // Seed the first replay with the prefill's (CPU-argmaxed)
            // token; from then on the graph's own gather refills x_a.
            let host_x = self.embed(&[token])?;
            self.exec.upload(&host_x, &s.x_a)?;
            for i in 0..n {
                // The position write is host-visible and becomes
                // visible to the device at submit; run()'s fence wait
                // orders the token-cell read after the GPU argmax.
                self.pos_buf.set(start_pos + i)?;
                let stats = prepared.run()?;
                self.note("prepared_total", &stats);
                token = self.token_buf.read()?;
                generated.push(token);
            }
        }
        self.pos = start_pos + n;
        Ok(generated)
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
        let (dh, t_max) = (self.cfg.dh, self.cfg.t_max);
        let eps = self.cfg.rms_eps;
        assert!(
            pos_addr == 0 || self.fused_attn,
            "pos-relative decode needs the fused attention (composed softmax has no pos read)"
        );
        // Position-dependent bases: zero when the position buffer
        // supplies the offset at execution time.
        let base = if pos_addr == 0 { pos } else { 0 };

        // Store-epilogue geometry for the fused q/k/v projections; the
        // scatter modes address element
        // `(base + p) * pos_scale + head * stride_head + dim * stride_dim`.
        let store = |pos_scale, stride_head, stride_dim| MatmulStoreDesc {
            head_dim: dh,
            pos_base: base,
            pos_scale,
            stride_head,
            stride_dim,
            pos_addr,
        };

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
            ops.push(ExecOp::Matmul(
                MatmulOp::new(mm(x_in, &layer.wq, &s.q))
                    .with_normed_a(&layer.attn_norm, eps)
                    .with_store_rope(&self.rope_table, store(0, 0, 0)),
            ));
            ops.push(ExecOp::Matmul(
                MatmulOp::new(mm(x_in, &layer.wk, &s.k))
                    .with_normed_a(&layer.attn_norm, eps)
                    // Kt append: one column per position.
                    .with_store_rope_scatter(
                        &self.rope_table,
                        &layer.kt_cache,
                        store(1, dh * t_max, t_max),
                    ),
            ));
            ops.push(ExecOp::Matmul(
                MatmulOp::new(mm(x_in, &layer.wv, &s.v))
                    .with_normed_a(&layer.attn_norm, eps)
                    // V append: one dh-row per position.
                    .with_store_scatter(&layer.v_cache, store(dh, t_max * dh, 1)),
            ));
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
