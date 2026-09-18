//! GGUF load: metadata, transposed weights, packed decode copies.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use tensor_ash::{Executor, Tensor, VulkanContext, f16w_row_tile_n, pack_f16w_row_tiles};

use super::{Config, DecodeMode, Layer, LoadOverrides, Model, Scratch, transpose};
use crate::gguf::{GGML_TYPE_F32, GgufFile};

impl Model {
    pub fn load(
        ctx: &Arc<VulkanContext>,
        exec: &Arc<Executor>,
        path: &Path,
        t_max: u32,
    ) -> Result<Self> {
        Self::load_with(ctx, exec, path, t_max, LoadOverrides::default())
    }

    /// [`load`](Self::load) with programmatic [`LoadOverrides`] for the
    /// decode-mode and KV-dtype environment knobs.
    pub fn load_with(
        ctx: &Arc<VulkanContext>,
        exec: &Arc<Executor>,
        path: &Path,
        t_max: u32,
        overrides: LoadOverrides,
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
        let load_linear_host = |gguf: &mut GgufFile, name: &str, n_in: u32, n_out: u32| {
            let info = gguf.info(name)?.clone();
            ensure!(
                info.ne == [n_in as u64, n_out as u64],
                "{name}: ne {:?} != expected [{n_in}, {n_out}]",
                info.ne
            );
            let host = gguf.read_f32(name)?;
            anyhow::Ok(transpose(&host, n_out as usize, n_in as usize))
        };
        let load_linear = |gguf: &mut GgufFile, name: &str, n_in: u32, n_out: u32| {
            let t = load_linear_host(gguf, name, n_in, n_out)?;
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
        let kv_f32 = overrides
            .kv_f32
            .unwrap_or_else(|| std::env::var("LLAMA_ASH_KV").as_deref() == Ok("f32"));
        let pack = |host: &[f32], k: u32, n: u32| -> Result<Tensor> {
            let tile = f16w_row_tile_n(k, n) as usize;
            let packed = pack_f16w_row_tiles(host, k as usize, n as usize, tile);
            let tensor = Tensor::uninit_device_f16(ctx, &[k, n])?;
            exec.upload(&packed, &tensor)?;
            Ok(tensor)
        };
        let mut checksum_logged = false;
        for i in 0..n_layers {
            let p = |suffix: &str| format!("blk.{i}.{suffix}");
            let wq_host = load_linear_host(&mut gguf, &p("attn_q.weight"), embd, embd)?;
            if !checksum_logged {
                // Sanity: checksum of transposed row 0 of blk.0.attn_q.
                let row_sum: f64 = wq_host[..embd as usize].iter().map(|&v| v as f64).sum();
                log::info!("blk.0.attn_q.weight transposed row0 sum = {row_sum:.6}");
                checksum_logged = true;
            }
            let wk_host = load_linear_host(&mut gguf, &p("attn_k.weight"), embd, kv_dim)?;
            let wv_host = load_linear_host(&mut gguf, &p("attn_v.weight"), embd, kv_dim)?;
            let qkv_n = embd + 2 * kv_dim;
            let mut qkv_host = vec![0.0_f32; embd as usize * qkv_n as usize];
            for row in 0..embd as usize {
                let dst = row * qkv_n as usize;
                let q = row * embd as usize;
                qkv_host[dst..dst + embd as usize].copy_from_slice(&wq_host[q..q + embd as usize]);
                let k = row * kv_dim as usize;
                qkv_host[dst + embd as usize..dst + embd as usize + kv_dim as usize]
                    .copy_from_slice(&wk_host[k..k + kv_dim as usize]);
                qkv_host[dst + embd as usize + kv_dim as usize..dst + qkv_n as usize]
                    .copy_from_slice(&wv_host[k..k + kv_dim as usize]);
            }
            let w_qkv = Tensor::uninit_device_f16(ctx, &[embd, qkv_n])?;
            exec.upload(&qkv_host, &w_qkv)?;
            let (wo, _) = load_linear(&mut gguf, &p("attn_output.weight"), embd, embd)?;
            let (w_gate, gate_host) = load_linear(&mut gguf, &p("ffn_gate.weight"), embd, ffn)?;
            let (w_up, up_host) = load_linear(&mut gguf, &p("ffn_up.weight"), embd, ffn)?;
            let (w_down, _) = load_linear(&mut gguf, &p("ffn_down.weight"), ffn, embd)?;
            let wq_p = pack(&wq_host, embd, embd)?;
            let wk_p = pack(&wk_host, embd, kv_dim)?;
            let wv_p = pack(&wv_host, embd, kv_dim)?;
            let w_gate_p = pack(&gate_host, embd, ffn)?;
            let w_up_p = pack(&up_host, embd, ffn)?;
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
                w_qkv,
                wo,
                w_gate,
                w_up,
                w_down,
                wq_p,
                wk_p,
                wv_p,
                w_gate_p,
                w_up_p,
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
        let (lm_head, lm_host) = if gguf.tensors.contains_key("output.weight") {
            load_linear(&mut gguf, "output.weight", embd, vocab)?
        } else {
            log::info!("output.weight absent; using tied token_embd.weight");
            let t = transpose(&embd_cpu, vocab as usize, embd as usize);
            let tensor = Tensor::uninit_device_f16(ctx, &[embd, vocab])?;
            exec.upload(&t, &tensor)?;
            (tensor, t)
        };
        let lm_head_p = pack(&lm_host, embd, vocab)?;

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

        let decode_scratch = Scratch::new(ctx, &cfg, 1, true, false)?;
        let pos_buf = exec.create_pos_buffer()?;
        let token_buf = exec.create_host_u32_buffer()?;
        let token_ids = exec.create_token_id_buffer(t_max)?;
        let prefill_act_f16 = !kv_f32
            && dh == 64
            && ctx.coopmat_enabled
            && ctx.coopmat2_enabled
            && ctx.f16_storage_enabled
            && embd.is_multiple_of(64)
            && kv_dim.is_multiple_of(64)
            && ffn.is_multiple_of(64)
            && std::env::var("LLAMA_ASH_ACT").as_deref() != Ok("f32");
        log::info!(
            "f16 prefill activations: {}",
            if prefill_act_f16 {
                "eligible (T >= 128 with T % 64 == 0)"
            } else {
                "off"
            }
        );
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
            lm_head_p,
            rope_table,
            pos: 0,
            decode_scratch,
            prefill_scratch: None,
            prefill_act_f16,
            pos_buf,
            token_buf,
            token_ids,
            breakdown: std::cell::RefCell::new(Vec::new()),
            decode_mode: match overrides.decode_mode {
                Some(mode) => mode,
                None => match std::env::var("LLAMA_ASH_DECODE").as_deref() {
                    Ok("flash") => DecodeMode::Flash,
                    Ok("composed") | Ok("perop") => DecodeMode::PerOp,
                    Ok("graph") => DecodeMode::Graph,
                    Ok("prepared") | Err(_) => DecodeMode::Prepared,
                    Ok(other) => {
                        bail!(
                            "LLAMA_ASH_DECODE must be prepared, graph, perop, or flash, got {other}"
                        )
                    }
                },
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
            prefill_prepared: None,
            decode_prepared: None,
            decode_unrolled: None,
        })
    }
}
