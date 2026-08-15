//! T1 / T5 / T7: model-level theses against a real GGUF model.
//!
//! T1 (decode-bandwidth-bound): decode GPU time/token is priced by
//! weight bytes/token over the device's achievable bandwidth — the
//! campaign's core claim about where decode time goes.
//!
//! T5 (prefill accounting): the pp512 graph's GPU time decomposes into
//! independently measured op-class terms (GEMM + flash attention +
//! elementwise + barrier drain) whose sum lands within tolerance of
//! the measured whole.  The piecewise inventory mirrors
//! `llama-ash/src/model.rs::prefill_ops` (T >= 256 branch); if that
//! structure drifts, THIS THESIS FAILING IS THE ALARM.
//!
//! T7 (token exactness): every decode mode x KV dtype x prefill width
//! emits bit-identical greedy tokens — the correctness invariant all
//! performance work is gated on.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, ensure};
use llama_ash::model::{DecodeMode, LoadOverrides, Model};
use tensor_ash::{
    BinaryOp, CopyDesc, Executor, FlashAttentionDesc, RopeDesc, RunStats, Tensor, VulkanContext,
};
use tensor_ash_test_support::fill_det;

use super::expect::Section;
use super::{Row, Verdict, check_min, key};
use crate::bench::cases::{BenchCase, SampleSummary};
use crate::bench::commands::run_case;

/// The same synthetic prompt family `llama_ash bench` uses: BOS then
/// arbitrary in-vocab ids.
fn synth_prompt(len: u32, vocab: u32) -> Result<Vec<u32>> {
    ensure!(
        vocab > 1100,
        "synthetic prompt ids need vocab > 1100, model has {vocab}"
    );
    Ok(std::iter::once(1)
        .chain((1..len).map(|i| 100 + (i % 1000)))
        .collect())
}

fn skip(thesis: &'static str, item: &str, reason: &str) -> Vec<Row> {
    vec![Row {
        thesis,
        item: item.into(),
        prediction: "-".into(),
        measured: reason.into(),
        verdict: Verdict::Skip,
    }]
}

/// Load the model in its DEFAULT configuration (prepared decode, f16
/// KV unless the environment says otherwise) for T1/T5.
pub(crate) fn load_default(
    ctx: &Arc<VulkanContext>,
    exec: &Arc<Executor>,
    path: &Path,
) -> Result<Model> {
    ensure!(
        ctx.buffer_device_address_enabled && ctx.f16_storage_enabled,
        "model theses need buffer-device-address and f16 storage support"
    );
    Model::load(ctx, exec, path, 2048)
}

/// Decode weight traffic per token in bytes: every f16 linear weight
/// is read once per token (q, k, v, o, gate, up, down per layer, plus
/// the LM head).  Norm weights, the embedding row, and KV-cache reads
/// are omitted — together well under 1% of the weight bytes.
fn weight_bytes_per_token(cfg: &llama_ash::model::Config) -> f64 {
    let (embd, ffn) = (cfg.embd as f64, cfg.ffn as f64);
    let kv_dim = (cfg.kv_heads * cfg.dh) as f64;
    let per_layer = 2.0 * embd * embd + 2.0 * embd * kv_dim + 3.0 * embd * ffn;
    2.0 * (cfg.n_layers as f64 * per_layer + embd * cfg.vocab as f64)
}

pub(crate) fn t1_decode_bandwidth(
    section: Option<&Section>,
    model: &mut Model,
) -> Result<Vec<Row>> {
    const ITEM: &str = "decode tg128 achieved weight bandwidth";
    const PP: u32 = 512;
    const TG: u32 = 128;
    let cfg = model.cfg;
    ensure!(
        cfg.t_max >= PP + TG + 17,
        "context too small for pp512+tg128"
    );
    let prompt = synth_prompt(PP, cfg.vocab)?;

    // Warmup prefill + decode burn-in, then the measured run — the
    // same discipline as `llama_ash bench`.
    let (warm, _) = model.prefill(&prompt)?;
    model.decode_many(warm, 16)?;
    model.reset()?;
    let (next, _) = model.prefill(&prompt)?;
    model.breakdown.borrow_mut().clear();
    let start = Instant::now();
    model.decode_many(next, TG)?;
    let wall_ms_per_token = start.elapsed().as_secs_f64() * 1e3 / TG as f64;

    let gpu_ns: u64 = model
        .breakdown
        .borrow()
        .iter()
        .filter(|(class, _)| matches!(*class, "graph_total" | "prepared_total"))
        .map(|(_, ns)| ns)
        .sum();
    model.reset()?;
    if gpu_ns == 0 {
        return Ok(skip(
            "T1",
            ITEM,
            "decode mode records no single-submission GPU total (perop/flash, or no timestamps)",
        ));
    }
    let gpu_ms_per_token = gpu_ns as f64 / 1e6 / TG as f64;
    let bytes = weight_bytes_per_token(&cfg);
    // bytes per token / ns per token is exactly GB/s.
    let measured_gbps = bytes / (gpu_ns as f64 / TG as f64);

    let bw = key(section, "mem_bw_gbps");
    let min_frac = key(section, "t1_min_bw_frac");
    let census = model.decode_graph_stats()?;
    let mut rows = vec![Row {
        thesis: "T1",
        item: ITEM.into(),
        prediction: match (bw, min_frac) {
            (Some(bw), Some(frac)) => {
                format!(">= {:.0} GB/s ({:.0}% of {bw:.0})", bw * frac, frac * 100.0)
            }
            _ => "unrecorded".into(),
        },
        measured: format!(
            "{measured_gbps:.0} GB/s ({:.3} GB/token / {gpu_ms_per_token:.3} ms GPU){}",
            bytes / 1e9,
            bw.map_or_else(String::new, |bw| {
                format!(" = {:.0}%", measured_gbps / bw * 100.0)
            }),
        ),
        verdict: match bw {
            Some(bw) => check_min(measured_gbps / bw, min_frac),
            None => Verdict::Info,
        },
    }];
    rows.push(Row {
        thesis: "T1",
        item: "decode step census / host overhead".into(),
        prediction: "-".into(),
        measured: format!(
            "{} dispatches, {} barriers/token; wall {wall_ms_per_token:.3} ms \
             (host {:+.3} ms)",
            census.dispatches,
            census.barriers,
            wall_ms_per_token - gpu_ms_per_token,
        ),
        verdict: Verdict::Info,
    });
    Ok(rows)
}

pub(crate) fn t5_prefill_accounting(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    section: Option<&Section>,
    model: &mut Model,
    barrier_us_measured: Option<f64>,
) -> Result<Vec<Row>> {
    const ITEM: &str = "pp512 piecewise accounting vs whole graph";
    const T: u32 = 512;
    let cfg = model.cfg;
    ensure!(cfg.t_max >= T, "context too small for pp512");
    let prompt = synth_prompt(T, cfg.vocab)?;

    // The whole: the real prefill graph's GPU-timestamped total,
    // median of 5 from-scratch runs.
    let mut whole_samples = Vec::with_capacity(5);
    for _ in 0..5 {
        model.reset()?;
        model.breakdown.borrow_mut().clear();
        model.prefill(&prompt)?;
        let ns: u64 = model
            .breakdown
            .borrow()
            .iter()
            .filter(|(class, _)| *class == "prefill_total")
            .map(|(_, ns)| ns)
            .sum();
        if ns == 0 {
            return Ok(skip("T5", ITEM, "device reports no GPU timestamps"));
        }
        whole_samples.push(ns as f64 / 1e3);
    }
    model.reset()?;
    let whole_us = SampleSummary::new(whole_samples)
        .map(|summary| summary.median)
        .unwrap_or(f64::NAN);
    let census = model.prefill_graph_stats(T)?;

    // The pieces: every op class of prefill_ops' T >= 256 branch,
    // measured standalone at the model's own shapes on synthetic
    // tensors (rates don't depend on values).
    let mut missing = false;
    let mut term = |value: Option<f64>| -> f64 {
        value.unwrap_or_else(|| {
            missing = true;
            0.0
        })
    };

    let mm_us = |m: u32, n: u32, k: u32| -> Result<Option<f64>> {
        let case = BenchCase {
            label: Cow::Owned(format!("t5 mm {m}x{n}x{k} f16w")),
            b: 1,
            m,
            n,
            k,
            b_f16: true,
        };
        Ok(run_case(ctx, exec, case, 10, 2)?
            .gpu_ms
            .map(|summary| summary.median * 1e3))
    };
    let (embd, heads, kv, dh, ffn, t_max) = (
        cfg.embd,
        cfg.heads,
        cfg.kv_heads,
        cfg.dh,
        cfg.ffn,
        cfg.t_max,
    );
    let kv_dim = kv * dh;
    let qo = term(mm_us(T, embd, embd)?);
    let kvp = term(mm_us(T, kv_dim, embd)?);
    let upgate = term(mm_us(T, ffn, embd)?);
    let down = term(mm_us(T, embd, ffn)?);
    let lm_head = term(mm_us(1, cfg.vocab, embd)?);

    fn med_gpu_us(mut op: impl FnMut() -> Result<RunStats>) -> Result<Option<f64>> {
        const ITERS: usize = 10;
        op()?;
        op()?;
        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            match op()?.gpu_time_ns {
                Some(ns) => samples.push(ns as f64 / 1e3),
                None => return Ok(None),
            }
        }
        Ok(SampleSummary::new(samples).map(|summary| summary.median))
    }

    let dev = |shape: &[u32]| Tensor::uninit_device(ctx, shape);
    let x = dev(&[T, embd])?;
    let xn = dev(&[T, embd])?;
    let attn_flat = dev(&[T, embd])?;
    let norm_w = dev(&[embd])?;
    let k_src = dev(&[T, kv_dim])?;
    let k_dst = dev(&[T, kv_dim])?;
    let table = dev(&[t_max, dh / 2, 2])?;
    let kt = Tensor::uninit_device_f16(ctx, &[kv, dh, t_max])?;
    let vc = Tensor::uninit_device_f16(ctx, &[kv, t_max, dh])?;
    let q_heads = dev(&[heads, T, dh])?;
    let attn_heads = dev(&[heads, T, dh])?;
    let up = dev(&[T, ffn])?;
    let gate = dev(&[T, ffn])?;
    let last = dev(&[1, embd])?;
    let last_n = dev(&[1, embd])?;
    for (seed, tensor) in [(21u64, &x), (22, &k_src), (23, &up), (24, &gate)] {
        let mut host = vec![0.0f32; tensor.len() as usize];
        fill_det(&mut host, seed);
        exec.upload(&host, tensor)?;
    }
    for tensor in [&norm_w, &table, &q_heads] {
        exec.upload(&vec![0.01f32; tensor.len() as usize], tensor)?;
    }
    // Zero caches: flash attention over zeros is numerically tame.
    let zeros = vec![0.0f32; (kv * dh * t_max) as usize];
    exec.upload(&zeros, &kt)?;
    exec.upload(&zeros, &vc)?;

    let rope = |heads| RopeDesc {
        heads,
        head_dim: dh,
        rot_dim: dh,
        pos_base: 0,
        ..Default::default()
    };
    let rms = term(med_gpu_us(|| {
        exec.run_rms_norm(&x, &norm_w, &xn, cfg.rms_eps)
    })?);
    let rope_q = term(med_gpu_us(|| exec.run_rope(&x, &table, &xn, rope(heads)))?);
    let rope_k = term(med_gpu_us(|| {
        exec.run_rope(&k_src, &table, &k_dst, rope(kv))
    })?);
    let copy_kt = term(med_gpu_us(|| {
        exec.run_copy_strided(
            &k_src,
            &kt,
            CopyDesc {
                extent: [dh, kv, T],
                src_strides: [1, dh, kv * dh],
                dst_strides: [t_max, dh * t_max, 1],
                ..Default::default()
            },
        )
    })?);
    let copy_v = term(med_gpu_us(|| {
        exec.run_copy_strided(
            &k_src,
            &vc,
            CopyDesc {
                extent: [dh, kv, T],
                src_strides: [1, dh, kv * dh],
                dst_strides: [1, t_max * dh, dh],
                ..Default::default()
            },
        )
    })?);
    let copy_qh = term(med_gpu_us(|| {
        exec.run_copy_strided(
            &x,
            &q_heads,
            CopyDesc {
                extent: [dh, T, heads],
                src_strides: [1, heads * dh, dh],
                dst_strides: [1, dh, T * dh],
                ..Default::default()
            },
        )
    })?);
    let copy_attn = term(med_gpu_us(|| {
        exec.run_copy_strided(
            &attn_heads,
            &attn_flat,
            CopyDesc {
                extent: [dh, T, heads],
                src_strides: [1, dh, T * dh],
                dst_strides: [1, heads * dh, dh],
                ..Default::default()
            },
        )
    })?);
    let flash = term(med_gpu_us(|| {
        exec.run_flash_attention(
            &q_heads,
            &kt,
            &vc,
            &attn_heads,
            FlashAttentionDesc {
                kv_len: T,
                pos_base: 0,
                scale: 1.0 / (dh as f32).sqrt(),
            },
        )
    })?);
    let add = term(med_gpu_us(|| {
        exec.run_binary(&x, &attn_flat, &xn, BinaryOp::AddScaled { beta: 1.0 })
    })?);
    let silu = term(med_gpu_us(|| {
        exec.run_binary(&gate, &up, &gate, BinaryOp::SiluMul)
    })?);
    let tail_copy = term(med_gpu_us(|| {
        exec.run_copy_strided(
            &x,
            &last,
            CopyDesc {
                extent: [embd, 1, 1],
                src_offset: (T - 1) * embd,
                src_strides: [1, 0, 0],
                dst_strides: [1, 0, 0],
                ..Default::default()
            },
        )
    })?);
    let tail_rms = term(med_gpu_us(|| {
        exec.run_rms_norm(&last, &norm_w, &last_n, cfg.rms_eps)
    })?);
    if missing {
        return Ok(skip("T5", ITEM, "device reports no GPU timestamps"));
    }

    // Per-layer inventory of prefill_ops' T >= 256 branch: 1 attn
    // rmsnorm + 3 qkv matmuls + 2 ropes + 2 KV appends + q permute +
    // flash + attn permute + [o matmul + residual add + ffn rmsnorm +
    // up/gate matmuls + silu-mul + down matmul + residual add].
    let layers = cfg.n_layers as f64;
    let gemm_total = (2.0 * qo + 2.0 * kvp + 2.0 * upgate + down) * layers + lm_head;
    let flash_total = flash * layers;
    let elem_per_layer =
        2.0 * rms + rope_q + rope_k + copy_kt + copy_v + copy_qh + copy_attn + 2.0 * add + silu;
    let elem_total = elem_per_layer * layers + tail_copy + tail_rms;
    let barrier_price_us = barrier_us_measured.or_else(|| key(section, "t3_barrier_us"));
    let barrier_total = barrier_price_us.map_or(0.0, |price| census.barriers as f64 * price);
    let sum_us = gemm_total + flash_total + elem_total + barrier_total;
    let gap = (sum_us - whole_us) / whole_us;

    let tolerance = key(section, "t5_rel_gap");
    let mut rows = vec![Row {
        thesis: "T5",
        item: ITEM.into(),
        prediction: tolerance.map_or_else(
            || "unrecorded".into(),
            |tolerance| format!("|sum - whole| <= {:.0}% of whole", tolerance * 100.0),
        ),
        measured: format!(
            "sum {:.2} ms vs whole {:.2} ms ({:+.1}%)",
            sum_us / 1e3,
            whole_us / 1e3,
            gap * 100.0
        ),
        verdict: match tolerance {
            Some(tolerance) if gap.abs() <= tolerance => Verdict::Pass,
            Some(_) => Verdict::Fail,
            None => Verdict::Info,
        },
    }];
    let percent = |value: f64| value / whole_us * 100.0;
    rows.push(Row {
        thesis: "T5",
        item: "  terms (GEMM / flash / elementwise / barriers)".into(),
        prediction: "-".into(),
        measured: format!(
            "{:.2} / {:.2} / {:.2} / {:.2} ms ({:.0}/{:.0}/{:.0}/{:.0}% of whole); \
             census {} dispatches, {} barriers{}",
            gemm_total / 1e3,
            flash_total / 1e3,
            elem_total / 1e3,
            barrier_total / 1e3,
            percent(gemm_total),
            percent(flash_total),
            percent(elem_total),
            percent(barrier_total),
            census.dispatches,
            census.barriers,
            if barrier_price_us.is_none() {
                " (barrier price unknown: term omitted)"
            } else {
                ""
            },
        ),
        verdict: Verdict::Info,
    });
    Ok(rows)
}

pub(crate) fn t7_token_exactness(
    ctx: &Arc<VulkanContext>,
    exec: &Arc<Executor>,
    path: &Path,
) -> Result<Vec<Row>> {
    const TG: u32 = 16;
    const LENS: [u32; 4] = [1, 64, 300, 512];
    // 512 prompt + 17 generated fits comfortably; a small context
    // keeps the six reloads' cache zeroing cheap.
    const CTX_LEN: u32 = 640;
    let modes = [
        (DecodeMode::Prepared, "prepared"),
        (DecodeMode::Graph, "graph"),
        (DecodeMode::PerOp, "perop"),
    ];
    let kvs = [(false, "kv16"), (true, "kv32")];

    let mut runs: Vec<(String, Vec<Vec<u32>>)> = Vec::new();
    for (mode, mode_name) in modes {
        for (kv_f32, kv_name) in kvs {
            let label = format!("{mode_name}/{kv_name}");
            log::info!("T7: loading model for {label}");
            let mut model = Model::load_with(
                ctx,
                exec,
                path,
                CTX_LEN,
                LoadOverrides {
                    decode_mode: Some(mode),
                    kv_f32: Some(kv_f32),
                },
            )?;
            let mut sequences = Vec::with_capacity(LENS.len());
            for &len in &LENS {
                model.reset()?;
                let prompt = synth_prompt(len, model.cfg.vocab)?;
                let (first, _) = model.prefill(&prompt)?;
                let mut sequence = vec![first];
                sequence.extend(model.decode_many(first, TG)?);
                sequences.push(sequence);
            }
            runs.push((label, sequences));
        }
    }

    let mut rows = Vec::new();
    for (index, len) in LENS.into_iter().enumerate() {
        let (reference_label, reference) = (&runs[0].0, &runs[0].1[index]);
        let mismatches: Vec<String> = runs[1..]
            .iter()
            .filter_map(|(label, sequences)| {
                let sequence = &sequences[index];
                if sequence == reference {
                    return None;
                }
                let at = reference
                    .iter()
                    .zip(sequence)
                    .position(|(a, b)| a != b)
                    .unwrap_or(reference.len().min(sequence.len()));
                Some(format!(
                    "{label} diverges from {reference_label} at token {at}"
                ))
            })
            .collect();
        rows.push(Row {
            thesis: "T7",
            item: format!("token exactness, prefill T={len} + {TG} decode"),
            prediction: format!("identical greedy tokens across {} configs", runs.len()),
            measured: if mismatches.is_empty() {
                format!(
                    "{} configs x {} tokens identical",
                    runs.len(),
                    reference.len()
                )
            } else {
                mismatches.join("; ")
            },
            verdict: if mismatches.is_empty() {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
        });
    }
    Ok(rows)
}
