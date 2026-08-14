//! End-to-end llama-style decoder layer composed purely from library
//! ops: RMSNorm -> QKV (f16 weights) -> RoPE -> KV-cache append ->
//! masked attention -> output projection with fused residual ->
//! RMSNorm -> SwiGLU MLP with fused gating and residual.
//!
//! This is the library's "can a real local model run on this?" proof:
//! one decode step over a zero-padded KV cache, validated against an
//! f64 CPU reference that mirrors the f16 weight rounding.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, CopyDesc, Epilogue, EpilogueBinary, MatmulCall, MatmulOp, RopeDesc, SoftmaxMask,
    Tensor,
};

const HIDDEN: usize = 256;
const HEADS: usize = 4;
const DH: usize = HIDDEN / HEADS;
const FFN: usize = 512;
const T_MAX: usize = 32;
const KV_LEN: usize = 11; // cached positions before this step

struct Weights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    norm_attn: Vec<f32>,
    norm_mlp: Vec<f32>,
    table: Vec<f32>,
}

fn f16_weights(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut host = vec![0.0; rows * cols];
    fill_det(&mut host, seed);
    for value in &mut host {
        // Keep magnitudes model-like and round through f16 exactly as
        // the upload path will.
        *value = round_f32_via_f16(*value * 0.125);
    }
    host
}

fn make_weights() -> Weights {
    let half = DH / 2;
    let mut table = vec![0.0_f32; T_MAX * half * 2];
    for pos in 0..T_MAX {
        for pair in 0..half {
            let theta = pos as f64 / 10000_f64.powf(2.0 * pair as f64 / DH as f64);
            table[(pos * half + pair) * 2] = theta.cos() as f32;
            table[(pos * half + pair) * 2 + 1] = theta.sin() as f32;
        }
    }
    let mut norm_attn = vec![0.0; HIDDEN];
    let mut norm_mlp = vec![0.0; HIDDEN];
    fill_det(&mut norm_attn, 3108);
    fill_det(&mut norm_mlp, 3109);
    for value in norm_attn.iter_mut().chain(norm_mlp.iter_mut()) {
        *value = 1.0 + 0.1 * *value;
    }
    Weights {
        wq: f16_weights(3101, HIDDEN, HIDDEN),
        wk: f16_weights(3102, HIDDEN, HIDDEN),
        wv: f16_weights(3103, HIDDEN, HIDDEN),
        wo: f16_weights(3104, HIDDEN, HIDDEN),
        w_gate: f16_weights(3105, HIDDEN, FFN),
        w_up: f16_weights(3106, HIDDEN, FFN),
        w_down: f16_weights(3107, FFN, HIDDEN),
        norm_attn,
        norm_mlp,
        table,
    }
}

fn cpu_rmsnorm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean_sq + 1e-5).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| ((v as f64 * inv) as f32) * w)
        .collect()
}

fn cpu_matvec(x: &[f32], w: &[f32], cols: usize) -> Vec<f32> {
    let rows = x.len();
    (0..cols)
        .map(|col| {
            (0..rows)
                .map(|row| x[row] as f64 * w[row * cols + col] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

fn cpu_rope_inplace(x: &mut [f32], table: &[f32], pos: usize) {
    let half = DH / 2;
    for head in 0..HEADS {
        for pair in 0..half {
            let c = table[(pos * half + pair) * 2];
            let s = table[(pos * half + pair) * 2 + 1];
            let i0 = head * DH + pair * 2;
            let (x0, x1) = (x[i0], x[i0 + 1]);
            x[i0] = x0 * c - x1 * s;
            x[i0 + 1] = x0 * s + x1 * c;
        }
    }
}

#[test]
#[ignore]
fn llama_decoder_layer_decode_step_matches_cpu() {
    let (ctx, exec) = make_setup(2, 16);
    if !ctx.buffer_device_address_enabled || !ctx.f16_storage_enabled {
        eprintln!("skipping: needs BDA + f16");
        return;
    }
    let weights = make_weights();

    // Device weights (f16) and constants (f32).
    let dev_f16 = |host: &[f32], shape: &[u32]| {
        let tensor = Tensor::uninit_device_f16(&ctx, shape).unwrap();
        exec.upload(host, &tensor).unwrap();
        tensor
    };
    let dev_f32 = |host: &[f32], shape: &[u32]| {
        let tensor = Tensor::uninit_device(&ctx, shape).unwrap();
        exec.upload(host, &tensor).unwrap();
        tensor
    };
    let (h, f) = (HIDDEN as u32, FFN as u32);
    let wq = dev_f16(&weights.wq, &[h, h]);
    let wk = dev_f16(&weights.wk, &[h, h]);
    let wv = dev_f16(&weights.wv, &[h, h]);
    let wo = dev_f16(&weights.wo, &[h, h]);
    let w_gate = dev_f16(&weights.w_gate, &[h, f]);
    let w_up = dev_f16(&weights.w_up, &[h, f]);
    let w_down = dev_f16(&weights.w_down, &[f, h]);
    let norm_attn = dev_f32(&weights.norm_attn, &[h]);
    let norm_mlp = dev_f32(&weights.norm_mlp, &[h]);
    let table = dev_f32(&weights.table, &[T_MAX as u32, (DH / 2) as u32, 2]);

    // Pre-filled KV caches: Kt [H, dh, T_max], V [H, T_max, dh],
    // valid prefix KV_LEN, zero-padded tail.
    let (heads, dh, t_max) = (HEADS as u32, DH as u32, T_MAX as u32);
    let mut host_kt = vec![0.0; HEADS * DH * T_MAX];
    let mut host_v = vec![0.0; HEADS * T_MAX * DH];
    let mut cache_fill = vec![0.0; HEADS * T_MAX * DH];
    fill_det(&mut cache_fill, 3200);
    for head in 0..HEADS {
        for step in 0..KV_LEN {
            for lane in 0..DH {
                let value = cache_fill[(head * T_MAX + step) * DH + lane] * 0.25;
                host_kt[(head * DH + lane) * T_MAX + step] = value;
                host_v[(head * T_MAX + step) * DH + lane] =
                    cache_fill[(head * T_MAX + (T_MAX - 1 - step)) * DH + lane] * 0.25;
            }
        }
    }
    let kt_cache = dev_f32(&host_kt, &[heads, dh, t_max]);
    let v_cache = dev_f32(&host_v, &[heads, t_max, dh]);

    // The incoming activation.
    let mut host_x = vec![0.0; HIDDEN];
    fill_det(&mut host_x, 3201);
    let x = dev_f32(&host_x, &[1, h]);

    // ---- GPU decoder layer ----
    let xn = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    exec.run_rms_norm(&x, &norm_attn, &xn, 1e-5).unwrap();
    let q = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    let k = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    let v_new = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    for (weight, out) in [(&wq, &q), (&wk, &k), (&wv, &v_new)] {
        exec.run_matmuls(&[MatmulCall {
            a: &xn,
            b: weight,
            c: out,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap();
    }
    let rope = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim: dh,
        pos_base: KV_LEN as u32,
        ..Default::default()
    };
    exec.run_rope(&q, &table, &q, rope).unwrap();
    exec.run_rope(&k, &table, &k, rope).unwrap();
    // Append K as column KV_LEN of Kt, V as row KV_LEN of V.
    exec.run_copy_strided(
        &k,
        &kt_cache,
        CopyDesc {
            extent: [dh, heads, 1],
            src_offset: 0,
            src_strides: [1, dh, 0],
            dst_offset: KV_LEN as u32,
            dst_strides: [t_max, dh * t_max, 0],
            ..Default::default()
        },
    )
    .unwrap();
    exec.run_copy_strided(
        &v_new,
        &v_cache,
        CopyDesc {
            extent: [dh, heads, 1],
            src_offset: 0,
            src_strides: [1, dh, 0],
            dst_offset: (KV_LEN as u32) * dh,
            dst_strides: [1, t_max * dh, 0],
            ..Default::default()
        },
    )
    .unwrap();
    // Attention over the zero-padded caches.
    let scores = Tensor::uninit_device(&ctx, &[heads, 1, t_max]).unwrap();
    let attn = Tensor::uninit_device(&ctx, &[heads, 1, dh]).unwrap();
    let q_heads = Tensor::uninit_device(&ctx, &[heads, 1, dh]).unwrap();
    // Reshape via straight copy (same memory order).
    exec.run_copy_strided(
        &q,
        &q_heads,
        CopyDesc {
            extent: [h, 1, 1],
            src_offset: 0,
            src_strides: [1, 0, 0],
            dst_offset: 0,
            dst_strides: [1, 0, 0],
            ..Default::default()
        },
    )
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &q_heads,
        b: &kt_cache,
        c: &scores,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    exec.run_softmax_rows(
        &scores,
        &scores,
        1.0 / (DH as f32).sqrt(),
        SoftmaxMask::Prefix {
            valid: (KV_LEN + 1) as u32,
        },
    )
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &scores,
        b: &v_cache,
        c: &attn,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    // Output projection with fused residual, then the SwiGLU MLP.
    let attn_flat = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    exec.run_copy_strided(
        &attn,
        &attn_flat,
        CopyDesc {
            extent: [h, 1, 1],
            src_offset: 0,
            src_strides: [1, 0, 0],
            dst_offset: 0,
            dst_strides: [1, 0, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let o = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &attn_flat,
            b: &wo,
            c: &o,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: None,
            activation: Activation::None,
            binary: EpilogueBinary::AddScaled { d: &x, beta: 1.0 },
        },
    )])
    .unwrap();
    let on = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    exec.run_rms_norm(&o, &norm_mlp, &on, 1e-5).unwrap();
    let up = Tensor::uninit_device(&ctx, &[1, f]).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &on,
        b: &w_up,
        c: &up,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let gated = Tensor::uninit_device(&ctx, &[1, f]).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &on,
            b: &w_gate,
            c: &gated,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: None,
            activation: Activation::Silu,
            binary: EpilogueBinary::Mul { d: &up },
        },
    )])
    .unwrap();
    let out = Tensor::uninit_device(&ctx, &[1, h]).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &gated,
            b: &w_down,
            c: &out,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: None,
            activation: Activation::None,
            binary: EpilogueBinary::AddScaled { d: &o, beta: 1.0 },
        },
    )])
    .unwrap();
    let mut gpu = vec![0.0; HIDDEN];
    exec.download(&out, &mut gpu).unwrap();

    // ---- CPU reference (f64 accumulation, same f16-rounded weights) ----
    let xn_ref = cpu_rmsnorm(&host_x, &weights.norm_attn);
    let mut q_ref = cpu_matvec(&xn_ref, &weights.wq, HIDDEN);
    let mut k_ref = cpu_matvec(&xn_ref, &weights.wk, HIDDEN);
    let v_ref = cpu_matvec(&xn_ref, &weights.wv, HIDDEN);
    cpu_rope_inplace(&mut q_ref, &weights.table, KV_LEN);
    cpu_rope_inplace(&mut k_ref, &weights.table, KV_LEN);
    // Append to the CPU caches.
    for head in 0..HEADS {
        for lane in 0..DH {
            host_kt[(head * DH + lane) * T_MAX + KV_LEN] = k_ref[head * DH + lane];
            host_v[(head * T_MAX + KV_LEN) * DH + lane] = v_ref[head * DH + lane];
        }
    }
    let mut attn_ref = vec![0.0_f32; HIDDEN];
    let scale = 1.0 / (DH as f64).sqrt();
    for head in 0..HEADS {
        let valid = KV_LEN + 1;
        let mut probs = vec![0.0_f64; valid];
        let mut row_max = f64::NEG_INFINITY;
        for step in 0..valid {
            let mut dot = 0.0_f64;
            for lane in 0..DH {
                dot += q_ref[head * DH + lane] as f64
                    * host_kt[(head * DH + lane) * T_MAX + step] as f64;
            }
            probs[step] = dot * scale;
            row_max = row_max.max(probs[step]);
        }
        let mut sum = 0.0_f64;
        for value in &mut probs {
            *value = (*value - row_max).exp();
            sum += *value;
        }
        for lane in 0..DH {
            let mut acc = 0.0_f64;
            for step in 0..valid {
                acc += probs[step] / sum * host_v[(head * T_MAX + step) * DH + lane] as f64;
            }
            attn_ref[head * DH + lane] = acc as f32;
        }
    }
    let o_proj = cpu_matvec(&attn_ref, &weights.wo, HIDDEN);
    let o_ref: Vec<f32> = o_proj
        .iter()
        .zip(&host_x)
        .map(|(&value, &residual)| value + residual)
        .collect();
    let on_ref = cpu_rmsnorm(&o_ref, &weights.norm_mlp);
    let up_ref = cpu_matvec(&on_ref, &weights.w_up, FFN);
    let gate_ref = cpu_matvec(&on_ref, &weights.w_gate, FFN);
    let gated_ref: Vec<f32> = gate_ref
        .iter()
        .zip(&up_ref)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect();
    let down_ref = cpu_matvec(&gated_ref, &weights.w_down, HIDDEN);
    let out_ref: Vec<f32> = down_ref
        .iter()
        .zip(&o_ref)
        .map(|(&value, &residual)| value + residual)
        .collect();

    let (error, index) = max_abs_err(&gpu, &out_ref);
    // Seven chained matmuls + two norms accumulate several rounds of
    // f32 summation error; 8x the single-matmul tolerance is generous
    // yet still catches any real defect (observed error is ~1e-5).
    assert!(
        error <= 8.0 * tolerance(HIDDEN as u32),
        "decoder layer error {error:.3e} at {index} (gpu {} vs cpu {})",
        gpu[index],
        out_ref[index]
    );
}
