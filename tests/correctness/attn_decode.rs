//! Fused split-K decode-attention correctness: against an f64 CPU
//! reference (prefix-masked softmax attention, one query row per
//! head), for f32 AND f16 caches, with poison past `kv_len` to prove
//! the op never reads beyond the valid prefix; agreement with the
//! composed scores/softmax/PV path on shared inputs; the graph-op
//! form; and geometry rejection.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{AttnDecodeDesc, ExecOp, MatmulCall, SoftmaxMask, Tensor};

/// f64 reference: each of the `group` query rows of each kv head
/// attends to positions `< kv_len` (decode prefix mask).
#[allow(clippy::too_many_arguments)]
fn cpu_attn_decode(
    q: &[f32],
    kt: &[f32],
    v: &[f32],
    kv_heads: usize,
    group: usize,
    t_max: usize,
    dh: usize,
    kv_len: usize,
    scale: f64,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; kv_heads * group * dh];
    for kv_head in 0..kv_heads {
        for row in 0..group {
            let q_base = (kv_head * group + row) * dh;
            let mut scores = vec![0.0_f64; kv_len];
            let mut row_max = f64::NEG_INFINITY;
            for (pos, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0_f64;
                for d in 0..dh {
                    dot += q[q_base + d] as f64 * kt[(kv_head * dh + d) * t_max + pos] as f64;
                }
                *score = dot * scale;
                row_max = row_max.max(*score);
            }
            let mut sum = 0.0_f64;
            for score in &mut scores {
                *score = (*score - row_max).exp();
                sum += *score;
            }
            for d in 0..dh {
                let mut acc = 0.0_f64;
                for (pos, score) in scores.iter().enumerate() {
                    acc += score / sum * v[(kv_head * t_max + pos) * dh + d] as f64;
                }
                out[q_base + d] = acc as f32;
            }
        }
    }
    out
}

const KV_HEADS: u32 = 4;
const GROUP: u32 = 8;
const T_MAX: u32 = 2048;
const DH: u32 = 64;

/// Deterministic caches with only the `kv_len` prefix valid; the tail
/// is poisoned with 1e30 (f16 storage rounds that to +inf), so any
/// read past the prefix blows up the result.  Returns the host copies
/// the f64 reference should see (f16: rounded through f16).
fn make_caches(kv_len: u32, f16: bool) -> (Vec<f32>, Vec<f32>) {
    let mut host_kt = vec![0.0_f32; (KV_HEADS * DH * T_MAX) as usize];
    let mut host_v = vec![0.0_f32; (KV_HEADS * T_MAX * DH) as usize];
    let mut fill = vec![0.0_f32; host_kt.len()];
    fill_det(&mut fill, 7301);
    for kv_head in 0..KV_HEADS as usize {
        for d in 0..DH as usize {
            for pos in 0..T_MAX as usize {
                let kt_index = (kv_head * DH as usize + d) * T_MAX as usize + pos;
                let v_index = (kv_head * T_MAX as usize + pos) * DH as usize + d;
                if pos < kv_len as usize {
                    host_kt[kt_index] = fill[kt_index % fill.len()];
                    host_v[v_index] = fill[(v_index * 7 + 3) % fill.len()];
                    if f16 {
                        host_kt[kt_index] = round_f32_via_f16(host_kt[kt_index]);
                        host_v[v_index] = round_f32_via_f16(host_v[v_index]);
                    }
                } else {
                    host_kt[kt_index] = 1.0e30;
                    host_v[v_index] = 1.0e30;
                }
            }
        }
    }
    (host_kt, host_v)
}

fn run_case(kv_len: u32, f16_kv: bool, via_graph: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    if f16_kv && !ctx.f16_storage_enabled {
        eprintln!("skipping: no f16 storage");
        return;
    }
    let label = format!("attn_decode kv_len={kv_len} f16={f16_kv} graph={via_graph}");
    let (q, host_q) = upload_det(&ctx, &exec, &[KV_HEADS, GROUP, DH], 7300);
    let (kt, v) = if f16_kv {
        (
            Tensor::uninit_device_f16(&ctx, &[KV_HEADS, DH, T_MAX]).unwrap(),
            Tensor::uninit_device_f16(&ctx, &[KV_HEADS, T_MAX, DH]).unwrap(),
        )
    } else {
        (
            Tensor::uninit_device(&ctx, &[KV_HEADS, DH, T_MAX]).unwrap(),
            Tensor::uninit_device(&ctx, &[KV_HEADS, T_MAX, DH]).unwrap(),
        )
    };
    let (host_kt, host_v) = make_caches(kv_len, f16_kv);
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();
    let scratch = Tensor::uninit_device(
        &ctx,
        &[KV_HEADS * tensor_ash::ATTN_DECODE_MAX_CHUNKS * GROUP * (DH + 2)],
    )
    .unwrap();
    let out = Tensor::uninit_device(&ctx, &[KV_HEADS, GROUP, DH]).unwrap();

    let scale = 1.0 / (DH as f32).sqrt();
    let desc = AttnDecodeDesc {
        kv_len,
        scale,
        ..Default::default()
    };
    if via_graph {
        exec.run_exec_ops(&[ExecOp::AttnDecode {
            q: &q,
            kt: &kt,
            v: &v,
            scratch: &scratch,
            out: &out,
            desc,
        }])
        .unwrap();
    } else {
        exec.run_attn_decode(&q, &kt, &v, &scratch, &out, desc)
            .unwrap();
    }
    let mut gpu = vec![0.0; (KV_HEADS * GROUP * DH) as usize];
    exec.download(&out, &mut gpu).unwrap();

    let cpu = cpu_attn_decode(
        &host_q,
        &host_kt,
        &host_v,
        KV_HEADS as usize,
        GROUP as usize,
        T_MAX as usize,
        DH as usize,
        kv_len as usize,
        scale as f64,
    );
    // Softmax weights sum to 1, so error grows gently with kv_len;
    // 1e-4 (like flash.rs) catches masking, chunk-merge, or rescale
    // bugs outright.  f16 caches only round the INPUTS (the reference
    // sees the rounded values), so the same tolerance holds.
    assert_close_tol(&gpu, &cpu, 1e-4, &label);
}

#[test]
#[ignore]
fn attn_decode_matches_cpu_f32_caches() {
    // kv_len sweep: single position (7 of 8 chunks empty), non-chunk-
    // aligned, mid-size, and the full T_max cache.
    for kv_len in [1, 33, 640, 2048] {
        run_case(kv_len, false, false);
    }
}

#[test]
#[ignore]
fn attn_decode_matches_cpu_f16_caches() {
    for kv_len in [1, 33, 640, 2048] {
        run_case(kv_len, true, false);
    }
}

#[test]
#[ignore]
fn attn_decode_graph_op_matches_cpu() {
    run_case(640, false, true);
    run_case(640, true, true);
}

#[test]
#[ignore]
fn attn_decode_matches_composed_path() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let kv_len = 517_u32;
    let (q, _) = upload_det(&ctx, &exec, &[KV_HEADS, GROUP, DH], 7400);
    let kt = Tensor::uninit_device(&ctx, &[KV_HEADS, DH, T_MAX]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[KV_HEADS, T_MAX, DH]).unwrap();
    // Zero tail (not poison): the composed path multiplies the padded
    // extent by the softmax's exact zeros, which is only finite for a
    // finite tail.
    let mut host_kt = vec![0.0_f32; (KV_HEADS * DH * T_MAX) as usize];
    let mut host_v = vec![0.0_f32; (KV_HEADS * T_MAX * DH) as usize];
    let mut fill = vec![0.0_f32; host_kt.len()];
    fill_det(&mut fill, 7401);
    for kv_head in 0..KV_HEADS as usize {
        for d in 0..DH as usize {
            for pos in 0..kv_len as usize {
                let kt_index = (kv_head * DH as usize + d) * T_MAX as usize + pos;
                let v_index = (kv_head * T_MAX as usize + pos) * DH as usize + d;
                host_kt[kt_index] = fill[kt_index % fill.len()];
                host_v[v_index] = fill[(v_index * 7 + 3) % fill.len()];
            }
        }
    }
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();
    let scale = 1.0 / (DH as f32).sqrt();

    let scratch = Tensor::uninit_device(
        &ctx,
        &[KV_HEADS * tensor_ash::ATTN_DECODE_MAX_CHUNKS * GROUP * (DH + 2)],
    )
    .unwrap();
    let fused = Tensor::uninit_device(&ctx, &[KV_HEADS, GROUP, DH]).unwrap();
    exec.run_attn_decode(
        &q,
        &kt,
        &v,
        &scratch,
        &fused,
        AttnDecodeDesc {
            kv_len,
            scale,
            ..Default::default()
        },
    )
    .unwrap();

    let scores = Tensor::uninit_device(&ctx, &[KV_HEADS, GROUP, T_MAX]).unwrap();
    let composed = Tensor::uninit_device(&ctx, &[KV_HEADS, GROUP, DH]).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &q,
        b: &kt,
        c: &scores,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    exec.run_softmax_rows(
        &scores,
        &scores,
        scale,
        SoftmaxMask::Prefix { valid: kv_len },
    )
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &scores,
        b: &v,
        c: &composed,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();

    let mut a = vec![0.0; (KV_HEADS * GROUP * DH) as usize];
    let mut b = vec![0.0; a.len()];
    exec.download(&fused, &mut a).unwrap();
    exec.download(&composed, &mut b).unwrap();
    assert_close_tol(&a, &b, 1e-5, "fused vs composed decode attention");
}

#[test]
#[ignore]
fn attn_decode_rejects_bad_geometry() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let scratch = Tensor::uninit_device(&ctx, &[4 * 32 * 8 * 130]).unwrap();
    let desc = AttnDecodeDesc {
        kv_len: 16,
        scale: 1.0,
        ..Default::default()
    };

    // dh 128 has no compiled stage-1 variant.
    let q = Tensor::uninit_device(&ctx, &[4, 8, 128]).unwrap();
    let kt = Tensor::uninit_device(&ctx, &[4, 128, 64]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[4, 64, 128]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[4, 8, 128]).unwrap();
    let err = exec
        .run_attn_decode(&q, &kt, &v, &scratch, &out, desc)
        .unwrap_err()
        .to_string();
    assert!(err.contains("head dimension 128"), "unexpected: {err}");

    // kv_len beyond the cache.
    let q = Tensor::uninit_device(&ctx, &[4, 8, 64]).unwrap();
    let kt = Tensor::uninit_device(&ctx, &[4, 64, 64]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[4, 64, 64]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[4, 8, 64]).unwrap();
    let err = exec
        .run_attn_decode(
            &q,
            &kt,
            &v,
            &scratch,
            &out,
            AttnDecodeDesc {
                kv_len: 65,
                scale: 1.0,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of range"), "unexpected: {err}");

    // Undersized scratch.
    let small = Tensor::uninit_device(&ctx, &[16]).unwrap();
    let err = exec
        .run_attn_decode(&q, &kt, &v, &small, &out, desc)
        .unwrap_err()
        .to_string();
    assert!(err.contains("scratch length"), "unexpected: {err}");
}
