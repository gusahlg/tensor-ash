//! Fused flash-attention correctness: against an f64 CPU reference,
//! against the composed path on shared inputs, GQA, warm-cache
//! offsets, ragged sizes, and masking robustness against garbage in
//! the cache tail.

use crate::common::*;

use tensor_ash::{FlashAttentionDesc, MatmulCall, SoftmaxMask, Tensor};

/// f64 reference: causal attention where query row `i` attends to
/// positions `< min(kv_len, pos_base + i + 1)`.
#[allow(clippy::too_many_arguments)]
fn cpu_flash(
    q: &[f32],
    kt: &[f32],
    v: &[f32],
    heads: usize,
    kv_heads: usize,
    t_q: usize,
    t_max: usize,
    dh: usize,
    kv_len: usize,
    pos_base: usize,
    scale: f64,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; heads * t_q * dh];
    let group = heads / kv_heads;
    for head in 0..heads {
        let kv_head = head / group;
        for row in 0..t_q {
            let valid = kv_len.min(pos_base + row + 1);
            if valid == 0 {
                continue;
            }
            let mut scores = vec![0.0_f64; valid];
            let mut row_max = f64::NEG_INFINITY;
            for pos in 0..valid {
                let mut dot = 0.0_f64;
                for d in 0..dh {
                    dot += q[(head * t_q + row) * dh + d] as f64
                        * kt[(kv_head * dh + d) * t_max + pos] as f64;
                }
                scores[pos] = dot * scale;
                row_max = row_max.max(scores[pos]);
            }
            let mut sum = 0.0_f64;
            for value in &mut scores {
                *value = (*value - row_max).exp();
                sum += *value;
            }
            for d in 0..dh {
                let mut acc = 0.0_f64;
                for pos in 0..valid {
                    acc += scores[pos] / sum * v[(kv_head * t_max + pos) * dh + d] as f64;
                }
                out[(head * t_q + row) * dh + d] = acc as f32;
            }
        }
    }
    out
}

struct FlashCase {
    heads: u32,
    kv_heads: u32,
    t_q: u32,
    t_max: u32,
    dh: u32,
    kv_len: u32,
    pos_base: u32,
}

fn run_case(label: &str, case: &FlashCase, poison_tail: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let FlashCase {
        heads,
        kv_heads,
        t_q,
        t_max,
        dh,
        kv_len,
        pos_base,
    } = *case;
    let (q, host_q) = upload_det(&ctx, &exec, &[heads, t_q, dh], 5100);
    let kt = Tensor::uninit_device(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[kv_heads, t_max, dh]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();

    // Fill valid cache positions deterministically; optionally poison
    // everything past kv_len with huge values that would blow up the
    // result if masking ever leaked.
    let mut host_kt = vec![0.0_f32; (kv_heads * dh * t_max) as usize];
    let mut host_v = vec![0.0_f32; (kv_heads * t_max * dh) as usize];
    let mut fill = vec![0.0_f32; host_kt.len().max(host_v.len())];
    fill_det(&mut fill, 5101);
    for kv_head in 0..kv_heads as usize {
        for d in 0..dh as usize {
            for pos in 0..t_max as usize {
                let kt_index = (kv_head * dh as usize + d) * t_max as usize + pos;
                let v_index = (kv_head * t_max as usize + pos) * dh as usize + d;
                if pos < kv_len as usize {
                    host_kt[kt_index] = fill[kt_index % fill.len()];
                    host_v[v_index] = fill[(v_index * 7 + 3) % fill.len()];
                } else if poison_tail {
                    host_kt[kt_index] = 1.0e30;
                    host_v[v_index] = 1.0e30;
                }
            }
        }
    }
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();

    let scale = 1.0 / (dh as f32).sqrt();
    exec.run_flash_attention(
        &q,
        &kt,
        &v,
        &out,
        FlashAttentionDesc {
            kv_len,
            pos_base,
            scale,
        },
    )
    .unwrap();
    let mut gpu = vec![0.0; (heads * t_q * dh) as usize];
    exec.download(&out, &mut gpu).unwrap();

    let cpu = cpu_flash(
        &host_q,
        &host_kt,
        &host_v,
        heads as usize,
        kv_heads as usize,
        t_q as usize,
        t_max as usize,
        dh as usize,
        kv_len as usize,
        pos_base as usize,
        scale as f64,
    );
    // Softmax weights sum to 1, so error scales with the number of
    // accumulated terms far more gently than a GEMM; 1e-4 is roomy for
    // f32 online softmax against f64 and still catches masking or
    // rescale bugs outright.
    assert_close_tol(&gpu, &cpu, 1e-4, label);
}

#[test]
#[ignore]
fn flash_matches_cpu_across_geometries() {
    // (label, H, H_kv, T_q, T_max, dh, kv_len, pos_base, poison)
    run_case(
        "fresh prefill dh64",
        &FlashCase {
            heads: 4,
            kv_heads: 4,
            t_q: 197,
            t_max: 256,
            dh: 64,
            kv_len: 197,
            pos_base: 0,
        },
        true,
    );
    run_case(
        "fresh prefill dh128 ragged",
        &FlashCase {
            heads: 3,
            kv_heads: 3,
            t_q: 129,
            t_max: 161,
            dh: 128,
            kv_len: 129,
            pos_base: 0,
        },
        true,
    );
    run_case(
        "warm cache offset",
        &FlashCase {
            heads: 4,
            kv_heads: 4,
            t_q: 64,
            t_max: 512,
            dh: 64,
            kv_len: 300 + 64,
            pos_base: 300,
        },
        true,
    );
    run_case(
        "gqa group of 4",
        &FlashCase {
            heads: 8,
            kv_heads: 2,
            t_q: 100,
            t_max: 128,
            dh: 64,
            kv_len: 100,
            pos_base: 0,
        },
        true,
    );
    run_case(
        "single-row decode style",
        &FlashCase {
            heads: 4,
            kv_heads: 4,
            t_q: 1,
            t_max: 96,
            dh: 128,
            kv_len: 77,
            pos_base: 76,
        },
        true,
    );
}

#[test]
#[ignore]
fn flash_matches_composed_path() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (heads, t, dh) = (8_u32, 256_u32, 64_u32);
    let (q, _) = upload_det(&ctx, &exec, &[heads, t, dh], 5200);
    let (kt, _) = upload_det(&ctx, &exec, &[heads, dh, t], 5201);
    let (v, _) = upload_det(&ctx, &exec, &[heads, t, dh], 5202);
    let scale = 1.0 / (dh as f32).sqrt();

    let fused = Tensor::uninit_device(&ctx, &[heads, t, dh]).unwrap();
    exec.run_flash_attention(
        &q,
        &kt,
        &v,
        &fused,
        FlashAttentionDesc {
            kv_len: t,
            pos_base: 0,
            scale,
        },
    )
    .unwrap();

    let scores = Tensor::uninit_device(&ctx, &[heads, t, t]).unwrap();
    let composed = Tensor::uninit_device(&ctx, &[heads, t, dh]).unwrap();
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
        SoftmaxMask::Causal {
            prefix: 0,
            rows_per_group: t,
        },
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

    let mut a = vec![0.0; (heads * t * dh) as usize];
    let mut b = vec![0.0; a.len()];
    exec.download(&fused, &mut a).unwrap();
    exec.download(&composed, &mut b).unwrap();
    assert_close_tol(&a, &b, 1e-5, "fused vs composed");
}

#[test]
#[ignore]
fn flash_rejects_bad_geometry() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let q = Tensor::uninit_device(&ctx, &[4, 32, 96]).unwrap();
    let kt = Tensor::uninit_device(&ctx, &[4, 96, 64]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[4, 64, 96]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[4, 32, 96]).unwrap();
    let desc = FlashAttentionDesc {
        kv_len: 32,
        pos_base: 0,
        scale: 1.0,
    };
    let err = exec
        .run_flash_attention(&q, &kt, &v, &out, desc)
        .unwrap_err()
        .to_string();
    assert!(err.contains("head dimension 96"), "unexpected: {err}");

    let q = Tensor::uninit_device(&ctx, &[4, 32, 64]).unwrap();
    let kt = Tensor::uninit_device(&ctx, &[4, 64, 64]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[4, 64, 64]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[4, 32, 64]).unwrap();
    let err = exec
        .run_flash_attention(
            &q,
            &kt,
            &v,
            &out,
            FlashAttentionDesc {
                kv_len: 65,
                pos_base: 0,
                scale: 1.0,
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds cache T_max"), "unexpected: {err}");
}
