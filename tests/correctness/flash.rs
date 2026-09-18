//! Fused flash-attention correctness: against an f64 CPU reference,
//! against the composed path on shared inputs, GQA, warm-cache
//! offsets, ragged sizes, and masking robustness against garbage in
//! the cache tail.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{ExecOp, FlashAttentionDesc, MatmulCall, SoftmaxMask, Tensor};

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

/// Deterministic K/V cache fill for `case`: valid positions from
/// `fill_det`, everything past `kv_len` optionally poisoned with
/// values that would blow up the result if masking ever leaked.
fn fill_kv(case: &FlashCase, poison: Option<f32>) -> (Vec<f32>, Vec<f32>) {
    let FlashCase {
        kv_heads,
        t_max,
        dh,
        kv_len,
        ..
    } = *case;
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
                } else if let Some(poison) = poison {
                    host_kt[kt_index] = poison;
                    host_v[v_index] = poison;
                }
            }
        }
    }
    (host_kt, host_v)
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

    let (host_kt, host_v) = fill_kv(case, poison_tail.then_some(1.0e30));
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();

    let scale = 1.0 / (dh as f32).sqrt();
    // Pinned to the SIMT kernels: this test asserts f32-path
    // tolerances; the coopmat2 route gets its own A/B cases below.
    exec.run_flash_attention_simt(
        &q,
        &kt,
        &v,
        &out,
        FlashAttentionDesc {
            kv_len,
            pos_base,
            scale,
            token_major_heads: None,
            out_token_major_heads: None,
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
    // SIMT-pinned for the same reason as `run_case`: 1e-5 is an
    // f32-vs-f32 tolerance the f16-operand coopmat2 kernels cannot hit.
    exec.run_flash_attention_simt(
        &q,
        &kt,
        &v,
        &fused,
        FlashAttentionDesc {
            kv_len: t,
            pos_base: 0,
            scale,
            token_major_heads: None,
            out_token_major_heads: None,
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
        token_major_heads: None,
        out_token_major_heads: None,
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
                token_major_heads: None,
                out_token_major_heads: None,
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds cache T_max"), "unexpected: {err}");
}

/// `ExecOp::FlashAttn` must be the standalone dispatch verbatim
/// (same routing, same kernel, same grid), so a graph containing only
/// it matches `run_flash_attention` BITWISE.  Warm-cache GQA geometry
/// with a poisoned tail keeps the desc plumbing honest.
#[test]
#[ignore]
fn flash_in_graph_matches_standalone_bitwise() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let case = FlashCase {
        heads: 8,
        kv_heads: 2,
        t_q: 100,
        t_max: 256,
        dh: 64,
        kv_len: 60 + 100,
        pos_base: 60,
    };
    let (q, _) = upload_det(&ctx, &exec, &[case.heads, case.t_q, case.dh], 5400);
    let kt = Tensor::uninit_device(&ctx, &[case.kv_heads, case.dh, case.t_max]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[case.kv_heads, case.t_max, case.dh]).unwrap();
    let (host_kt, host_v) = fill_kv(&case, Some(1.0e30));
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();

    let desc = FlashAttentionDesc {
        kv_len: case.kv_len,
        pos_base: case.pos_base,
        scale: 1.0 / (case.dh as f32).sqrt(),
        token_major_heads: None,
        out_token_major_heads: None,
    };
    let out_standalone = Tensor::uninit_device(&ctx, &[case.heads, case.t_q, case.dh]).unwrap();
    let out_graph = Tensor::uninit_device(&ctx, &[case.heads, case.t_q, case.dh]).unwrap();
    exec.run_flash_attention(&q, &kt, &v, &out_standalone, desc)
        .unwrap();
    exec.run_exec_ops(&[ExecOp::FlashAttn {
        q: &q,
        kt: &kt,
        v: &v,
        out: &out_graph,
        desc,
    }])
    .unwrap();

    let mut a = vec![0.0; (case.heads * case.t_q * case.dh) as usize];
    let mut b = vec![0.0; a.len()];
    exec.download(&out_standalone, &mut a).unwrap();
    exec.download(&out_graph, &mut b).unwrap();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "graph flash vs standalone (bitwise)"
    );
}

/// A/B one geometry across the NV_cooperative_matrix2 route (the
/// default `run_flash_attention` dispatch where supported), the SIMT
/// kernels, and the f64 CPU reference.  Skips on devices without the
/// coopmat2 path.
fn run_cm2_case(label: &str, case: &FlashCase, kv_f16: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.coopmat2_enabled {
        eprintln!("skipping: no NV_cooperative_matrix2");
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
    let (q, host_q) = upload_det(&ctx, &exec, &[heads, t_q, dh], 5300);
    // Poison the cache tail past kv_len: masking leaks blow up both
    // GPU outputs against the reference.  For f16 caches stay finite
    // (1e30 would round to inf; 6e4 still wrecks any leaked softmax).
    let poison = if kv_f16 { 6.0e4 } else { 1.0e30 };
    let (mut host_kt, mut host_v) = fill_kv(case, Some(poison));
    let (kt, v) = if kv_f16 {
        (
            Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap(),
            Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap(),
        )
    } else {
        (
            Tensor::uninit_device(&ctx, &[kv_heads, dh, t_max]).unwrap(),
            Tensor::uninit_device(&ctx, &[kv_heads, t_max, dh]).unwrap(),
        )
    };
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();
    if kv_f16 {
        // The CPU reference sees exactly what the GPU stores.
        for value in host_kt.iter_mut().chain(host_v.iter_mut()) {
            *value = round_f32_via_f16(*value);
        }
    }

    let scale = 1.0 / (dh as f32).sqrt();
    let desc = FlashAttentionDesc {
        kv_len,
        pos_base,
        scale,
        token_major_heads: None,
        out_token_major_heads: None,
    };
    let out_cm2 = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    let out_simt = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    exec.run_flash_attention(&q, &kt, &v, &out_cm2, desc)
        .unwrap();
    exec.run_flash_attention_simt(&q, &kt, &v, &out_simt, desc)
        .unwrap();
    let mut gpu_cm2 = vec![0.0; (heads * t_q * dh) as usize];
    let mut gpu_simt = vec![0.0; gpu_cm2.len()];
    exec.download(&out_cm2, &mut gpu_cm2).unwrap();
    exec.download(&out_simt, &mut gpu_simt).unwrap();

    // The cm2 kernels round Q*scale, K, V, and the softmax weights
    // through f16 (accumulation stays f32), so f16-operand headroom
    // applies against both references; weights summing to 1 keep the
    // error absolute-bounded.  Provisional until measured on hardware.
    const CM2_TOL: f32 = 2.0e-2;
    assert_close_tol(
        &gpu_cm2,
        &gpu_simt,
        CM2_TOL,
        &format!("{label} (cm2 vs simt)"),
    );
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
    assert_close_tol(&gpu_cm2, &cpu, CM2_TOL, &format!("{label} (cm2 vs f64)"));
}

#[test]
#[ignore]
fn flash_cm2_matches_simt_and_cpu() {
    let t512 = FlashCase {
        heads: 4,
        kv_heads: 4,
        t_q: 512,
        t_max: 512,
        dh: 64,
        kv_len: 512,
        pos_base: 0,
    };
    run_cm2_case("cm2 dh64 T512", &t512, false);
    run_cm2_case("cm2 dh64 T512 kv16", &t512, true);

    // Few heads: the f64 reference dominates the runtime at T=2048.
    let t2048 = FlashCase {
        heads: 2,
        kv_heads: 2,
        t_q: 2048,
        t_max: 2048,
        dh: 64,
        kv_len: 2048,
        pos_base: 0,
    };
    run_cm2_case("cm2 dh64 T2048", &t2048, false);
    run_cm2_case("cm2 dh64 T2048 kv16", &t2048, true);

    let dh128 = FlashCase {
        heads: 2,
        kv_heads: 2,
        t_q: 512,
        t_max: 512,
        dh: 128,
        kv_len: 512,
        pos_base: 0,
    };
    run_cm2_case("cm2 dh128 T512", &dh128, false);
    run_cm2_case("cm2 dh128 T512 kv16", &dh128, true);

    // GQA group of 4, warm cache, ragged t_q (not a Br multiple):
    // exercises the per-element causal/ragged mask and the clamped
    // edge loads/stores in one go.
    let ragged = FlashCase {
        heads: 8,
        kv_heads: 2,
        t_q: 333,
        t_max: 1024,
        dh: 64,
        kv_len: 200 + 333,
        pos_base: 200,
    };
    run_cm2_case("cm2 gqa warm ragged", &ragged, true);
}

/// f16 q/out (f16 activations) through the CM2 kv16 io16 kernel: the
/// f32-q CM2 path on identical (pre-rounded) inputs is the reference —
/// both kernels run the same f16-operand/f32-accumulate pipeline, so
/// the io16 output must equal the f32 output narrowed through f16
/// (compared with a whisker of tolerance for any scheduling-order
/// difference between the two compiled pipelines).
fn run_cm2_io16_case(label: &str, case: &FlashCase) {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.coopmat2_enabled || !ctx.f16_storage_enabled {
        eprintln!("skipping: no NV_cooperative_matrix2 / f16 storage");
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
    let mut host_q = vec![0.0_f32; (heads * t_q * dh) as usize];
    fill_det(&mut host_q, 5400);
    for value in &mut host_q {
        *value = round_f32_via_f16(*value);
    }
    let q16 = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    let q32 = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    exec.upload(&host_q, &q16).unwrap();
    exec.upload(&host_q, &q32).unwrap();
    let (host_kt, host_v) = fill_kv(case, Some(6.0e4));
    let kt = Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v = Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();

    let desc = FlashAttentionDesc {
        kv_len,
        pos_base,
        scale: 1.0 / (dh as f32).sqrt(),
        token_major_heads: None,
        out_token_major_heads: None,
    };
    let out16 = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    let out32 = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    exec.run_flash_attention(&q16, &kt, &v, &out16, desc)
        .unwrap();
    exec.run_flash_attention(&q32, &kt, &v, &out32, desc)
        .unwrap();
    let mut gpu16 = vec![0.0; (heads * t_q * dh) as usize];
    let mut gpu32 = vec![0.0; gpu16.len()];
    exec.download(&out16, &mut gpu16).unwrap();
    exec.download(&out32, &mut gpu32).unwrap();
    // Attention outputs are convex combinations of f16 V rows, so
    // |out| <= max|V| ~ 3; one f16 ulp there is ~2^-9.
    assert_close_tol(&gpu16, &gpu32, 4.0e-3, label);
}

#[test]
#[ignore]
fn flash_cm2_io16_matches_f32_io_route() {
    run_cm2_io16_case(
        "cm2 io16 T512",
        &FlashCase {
            heads: 4,
            kv_heads: 4,
            t_q: 512,
            t_max: 512,
            dh: 64,
            kv_len: 512,
            pos_base: 0,
        },
    );
    // GQA + warm cache + ragged t_q.
    run_cm2_io16_case(
        "cm2 io16 gqa warm ragged",
        &FlashCase {
            heads: 8,
            kv_heads: 2,
            t_q: 333,
            t_max: 1024,
            dh: 64,
            kv_len: 200 + 333,
            pos_base: 200,
        },
    );
}

#[test]
#[ignore]
fn flash_io16_invalid_combinations_are_rejected() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.coopmat2_enabled || !ctx.f16_storage_enabled {
        eprintln!("skipping: no NV_cooperative_matrix2 / f16 storage");
        return;
    }
    let (heads, t_q, t_max, dh) = (2_u32, 64_u32, 64_u32, 64_u32);
    let q16 = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    let out16 = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    let out32 = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    let kt32 = Tensor::uninit_device(&ctx, &[heads, dh, t_max]).unwrap();
    let v32 = Tensor::uninit_device(&ctx, &[heads, t_max, dh]).unwrap();
    let desc = FlashAttentionDesc {
        kv_len: t_max,
        pos_base: 0,
        scale: 0.125,
        token_major_heads: None,
        out_token_major_heads: None,
    };
    // f16 q with f32 KV caches has no compiled kernel.
    let err = exec
        .run_flash_attention(&q16, &kt32, &v32, &out16, desc)
        .unwrap_err()
        .to_string();
    assert!(err.contains("f16 KV caches"), "unexpected: {err}");
    // Mixed q/out storage is rejected.
    let kt16 = Tensor::uninit_device_f16(&ctx, &[heads, dh, t_max]).unwrap();
    let v16 = Tensor::uninit_device_f16(&ctx, &[heads, t_max, dh]).unwrap();
    let err = exec
        .run_flash_attention(&q16, &kt16, &v16, &out32, desc)
        .unwrap_err()
        .to_string();
    assert!(err.contains("share a storage type"), "unexpected: {err}");
}

fn permute_head_major_to_token_major(q: &[f32], heads: usize, t: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0; heads * t * dh];
    for head in 0..heads {
        for row in 0..t {
            for d in 0..dh {
                out[(row * heads + head) * dh + d] = q[(head * t + row) * dh + d];
            }
        }
    }
    out
}

fn permute_token_major_to_head_major(q: &[f32], heads: usize, t: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0; heads * t * dh];
    for head in 0..heads {
        for row in 0..t {
            for d in 0..dh {
                out[(head * t + row) * dh + d] = q[(row * heads + head) * dh + d];
            }
        }
    }
    out
}

/// Token-major `[T, H*dh]` Q/out must match the default `[H, T, dh]`
/// layout after a host permute — this is the wide-prefill addressing
/// that drops the two head-permute copies.
#[test]
#[ignore]
fn flash_token_major_matches_head_major() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.buffer_device_address_enabled {
        eprintln!("skipping: no BDA");
        return;
    }
    let (heads, kv_heads, t_q, t_max, dh) = (8_u32, 2_u32, 128_u32, 256_u32, 64_u32);
    let case = FlashCase {
        heads,
        kv_heads,
        t_q,
        t_max,
        dh,
        kv_len: t_q,
        pos_base: 0,
    };
    let (q_hm, host_q) = upload_det(&ctx, &exec, &[heads, t_q, dh], 5500);
    let (host_kt, host_v) = fill_kv(&case, None);
    let kt = Tensor::uninit_device(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();
    let scale = 1.0 / (dh as f32).sqrt();
    let out_hm = Tensor::uninit_device(&ctx, &[heads, t_q, dh]).unwrap();
    exec.run_flash_attention_simt(
        &q_hm,
        &kt,
        &v,
        &out_hm,
        FlashAttentionDesc {
            kv_len: t_q,
            pos_base: 0,
            scale,
            token_major_heads: None,
            out_token_major_heads: None,
        },
    )
    .unwrap();
    let host_q_tm =
        permute_head_major_to_token_major(&host_q, heads as usize, t_q as usize, dh as usize);
    let q_tm = Tensor::uninit_device(&ctx, &[t_q, heads * dh]).unwrap();
    exec.upload(&host_q_tm, &q_tm).unwrap();
    let out_tm = Tensor::uninit_device(&ctx, &[t_q, heads * dh]).unwrap();
    exec.run_flash_attention_simt(
        &q_tm,
        &kt,
        &v,
        &out_tm,
        FlashAttentionDesc {
            kv_len: t_q,
            pos_base: 0,
            scale,
            token_major_heads: Some(heads),
            out_token_major_heads: None,
        },
    )
    .unwrap();
    let mut gpu_hm = vec![0.0; host_q.len()];
    let mut gpu_tm = vec![0.0; host_q.len()];
    exec.download(&out_hm, &mut gpu_hm).unwrap();
    exec.download(&out_tm, &mut gpu_tm).unwrap();
    let gpu_tm_hm =
        permute_token_major_to_head_major(&gpu_tm, heads as usize, t_q as usize, dh as usize);
    assert_close_tol(&gpu_tm_hm, &gpu_hm, 1e-5, "token-major vs head-major SIMT");

    if !ctx.coopmat2_enabled || !ctx.f16_storage_enabled {
        return;
    }
    let mut host_q16 = host_q.clone();
    for value in &mut host_q16 {
        *value = round_f32_via_f16(*value);
    }
    let q16_hm = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    exec.upload(&host_q16, &q16_hm).unwrap();
    let kt16 = Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v16 = Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&host_kt, &kt16).unwrap();
    exec.upload(&host_v, &v16).unwrap();
    let out16_hm = Tensor::uninit_device_f16(&ctx, &[heads, t_q, dh]).unwrap();
    let desc_hm = FlashAttentionDesc {
        kv_len: t_q,
        pos_base: 0,
        scale,
        token_major_heads: None,
        out_token_major_heads: None,
    };
    exec.run_flash_attention(&q16_hm, &kt16, &v16, &out16_hm, desc_hm)
        .unwrap();
    let host_q16_tm =
        permute_head_major_to_token_major(&host_q16, heads as usize, t_q as usize, dh as usize);
    let q16_tm = Tensor::uninit_device_f16(&ctx, &[t_q, heads * dh]).unwrap();
    exec.upload(&host_q16_tm, &q16_tm).unwrap();
    let out16_tm = Tensor::uninit_device_f16(&ctx, &[t_q, heads * dh]).unwrap();
    exec.run_flash_attention(
        &q16_tm,
        &kt16,
        &v16,
        &out16_tm,
        FlashAttentionDesc {
            kv_len: t_q,
            pos_base: 0,
            scale,
            token_major_heads: Some(heads),
            out_token_major_heads: None,
        },
    )
    .unwrap();
    let mut gpu16_hm = vec![0.0; host_q.len()];
    let mut gpu16_tm = vec![0.0; host_q.len()];
    exec.download(&out16_hm, &mut gpu16_hm).unwrap();
    exec.download(&out16_tm, &mut gpu16_tm).unwrap();
    let gpu16_tm_hm =
        permute_token_major_to_head_major(&gpu16_tm, heads as usize, t_q as usize, dh as usize);
    assert_close_tol(
        &gpu16_tm_hm,
        &gpu16_hm,
        4.0e-3,
        "token-major vs head-major io16",
    );

    // Mixed: head-major Q, token-major O — the wide-prefill addressing.
    let out16_mixed = Tensor::uninit_device_f16(&ctx, &[t_q, heads * dh]).unwrap();
    exec.run_flash_attention(
        &q16_hm,
        &kt16,
        &v16,
        &out16_mixed,
        FlashAttentionDesc {
            kv_len: t_q,
            pos_base: 0,
            scale,
            token_major_heads: None,
            out_token_major_heads: Some(heads),
        },
    )
    .unwrap();
    let mut gpu16_mixed = vec![0.0; host_q.len()];
    exec.download(&out16_mixed, &mut gpu16_mixed).unwrap();
    let gpu16_mixed_hm =
        permute_token_major_to_head_major(&gpu16_mixed, heads as usize, t_q as usize, dh as usize);
    assert_close_tol(
        &gpu16_mixed_hm,
        &gpu16_hm,
        4.0e-3,
        "mixed head-Q/token-O vs head-major io16",
    );
}
