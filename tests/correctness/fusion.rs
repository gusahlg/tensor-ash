//! Decode-fusion correctness: the normed-A row GEMV (RMSNorm folded
//! into the matmul) and the fused RoPE + KV-append scatter.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, CopyDesc, Epilogue, EpilogueBinary, ExecOp, MatmulCall, MatmulOp,
    PrefillQkvPackDesc, RopeDesc, RopeScatterDesc, Tensor,
};

fn fusion_available(ctx: &std::sync::Arc<tensor_ash::VulkanContext>) -> bool {
    ctx.buffer_device_address_enabled
}

/// f64 RMSNorm reference: `x * w / sqrt(mean(x^2) + eps)`.
fn cpu_rms_norm_f64(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + eps as f64).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &w)| (v as f64 * inv * w as f64) as f32)
        .collect()
}

fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Upload one normed-A GEMV case; B is f16 when `b_f16`, else f32.
/// Returns (a, b, w, host_a, host_b_rounded, host_w).
#[allow(clippy::type_complexity)]
fn setup_normed_case(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    k: u32,
    n: u32,
    b_f16: bool,
    seed: u64,
) -> (Tensor, Tensor, Tensor, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (a, host_a) = upload_det(ctx, exec, &[1, k], seed);
    let (w, host_w) = upload_det(ctx, exec, &[k], seed + 1);
    let mut host_b = vec![0.0; (k * n) as usize];
    fill_det(&mut host_b, seed + 2);
    let b = if b_f16 {
        let b = Tensor::uninit_device_f16(ctx, &[k, n]).unwrap();
        exec.upload(&host_b, &b).unwrap();
        for value in &mut host_b {
            *value = round_f32_via_f16(*value);
        }
        b
    } else {
        let b = Tensor::uninit_device(ctx, &[k, n]).unwrap();
        exec.upload(&host_b, &b).unwrap();
        b
    };
    (a, b, w, host_a, host_b, host_w)
}

#[test]
#[ignore]
fn normed_a_gemv_matches_f64_rms_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) {
        eprintln!("skipping: no BDA support");
        return;
    }
    let eps = 1e-5_f32;
    // (k, n, b_f16): deep-K narrow-N hits the k16 row variant, the
    // moderate case the k8 variant, plus the f32 row kernel and odd
    // sizes for the tail paths.
    let cases: &[(u32, u32, bool)] = &[
        (2048, 256, true),
        (256, 1024, true),
        (203, 35, true),
        (512, 300, false),
    ];
    for &(k, n, b_f16) in cases {
        if b_f16 && !ctx.f16_storage_enabled {
            continue;
        }
        let route = exec.dispatch_info_for(1, 1, n, k, b_f16).kernel;
        assert!(
            route.contains("row_bda"),
            "expected a row route for 1x{n}x{k}, got {route}"
        );
        let (a, b, w, host_a, host_b, host_w) =
            setup_normed_case(&ctx, &exec, k, n, b_f16, 11_000 + k as u64);
        let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
        exec.run_ops(&[MatmulOp::new(MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        })
        .with_normed_a(&w, eps)])
            .unwrap();
        let mut gpu = vec![0.0; n as usize];
        exec.download(&c, &mut gpu).unwrap();
        let normed = cpu_rms_norm_f64(&host_a, &host_w, eps);
        let cpu = cpu_bmm(&normed, &host_b, None, 1, 1, n, k, 1.0, false);
        assert_close(&gpu, &cpu, k, &format!("normed-A 1x{n}x{k} f16={b_f16}"));
    }
}

#[test]
#[ignore]
fn normed_a_composes_with_silu_mul_epilogue_in_graph() {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    // The llama decode MLP pattern: up = norm(x)@Wu (plain), then
    // gate = silu(norm(x)@Wg) * up (Mul epilogue), both with the
    // RMSNorm folded in, chained in one exec graph.
    let (k, n) = (512_u32, 640_u32);
    let eps = 2e-5_f32;
    let (a, b_up, w, host_a, host_b_up, host_w) =
        setup_normed_case(&ctx, &exec, k, n, true, 21_000);
    let mut host_b_gate = vec![0.0; (k * n) as usize];
    fill_det(&mut host_b_gate, 21_500);
    let b_gate = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
    exec.upload(&host_b_gate, &b_gate).unwrap();
    for value in &mut host_b_gate {
        *value = round_f32_via_f16(*value);
    }
    let up = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let gate = Tensor::uninit_device(&ctx, &[1, n]).unwrap();

    let mm = |b, c| MatmulCall {
        a: &a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    };
    exec.run_exec_ops(&[
        ExecOp::Matmul(MatmulOp::new(mm(&b_up, &up)).with_normed_a(&w, eps)),
        ExecOp::Matmul(
            MatmulOp::with_epilogue(
                mm(&b_gate, &gate),
                Epilogue {
                    bias: None,
                    activation: Activation::Silu,
                    binary: EpilogueBinary::Mul { d: &up },
                },
            )
            .with_normed_a(&w, eps),
        ),
    ])
    .unwrap();

    let mut gpu_up = vec![0.0; n as usize];
    let mut gpu_gate = vec![0.0; n as usize];
    exec.download(&up, &mut gpu_up).unwrap();
    exec.download(&gate, &mut gpu_gate).unwrap();

    let normed = cpu_rms_norm_f64(&host_a, &host_w, eps);
    let cpu_up = cpu_bmm(&normed, &host_b_up, None, 1, 1, n, k, 1.0, false);
    let mut cpu_gate = cpu_bmm(&normed, &host_b_gate, None, 1, 1, n, k, 1.0, false);
    for (gate_value, up_value) in cpu_gate.iter_mut().zip(&cpu_up) {
        *gate_value = cpu_silu(*gate_value) * up_value;
    }
    assert_close(&gpu_up, &cpu_up, k, "normed-A up (graph)");
    assert_close(&gpu_gate, &cpu_gate, k, "normed-A silu*mul gate (graph)");
}

#[test]
#[ignore]
fn normed_a_rejects_conflicting_forms() {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) {
        eprintln!("skipping: no BDA support");
        return;
    }
    let (k, n) = (64_u32, 32_u32);
    let (a, _) = upload_det(&ctx, &exec, &[1, k], 31_000);
    let (a_wide, _) = upload_det(&ctx, &exec, &[8, k], 31_001);
    let (b, _) = upload_det(&ctx, &exec, &[k, n], 31_002);
    let (w, _) = upload_det(&ctx, &exec, &[k], 31_003);
    let (w_bad, _) = upload_det(&ctx, &exec, &[k + 1], 31_004);
    let (bias, _) = upload_det(&ctx, &exec, &[n], 31_005);
    let (d, _) = upload_det(&ctx, &exec, &[1, n], 31_006);
    let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_wide = Tensor::uninit_device(&ctx, &[8, n]).unwrap();
    let call = MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    };

    let err = exec
        .run_ops(&[MatmulOp::new(call).with_normed_a(&w_bad, 1e-5)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("must equal K"), "unexpected: {err}");

    let err = exec
        .run_ops(&[MatmulOp::with_epilogue(
            call,
            Epilogue {
                bias: Some(&bias),
                activation: Activation::None,
                binary: EpilogueBinary::None,
            },
        )
        .with_normed_a(&w, 1e-5)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("epilogue bias"), "unexpected: {err}");

    let err = exec
        .run_ops(&[MatmulOp::with_epilogue(
            call,
            Epilogue {
                bias: None,
                activation: Activation::None,
                binary: EpilogueBinary::AddScaled { d: &d, beta: 1.0 },
            },
        )
        .with_normed_a(&w, 1e-5)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("AddScaled"), "unexpected: {err}");

    // M > 1 has no row route; the record path must fail loudly rather
    // than silently skip the normalization.
    let err = exec
        .run_ops(&[MatmulOp::new(MatmulCall {
            a: &a_wide,
            b: &b,
            c: &c_wide,
            alpha: 1.0,
            accumulate: false,
        })
        .with_normed_a(&w, 1e-5)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("normed-A"), "unexpected: {err}");
}

/// Run the fused rope-scatter and the composed rope + strided-copy
/// reference into two caches (same poison fill) and demand bitwise
/// equality — the rotation math is identical, so even the f16
/// narrowing rounds identically.
fn rope_scatter_case(kv_f16: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) {
        eprintln!("skipping: no BDA support");
        return;
    }
    if kv_f16 && !ctx.f16_storage_enabled {
        eprintln!("skipping: no f16 storage support");
        return;
    }
    // Partial rotary (rot_dim < head_dim) exercises the pass-through
    // lanes; pos_base > 0 exercises the warm-cache offset.
    let (tokens, heads, dh, rot_dim, t_max, pos_base) = (3_u32, 2_u32, 8_u32, 6_u32, 16_u32, 5_u32);
    let (input, _) = upload_det(&ctx, &exec, &[tokens, heads * dh], 41_000);
    let (table, _) = upload_det(&ctx, &exec, &[t_max, rot_dim / 2, 2], 41_001);
    let cache_shape = [heads, dh, t_max];
    let poison = vec![7.0_f32; Tensor::numel(&cache_shape) as usize];
    let make_cache = || {
        let cache = if kv_f16 {
            Tensor::uninit_device_f16(&ctx, &cache_shape).unwrap()
        } else {
            Tensor::uninit_device(&ctx, &cache_shape).unwrap()
        };
        exec.upload(&poison, &cache).unwrap();
        cache
    };
    let cache_ref = make_cache();
    let cache_fused = make_cache();
    let desc = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim,
        pos_base,
        ..Default::default()
    };

    // Reference: rope out-of-place, then strided-copy append.
    let roped = Tensor::uninit_device(&ctx, &[tokens, heads * dh]).unwrap();
    exec.run_rope(&input, &table, &roped, desc).unwrap();
    exec.run_copy_strided(
        &roped,
        &cache_ref,
        CopyDesc {
            extent: [dh, heads, tokens],
            src_offset: 0,
            src_strides: [1, dh, heads * dh],
            dst_offset: pos_base,
            dst_strides: [t_max, dh * t_max, 1],
            ..Default::default()
        },
    )
    .unwrap();

    exec.run_rope_scatter(
        &input,
        &table,
        &cache_fused,
        desc,
        RopeScatterDesc {
            dst_offset: pos_base,
            dst_strides: [1, dh * t_max, t_max],
            ..Default::default()
        },
    )
    .unwrap();

    let mut expected = vec![0.0; poison.len()];
    let mut actual = vec![0.0; poison.len()];
    exec.download(&cache_ref, &mut expected).unwrap();
    exec.download(&cache_fused, &mut actual).unwrap();
    for (index, (&want, &got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "cache element {index} (kv_f16={kv_f16}): ref {want}, fused {got}"
        );
    }
}

#[test]
#[ignore]
fn rope_scatter_matches_rope_plus_copy_f32() {
    rope_scatter_case(false);
}

#[test]
#[ignore]
fn rope_scatter_matches_rope_plus_copy_f16() {
    rope_scatter_case(true);
}

/// Wide-prefill QKV pack vs the composed split + RoPE + KV-append
/// chain on f16 activations.  Covers GQA, a warm `pos_base`, and
/// partial rotary so the pass-through tail is not a dead path.
#[test]
#[ignore]
fn prefill_qkv_pack_matches_composed() {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: needs BDA + f16");
        return;
    }
    let (tokens, heads, kv_heads, dh, rot_dim, t_max, pos_base) =
        (5_u32, 4_u32, 2_u32, 8_u32, 6_u32, 16_u32, 3_u32);
    let embd = heads * dh;
    let kv_dim = kv_heads * dh;
    let n_qkv = embd + 2 * kv_dim;
    let mut host_qkv = vec![0.0; (tokens * n_qkv) as usize];
    fill_det(&mut host_qkv, 61_000);
    for value in &mut host_qkv {
        *value = round_f32_via_f16(*value);
    }
    let qkv = Tensor::uninit_device_f16(&ctx, &[tokens, n_qkv]).unwrap();
    exec.upload(&host_qkv, &qkv).unwrap();
    let (table, _) = upload_det(&ctx, &exec, &[t_max, rot_dim / 2, 2], 61_001);

    let q_ref = Tensor::uninit_device_f16(&ctx, &[tokens, embd]).unwrap();
    let k_ref = Tensor::uninit_device_f16(&ctx, &[tokens, kv_dim]).unwrap();
    let v_ref = Tensor::uninit_device_f16(&ctx, &[tokens, kv_dim]).unwrap();
    exec.run_copy_strided(
        &qkv,
        &q_ref,
        CopyDesc {
            extent: [embd, tokens, 1],
            src_offset: 0,
            src_strides: [1, n_qkv, 0],
            dst_offset: 0,
            dst_strides: [1, embd, 0],
            ..Default::default()
        },
    )
    .unwrap();
    exec.run_copy_strided(
        &qkv,
        &k_ref,
        CopyDesc {
            extent: [kv_dim, tokens, 1],
            src_offset: embd,
            src_strides: [1, n_qkv, 0],
            dst_offset: 0,
            dst_strides: [1, kv_dim, 0],
            ..Default::default()
        },
    )
    .unwrap();
    exec.run_copy_strided(
        &qkv,
        &v_ref,
        CopyDesc {
            extent: [kv_dim, tokens, 1],
            src_offset: embd + kv_dim,
            src_strides: [1, n_qkv, 0],
            dst_offset: 0,
            dst_strides: [1, kv_dim, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let rope_q = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim,
        pos_base,
        ..Default::default()
    };
    let rope_k = RopeDesc {
        heads: kv_heads,
        head_dim: dh,
        rot_dim,
        pos_base,
        ..Default::default()
    };
    exec.run_rope(&q_ref, &table, &q_ref, rope_q).unwrap();
    exec.run_rope(&k_ref, &table, &k_ref, rope_k).unwrap();
    let poison_kt = vec![7.0_f32; (kv_heads * dh * t_max) as usize];
    let poison_v = vec![7.0_f32; (kv_heads * t_max * dh) as usize];
    let kt_ref = Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v_cache_ref = Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&poison_kt, &kt_ref).unwrap();
    exec.upload(&poison_v, &v_cache_ref).unwrap();
    exec.run_copy_strided(
        &k_ref,
        &kt_ref,
        CopyDesc {
            extent: [dh, kv_heads, tokens],
            src_offset: 0,
            src_strides: [1, dh, kv_dim],
            dst_offset: pos_base,
            dst_strides: [t_max, dh * t_max, 1],
            ..Default::default()
        },
    )
    .unwrap();
    exec.run_copy_strided(
        &v_ref,
        &v_cache_ref,
        CopyDesc {
            extent: [dh, kv_heads, tokens],
            src_offset: 0,
            src_strides: [1, dh, kv_dim],
            dst_offset: pos_base * dh,
            dst_strides: [1, t_max * dh, dh],
            ..Default::default()
        },
    )
    .unwrap();

    let q_got = Tensor::uninit_device_f16(&ctx, &[heads, tokens, dh]).unwrap();
    let kt_got = Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v_got = Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&vec![0.0; q_got.len() as usize], &q_got)
        .unwrap();
    exec.upload(&poison_kt, &kt_got).unwrap();
    exec.upload(&poison_v, &v_got).unwrap();
    let pack = PrefillQkvPackDesc {
        heads,
        kv_heads,
        head_dim: dh,
        rot_dim,
        pos_base,
    };
    exec.run_prefill_qkv_pack(&qkv, &table, &q_got, &kt_got, &v_got, pack)
        .unwrap();

    let n_q = q_got.len() as usize;
    let n_kt = kt_got.len() as usize;
    let n_v = v_got.len() as usize;
    let mut q_want = vec![0.0; n_q];
    let mut q_tm = vec![0.0; n_q];
    exec.download(&q_ref, &mut q_tm).unwrap();
    for token in 0..tokens as usize {
        for head in 0..heads as usize {
            for d in 0..dh as usize {
                q_want[(head * tokens as usize + token) * dh as usize + d] =
                    q_tm[(token * heads as usize + head) * dh as usize + d];
            }
        }
    }
    let mut kt_want = vec![0.0; n_kt];
    let mut v_want = vec![0.0; n_v];
    exec.download(&kt_ref, &mut kt_want).unwrap();
    exec.download(&v_cache_ref, &mut v_want).unwrap();

    let assert_pack = |label: &str, q: &Tensor, kt: &Tensor, v: &Tensor| {
        let mut q_actual = vec![0.0; n_q];
        let mut kt_actual = vec![0.0; n_kt];
        let mut v_actual = vec![0.0; n_v];
        exec.download(q, &mut q_actual).unwrap();
        exec.download(kt, &mut kt_actual).unwrap();
        exec.download(v, &mut v_actual).unwrap();
        for (name, want, got) in [
            ("q", &q_want, &q_actual),
            ("kt", &kt_want, &kt_actual),
            ("v", &v_want, &v_actual),
        ] {
            for (index, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    w.to_bits(),
                    g.to_bits(),
                    "{label} {name}[{index}]: composed {w} vs pack {g}"
                );
            }
        }
    };
    assert_pack("standalone", &q_got, &kt_got, &v_got);

    let q_graph = Tensor::uninit_device_f16(&ctx, &[heads, tokens, dh]).unwrap();
    let kt_graph = Tensor::uninit_device_f16(&ctx, &[kv_heads, dh, t_max]).unwrap();
    let v_graph = Tensor::uninit_device_f16(&ctx, &[kv_heads, t_max, dh]).unwrap();
    exec.upload(&vec![0.0; q_graph.len() as usize], &q_graph)
        .unwrap();
    exec.upload(&poison_kt, &kt_graph).unwrap();
    exec.upload(&poison_v, &v_graph).unwrap();
    exec.run_exec_ops(&[ExecOp::PrefillQkvPack {
        qkv: &qkv,
        table: &table,
        q: &q_graph,
        kt: &kt_graph,
        v: &v_graph,
        desc: pack,
    }])
    .unwrap();
    assert_pack("graph", &q_graph, &kt_graph, &v_graph);
}

#[test]
#[ignore]
fn rope_scatter_composes_in_exec_graph_and_bounds_check() {
    let (ctx, exec) = make_setup(2, 8);
    if !fusion_available(&ctx) {
        eprintln!("skipping: no BDA support");
        return;
    }
    let (heads, dh, t_max, pos) = (2_u32, 8_u32, 12_u32, 4_u32);
    let k = 32_u32;
    let (x, host_x) = upload_det(&ctx, &exec, &[1, k], 51_000);
    let (wk, host_wk) = upload_det(&ctx, &exec, &[k, heads * dh], 51_001);
    let (table, host_table) = upload_det(&ctx, &exec, &[t_max, dh / 2, 2], 51_002);
    let k_vec = Tensor::uninit_device(&ctx, &[1, heads * dh]).unwrap();
    let cache = Tensor::uninit_device(&ctx, &[heads, dh, t_max]).unwrap();
    exec.upload(
        &vec![0.0; Tensor::numel(&[heads, dh, t_max]) as usize],
        &cache,
    )
    .unwrap();
    let desc = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim: dh,
        pos_base: pos,
        ..Default::default()
    };

    // The scatter reads the matmul's output: the graph must barrier.
    exec.run_exec_ops(&[
        ExecOp::Matmul(MatmulOp::new(MatmulCall {
            a: &x,
            b: &wk,
            c: &k_vec,
            alpha: 1.0,
            accumulate: false,
        })),
        ExecOp::RopeScatter {
            input: &k_vec,
            table: &table,
            dst: &cache,
            desc,
            scatter: RopeScatterDesc {
                dst_offset: pos,
                dst_strides: [1, dh * t_max, t_max],
                ..Default::default()
            },
        },
    ])
    .unwrap();

    // CPU reference: matvec, rotate adjacent pairs, place at column pos.
    let k_row = cpu_bmm(&host_x, &host_wk, None, 1, 1, heads * dh, k, 1.0, false);
    let mut expected = vec![0.0_f32; (heads * dh * t_max) as usize];
    for head in 0..heads as usize {
        for pair in 0..(dh / 2) as usize {
            let x0 = k_row[head * dh as usize + pair * 2];
            let x1 = k_row[head * dh as usize + pair * 2 + 1];
            let angle = (pos as usize * (dh / 2) as usize + pair) * 2;
            let (c, s) = (host_table[angle], host_table[angle + 1]);
            let r0 = x0 * c - x1 * s;
            let r1 = x0 * s + x1 * c;
            let base = head * (dh * t_max) as usize + pos as usize;
            expected[base + (pair * 2) * t_max as usize] = r0;
            expected[base + (pair * 2 + 1) * t_max as usize] = r1;
        }
    }
    let mut actual = vec![0.0; expected.len()];
    exec.download(&cache, &mut actual).unwrap();
    assert_close(&actual, &expected, k, "rope-scatter exec graph");

    // Out-of-bounds destinations must fail validation up front.
    let err = exec
        .run_rope_scatter(
            &k_vec,
            &table,
            &cache,
            desc,
            RopeScatterDesc {
                dst_offset: t_max, // last dim index lands past the end
                dst_strides: [1, dh * t_max, t_max],
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("destination access"), "unexpected: {err}");
}
