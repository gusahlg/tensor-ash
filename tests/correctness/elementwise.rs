//! Model-op correctness: row softmax, RMS/LayerNorm, RoPE, strided
//! copy — and a full decode-attention step composed from them.

use crate::common::*;

use tensor_ash::{
    BinaryOp, CopyDesc, Epilogue, EpilogueBinary, ExecOp, MatmulCall, MatmulOp, RopeDesc,
    SoftmaxMask, Tensor,
};

fn ops_available(ctx: &std::sync::Arc<tensor_ash::VulkanContext>) -> bool {
    ctx.buffer_device_address_enabled
}

#[test]
#[ignore]
fn softmax_rows_full_prefix_and_causal() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let (rows, cols) = (6_u32, 517_u32);
    let (input, host_in) = upload_det(&ctx, &exec, &[rows, cols], 2100);
    let output = Tensor::uninit_device(&ctx, &[rows, cols]).unwrap();
    let scale = 0.125_f32;

    let cpu_softmax = |valid: &dyn Fn(usize) -> usize| -> Vec<f32> {
        let mut out = vec![0.0; (rows * cols) as usize];
        for row in 0..rows as usize {
            let valid = valid(row);
            let data = &host_in[row * cols as usize..][..cols as usize];
            let row_max = data[..valid]
                .iter()
                .map(|&v| v * scale)
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = data[..valid]
                .iter()
                .map(|&v| (v * scale - row_max).exp())
                .sum();
            for col in 0..valid {
                out[row * cols as usize + col] = (data[col] * scale - row_max).exp() / sum;
            }
        }
        out
    };

    // Full.
    exec.run_softmax_rows(&input, &output, scale, SoftmaxMask::Full)
        .unwrap();
    let mut gpu = vec![0.0; (rows * cols) as usize];
    exec.download(&output, &mut gpu).unwrap();
    let cpu = cpu_softmax(&|_| cols as usize);
    assert_close_tol(&gpu, &cpu, 1e-6, "full softmax");
    for row in 0..rows as usize {
        let sum: f32 = gpu[row * cols as usize..][..cols as usize].iter().sum();
        assert!((sum - 1.0).abs() <= 1e-5, "row {row} sums to {sum}");
    }

    // Prefix mask: tail must be exactly zero.
    let valid = 130_usize;
    exec.run_softmax_rows(
        &input,
        &output,
        scale,
        SoftmaxMask::Prefix {
            valid: valid as u32,
        },
    )
    .unwrap();
    exec.download(&output, &mut gpu).unwrap();
    let cpu = cpu_softmax(&|_| valid);
    assert_close_tol(&gpu, &cpu, 1e-6, "prefix softmax");
    for row in 0..rows as usize {
        for col in valid..cols as usize {
            assert_eq!(gpu[row * cols as usize + col], 0.0, "masked ({row},{col})");
        }
    }

    // Causal over two groups of three rows, prefix 100:
    // row i in its group sees 100 + (i % 3) + 1 columns.
    exec.run_softmax_rows(
        &input,
        &output,
        scale,
        SoftmaxMask::Causal {
            prefix: 100,
            rows_per_group: 3,
        },
    )
    .unwrap();
    exec.download(&output, &mut gpu).unwrap();
    let cpu = cpu_softmax(&|row| 100 + (row % 3) + 1);
    assert_close_tol(&gpu, &cpu, 1e-6, "causal softmax");
}

#[test]
#[ignore]
fn rms_and_layer_norm_match_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let eps = 1e-5_f32;
    // 771: scalar tail (cols % 4 != 0). 768: vec4 row walk.
    for (rows, cols, seed) in [(5_u32, 771_u32, 2200_u64), (4_u32, 768_u32, 2210_u64)] {
        let (input, host_in) = upload_det(&ctx, &exec, &[rows, cols], seed);
        let (weight, host_w) = upload_det(&ctx, &exec, &[cols], seed + 1);
        let (bias, host_b) = upload_det(&ctx, &exec, &[cols], seed + 2);
        let output = Tensor::uninit_device(&ctx, &[rows, cols]).unwrap();
        let mut gpu = vec![0.0; (rows * cols) as usize];

        exec.run_rms_norm(&input, &weight, &output, eps).unwrap();
        exec.download(&output, &mut gpu).unwrap();
        let mut cpu = vec![0.0; gpu.len()];
        for row in 0..rows as usize {
            let data = &host_in[row * cols as usize..][..cols as usize];
            let mean_sq = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / cols as f64;
            let inv = 1.0 / (mean_sq + eps as f64).sqrt();
            for col in 0..cols as usize {
                cpu[row * cols as usize + col] = (data[col] as f64 * inv) as f32 * host_w[col];
            }
        }
        assert_close_tol(&gpu, &cpu, 1e-4, &format!("rmsnorm cols={cols}"));

        exec.run_layer_norm(&input, &weight, &bias, &output, eps)
            .unwrap();
        exec.download(&output, &mut gpu).unwrap();
        for row in 0..rows as usize {
            let data = &host_in[row * cols as usize..][..cols as usize];
            let mean = data.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            for col in 0..cols as usize {
                cpu[row * cols as usize + col] =
                    ((data[col] as f64 - mean) * inv) as f32 * host_w[col] + host_b[col];
            }
        }
        assert_close_tol(&gpu, &cpu, 1e-4, &format!("layernorm cols={cols}"));
    }
}

#[test]
#[ignore]
fn rope_rotates_pairs_with_partial_dim_and_offset() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let (tokens, heads, head_dim, rot_dim) = (3_u32, 4_u32, 16_u32, 12_u32);
    let t_max = 32_u32;
    let pos_base = 7_u32;
    let (input, host_in) = upload_det(&ctx, &exec, &[tokens, heads * head_dim], 2300);
    let output = Tensor::uninit_device(&ctx, &[tokens, heads * head_dim]).unwrap();

    // Llama-style table: theta = pos / 10000^(2i/rot_dim).
    let half = (rot_dim / 2) as usize;
    let mut host_table = vec![0.0_f32; (t_max as usize) * half * 2];
    for pos in 0..t_max as usize {
        for pair in 0..half {
            let theta = pos as f64 / 10000_f64.powf(2.0 * pair as f64 / rot_dim as f64);
            host_table[(pos * half + pair) * 2] = theta.cos() as f32;
            host_table[(pos * half + pair) * 2 + 1] = theta.sin() as f32;
        }
    }
    let table = Tensor::uninit_device(&ctx, &[t_max, rot_dim / 2, 2]).unwrap();
    exec.upload(&host_table, &table).unwrap();

    exec.run_rope(
        &input,
        &table,
        &output,
        RopeDesc {
            heads,
            head_dim,
            rot_dim,
            pos_base,
            ..Default::default()
        },
    )
    .unwrap();
    let mut gpu = vec![0.0; host_in.len()];
    exec.download(&output, &mut gpu).unwrap();

    let mut cpu = host_in.clone();
    for token in 0..tokens as usize {
        for head in 0..heads as usize {
            let base = (token * heads as usize + head) * head_dim as usize;
            for pair in 0..half {
                let pos = pos_base as usize + token;
                let c = host_table[(pos * half + pair) * 2];
                let s = host_table[(pos * half + pair) * 2 + 1];
                let x0 = host_in[base + pair * 2];
                let x1 = host_in[base + pair * 2 + 1];
                cpu[base + pair * 2] = x0 * c - x1 * s;
                cpu[base + pair * 2 + 1] = x0 * s + x1 * c;
            }
            // Lanes past rot_dim pass through — cpu already holds them.
        }
    }
    assert_close_tol(&gpu, &cpu, 1e-5, "rope");

    // In-place run matches the out-of-place result.
    exec.run_rope(
        &input,
        &table,
        &input,
        RopeDesc {
            heads,
            head_dim,
            rot_dim,
            pos_base,
            ..Default::default()
        },
    )
    .unwrap();
    let mut inplace = vec![0.0; host_in.len()];
    exec.download(&input, &mut inplace).unwrap();
    assert_close_tol(&inplace, &gpu, 0.0, "in-place rope diverges");
}

#[test]
#[ignore]
fn copy_strided_transposes_and_appends() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    // Transpose a 37x53 matrix via stride swap.
    let (rows, cols) = (37_u32, 53_u32);
    let (src, host_src) = upload_det(&ctx, &exec, &[rows, cols], 2400);
    let dst = Tensor::uninit_device(&ctx, &[cols, rows]).unwrap();
    exec.run_copy_strided(
        &src,
        &dst,
        CopyDesc {
            extent: [cols, rows, 1],
            src_offset: 0,
            src_strides: [1, cols, 0],
            dst_offset: 0,
            dst_strides: [rows, 1, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let mut gpu = vec![0.0; (rows * cols) as usize];
    exec.download(&dst, &mut gpu).unwrap();
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            assert_eq!(
                gpu[col * rows as usize + row],
                host_src[row * cols as usize + col],
                "transpose mismatch at ({row},{col})"
            );
        }
    }

    // KV-append: scatter a [heads, dh] K vector as column `t` of a
    // zeroed [heads, dh, t_max] transposed cache.
    let (heads, dh, t_max, t) = (4_u32, 8_u32, 16_u32, 5_u32);
    let (kvec, host_k) = upload_det(&ctx, &exec, &[heads, dh], 2401);
    let cache = Tensor::uninit_device(&ctx, &[heads, dh, t_max]).unwrap();
    exec.upload(&vec![0.0; (heads * dh * t_max) as usize], &cache)
        .unwrap();
    exec.run_copy_strided(
        &kvec,
        &cache,
        CopyDesc {
            extent: [dh, heads, 1],
            src_offset: 0,
            src_strides: [1, dh, 0],
            dst_offset: t,
            dst_strides: [t_max, dh * t_max, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let mut cache_host = vec![0.0; (heads * dh * t_max) as usize];
    exec.download(&cache, &mut cache_host).unwrap();
    for head in 0..heads as usize {
        for lane in 0..dh as usize {
            for step in 0..t_max as usize {
                let got = cache_host[(head * dh as usize + lane) * t_max as usize + step];
                let want = if step == t as usize {
                    host_k[head * dh as usize + lane]
                } else {
                    0.0
                };
                assert_eq!(got, want, "cache ({head},{lane},{step})");
            }
        }
    }

    // Out-of-bounds descriptors are rejected before dispatch.
    let err = exec
        .run_copy_strided(
            &kvec,
            &cache,
            CopyDesc {
                extent: [dh, heads, 1],
                src_offset: 0,
                src_strides: [1, dh, 0],
                dst_offset: t_max,
                dst_strides: [t_max, dh * t_max, 0],
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("destination access"), "unexpected: {err}");
}

/// The integration payoff: one decode-attention step composed purely
/// from library ops — scores = q @ Kt (zero-padded T_max cache),
/// probs = masked softmax, out = probs @ V — validated against a CPU
/// reference over the valid prefix only.
#[test]
#[ignore]
fn decode_attention_step_composes() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let (heads, dh, t_max, kv_len) = (8_u32, 64_u32, 128_u32, 77_u32);
    let scale = 1.0 / (dh as f32).sqrt();

    // Zero-padded caches: Kt [H, dh, T_max], V [H, T_max, dh].
    let (q, host_q) = upload_det(&ctx, &exec, &[heads, 1, dh], 2500);
    let kt = Tensor::uninit_device(&ctx, &[heads, dh, t_max]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[heads, t_max, dh]).unwrap();
    let mut host_kt = vec![0.0; (heads * dh * t_max) as usize];
    let mut host_v = vec![0.0; (heads * t_max * dh) as usize];
    let mut kt_fill = vec![0.0; host_kt.len()];
    let mut v_fill = vec![0.0; host_v.len()];
    fill_det(&mut kt_fill, 2501);
    fill_det(&mut v_fill, 2502);
    // Populate only the valid prefix; the tail stays zero.
    for head in 0..heads as usize {
        for lane in 0..dh as usize {
            for step in 0..kv_len as usize {
                let idx = (head * dh as usize + lane) * t_max as usize + step;
                host_kt[idx] = kt_fill[idx];
            }
        }
        for step in 0..kv_len as usize {
            for lane in 0..dh as usize {
                let idx = (head * t_max as usize + step) * dh as usize + lane;
                host_v[idx] = v_fill[idx];
            }
        }
    }
    exec.upload(&host_kt, &kt).unwrap();
    exec.upload(&host_v, &v).unwrap();

    // GPU: scores -> masked softmax (in-place) -> out.
    let scores = Tensor::uninit_device(&ctx, &[heads, 1, t_max]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[heads, 1, dh]).unwrap();
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
        c: &out,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let mut gpu = vec![0.0; (heads * dh) as usize];
    exec.download(&out, &mut gpu).unwrap();

    // CPU reference over the valid prefix.
    let mut cpu = vec![0.0; (heads * dh) as usize];
    for head in 0..heads as usize {
        let mut probs = vec![0.0_f64; kv_len as usize];
        let mut row_max = f64::NEG_INFINITY;
        for step in 0..kv_len as usize {
            let mut dot = 0.0_f64;
            for lane in 0..dh as usize {
                dot += host_q[head * dh as usize + lane] as f64
                    * host_kt[(head * dh as usize + lane) * t_max as usize + step] as f64;
            }
            probs[step] = dot * scale as f64;
            row_max = row_max.max(probs[step]);
        }
        let mut sum = 0.0_f64;
        for value in &mut probs {
            *value = (*value - row_max).exp();
            sum += *value;
        }
        for lane in 0..dh as usize {
            let mut acc = 0.0_f64;
            for step in 0..kv_len as usize {
                acc += probs[step] / sum
                    * host_v[(head * t_max as usize + step) * dh as usize + lane] as f64;
            }
            cpu[head * dh as usize + lane] = acc as f32;
        }
    }
    assert_close_tol(&gpu, &cpu, 1e-4, "decode attention");
}

/// A mixed decode-style chain through `run_exec_ops` must match the
/// same ops submitted one by one — same kernels, same order, so the
/// results are bitwise identical; only the submission count differs.
#[test]
#[ignore]
fn exec_ops_chain_matches_per_op_path() {
    let (ctx, exec) = make_setup(2, 64);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let (rows, hidden, ffn) = (4_u32, 256_u32, 512_u32);
    let (x, _) = upload_det(&ctx, &exec, &[rows, hidden], 6100);
    let (norm_w, _) = upload_det(&ctx, &exec, &[hidden], 6101);
    let w_up = Tensor::uninit_device_f16(&ctx, &[hidden, ffn]).unwrap();
    let w_gate = Tensor::uninit_device_f16(&ctx, &[hidden, ffn]).unwrap();
    let mut host_w = vec![0.0; (hidden * ffn) as usize];
    fill_det(&mut host_w, 6102);
    exec.upload(&host_w, &w_up).unwrap();
    fill_det(&mut host_w, 6103);
    exec.upload(&host_w, &w_gate).unwrap();

    let run = |xn: &Tensor, up: &Tensor, gated: &Tensor, graph: bool| {
        let ops = [
            ExecOp::RmsNorm {
                input: &x,
                weight: &norm_w,
                output: xn,
                eps: 1e-5,
            },
            ExecOp::Matmul(MatmulOp::new(MatmulCall {
                a: xn,
                b: &w_up,
                c: up,
                alpha: 1.0,
                accumulate: false,
            })),
            ExecOp::Matmul(MatmulOp::with_epilogue(
                MatmulCall {
                    a: xn,
                    b: &w_gate,
                    c: gated,
                    alpha: 1.0,
                    accumulate: false,
                },
                Epilogue {
                    bias: None,
                    activation: tensor_ash::Activation::Silu,
                    binary: EpilogueBinary::Mul { d: up },
                },
            )),
            ExecOp::SoftmaxRows {
                input: gated,
                output: gated,
                scale: 1.0,
                mask: SoftmaxMask::Full,
            },
        ];
        if graph {
            let stats = exec.run_exec_ops(&ops).unwrap();
            assert_eq!(stats.n_calls, ops.len());
        } else {
            for op in ops {
                match op {
                    ExecOp::RmsNorm {
                        input,
                        weight,
                        output,
                        eps,
                    } => {
                        exec.run_rms_norm(input, weight, output, eps).unwrap();
                    }
                    ExecOp::Matmul(op) => {
                        exec.run_ops(std::slice::from_ref(&op)).unwrap();
                    }
                    ExecOp::SoftmaxRows {
                        input,
                        output,
                        scale,
                        mask,
                    } => {
                        exec.run_softmax_rows(input, output, scale, mask).unwrap();
                    }
                    _ => unreachable!(),
                }
            }
        }
    };

    let xn_a = Tensor::uninit_device(&ctx, &[rows, hidden]).unwrap();
    let up_a = Tensor::uninit_device(&ctx, &[rows, ffn]).unwrap();
    let out_a = Tensor::uninit_device(&ctx, &[rows, ffn]).unwrap();
    run(&xn_a, &up_a, &out_a, true);
    let xn_b = Tensor::uninit_device(&ctx, &[rows, hidden]).unwrap();
    let up_b = Tensor::uninit_device(&ctx, &[rows, ffn]).unwrap();
    let out_b = Tensor::uninit_device(&ctx, &[rows, ffn]).unwrap();
    run(&xn_b, &up_b, &out_b, false);

    let mut a = vec![0.0; (rows * ffn) as usize];
    let mut b = vec![0.0; (rows * ffn) as usize];
    exec.download(&out_a, &mut a).unwrap();
    exec.download(&out_b, &mut b).unwrap();
    for (i, (&va, &vb)) in a.iter().zip(&b).enumerate() {
        assert_eq!(va.to_bits(), vb.to_bits(), "graph vs per-op diverge at {i}");
    }
}

#[test]
#[ignore]
fn binary_add_scaled_and_silu_mul_match_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) {
        eprintln!("skipping: no BDA");
        return;
    }
    let n = 4099_u32; // odd length exercises the tail guard
    let (a, host_a) = upload_det(&ctx, &exec, &[n], 7100);
    let (b, host_b) = upload_det(&ctx, &exec, &[n], 7101);
    let out = Tensor::uninit_device(&ctx, &[n]).unwrap();
    let mut gpu = vec![0.0; n as usize];

    exec.run_binary(&a, &b, &out, BinaryOp::AddScaled { beta: 0.5 })
        .unwrap();
    exec.download(&out, &mut gpu).unwrap();
    let cpu: Vec<f32> = host_a
        .iter()
        .zip(&host_b)
        .map(|(&x, &y)| x + 0.5 * y)
        .collect();
    assert_close_tol(&gpu, &cpu, 1e-6, "add_scaled");

    // In-place SiluMul must match the fused-epilogue definition.
    exec.run_binary(&a, &b, &a, BinaryOp::SiluMul).unwrap();
    exec.download(&a, &mut gpu).unwrap();
    let cpu: Vec<f32> = host_a
        .iter()
        .zip(&host_b)
        .map(|(&x, &y)| (x / (1.0 + (-x).exp())) * y)
        .collect();
    assert_close_tol(&gpu, &cpu, 1e-5, "silu_mul in-place");
}

/// Shared setup for the f16-IO variants: an f16 tensor and an f32
/// tensor holding IDENTICAL values (the f32 copy pre-rounded through
/// f16), so the f32-path result narrowed to f16 must be bit-equal to
/// the f16-path result — both kernels run the same f32 arithmetic in
/// the same deterministic order.
fn upload_f16_and_f32_twin(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    shape: &[u32],
    seed: u64,
) -> (Tensor, Tensor, Vec<f32>) {
    use tensor_ash::dtype::round_f32_via_f16;
    let mut host = vec![0.0_f32; Tensor::numel(shape) as usize];
    fill_det(&mut host, seed);
    for value in &mut host {
        *value = round_f32_via_f16(*value);
    }
    let t16 = Tensor::uninit_device_f16(ctx, shape).unwrap();
    let t32 = Tensor::uninit_device(ctx, shape).unwrap();
    exec.upload(&host, &t16).unwrap();
    exec.upload(&host, &t32).unwrap();
    (t16, t32, host)
}

/// Bit-exact comparison of an f16-path result against the f32-path
/// result narrowed through f16.
fn assert_matches_narrowed(gpu16: &[f32], gpu32: &[f32], label: &str) {
    use tensor_ash::dtype::round_f32_via_f16;
    for (index, (&got, &full)) in gpu16.iter().zip(gpu32).enumerate() {
        let expect = round_f32_via_f16(full);
        assert_eq!(
            got.to_bits(),
            expect.to_bits(),
            "{label}: element {index}: f16 path {got}, narrowed f32 path {expect}"
        );
    }
}

#[test]
#[ignore]
fn rms_norm_f16_io_matches_narrowed_f32_path() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (rows, cols) = (7_u32, 2048_u32);
    let (in16, in32, _) = upload_f16_and_f32_twin(&ctx, &exec, &[rows, cols], 6100);
    let (weight, _) = upload_det(&ctx, &exec, &[cols], 6101);
    let out16 = Tensor::uninit_device_f16(&ctx, &[rows, cols]).unwrap();
    let out32 = Tensor::uninit_device(&ctx, &[rows, cols]).unwrap();
    exec.run_rms_norm(&in16, &weight, &out16, 1e-5).unwrap();
    exec.run_rms_norm(&in32, &weight, &out32, 1e-5).unwrap();
    let mut gpu16 = vec![0.0; (rows * cols) as usize];
    let mut gpu32 = vec![0.0; gpu16.len()];
    exec.download(&out16, &mut gpu16).unwrap();
    exec.download(&out32, &mut gpu32).unwrap();
    assert_matches_narrowed(&gpu16, &gpu32, "rmsnorm f16");

    // Mixed f16 input / f32 output is rejected.
    let err = exec
        .run_rms_norm(&in16, &weight, &out32, 1e-5)
        .unwrap_err()
        .to_string();
    assert!(err.contains("share a storage type"), "unexpected: {err}");
}

#[test]
#[ignore]
fn rope_f16_io_matches_narrowed_f32_path() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (tokens, heads, dh) = (12_u32, 4_u32, 64_u32);
    let t_max = 64_u32;
    let (in16, in32, _) = upload_f16_and_f32_twin(&ctx, &exec, &[tokens, heads * dh], 6200);
    // Real cos/sin table (f32 both paths).
    let half = (dh / 2) as usize;
    let mut table_host = vec![0.0_f32; t_max as usize * half * 2];
    for pos in 0..t_max as usize {
        for pair in 0..half {
            let theta = pos as f64 / 10000_f64.powf(2.0 * pair as f64 / dh as f64);
            table_host[(pos * half + pair) * 2] = theta.cos() as f32;
            table_host[(pos * half + pair) * 2 + 1] = theta.sin() as f32;
        }
    }
    let table = Tensor::uninit_device(&ctx, &[t_max, dh / 2, 2]).unwrap();
    exec.upload(&table_host, &table).unwrap();
    let desc = RopeDesc {
        heads,
        head_dim: dh,
        rot_dim: dh,
        pos_base: 3,
        ..Default::default()
    };
    // In-place on both (the prefill usage).
    exec.run_rope(&in16, &table, &in16, desc).unwrap();
    exec.run_rope(&in32, &table, &in32, desc).unwrap();
    let mut gpu16 = vec![0.0; (tokens * heads * dh) as usize];
    let mut gpu32 = vec![0.0; gpu16.len()];
    exec.download(&in16, &mut gpu16).unwrap();
    exec.download(&in32, &mut gpu32).unwrap();
    assert_matches_narrowed(&gpu16, &gpu32, "rope f16");
}

#[test]
#[ignore]
fn binary_f16_io_matches_narrowed_f32_path() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let n = 4099_u32;
    let (a16, a32, _) = upload_f16_and_f32_twin(&ctx, &exec, &[n], 6300);
    let (b16, b32, _) = upload_f16_and_f32_twin(&ctx, &exec, &[n], 6301);
    for op in [BinaryOp::AddScaled { beta: 0.5 }, BinaryOp::SiluMul] {
        let out16 = Tensor::uninit_device_f16(&ctx, &[n]).unwrap();
        let out32 = Tensor::uninit_device(&ctx, &[n]).unwrap();
        exec.run_binary(&a16, &b16, &out16, op).unwrap();
        exec.run_binary(&a32, &b32, &out32, op).unwrap();
        let mut gpu16 = vec![0.0; n as usize];
        let mut gpu32 = vec![0.0; n as usize];
        exec.download(&out16, &mut gpu16).unwrap();
        exec.download(&out32, &mut gpu32).unwrap();
        assert_matches_narrowed(&gpu16, &gpu32, &format!("binary f16 {op:?}"));
    }
    // Mixed storage is rejected.
    let out16 = Tensor::uninit_device_f16(&ctx, &[n]).unwrap();
    let err = exec
        .run_binary(&a16, &b32, &out16, BinaryOp::SiluMul)
        .unwrap_err()
        .to_string();
    assert!(err.contains("must match a"), "unexpected: {err}");
}

#[test]
#[ignore]
fn copy_strided_f16_pairings_roundtrip() {
    let (ctx, exec) = make_setup(2, 8);
    if !ops_available(&ctx) || !ctx.f16_storage_enabled {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (rows, cols) = (24_u32, 96_u32);
    let (src16, _, host) = upload_f16_and_f32_twin(&ctx, &exec, &[rows, cols], 6400);
    // f16 -> f16 transpose, then f16 -> f32 transpose back: the double
    // permute must reproduce the (pre-rounded) source exactly.
    let mid16 = Tensor::uninit_device_f16(&ctx, &[cols, rows]).unwrap();
    exec.run_copy_strided(
        &src16,
        &mid16,
        CopyDesc {
            extent: [cols, rows, 1],
            src_offset: 0,
            src_strides: [1, cols, 0],
            dst_offset: 0,
            dst_strides: [rows, 1, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let back32 = Tensor::uninit_device(&ctx, &[rows, cols]).unwrap();
    exec.run_copy_strided(
        &mid16,
        &back32,
        CopyDesc {
            extent: [rows, cols, 1],
            src_offset: 0,
            src_strides: [1, rows, 0],
            dst_offset: 0,
            dst_strides: [cols, 1, 0],
            ..Default::default()
        },
    )
    .unwrap();
    let mut gpu = vec![0.0; (rows * cols) as usize];
    exec.download(&back32, &mut gpu).unwrap();
    for (index, (&got, &sent)) in gpu.iter().zip(&host).enumerate() {
        assert_eq!(
            got.to_bits(),
            sent.to_bits(),
            "copy roundtrip element {index}: got {got}, sent {sent}"
        );
    }
}
