//! Correctness for fused GEMM epilogues: bias, activations, and the
//! residual-add / gating-mul binary ops, across interior-vec4 and
//! bounds-checked edge store paths.

use crate::common::*;

use tensor_ash::{Activation, Epilogue, EpilogueBinary, MatmulCall, MatmulOp, Tensor};

fn cpu_activate(x: f64, act: Activation) -> f64 {
    match act {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::Gelu => {
            let k0 = 0.7978845608028654f64;
            0.5 * x * (1.0 + (k0 * (x + 0.044715 * x * x * x)).tanh())
        }
    }
}

/// Apply the epilogue on the CPU reference, mirroring the shader order:
/// (value already includes alpha-scale and accumulate) → +bias[col] →
/// activation → binary with D.
fn cpu_epilogue(
    c: &mut [f32],
    n: u32,
    bias: Option<&[f32]>,
    act: Activation,
    binary: Option<(&[f32], f32, bool)>, // (d, beta, is_mul)
) {
    for (i, v) in c.iter_mut().enumerate() {
        let col = i % n as usize;
        let mut x = *v as f64;
        if let Some(b) = bias {
            x += b[col] as f64;
        }
        x = cpu_activate(x, act);
        if let Some((d, beta, is_mul)) = binary {
            if is_mul {
                x *= d[i] as f64;
            } else {
                x += beta as f64 * d[i] as f64;
            }
        }
        *v = x as f32;
    }
}

/// Nonlinear activations evaluate transcendentals whose GPU and libm
/// implementations differ by a few ulp at operand magnitude ~sqrt(K);
/// budget that on top of the GEMM tolerance.
fn epi_tolerance(k: u32) -> f32 {
    2.0 * tolerance(k) + 1e-4
}

#[allow(clippy::too_many_arguments)]
fn run_epilogue_case(
    shape: (u32, u32, u32),
    alpha: f32,
    accumulate: bool,
    bias: bool,
    act: Activation,
    binary: Option<(f32, bool)>, // (beta, is_mul)
) {
    let (m, n, k) = shape;
    let (ctx, exec) = make_setup(2, 8);

    let (a, ha) = upload_det(&ctx, &exec, &[m, k], 71);
    let (b, hb) = upload_det(&ctx, &exec, &[k, n], 72);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    let mut hc0 = vec![0.0f32; (m * n) as usize];
    if accumulate {
        fill_det(&mut hc0, 73);
        exec.upload(&hc0, &c).unwrap();
    }

    let bias_pair = bias.then(|| upload_det(&ctx, &exec, &[n], 74));
    let bias_ref = bias_pair.as_ref().map(|(bias_t, _)| bias_t);

    let d_pair = binary.map(|_| upload_det(&ctx, &exec, &[m, n], 75));
    let binary_op = match (binary, &d_pair) {
        (Some((_, true)), Some((d_t, _))) => EpilogueBinary::Mul { d: d_t },
        (Some((beta, false)), Some((d_t, _))) => EpilogueBinary::AddScaled { d: d_t, beta },
        _ => EpilogueBinary::None,
    };

    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha,
            accumulate,
        },
        Epilogue {
            bias: bias_ref,
            activation: act,
            binary: binary_op,
        },
    )])
    .unwrap();

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&c, &mut got).unwrap();

    let mut expect = cpu_bmm(
        &ha,
        &hb,
        accumulate.then_some(&hc0[..]),
        1,
        m,
        n,
        k,
        alpha,
        accumulate,
    );
    cpu_epilogue(
        &mut expect,
        n,
        bias_pair.as_ref().map(|(_, hbias)| &hbias[..]),
        act,
        binary.map(|(beta, is_mul)| (&d_pair.as_ref().unwrap().1[..], beta, is_mul)),
    );

    assert_close_tol(
        &got,
        &expect,
        epi_tolerance(k),
        &format!(
            "epilogue case {shape:?} alpha={alpha} acc={accumulate} bias={bias} act={act:?} \
             binary={binary:?}"
        ),
    );
}

// -- Aligned shape (interior vec4 store path on the V4 kernels) --

#[test]
#[ignore]
fn bias_only_aligned() {
    run_epilogue_case((512, 512, 512), 1.0, false, true, Activation::None, None);
}

#[test]
#[ignore]
fn bias_relu_aligned() {
    run_epilogue_case((512, 512, 512), 1.0, false, true, Activation::Relu, None);
}

#[test]
#[ignore]
fn silu_gate_mul_aligned() {
    // SwiGLU shape: silu(x@W_gate) * up.
    run_epilogue_case(
        (256, 1024, 512),
        1.0,
        false,
        false,
        Activation::Silu,
        Some((0.0, true)),
    );
}

#[test]
#[ignore]
fn bias_gelu_residual_aligned() {
    run_epilogue_case(
        (512, 512, 256),
        1.0,
        false,
        true,
        Activation::Gelu,
        Some((0.75, false)),
    );
}

#[test]
#[ignore]
fn alpha_bias_act_accumulate_aligned() {
    run_epilogue_case((256, 256, 256), 0.5, true, true, Activation::Silu, None);
}

// -- Odd shapes (bounds-checked scalar store paths) --

#[test]
#[ignore]
fn bias_relu_odd_shape() {
    run_epilogue_case((255, 257, 263), 1.0, false, true, Activation::Relu, None);
}

#[test]
#[ignore]
fn silu_mul_odd_shape() {
    run_epilogue_case(
        (123, 517, 129),
        1.0,
        false,
        false,
        Activation::Silu,
        Some((0.0, true)),
    );
}

#[test]
#[ignore]
fn bias_residual_odd_shape() {
    run_epilogue_case(
        (1023, 129, 511),
        2.0,
        false,
        true,
        Activation::None,
        Some((-1.5, false)),
    );
}

#[test]
#[ignore]
fn batched_bias_uses_one_row_per_batch() {
    let (ctx, exec) = make_setup(2, 8);
    let (batch, m, n, k) = (3u32, 2u32, 7u32, 5u32);
    let (a, ha) = upload_det(&ctx, &exec, &[batch, m, k], 91);
    let (b, hb) = upload_det(&ctx, &exec, &[batch, k, n], 92);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let bias = Tensor::uninit_device(&ctx, &[batch, n]).unwrap();

    let mut hbias = vec![0.0; (batch * n) as usize];
    for batch_index in 0..batch as usize {
        for col in 0..n as usize {
            hbias[batch_index * n as usize + col] = 10.0 * batch_index as f32 + col as f32;
        }
    }
    exec.upload(&hbias, &bias).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: Some(&bias),
            activation: Activation::None,
            binary: EpilogueBinary::None,
        },
    )])
    .unwrap();

    let mut got = vec![0.0; (batch * m * n) as usize];
    exec.download(&c, &mut got).unwrap();
    let mut expect = cpu_bmm(&ha, &hb, None, batch, m, n, k, 1.0, false);
    for batch_index in 0..batch as usize {
        for row in 0..m as usize {
            for col in 0..n as usize {
                expect[(batch_index * m as usize + row) * n as usize + col] +=
                    hbias[batch_index * n as usize + col];
            }
        }
    }

    assert_close_tol(&got, &expect, epi_tolerance(k), "batched bias");
}

// -- Epilogue tensors participate in graph hazard tracking --

#[test]
#[ignore]
fn graph_hazard_on_epilogue_d_operand() {
    // up = x@W_up  (no epilogue); gated = silu(x@W_gate) * up — the
    // second op's D operand is the first op's output, so the graph
    // recorder must fence between them.
    let (ctx, exec) = make_setup(2, 8);
    let (m, h, i_) = (64u32, 128u32, 256u32);

    let (x, hx) = upload_det(&ctx, &exec, &[m, h], 81);
    let (w_up, hup) = upload_det(&ctx, &exec, &[h, i_], 82);
    let (w_gate, hgate) = upload_det(&ctx, &exec, &[h, i_], 83);
    let up = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();
    let gated = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();

    exec.run_op_graph(&[
        MatmulOp::new(MatmulCall {
            a: &x,
            b: &w_up,
            c: &up,
            alpha: 1.0,
            accumulate: false,
        }),
        MatmulOp::with_epilogue(
            MatmulCall {
                a: &x,
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
        ),
    ])
    .unwrap();

    let mut got = vec![0.0f32; (m * i_) as usize];
    exec.download(&gated, &mut got).unwrap();

    let hup_out = cpu_bmm(&hx, &hup, None, 1, m, i_, h, 1.0, false);
    let mut expect = cpu_bmm(&hx, &hgate, None, 1, m, i_, h, 1.0, false);
    cpu_epilogue(
        &mut expect,
        i_,
        None,
        Activation::Silu,
        Some((&hup_out[..], 0.0, true)),
    );

    assert_close_tol(&got, &expect, epi_tolerance(h), "swiglu graph");
}

/// Bias shape validation: wrong-length bias must be rejected.
#[test]
#[ignore]
fn rejects_wrong_bias_length() {
    let (ctx, exec) = make_setup(2, 8);
    let a = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let bad_bias = Tensor::uninit_device(&ctx, &[32]).unwrap();

    let err = exec
        .run_ops(&[MatmulOp::with_epilogue(
            MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            },
            Epilogue {
                bias: Some(&bad_bias),
                activation: Activation::None,
                binary: EpilogueBinary::None,
            },
        )])
        .unwrap_err()
        .to_string();
    assert!(err.contains("bias"), "unexpected error: {err}");
}
