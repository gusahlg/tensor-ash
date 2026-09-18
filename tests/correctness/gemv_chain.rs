//! Persistent GEMV-chain vs composed row-GEMV graph: bitwise on the
//! decode-MLP recipe (o + residual, normed up, normed silu-gate, down
//! + residual) and a couple of isolated shapes.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, ExecOp, GEMV_CHAIN_MAX_JOBS, MatmulCall, MatmulOp, Tensor,
};

fn mm<'a>(a: &'a Tensor, b: &'a Tensor, c: &'a Tensor) -> MatmulCall<'a> {
    MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }
}

fn upload_f16(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    rows: u32,
    cols: u32,
    seed: u64,
) -> (Tensor, Vec<f32>) {
    let mut host = vec![0.0; (rows * cols) as usize];
    fill_det(&mut host, seed);
    let t = Tensor::uninit_device_f16(ctx, &[rows, cols]).unwrap();
    exec.upload(&host, &t).unwrap();
    for v in &mut host {
        *v = round_f32_via_f16(*v);
    }
    (t, host)
}

#[test]
#[ignore]
fn gemv_chain_mlp_matches_composed_graph_bitwise() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.memory_model_device_scope_enabled || !ctx.f16_storage_enabled {
        eprintln!("skipping: no memory-model device scope or f16 storage");
        return;
    }
    let (k, n_o, n_ff) = (256u32, 256u32, 384u32);
    let (x, _) = upload_det(&ctx, &exec, &[1, k], 1);
    let (attn, _) = upload_det(&ctx, &exec, &[1, k], 2);
    let (wo, _) = upload_f16(&ctx, &exec, k, n_o, 3);
    let (w_up, _) = upload_f16(&ctx, &exec, n_o, n_ff, 4);
    let (w_gate, _) = upload_f16(&ctx, &exec, n_o, n_ff, 5);
    let (w_down, _) = upload_f16(&ctx, &exec, n_ff, n_o, 6);
    let (ffn_norm, _) = upload_det(&ctx, &exec, &[n_o], 7);

    let o = Tensor::uninit_device(&ctx, &[1, n_o]).unwrap();
    let up = Tensor::uninit_device(&ctx, &[1, n_ff]).unwrap();
    let gate = Tensor::uninit_device(&ctx, &[1, n_ff]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[1, n_o]).unwrap();
    let o2 = Tensor::uninit_device(&ctx, &[1, n_o]).unwrap();
    let up2 = Tensor::uninit_device(&ctx, &[1, n_ff]).unwrap();
    let gate2 = Tensor::uninit_device(&ctx, &[1, n_ff]).unwrap();
    let out2 = Tensor::uninit_device(&ctx, &[1, n_o]).unwrap();

    let o_proj = MatmulOp::with_epilogue(
        mm(&attn, &wo, &o),
        Epilogue {
            bias: None,
            activation: Activation::None,
            binary: EpilogueBinary::AddScaled { d: &x, beta: 1.0 },
        },
    );
    let up_op = MatmulOp::new(mm(&o, &w_up, &up)).with_normed_a(&ffn_norm, 1e-5);
    let gate_op = MatmulOp::with_epilogue(
        mm(&o, &w_gate, &gate),
        Epilogue {
            bias: None,
            activation: Activation::Silu,
            binary: EpilogueBinary::Mul { d: &up },
        },
    )
    .with_normed_a(&ffn_norm, 1e-5);
    let down_op = MatmulOp::with_epilogue(
        mm(&gate, &w_down, &out),
        Epilogue {
            bias: None,
            activation: Activation::None,
            binary: EpilogueBinary::AddScaled { d: &o, beta: 1.0 },
        },
    );

    let mut jobs = [o_proj; GEMV_CHAIN_MAX_JOBS];
    jobs[0] = o_proj;
    jobs[1] = up_op;
    jobs[2] = gate_op;
    jobs[3] = down_op;

    exec.run_exec_ops(&[
        ExecOp::Matmul(MatmulOp::with_epilogue(
            mm(&attn, &wo, &o2),
            Epilogue {
                bias: None,
                activation: Activation::None,
                binary: EpilogueBinary::AddScaled { d: &x, beta: 1.0 },
            },
        )),
        ExecOp::Matmul(MatmulOp::new(mm(&o2, &w_up, &up2)).with_normed_a(&ffn_norm, 1e-5)),
        ExecOp::Matmul(
            MatmulOp::with_epilogue(
                mm(&o2, &w_gate, &gate2),
                Epilogue {
                    bias: None,
                    activation: Activation::Silu,
                    binary: EpilogueBinary::Mul { d: &up2 },
                },
            )
            .with_normed_a(&ffn_norm, 1e-5),
        ),
        ExecOp::Matmul(MatmulOp::with_epilogue(
            mm(&gate2, &w_down, &out2),
            Epilogue {
                bias: None,
                activation: Activation::None,
                binary: EpilogueBinary::AddScaled { d: &o2, beta: 1.0 },
            },
        )),
    ])
    .unwrap();

    exec.run_exec_ops(&[ExecOp::GemvChain {
        jobs: Box::new(jobs),
        n: 4,
    }])
    .unwrap();

    let mut host = vec![0.0; n_o as usize];
    let mut host2 = vec![0.0; n_o as usize];
    exec.download(&out, &mut host).unwrap();
    exec.download(&out2, &mut host2).unwrap();
    assert_eq!(host, host2, "chain vs composed output diverged");
}

#[test]
#[ignore]
fn gemv_chain_single_job_matches_row_kernel() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.memory_model_device_scope_enabled || !ctx.f16_storage_enabled {
        eprintln!("skipping: no memory-model device scope or f16 storage");
        return;
    }
    let (k, n) = (128u32, 96u32);
    let (a, _) = upload_det(&ctx, &exec, &[1, k], 11);
    let (b, _) = upload_f16(&ctx, &exec, k, n, 12);
    let c_chain = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_ref = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let op_chain = MatmulOp::new(mm(&a, &b, &c_chain));
    let op_ref = MatmulOp::new(mm(&a, &b, &c_ref));
    let mut jobs = [op_chain; GEMV_CHAIN_MAX_JOBS];
    jobs[0] = op_chain;
    let mut host_a = vec![0.0; k as usize];
    exec.download(&a, &mut host_a).unwrap();
    let mut host_b = vec![0.0; (k * n) as usize];
    exec.download(&b, &mut host_b).unwrap();
    exec.run_gemv_chain(&jobs[..1]).unwrap();
    exec.run_ops(&[op_ref]).unwrap();
    let mut h1 = vec![0.0; n as usize];
    let mut h2 = vec![0.0; n as usize];
    exec.download(&c_chain, &mut h1).unwrap();
    exec.download(&c_ref, &mut h2).unwrap();
    assert_eq!(h1, h2, "single-job chain vs row kernel diverged");
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, 1, n, k, 1.0, false);
    assert_close(&h1, &cpu, k, "gemv_chain vs f64 CPU reference");
}
