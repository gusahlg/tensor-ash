use crate::common::*;

use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

#[test]
#[ignore]
fn prepared_matches_reference_and_replays_fresh_uploads() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (96_u32, 112_u32, 80_u32);
    let (a, mut host_a) = upload_det(&ctx, &exec, &[m, k], 1201);
    let (b, host_b) = upload_det(&ctx, &exec, &[k, n], 1202);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    let mut prepared = exec
        .prepare_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap();

    // First replay matches the CPU reference.
    let stats = prepared.run().unwrap();
    assert_eq!(stats.n_calls, 1);
    let mut gpu = vec![0.0; (m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    assert_close(&gpu, &cpu, k, "prepared");

    // A second replay after re-uploading A sees the new contents —
    // the recorded command buffer points at memory, not values.
    fill_det(&mut host_a, 1203);
    exec.upload(&host_a, &a).unwrap();
    prepared.run().unwrap();
    exec.download(&c, &mut gpu).unwrap();
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    assert_close(&gpu, &cpu, k, "replay");
}

#[test]
#[ignore]
fn prepared_graph_orders_dependent_chain() {
    let (ctx, exec) = make_setup(2, 8);
    let s = 64_u32;
    let (a, host_a) = upload_det(&ctx, &exec, &[s, s], 1301);
    let (b, host_b) = upload_det(&ctx, &exec, &[s, s], 1302);
    let c1 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c2 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();

    let calls = [
        MatmulCall {
            a: &a,
            b: &b,
            c: &c1,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &c1,
            b: &b,
            c: &c2,
            alpha: 1.0,
            accumulate: false,
        },
    ];
    let ops: Vec<tensor_ash::MatmulOp<'_>> = calls
        .iter()
        .copied()
        .map(tensor_ash::MatmulOp::new)
        .collect();
    let mut prepared = exec.prepare_op_graph(&ops).unwrap();
    let stats = prepared.run().unwrap();
    assert_eq!(stats.n_calls, 2);

    let mut gpu = vec![0.0; (s * s) as usize];
    exec.download(&c2, &mut gpu).unwrap();
    let cpu_c1 = cpu_bmm(&host_a, &host_b, None, 1, s, s, s, 1.0, false);
    let cpu_c2 = cpu_bmm(&cpu_c1, &host_b, None, 1, s, s, s, 1.0, false);
    // Two chained reductions accumulate two rounds of error.
    assert_close_tol(&gpu, &cpu_c2, 2.0 * tolerance(s), "prepared graph");
}

#[test]
#[ignore]
fn prepared_replays_batched_broadcast_epilogue_op() {
    let (ctx, exec) = make_setup(2, 8);
    // B is broadcast across batches; bias + ReLU + residual exercise the
    // epilogue device addresses baked into the replayed command buffer.
    let (batch, m, n, k) = (3_u32, 21_u32, 45_u32, 33_u32);
    let (a, host_a) = upload_det(&ctx, &exec, &[batch, m, k], 1501);
    let (b, host_b) = upload_det(&ctx, &exec, &[1, k, n], 1502);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[batch, n], 1503);
    let (residual, host_residual) = upload_det(&ctx, &exec, &[batch, m, n], 1504);

    let beta = 0.25;
    let ops = [MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: Some(&bias),
            activation: Activation::Relu,
            binary: EpilogueBinary::AddScaled { d: &residual, beta },
        },
    )];
    let mut prepared = exec.prepare_ops(&ops).unwrap();
    // Replay twice; the second run overwrites the first identically.
    prepared.run().unwrap();
    prepared.run().unwrap();

    let mut actual = vec![0.0; Tensor::numel(c.shape()) as usize];
    exec.download(&c, &mut actual).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
    cpu_bias_relu_residual(&mut expected, &host_bias, &host_residual, batch, m, n, beta);
    assert_close(&actual, &expected, k, "prepared epilogue");
}

#[test]
#[ignore]
fn prepared_rejects_descriptor_kernels_and_forces_dp_on_deep_k() {
    // Explicit descriptor-bound selection cannot be prepared.
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::Large);
    let a = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let call = MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    };
    let err = match exec.prepare_matmuls(&[call]) {
        Ok(_) => panic!("descriptor kernel must not prepare"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("binds descriptor sets"),
        "unexpected error: {err}"
    );

    // A shape whose auto route is split-K2 still prepares (via the forced
    // data-parallel plan) and matches the CPU reference — this pins the
    // documented data-parallel-only contract of the prepared path.
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (37_u32, 41_u32, 1088_u32);
    if exec.dispatch_info(1, m, n, k).split_k2_splits.is_some() {
        let (a, host_a) = upload_det(&ctx, &exec, &[m, k], 1601);
        let (b, host_b) = upload_det(&ctx, &exec, &[k, n], 1602);
        let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
        let mut prepared = exec
            .prepare_matmuls(&[MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            }])
            .unwrap();
        prepared.run().unwrap();
        let mut gpu = vec![0.0; (m * n) as usize];
        exec.download(&c, &mut gpu).unwrap();
        let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
        assert_close(&gpu, &cpu, k, "prepared deep-K DP");
    }
}

#[test]
#[ignore]
fn prepared_pingpong_overlaps_and_rejects_misuse() {
    let (ctx, exec) = make_setup(2, 8);
    let s = 48_u32;
    let (a, host_a) = upload_det(&ctx, &exec, &[s, s], 1401);
    let (b, host_b) = upload_det(&ctx, &exec, &[s, s], 1402);
    let c1 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c2 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();

    let call = |c| MatmulCall {
        a: &a,
        b: &b,
        c,
        alpha: 1.0,
        accumulate: false,
    };
    let mut first = exec.prepare_matmuls(&[call(&c1)]).unwrap();
    let mut second = exec.prepare_matmuls(&[call(&c2)]).unwrap();

    // Misuse is rejected without touching the queue.  SAFETY: neither
    // object is leaked; both are waited below before any drop.
    assert!(first.wait().is_err(), "wait before submit must fail");
    unsafe { first.submit().unwrap() };
    assert!(
        unsafe { first.submit() }.is_err(),
        "double submit must fail"
    );

    // Ping-pong: both in flight, waited in order, repeatedly.
    unsafe { second.submit().unwrap() };
    for _ in 0..8 {
        first.wait().unwrap();
        unsafe { first.submit().unwrap() };
        second.wait().unwrap();
        unsafe { second.submit().unwrap() };
    }
    first.wait().unwrap();
    second.wait().unwrap();

    let cpu = cpu_bmm(&host_a, &host_b, None, 1, s, s, s, 1.0, false);
    for c in [&c1, &c2] {
        let mut gpu = vec![0.0; (s * s) as usize];
        exec.download(c, &mut gpu).unwrap();
        assert_close(&gpu, &cpu, s, "pingpong");
    }
}
