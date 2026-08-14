use crate::common::*;

use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

#[test]
#[ignore]
fn prepared_matches_reference_and_replays_fresh_uploads() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (96_u32, 112_u32, 80_u32);
    let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let mut host_a = vec![0.0; (m * k) as usize];
    let mut host_b = vec![0.0; (k * n) as usize];
    fill_det(&mut host_a, 1201);
    fill_det(&mut host_b, 1202);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();

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
    let (error, index) = max_abs_err(&gpu, &cpu);
    assert!(
        error <= tolerance(k),
        "prepared error {error:.3e} at {index}"
    );

    // A second replay after re-uploading A sees the new contents —
    // the recorded command buffer points at memory, not values.
    fill_det(&mut host_a, 1203);
    exec.upload(&host_a, &a).unwrap();
    prepared.run().unwrap();
    exec.download(&c, &mut gpu).unwrap();
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    let (error, index) = max_abs_err(&gpu, &cpu);
    assert!(error <= tolerance(k), "replay error {error:.3e} at {index}");
}

#[test]
#[ignore]
fn prepared_graph_orders_dependent_chain() {
    let (ctx, exec) = make_setup(2, 8);
    let s = 64_u32;
    let a = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c1 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c2 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let mut host_a = vec![0.0; (s * s) as usize];
    let mut host_b = vec![0.0; (s * s) as usize];
    fill_det(&mut host_a, 1301);
    fill_det(&mut host_b, 1302);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();

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
    let (error, index) = max_abs_err(&gpu, &cpu_c2);
    // Two chained reductions accumulate two rounds of error.
    assert!(
        error <= 2.0 * tolerance(s),
        "prepared graph error {error:.3e} at {index}"
    );
}

#[test]
#[ignore]
fn prepared_replays_batched_broadcast_epilogue_op() {
    let (ctx, exec) = make_setup(2, 8);
    // B is broadcast across batches; bias + ReLU + residual exercise the
    // epilogue device addresses baked into the replayed command buffer.
    let (batch, m, n, k) = (3_u32, 21_u32, 45_u32, 33_u32);
    let a = Tensor::uninit_device(&ctx, &[batch, m, k]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[1, k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let bias = Tensor::uninit_device(&ctx, &[batch, n]).unwrap();
    let residual = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let mut host_a = vec![0.0; Tensor::numel(a.shape()) as usize];
    let mut host_b = vec![0.0; Tensor::numel(b.shape()) as usize];
    let mut host_bias = vec![0.0; Tensor::numel(bias.shape()) as usize];
    let mut host_residual = vec![0.0; Tensor::numel(residual.shape()) as usize];
    fill_det(&mut host_a, 1501);
    fill_det(&mut host_b, 1502);
    fill_det(&mut host_bias, 1503);
    fill_det(&mut host_residual, 1504);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();
    exec.upload(&host_bias, &bias).unwrap();
    exec.upload(&host_residual, &residual).unwrap();

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
    for batch_index in 0..batch as usize {
        for row in 0..m as usize {
            for col in 0..n as usize {
                let index = (batch_index * m as usize + row) * n as usize + col;
                let with_bias = expected[index] + host_bias[batch_index * n as usize + col];
                expected[index] = with_bias.max(0.0) + beta * host_residual[index];
            }
        }
    }
    let (error, index) = max_abs_err(&actual, &expected);
    assert!(
        error <= tolerance(k),
        "prepared epilogue error {error:.3e} at {index}"
    );
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
        let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
        let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
        let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
        let mut host_a = vec![0.0; (m * k) as usize];
        let mut host_b = vec![0.0; (k * n) as usize];
        fill_det(&mut host_a, 1601);
        fill_det(&mut host_b, 1602);
        exec.upload(&host_a, &a).unwrap();
        exec.upload(&host_b, &b).unwrap();
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
        let (error, index) = max_abs_err(&gpu, &cpu);
        assert!(
            error <= tolerance(k),
            "prepared deep-K DP error {error:.3e} at {index}"
        );
    }
}

#[test]
#[ignore]
fn prepared_pingpong_overlaps_and_rejects_misuse() {
    let (ctx, exec) = make_setup(2, 8);
    let s = 48_u32;
    let a = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c1 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let c2 = Tensor::uninit_device(&ctx, &[s, s]).unwrap();
    let mut host_a = vec![0.0; (s * s) as usize];
    let mut host_b = vec![0.0; (s * s) as usize];
    fill_det(&mut host_a, 1401);
    fill_det(&mut host_b, 1402);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();

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
        let (error, index) = max_abs_err(&gpu, &cpu);
        assert!(
            error <= tolerance(s),
            "pingpong error {error:.3e} at {index}"
        );
    }
}
