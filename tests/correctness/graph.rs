//! Correctness for `Executor::run_matmul_graph` — dependent chains in
//! one submission with automatic hazard barriers.

use crate::common::*;

use tensor_ash::{MatmulCall, Tensor};

/// D = (A@B)@C, chained in one submission.  The second call reads the
/// first call's output, so the graph recorder must emit a barrier
/// between them.
#[test]
#[ignore]
fn chained_two_stage() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, k1, k2, n) = (128u32, 96u32, 80u32, 64u32);

    let (a, ha) = upload_det(&ctx, &exec, &[m, k1], 11);
    let (b, hb) = upload_det(&ctx, &exec, &[k1, k2], 22);
    let t = Tensor::uninit_device(&ctx, &[m, k2]).unwrap();
    let (c, hc) = upload_det(&ctx, &exec, &[k2, n], 33);
    let d = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    exec.run_matmul_graph(&[
        MatmulCall {
            a: &a,
            b: &b,
            c: &t,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &t,
            b: &c,
            c: &d,
            alpha: 1.0,
            accumulate: false,
        },
    ])
    .unwrap();

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&d, &mut got).unwrap();

    let ht = cpu_bmm(&ha, &hb, None, 1, m, k2, k1, 1.0, false);
    let hd = cpu_bmm(&ht, &hc, None, 1, m, n, k2, 1.0, false);
    // The second GEMM's operands are first-GEMM outputs with magnitude
    // ~sqrt(k1), so its absolute rounding error scales by that factor.
    let budget = tolerance(k1) + (k1 as f32).sqrt() * tolerance(k2);
    assert_close_tol(&got, &hd, budget, "chained graph");
}

/// Diamond: two independent producers into one consumer.  The two
/// producer calls need no barrier between themselves; the consumer
/// needs one barrier that covers both.
#[test]
#[ignore]
fn diamond_dependency() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, k, h, n) = (64u32, 64u32, 96u32, 64u32);

    let (x, hx) = upload_det(&ctx, &exec, &[m, k], 1);
    let (w1, hw1) = upload_det(&ctx, &exec, &[k, h], 2);
    let (w2, hw2) = upload_det(&ctx, &exec, &[h, n], 3);
    let u = Tensor::uninit_device(&ctx, &[m, h]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[h, n]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let (w3, hw3) = upload_det(&ctx, &exec, &[h, h], 4);

    // u = x@w1  and  v = w3@w2  are independent; out = u@v depends on both.
    exec.run_matmul_graph(&[
        MatmulCall {
            a: &x,
            b: &w1,
            c: &u,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &w3,
            b: &w2,
            c: &v,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &u,
            b: &v,
            c: &out,
            alpha: 1.0,
            accumulate: false,
        },
    ])
    .unwrap();

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&out, &mut got).unwrap();

    let hu = cpu_bmm(&hx, &hw1, None, 1, m, h, k, 1.0, false);
    let hv = cpu_bmm(&hw3, &hw2, None, 1, h, n, h, 1.0, false);
    let hout = cpu_bmm(&hu, &hv, None, 1, m, n, h, 1.0, false);
    // Consumer operands are producer outputs with magnitude ~sqrt(k)
    // and ~sqrt(h); scale the consumer stage's error budget accordingly.
    let budget = tolerance(k) + tolerance(h) + (k.max(h) as f32).sqrt() * tolerance(h);
    assert_close_tol(&got, &hout, budget, "diamond graph");
}

/// Write-after-write + read-after-write on the same C: first call
/// writes C, second accumulates into it.
#[test]
#[ignore]
fn accumulate_into_prior_output() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (96u32, 64u32, 128u32);

    let (a1, ha1) = upload_det(&ctx, &exec, &[m, k], 5);
    let (b1, hb1) = upload_det(&ctx, &exec, &[k, n], 6);
    let (a2, ha2) = upload_det(&ctx, &exec, &[m, k], 7);
    let (b2, hb2) = upload_det(&ctx, &exec, &[k, n], 8);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    exec.run_matmul_graph(&[
        MatmulCall {
            a: &a1,
            b: &b1,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &a2,
            b: &b2,
            c: &c,
            alpha: 0.5,
            accumulate: true,
        },
    ])
    .unwrap();

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&c, &mut got).unwrap();

    let first = cpu_bmm(&ha1, &hb1, None, 1, m, n, k, 1.0, false);
    let both = cpu_bmm(&ha2, &hb2, Some(&first), 1, m, n, k, 0.5, true);
    assert_close_tol(&got, &both, 2.0 * tolerance(k), "accumulate graph");
}

/// A longer llama-style chain: qkv → down-projection style shapes, all
/// in one submission, compared against sequential run_matmuls calls.
#[test]
#[ignore]
fn graph_matches_sequential_submissions() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, h, i_) = (32u32, 256u32, 512u32);

    let (x, _) = upload_det(&ctx, &exec, &[m, h], 41);
    let (w_up, _) = upload_det(&ctx, &exec, &[h, i_], 42);
    let (w_down, _) = upload_det(&ctx, &exec, &[i_, h], 43);
    let up_g = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();
    let out_g = Tensor::uninit_device(&ctx, &[m, h]).unwrap();
    let up_s = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();
    let out_s = Tensor::uninit_device(&ctx, &[m, h]).unwrap();

    exec.run_matmul_graph(&[
        MatmulCall {
            a: &x,
            b: &w_up,
            c: &up_g,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &up_g,
            b: &w_down,
            c: &out_g,
            alpha: 1.0,
            accumulate: false,
        },
    ])
    .unwrap();

    exec.run_matmuls(&[MatmulCall {
        a: &x,
        b: &w_up,
        c: &up_s,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &up_s,
        b: &w_down,
        c: &out_s,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();

    let mut got_g = vec![0.0f32; (m * h) as usize];
    let mut got_s = vec![0.0f32; (m * h) as usize];
    exec.download(&out_g, &mut got_g).unwrap();
    exec.download(&out_s, &mut got_s).unwrap();

    // Same kernels, same order, same rounding — bit-identical.
    assert_eq!(got_g, got_s, "graph vs sequential mismatch");
}
