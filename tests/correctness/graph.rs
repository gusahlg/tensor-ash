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

    let a = Tensor::uninit_device(&ctx, &[m, k1]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[k1, k2]).unwrap();
    let t = Tensor::uninit_device(&ctx, &[m, k2]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[k2, n]).unwrap();
    let d = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    let mut ha = vec![0.0f32; (m * k1) as usize];
    let mut hb = vec![0.0f32; (k1 * k2) as usize];
    let mut hc = vec![0.0f32; (k2 * n) as usize];
    fill_det(&mut ha, 11);
    fill_det(&mut hb, 22);
    fill_det(&mut hc, 33);
    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();
    exec.upload(&hc, &c).unwrap();

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
    let (e, _) = max_abs_err(&got, &hd);
    // The second GEMM's operands are first-GEMM outputs with magnitude
    // ~sqrt(k1), so its absolute rounding error scales by that factor.
    let budget = tolerance(k1) + (k1 as f32).sqrt() * tolerance(k2);
    assert!(e <= budget, "chained graph err {e:.3e} > {budget:.3e}");
}

/// Diamond: two independent producers into one consumer.  The two
/// producer calls need no barrier between themselves; the consumer
/// needs one barrier that covers both.
#[test]
#[ignore]
fn diamond_dependency() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, k, h, n) = (64u32, 64u32, 96u32, 64u32);

    let x = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let w1 = Tensor::uninit_device(&ctx, &[k, h]).unwrap();
    let w2 = Tensor::uninit_device(&ctx, &[h, n]).unwrap();
    let u = Tensor::uninit_device(&ctx, &[m, h]).unwrap();
    let v = Tensor::uninit_device(&ctx, &[h, n]).unwrap();
    let out = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let w3 = Tensor::uninit_device(&ctx, &[h, h]).unwrap();

    let mut hx = vec![0.0f32; (m * k) as usize];
    let mut hw1 = vec![0.0f32; (k * h) as usize];
    let mut hw2 = vec![0.0f32; (h * n) as usize];
    let mut hw3 = vec![0.0f32; (h * h) as usize];
    fill_det(&mut hx, 1);
    fill_det(&mut hw1, 2);
    fill_det(&mut hw2, 3);
    fill_det(&mut hw3, 4);
    exec.upload(&hx, &x).unwrap();
    exec.upload(&hw1, &w1).unwrap();
    exec.upload(&hw2, &w2).unwrap();
    exec.upload(&hw3, &w3).unwrap();

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
    let (e, _) = max_abs_err(&got, &hout);
    // Consumer operands are producer outputs with magnitude ~sqrt(k)
    // and ~sqrt(h); scale the consumer stage's error budget accordingly.
    let budget = tolerance(k) + tolerance(h) + (k.max(h) as f32).sqrt() * tolerance(h);
    assert!(e <= budget, "diamond graph err {e:.3e} > {budget:.3e}");
}

/// Write-after-write + read-after-write on the same C: first call
/// writes C, second accumulates into it.
#[test]
#[ignore]
fn accumulate_into_prior_output() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (96u32, 64u32, 128u32);

    let a1 = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let b1 = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
    let a2 = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let b2 = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

    let mut ha1 = vec![0.0f32; (m * k) as usize];
    let mut hb1 = vec![0.0f32; (k * n) as usize];
    let mut ha2 = vec![0.0f32; (m * k) as usize];
    let mut hb2 = vec![0.0f32; (k * n) as usize];
    fill_det(&mut ha1, 5);
    fill_det(&mut hb1, 6);
    fill_det(&mut ha2, 7);
    fill_det(&mut hb2, 8);
    exec.upload(&ha1, &a1).unwrap();
    exec.upload(&hb1, &b1).unwrap();
    exec.upload(&ha2, &a2).unwrap();
    exec.upload(&hb2, &b2).unwrap();

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
    let (e, _) = max_abs_err(&got, &both);
    assert!(e <= 2.0 * tolerance(k), "accumulate graph err {e:.3e}");
}

/// A longer llama-style chain: qkv → down-projection style shapes, all
/// in one submission, compared against sequential run_matmuls calls.
#[test]
#[ignore]
fn graph_matches_sequential_submissions() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, h, i_) = (32u32, 256u32, 512u32);

    let x = Tensor::uninit_device(&ctx, &[m, h]).unwrap();
    let w_up = Tensor::uninit_device(&ctx, &[h, i_]).unwrap();
    let w_down = Tensor::uninit_device(&ctx, &[i_, h]).unwrap();
    let up_g = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();
    let out_g = Tensor::uninit_device(&ctx, &[m, h]).unwrap();
    let up_s = Tensor::uninit_device(&ctx, &[m, i_]).unwrap();
    let out_s = Tensor::uninit_device(&ctx, &[m, h]).unwrap();

    let mut hx = vec![0.0f32; (m * h) as usize];
    let mut hup = vec![0.0f32; (h * i_) as usize];
    let mut hdown = vec![0.0f32; (i_ * h) as usize];
    fill_det(&mut hx, 41);
    fill_det(&mut hup, 42);
    fill_det(&mut hdown, 43);
    exec.upload(&hx, &x).unwrap();
    exec.upload(&hup, &w_up).unwrap();
    exec.upload(&hdown, &w_down).unwrap();

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
