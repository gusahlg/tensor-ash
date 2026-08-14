use crate::common::*;

use tensor_ash::{MatmulCall, Tensor};

#[test]
#[ignore]
fn rejects_output_aliases_with_inputs() {
    let (ctx, exec) = make_setup(1, 1);
    let a = Tensor::uninit_device(&ctx, &[16, 16]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[16, 16]).unwrap();
    for call in [
        MatmulCall {
            a: &a,
            b: &b,
            c: &a,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &a,
            b: &b,
            c: &b,
            alpha: 1.0,
            accumulate: false,
        },
    ] {
        let err = exec.run_matmuls(&[call]).unwrap_err().to_string();
        assert!(err.contains("must not alias"), "unexpected error: {err}");
    }
}

#[test]
#[ignore]
fn alpha_and_accumulate() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (37u32, 41u32, 29u32);
    let mut c_init = vec![0.0f32; (m * n) as usize];
    fill_det(&mut c_init, 99);
    let alpha: f32 = 0.5;

    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[m, k],
        &[k, n],
        &[m, n],
        alpha,
        true,
        51,
        53,
        Some(&c_init),
    );
    assert_close_tol(
        &gpu,
        &cpu,
        tolerance(k) + alpha.abs() * 1e-5,
        "alpha+accumulate",
    );
}

#[test]
#[ignore]
fn identity_matmul() {
    let (ctx, exec) = make_setup(2, 4);
    let n = 64u32;
    let mut id = vec![0.0f32; (n * n) as usize];
    for i in 0..n as usize {
        id[i * n as usize + i] = 1.0;
    }
    let a = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    exec.upload(&id, &a).unwrap();
    let (b, x) = upload_det(&ctx, &exec, &[n, n], 71);
    let c = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();

    let (e, _) = max_abs_err(&got, &x);
    assert!(e == 0.0, "identity matmul should be exact, err={e}");
}

#[test]
#[ignore]
fn nan_input_propagates_to_output() {
    let (ctx, exec) = make_setup(1, 1);
    let n = 32u32;
    let mut ha = vec![1.0f32; (n * n) as usize];
    let hb = vec![1.0f32; (n * n) as usize];
    ha[0] = f32::NAN;
    let a = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();

    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();
    assert!(
        got.iter().any(|v| v.is_nan()),
        "expected at least one NaN in output, got none"
    );
}

#[test]
#[ignore]
fn zero_inputs_produce_zeros() {
    let (ctx, exec) = make_setup(1, 1);
    let n = 96u32;
    let ha = vec![0.0f32; (n * n) as usize];
    let hb = vec![0.0f32; (n * n) as usize];
    let a = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();
    assert!(
        got.iter().all(|&v| v == 0.0),
        "expected all-zero output, found non-zero values"
    );
}

#[test]
#[ignore]
fn alpha_zero_zeros_output_without_accumulate() {
    let (ctx, exec) = make_setup(1, 1);
    let n = 64u32;
    let (a, _) = upload_det(&ctx, &exec, &[n, n], 31);
    let (b, _) = upload_det(&ctx, &exec, &[n, n], 32);
    let (c, _) = upload_det(&ctx, &exec, &[n, n], 33);
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 0.0,
        accumulate: false,
    }])
    .unwrap();

    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();
    assert!(
        got.iter().all(|&v| v == 0.0),
        "expected all-zero output with alpha=0, sample={:.6}",
        got[0]
    );
}

#[test]
#[ignore]
fn large_k_remains_correct() {
    let (ctx, exec) = make_setup(2, 4);
    let (m, n, k) = (64u32, 64u32, 1023u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[m, k],
        &[k, n],
        &[m, n],
        1.0,
        false,
        77,
        78,
        None,
    );
    assert_close(&gpu, &cpu, k, "large K");
}

#[test]
#[ignore]
fn tensor_accessors_match_constructor_input() {
    let (ctx, _exec) = make_setup(1, 1);
    let t = Tensor::uninit_device(&ctx, &[4, 8, 12]).unwrap();
    assert_eq!(t.shape(), &[4, 8, 12]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.len(), 384);
    assert!(!t.is_empty());
    assert_eq!(t.size_bytes(), 384 * 4);
    let (b, m, k) = t.as_3d().unwrap();
    assert_eq!((b, m, k), (4, 8, 12));
}
