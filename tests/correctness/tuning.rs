//! Measured-tuning path: explicit `tune_shape` warmup + correctness of
//! results computed through a tuned winner.

use crate::common::*;

use tensor_ash::{MatmulCall, Tensor};

#[test]
#[ignore]
fn tune_shape_then_compute_is_correct() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (320u32, 192u32, 448u32);

    // Explicit warmup: measures all candidates on scratch tensors and
    // records a winner for the shape (also persists it; a stale entry
    // for this odd shape is harmless).
    exec.tune_shape(1, m, n, k).expect("tune_shape");

    // A real matmul on the tuned shape must still be correct through
    // whatever kernel won.
    let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let mut ha = vec![0.0f32; (m * k) as usize];
    let mut hb = vec![0.0f32; (k * n) as usize];
    fill_det(&mut ha, 91);
    fill_det(&mut hb, 92);
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

    let mut got = vec![0.0f32; (m * n) as usize];
    exec.download(&c, &mut got).unwrap();
    let expect = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
    let (e, _) = max_abs_err(&got, &expect);
    assert!(e <= tolerance(k), "tuned-path matmul err {e:.3e}");

    // Tuning the same shape again is a no-op (already recorded).
    exec.tune_shape(1, m, n, k).expect("tune_shape (repeat)");
}
