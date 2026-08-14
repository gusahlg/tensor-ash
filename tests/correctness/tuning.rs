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
    let (a, ha) = upload_det(&ctx, &exec, &[m, k], 91);
    let (b, hb) = upload_det(&ctx, &exec, &[k, n], 92);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();

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
    assert_close(&got, &expect, k, "tuned-path matmul");

    // Tuning the same shape again is a no-op (already recorded).
    exec.tune_shape(1, m, n, k).expect("tune_shape (repeat)");
}
