//! Correctness for the two-stage split-K path (scratch partials +
//! reduce, no atomics).

use crate::common::*;

use tensor_ash::{DeviceKind, KernelSelection, MatmulCall, Tensor};

fn run_splitk2_case(
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    num_splits: u32,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (ctx, exec) = make_setup(2, 8);
    let shape = |rows: u32, cols: u32| -> Vec<u32> {
        if batch == 1 {
            vec![rows, cols]
        } else {
            vec![batch, rows, cols]
        }
    };
    let (a, ha) = upload_det(&ctx, &exec, &shape(m, k), 101);
    let (b, hb) = upload_det(&ctx, &exec, &shape(k, n), 102);
    let c = Tensor::uninit_device(&ctx, &shape(m, n)).unwrap();

    exec.run_matmuls_split_k2(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha,
            accumulate: false,
        },
        num_splits,
    )
    .unwrap();

    let mut got = vec![0.0f32; (batch * m * n) as usize];
    exec.download(&c, &mut got).unwrap();
    let expect = cpu_bmm(&ha, &hb, None, batch, m, n, k, alpha, false);
    (got, expect)
}

#[test]
#[ignore]
fn splitk2_deep_k_tiny_mn() {
    let (got, expect) = run_splitk2_case(1, 64, 64, 8192, 0, 1.0);
    assert_close_tol(&got, &expect, 2.0 * tolerance(8192), "splitk2 64x64x8192");
}

#[test]
#[ignore]
fn auto_routes_deep_k_only_and_matches_cpu() {
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::Auto);
    if ctx.device_kind() != DeviceKind::DiscreteGpu || !ctx.buffer_device_address_enabled {
        return;
    }

    // Odd dimensions exercise stage-1 bounds and make a pre-existing tuned
    // store entry for this integration-only shape exceedingly unlikely.
    let (m, n, k) = (37, 41, 1088);
    let route = exec.dispatch_info(1, m, n, k);
    assert_eq!(
        route.split_k2_splits,
        Some(16),
        "unexpected route: {route:?}"
    );
    assert!(route.kernel.starts_with("split_k2_"));

    let (got, expect) = run_one(
        &ctx,
        &exec,
        &[m, k],
        &[k, n],
        &[m, n],
        0.75,
        false,
        131,
        137,
        None,
    );
    assert_close_tol(&got, &expect, 2.0 * tolerance(k), "auto split-K2");

    let dp = exec.dispatch_info(1, m, n, 1000);
    assert_eq!(dp.split_k2_splits, None, "unexpected route: {dp:?}");
    assert!(!dp.kernel.starts_with("split_k2_"));
}

#[test]
#[ignore]
fn splitk2_deep_k_mid_mn() {
    let (got, expect) = run_splitk2_case(1, 256, 256, 4096, 16, 1.0);
    assert_close_tol(&got, &expect, 2.0 * tolerance(4096), "splitk2 256x256x4096");
}

#[test]
#[ignore]
fn splitk2_odd_shape_scalar_reduce() {
    // mn % 4 != 0 exercises the reducer's scalar plane-crossing path;
    // 7 splits exercise the uneven K partition.
    let (got, expect) = run_splitk2_case(1, 101, 103, 999, 7, 1.0);
    assert_close_tol(&got, &expect, 2.0 * tolerance(999), "splitk2 odd shape");
}

#[test]
#[ignore]
fn splitk2_batched_with_alpha() {
    let (got, expect) = run_splitk2_case(3, 64, 64, 2048, 8, 0.25);
    assert_close_tol(&got, &expect, 2.0 * tolerance(2048), "splitk2 batched");
}

#[test]
#[ignore]
fn splitk2_is_deterministic() {
    // Unlike the atomic split-K, plane reduction has a fixed order —
    // two runs must be bit-identical.
    let (got1, _) = run_splitk2_case(1, 128, 128, 4096, 16, 1.0);
    let (got2, _) = run_splitk2_case(1, 128, 128, 4096, 16, 1.0);
    assert_eq!(got1, got2, "split-K2 results differ across runs");
}

#[test]
#[ignore]
fn splitk2_rejects_accumulate() {
    let (ctx, exec) = make_setup(2, 8);
    let a = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let err = exec
        .run_matmuls_split_k2(
            MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: true,
            },
            4,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("accumulate"), "unexpected error: {err}");
}

#[test]
#[ignore]
fn splitk_one_falls_back_and_rejects_splits_past_k() {
    let (ctx, exec) = make_setup(1, 4);
    let (a, ha) = upload_det(&ctx, &exec, &[16, 16], 211);
    let (b, hb) = upload_det(&ctx, &exec, &[16, 16], 223);
    let c = Tensor::uninit_device(&ctx, &[16, 16]).unwrap();
    let call = MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    };
    let expected = cpu_bmm(&ha, &hb, None, 1, 16, 16, 16, 1.0, false);

    exec.run_matmuls_split_k2(call, 1).unwrap();
    let mut got = vec![0.0; 16 * 16];
    exec.download(&c, &mut got).unwrap();
    assert_close(&got, &expected, 16, "splitk one fallback");

    let err = exec.run_matmuls_split_k2(call, 17).unwrap_err().to_string();
    assert!(err.contains("exceeds K=16"), "unexpected error: {err}");
}

/// A tuned deep-K shape inside a dependent graph must route through the
/// inline split-K2 path (when the probe recorded one) and still produce
/// correct results for the downstream consumer.
#[test]
#[ignore]
fn splitk2_inside_graph_after_tuning() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k, n2) = (64u32, 64u32, 4096u32, 96u32);

    // Records a winner for the shape; on deep-K low-tile shapes this
    // usually includes a splitk2 split count, which the graph recorder
    // then applies inline.  If the probe kept plain DP the test still
    // validates the graph path.
    exec.tune_shape(1, m, n, k).expect("tune_shape");

    let (a, ha) = upload_det(&ctx, &exec, &[m, k], 111);
    let (b, hb) = upload_det(&ctx, &exec, &[k, n], 112);
    let t = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let (w, hw) = upload_det(&ctx, &exec, &[n, n2], 113);
    let out = Tensor::uninit_device(&ctx, &[m, n2]).unwrap();

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
            b: &w,
            c: &out,
            alpha: 1.0,
            accumulate: false,
        },
    ])
    .unwrap();

    let mut got = vec![0.0f32; (m * n2) as usize];
    exec.download(&out, &mut got).unwrap();

    let ht = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
    let hout = cpu_bmm(&ht, &hw, None, 1, m, n2, n, 1.0, false);
    // Deep-K first stage: its outputs (magnitude ~sqrt(k)) feed the
    // second GEMM, scaling the consumer's absolute error.
    let budget = 2.0 * tolerance(k) + (k as f32).sqrt() * tolerance(n);
    assert_close_tol(&got, &hout, budget, "splitk2-in-graph");
}
