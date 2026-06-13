//! Correctness tests for the Stream-K SGEMM dispatch.
//!
//! Each test runs the same matmul through both the regular
//! `Executor::run_matmuls` path and the `Executor::run_matmuls_stream_k`
//! path, then diffs the outputs.
use crate::common::*;

use std::sync::Arc;
use tensor_ash::{Executor, MatmulCall, Tensor, VulkanContext};

/// Thin wrapper around `run_matmuls_stream_k` so the comparison helper
/// can take a `&[MatmulCall]` like the regular path does.  Stream-K
/// only dispatches one matmul per call, so we loop once per element.
fn streamk_dispatch(exec: &Executor, calls: &[MatmulCall<'_>]) {
    for c in calls {
        let call = MatmulCall {
            a: c.a,
            b: c.b,
            c: c.c,
            alpha: c.alpha,
            accumulate: c.accumulate,
        };
        exec.run_matmuls_stream_k(call).unwrap();
    }
}

/// Run the same `(A, B, C)` matmul through the regular `run_matmuls`
/// path and the Stream-K path, returning `(regular_out, streamk_out)`.
///
/// Both runs share identical, deterministically-filled inputs derived
/// from `seed_a` / `seed_b` so any divergence is attributable to the
/// dispatch path, not input drift.
fn run_regular_and_streamk(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    shape_a: &[u32],
    shape_b: &[u32],
    shape_c: &[u32],
    seed_a: u64,
    seed_b: u64,
) -> (Vec<f32>, Vec<f32>) {
    let na = Tensor::numel(shape_a) as usize;
    let nb = Tensor::numel(shape_b) as usize;
    let nc = Tensor::numel(shape_c) as usize;

    let mut h_a = vec![0.0f32; na];
    let mut h_b = vec![0.0f32; nb];
    fill_det(&mut h_a, seed_a);
    fill_det(&mut h_b, seed_b);

    // --- Regular dispatch ---
    let a_r = Tensor::uninit_device(ctx, shape_a).unwrap();
    let b_r = Tensor::uninit_device(ctx, shape_b).unwrap();
    let c_r = Tensor::uninit_device(ctx, shape_c).unwrap();
    exec.upload(&h_a, &a_r).unwrap();
    exec.upload(&h_b, &b_r).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a_r,
        b: &b_r,
        c: &c_r,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let mut regular_out = vec![0.0f32; nc];
    exec.download(&c_r, &mut regular_out).unwrap();

    // --- Stream-K dispatch ---
    let a_s = Tensor::uninit_device(ctx, shape_a).unwrap();
    let b_s = Tensor::uninit_device(ctx, shape_b).unwrap();
    let c_s = Tensor::uninit_device(ctx, shape_c).unwrap();
    exec.upload(&h_a, &a_s).unwrap();
    exec.upload(&h_b, &b_s).unwrap();
    streamk_dispatch(
        exec,
        &[MatmulCall {
            a: &a_s,
            b: &b_s,
            c: &c_s,
            alpha: 1.0,
            accumulate: false,
        }],
    );
    let mut streamk_out = vec![0.0f32; nc];
    exec.download(&c_s, &mut streamk_out).unwrap();

    (regular_out, streamk_out)
}

/// Tolerance is the same `8 * K * eps` bound used by `tolerance(k)` in
/// `cross_kernel_equivalence` / `shape_sweep_rank2`.  Both paths are
/// f32-accumulated SGEMM so the difference is governed by the same
/// rounding error envelope.
fn assert_match(regular: &[f32], streamk: &[f32], m: u32, n: u32, k: u32, label: &str) {
    let (e, idx) = max_abs_err(regular, streamk);
    let tol = tolerance(k);
    assert!(
        e <= tol,
        "{label} M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
         at idx {idx}: regular={:.6} streamk={:.6}",
        regular[idx],
        streamk[idx],
    );
}

fn run_case(m: u32, n: u32, k: u32, seed_a: u64, seed_b: u64, label: &str) {
    let (ctx, exec) = make_setup(2, 8);
    let (regular, streamk) =
        run_regular_and_streamk(&ctx, &exec, &[m, k], &[k, n], &[m, n], seed_a, seed_b);
    assert_match(&regular, &streamk, m, n, k, label);
}

#[test]
fn streamk_matches_regular_aligned_128() {
    run_case(128, 128, 128, 0x1001, 0x1002, "aligned_128");
}

#[test]
fn streamk_matches_regular_aligned_256() {
    run_case(256, 256, 256, 0x2001, 0x2002, "aligned_256");
}

#[test]
fn streamk_matches_regular_aligned_512() {
    run_case(512, 512, 512, 0x3001, 0x3002, "aligned_512");
}

#[test]
fn streamk_matches_regular_aligned_1024() {
    run_case(1024, 1024, 1024, 0x4001, 0x4002, "aligned_1024");
}

#[test]
fn streamk_matches_regular_aligned_2048() {
    run_case(2048, 2048, 2048, 0x5001, 0x5002, "aligned_2048");
}

#[test]
fn streamk_matches_regular_tall_256x1024x4096() {
    run_case(256, 1024, 4096, 0x6001, 0x6002, "tall_256x1024x4096");
}

#[test]
fn streamk_matches_regular_wide_1024x256x4096() {
    run_case(1024, 256, 4096, 0x7001, 0x7002, "wide_1024x256x4096");
}

#[test]
fn streamk_matches_regular_deep_k_256x256x8192() {
    run_case(256, 256, 8192, 0x8001, 0x8002, "deep_k_256x256x8192");
}

#[test]
fn streamk_matches_regular_aligned_non_square_1024x2048x512() {
    run_case(
        1024,
        2048,
        512,
        0x9001,
        0x9002,
        "aligned_non_square_1024x2048x512",
    );
}
