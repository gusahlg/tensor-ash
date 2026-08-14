//! f16-storage-B (half-precision weights) correctness.
//!
//! The CPU reference rounds B through f16 exactly like the upload path,
//! so the GPU result differs only by f32 summation order and the usual
//! `tolerance(k)` applies.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

fn f16_available(ctx: &std::sync::Arc<tensor_ash::VulkanContext>) -> bool {
    ctx.f16_storage_enabled && ctx.buffer_device_address_enabled
}

/// Upload helper: fill A (f32) and B (f16) deterministically, return
/// the host copies with B already rounded to what the GPU stores.
fn setup_f16_case(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    a_shape: &[u32],
    b_shape: &[u32],
    seed_a: u64,
    seed_b: u64,
) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let a = Tensor::uninit_device(ctx, a_shape).unwrap();
    let b = Tensor::uninit_device_f16(ctx, b_shape).unwrap();
    let mut host_a = vec![0.0; Tensor::numel(a_shape) as usize];
    let mut host_b = vec![0.0; Tensor::numel(b_shape) as usize];
    fill_det(&mut host_a, seed_a);
    fill_det(&mut host_b, seed_b);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();
    for value in &mut host_b {
        *value = round_f32_via_f16(*value);
    }
    (a, b, host_a, host_b)
}

#[test]
#[ignore]
fn f16_upload_download_roundtrip() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.f16_storage_enabled {
        eprintln!("skipping: no f16 storage support");
        return;
    }
    let t = Tensor::uninit_device_f16(&ctx, &[37, 53]).unwrap();
    let mut host = vec![0.0; 37 * 53];
    fill_det(&mut host, 4242);
    exec.upload(&host, &t).unwrap();
    let mut back = vec![0.0; 37 * 53];
    exec.download(&t, &mut back).unwrap();
    for (i, (&sent, &got)) in host.iter().zip(&back).enumerate() {
        assert_eq!(
            round_f32_via_f16(sent).to_bits(),
            got.to_bits(),
            "element {i}: sent {sent}, got {got}"
        );
    }
}

#[test]
#[ignore]
fn f16_weights_match_rounded_reference_across_routes() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    // Shapes chosen to hit every f16w tile class: large (512^3),
    // k64-class (384), small/batched, interior and ragged edges.
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 512, 512, 512),
        (1, 384, 384, 384),
        (1, 127, 129, 65),
        (1, 64, 64, 64),
        (4, 96, 112, 80),
        (1, 1023, 1025, 127),
    ];
    for &(batch, m, n, k) in cases {
        let shape = |rows: u32, cols: u32| -> Vec<u32> {
            if batch == 1 {
                vec![rows, cols]
            } else {
                vec![batch, rows, cols]
            }
        };
        let (a, b, host_a, host_b) = setup_f16_case(
            &ctx,
            &exec,
            &shape(m, k),
            &shape(k, n),
            9000 + k as u64,
            9100 + n as u64,
        );
        let c = Tensor::uninit_device(&ctx, &shape(m, n)).unwrap();
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap();
        let mut gpu = vec![0.0; Tensor::numel(&shape(m, n)) as usize];
        exec.download(&c, &mut gpu).unwrap();
        let cpu = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
        let (error, index) = max_abs_err(&gpu, &cpu);
        assert!(
            error <= tolerance(k),
            "B={batch} {m}x{n}x{k}: f16w error {error:.3e} at {index}"
        );
    }
}

#[test]
#[ignore]
fn f16_routes_pick_f16w_kernels_and_skip_splitk2() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    let large = exec.dispatch_info_for(1, 2048, 2048, 2048, true);
    assert_eq!(large.kernel, "f16w_large_bda_v4", "large route: {large:?}");
    // 1024^3 mirrors its f32 route class (m128n64k64).
    let mid = exec.dispatch_info_for(1, 1024, 1024, 1024, true);
    assert_eq!(mid.kernel, "f16w_m128n64k64_bda_v4", "mid route: {mid:?}");
    let row = exec.dispatch_info_for(1, 1, 4096, 4096, true);
    assert_eq!(row.kernel, "f16w_row_bda", "row route: {row:?}");
    // Deep-K would tempt split-K2 on f32; f16 must stay data-parallel
    // on an f16w kernel.
    let deep = exec.dispatch_info_for(1, 37, 41, 4096, true);
    assert_eq!(deep.split_k2_splits, None, "deep-K route: {deep:?}");
    assert!(deep.kernel.starts_with("f16w_"), "deep-K kernel: {deep:?}");
    // And the same shapes with f32 B keep their f32 routes.
    let f32_large = exec.dispatch_info_for(1, 1024, 1024, 1024, false);
    assert!(!f32_large.kernel.starts_with("f16w_"), "{f32_large:?}");
}

#[test]
#[ignore]
fn f16_row_gemv_and_batched_epilogue() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    // Lone-row decode-style GEMV.
    let (m, n, k) = (1_u32, 2048_u32, 1024_u32);
    let (a, b, host_a, host_b) = setup_f16_case(&ctx, &exec, &[m, k], &[k, n], 9500, 9501);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    let mut gpu = vec![0.0; (m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    let (error, index) = max_abs_err(&gpu, &cpu);
    assert!(error <= tolerance(k), "gemv error {error:.3e} at {index}");

    // Batched broadcast-B with bias + SiLU + scaled residual: the
    // epilogue operands stay f32 while B is f16.
    let (batch, m, n, k) = (3_u32, 33_u32, 47_u32, 65_u32);
    let (a, b, host_a, host_b) =
        setup_f16_case(&ctx, &exec, &[batch, m, k], &[1, k, n], 9600, 9601);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let bias = Tensor::uninit_device(&ctx, &[n]).unwrap();
    let residual = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let mut host_bias = vec![0.0; n as usize];
    let mut host_residual = vec![0.0; (batch * m * n) as usize];
    fill_det(&mut host_bias, 9602);
    fill_det(&mut host_residual, 9603);
    exec.upload(&host_bias, &bias).unwrap();
    exec.upload(&host_residual, &residual).unwrap();
    let beta = 0.5;
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: Some(&bias),
            activation: Activation::Silu,
            binary: EpilogueBinary::AddScaled { d: &residual, beta },
        },
    )])
    .unwrap();
    let mut gpu = vec![0.0; (batch * m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let col = index % n as usize;
        let with_bias = *value + host_bias[col];
        let act = with_bias / (1.0 + (-with_bias).exp());
        *value = act + beta * host_residual[index];
    }
    let (error, index) = max_abs_err(&gpu, &expected);
    assert!(
        error <= tolerance(k),
        "f16w epilogue error {error:.3e} at {index}"
    );
}

#[test]
#[ignore]
fn f16_b_with_explicit_f32_kernel_is_rejected() {
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::LargeBdaV4);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    let a = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let b = Tensor::uninit_device_f16(&ctx, &[64, 64]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("expects f32 B storage"),
        "unexpected error: {err}"
    );

    // And A/C in f16 are rejected outright.
    let a16 = Tensor::uninit_device_f16(&ctx, &[64, 64]).unwrap();
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a16,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(err.contains("A must be f32"), "unexpected error: {err}");
}
