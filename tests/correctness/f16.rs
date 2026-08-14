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
    let (a, host_a) = upload_det(ctx, exec, a_shape, seed_a);
    let b = Tensor::uninit_device_f16(ctx, b_shape).unwrap();
    let mut host_b = vec![0.0; Tensor::numel(b_shape) as usize];
    fill_det(&mut host_b, seed_b);
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
        let (a, b, mut host_a, host_b) = setup_f16_case(
            &ctx,
            &exec,
            &shape(m, k),
            &shape(k, n),
            9000 + k as u64,
            9100 + n as u64,
        );
        // The tensor-core route quantizes A to f16 while staging;
        // mirror it in the reference so products stay exact.
        if exec
            .dispatch_info_for(batch, m, n, k, true)
            .kernel
            .contains("coopmat")
        {
            for value in &mut host_a {
                *value = round_f32_via_f16(*value);
            }
            exec.upload(&host_a, &a).unwrap();
        }
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
        assert_close(&gpu, &cpu, k, &format!("B={batch} {m}x{n}x{k} f16w"));
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
    // Big aligned shapes take the tensor cores when available; the
    // SIMT large tile handles the unaligned siblings.
    let large = exec.dispatch_info_for(1, 2048, 2048, 2048, true);
    if ctx.coopmat_enabled {
        assert_eq!(
            large.kernel, "f16w_coopmat_aligned",
            "large route: {large:?}"
        );
    } else {
        assert_eq!(large.kernel, "f16w_large_bda_v4", "large route: {large:?}");
    }
    let unaligned = exec.dispatch_info_for(1, 2048, 2040, 2048, true);
    assert!(
        unaligned.kernel.starts_with("f16w_") && !unaligned.kernel.contains("coopmat"),
        "unaligned route: {unaligned:?}"
    );
    // 1024^3: tensor cores when available, else the f32 route class
    // mirror (m128n64k64).
    let mid = exec.dispatch_info_for(1, 1024, 1024, 1024, true);
    if ctx.coopmat_enabled {
        assert_eq!(mid.kernel, "f16w_coopmat_aligned", "mid route: {mid:?}");
    } else {
        assert_eq!(mid.kernel, "f16w_m128n64k64_bda_v4", "mid route: {mid:?}");
    }
    let row = exec.dispatch_info_for(1, 1, 4096, 4096, true);
    assert_eq!(row.kernel, "f16w_row_bda_k16", "row route: {row:?}");
    let row_shallow = exec.dispatch_info_for(1, 1, 4096, 2048, true);
    assert_eq!(
        row_shallow.kernel, "f16w_row_bda",
        "row route: {row_shallow:?}"
    );
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
    assert_close(&gpu, &cpu, k, "gemv");

    // Batched broadcast-B with bias + SiLU + scaled residual: the
    // epilogue operands stay f32 while B is f16.
    let (batch, m, n, k) = (3_u32, 33_u32, 47_u32, 65_u32);
    let (a, b, host_a, host_b) =
        setup_f16_case(&ctx, &exec, &[batch, m, k], &[1, k, n], 9600, 9601);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 9602);
    let (residual, host_residual) = upload_det(&ctx, &exec, &[batch, m, n], 9603);
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
    assert_close(&gpu, &expected, k, "f16w epilogue");
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

#[test]
#[ignore]
fn coopmat_routes_and_matches_dual_rounded_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Aligned + big => tensor cores; small or ragged stays SIMT.
    let big = exec.dispatch_info_for(1, 1024, 1024, 1024, true);
    assert_eq!(big.kernel, "f16w_coopmat_aligned", "{big:?}");
    let small = exec.dispatch_info_for(1, 128, 128, 128, true);
    assert_ne!(small.kernel, "f16w_coopmat_aligned", "{small:?}");
    let ragged = exec.dispatch_info_for(1, 1024, 1000, 1024, true);
    assert_ne!(ragged.kernel, "f16w_coopmat_aligned", "{ragged:?}");

    // Batched + alpha + accumulate against a dual-rounded reference:
    // f16 x f16 products are exact in f32, so tolerance(k) holds.
    let (batch, m, n, k) = (2_u32, 256_u32, 384_u32, 256_u32);
    let (a, b, mut host_a, host_b) =
        setup_f16_case(&ctx, &exec, &[batch, m, k], &[batch, k, n], 9700, 9701);
    for value in &mut host_a {
        *value = round_f32_via_f16(*value);
    }
    exec.upload(&host_a, &a).unwrap();
    let (c, host_c) = upload_det(&ctx, &exec, &[batch, m, n], 9702);
    let alpha = 0.75;
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha,
        accumulate: true,
    }])
    .unwrap();
    let mut gpu = vec![0.0; (batch * m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let cpu = cpu_bmm(&host_a, &host_b, Some(&host_c), batch, m, n, k, alpha, true);
    assert_close(&gpu, &cpu, k, "coopmat");
}

/// Regression: TinyLlama prefill surfaced that aligned f16 shapes
/// route to the coopmat kernel, which cannot fuse epilogues — fused
/// ops must demote to the SIMT sibling while plain ops keep the
/// tensor cores, and explicit selections keep their loud failure.
#[test]
#[ignore]
fn f16_epilogue_on_coopmat_shape_demotes_and_matches() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Plain route for this aligned shape is the tensor-core kernel.
    let (m, n, k) = (256_u32, 384_u32, 256_u32);
    let plain = exec.dispatch_info_for(1, m, n, k, true);
    assert_eq!(plain.kernel, "f16w_coopmat_aligned", "{plain:?}");

    // A fused bias+SiLU+gate op on the same shape must run (demoted)
    // and match the reference computed from f16-rounded inputs.
    let (a, b, mut host_a, host_b) = setup_f16_case(&ctx, &exec, &[m, k], &[k, n], 9800, 9801);
    // The demoted SIMT kernel reads A as f32 (no rounding); keep the
    // reference on the raw A.
    let _ = &mut host_a;
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 9802);
    let (gate, host_gate) = upload_det(&ctx, &exec, &[m, n], 9803);
    exec.run_ops(&[tensor_ash::MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        tensor_ash::Epilogue {
            bias: Some(&bias),
            activation: tensor_ash::Activation::Silu,
            binary: tensor_ash::EpilogueBinary::Mul { d: &gate },
        },
    )])
    .unwrap();
    let mut gpu = vec![0.0; (m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let with_bias = *value + host_bias[index % n as usize];
        let act = with_bias / (1.0 + (-with_bias).exp());
        *value = act * host_gate[index];
    }
    assert_close(&gpu, &expected, k, "demoted epilogue on coopmat shape");
}
