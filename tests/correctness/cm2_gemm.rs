//! CM2 (NV_cooperative_matrix2) tensor-core GEMM correctness.
//!
//! The `f16w_cm2` kernel quantizes A to f16 in its tensor-load decode
//! callback, so every CPU reference here rounds BOTH operands through
//! f16 (the coopmat1 dual-rounded idiom from `f16.rs`); products are
//! then exact in f32 and the usual `tolerance(k)` applies.
//!
//! Unlike `f16w_coopmat_aligned` the CM2 body is general-shape correct
//! (tensor layouts clamp loads to 0.0 and drop out-of-range stores),
//! so ragged batch=1 shapes are exercised here via explicit selection
//! even though auto routing keeps the kernel on coopmat-aligned
//! shapes.  Batched cases stay aligned: batched tensor base addresses
//! must be 16-byte aligned per the tensor-addressing spec.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

/// Probe device support with a throwaway auto-routing context; an
/// explicit `F16wCm2` selection fails pipeline creation outright when
/// the registry slot is gated off.
fn cm2_ready() -> bool {
    let (ctx, _exec) = make_setup(1, 4);
    let ready =
        ctx.coopmat2_enabled && ctx.f16_storage_enabled && ctx.buffer_device_address_enabled;
    if !ready {
        eprintln!("skipping: no coopmat2 support");
    }
    ready
}

/// Upload A (f32, pre-rounded through f16 to mirror the kernel's load
/// decode) and B (f16); return host copies matching what the GPU sees.
fn setup_cm2_case(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    a_shape: &[u32],
    b_shape: &[u32],
    seed_a: u64,
    seed_b: u64,
) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let (a, mut host_a) = upload_det(ctx, exec, a_shape, seed_a);
    for value in &mut host_a {
        *value = round_f32_via_f16(*value);
    }
    exec.upload(&host_a, &a).unwrap();
    let b = Tensor::uninit_device_f16(ctx, b_shape).unwrap();
    let mut host_b = vec![0.0; Tensor::numel(b_shape) as usize];
    fill_det(&mut host_b, seed_b);
    exec.upload(&host_b, &b).unwrap();
    for value in &mut host_b {
        *value = round_f32_via_f16(*value);
    }
    (a, b, host_a, host_b)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu_tanh(x: f32) -> f32 {
    const K0: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (K0 * (0.044715 * x * x).mul_add(x, x)).tanh())
}

#[test]
#[ignore]
fn cm2_plain_matches_dual_rounded_reference() {
    if !cm2_ready() {
        return;
    }
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::F16wCm2);
    // Aligned tiles, a K % BK != 0 tail (96 = 64 + 32), fully ragged
    // M/N/K at batch=1 (clamped edges), a single tile, and an aligned
    // batched case.
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 256, 256, 256),
        (1, 384, 512, 96),
        (1, 257, 130, 100),
        (1, 128, 64, 64),
        (2, 256, 128, 64),
    ];
    for &(batch, m, n, k) in cases {
        let shape = |rows: u32, cols: u32| -> Vec<u32> {
            if batch == 1 {
                vec![rows, cols]
            } else {
                vec![batch, rows, cols]
            }
        };
        let (a, b, host_a, host_b) = setup_cm2_case(
            &ctx,
            &exec,
            &shape(m, k),
            &shape(k, n),
            11_000 + k as u64,
            11_100 + n as u64,
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
        assert_close(&gpu, &cpu, k, &format!("B={batch} {m}x{n}x{k} cm2 plain"));
    }

    // Alpha + accumulate: the scale applies before the prior-C add,
    // and the prior C rides a bounds-clamped tensor load.
    let (batch, m, n, k) = (2_u32, 256_u32, 128_u32, 96_u32);
    let (a, b, host_a, host_b) =
        setup_cm2_case(&ctx, &exec, &[batch, m, k], &[batch, k, n], 11_200, 11_201);
    let (c, host_c) = upload_det(&ctx, &exec, &[batch, m, n], 11_202);
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
    assert_close(&gpu, &cpu, k, "cm2 alpha+accumulate");
}

/// Every epilogue form the CM2 store callback implements, each checked
/// against the composed reference (unfused matmul + host-side epilogue
/// math in the shader's order: bias -> activation -> binary).
#[test]
#[ignore]
fn cm2_epilogue_forms_match_composed_reference() {
    if !cm2_ready() {
        return;
    }
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::F16wCm2);
    // One aligned and one ragged shape; the ragged one proves the
    // per-element callback's clamped bias/D reads stay in bounds while
    // the clamped store drops the edge lanes.
    for &(m, n, k) in &[(256_u32, 384_u32, 128_u32), (200, 100, 96)] {
        let (a, b, host_a, host_b) = setup_cm2_case(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            12_000 + m as u64,
            12_100 + n as u64,
        );
        let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 12_200);
        let (d, host_d) = upload_det(&ctx, &exec, &[m, n], 12_201);
        // AddScaled cases use beta = 0.5, baked into the fn-pointer
        // references below (they cannot capture).
        let plain = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);

        type Reference = fn(f32, f32, f32) -> f32;
        let cases: &[(&str, Epilogue<'_>, Reference)] = &[
            (
                "bias",
                Epilogue {
                    bias: Some(&bias),
                    ..Epilogue::NONE
                },
                |v, bias, _d| v + bias,
            ),
            (
                "relu",
                Epilogue {
                    activation: Activation::Relu,
                    ..Epilogue::NONE
                },
                |v, _bias, _d| v.max(0.0),
            ),
            (
                "silu",
                Epilogue {
                    activation: Activation::Silu,
                    ..Epilogue::NONE
                },
                |v, _bias, _d| silu(v),
            ),
            (
                "gelu",
                Epilogue {
                    activation: Activation::Gelu,
                    ..Epilogue::NONE
                },
                |v, _bias, _d| gelu_tanh(v),
            ),
            (
                "add_scaled",
                Epilogue {
                    binary: EpilogueBinary::AddScaled { d: &d, beta: 0.5 },
                    ..Epilogue::NONE
                },
                |v, _bias, d| v + 0.5 * d,
            ),
            (
                "mul",
                Epilogue {
                    binary: EpilogueBinary::Mul { d: &d },
                    ..Epilogue::NONE
                },
                |v, _bias, d| v * d,
            ),
            (
                "silu_mul",
                Epilogue {
                    activation: Activation::Silu,
                    binary: EpilogueBinary::Mul { d: &d },
                    ..Epilogue::NONE
                },
                |v, _bias, d| silu(v) * d,
            ),
            (
                "bias_silu_add_scaled",
                Epilogue {
                    bias: Some(&bias),
                    activation: Activation::Silu,
                    binary: EpilogueBinary::AddScaled { d: &d, beta: 0.5 },
                },
                |v, bias, d| silu(v + bias) + 0.5 * d,
            ),
        ];
        for (label, epilogue, reference) in cases {
            let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
            exec.run_ops(&[MatmulOp::with_epilogue(
                MatmulCall {
                    a: &a,
                    b: &b,
                    c: &c,
                    alpha: 1.0,
                    accumulate: false,
                },
                *epilogue,
            )])
            .unwrap();
            let mut gpu = vec![0.0; (m * n) as usize];
            exec.download(&c, &mut gpu).unwrap();
            let expected: Vec<f32> = plain
                .iter()
                .enumerate()
                .map(|(index, &v)| reference(v, host_bias[index % n as usize], host_d[index]))
                .collect();
            assert_close(&gpu, &expected, k, &format!("cm2 {m}x{n}x{k} {label}"));
        }
    }

    // Batched bias `[B, N]` (bias_batch_stride path) with SiLU + gate.
    let (batch, m, n, k) = (2_u32, 128_u32, 128_u32, 64_u32);
    let (a, b, host_a, host_b) =
        setup_cm2_case(&ctx, &exec, &[batch, m, k], &[batch, k, n], 12_300, 12_301);
    let (bias, host_bias) = upload_det(&ctx, &exec, &[batch, n], 12_302);
    let (gate, host_gate) = upload_det(&ctx, &exec, &[batch, m, n], 12_303);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
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
            binary: EpilogueBinary::Mul { d: &gate },
        },
    )])
    .unwrap();
    let mut gpu = vec![0.0; (batch * m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let b_idx = index as u32 / (m * n);
        let col = index as u32 % n;
        let with_bias = *value + host_bias[(b_idx * n + col) as usize];
        *value = silu(with_bias) * host_gate[index];
    }
    assert_close(&gpu, &expected, k, "cm2 batched bias epilogue");
}

/// The un-demote itself: on a device with coopmat2, a fused-epilogue
/// op on a coopmat-eligible shape must produce output matching the
/// CM2 (f16-quantized A) reference while the PLAIN route for the same
/// shape stays on the measured coopmat1 kernel.
#[test]
#[ignore]
fn cm2_epilogue_on_coopmat_shape_stays_on_tensor_cores() {
    let (ctx, exec) = make_setup(2, 8);
    if !(ctx.coopmat2_enabled
        && ctx.coopmat_enabled
        && ctx.f16_storage_enabled
        && ctx.buffer_device_address_enabled)
    {
        eprintln!("skipping: no coopmat2 support");
        return;
    }
    // Plain routes are untouched: coopmat1 keeps the eligible shape.
    let plain = exec.dispatch_info_for(1, 512, 512, 512, true);
    assert_eq!(plain.kernel, "f16w_coopmat_m64n64", "{plain:?}");

    // Fused op on an eligible shape: rides the CM2 GEMM (A quantized
    // to f16 at load — the reference mirrors it) instead of demoting.
    let (m, n, k) = (256_u32, 384_u32, 256_u32);
    let (a, b, host_a, host_b) = setup_cm2_case(&ctx, &exec, &[m, k], &[k, n], 13_000, 13_001);
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 13_002);
    let (residual, host_residual) = upload_det(&ctx, &exec, &[m, n], 13_003);
    let beta = 1.0_f32;
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
    let mut gpu = vec![0.0; (m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let with_bias = *value + host_bias[index % n as usize];
        *value = silu(with_bias) + beta * host_residual[index];
    }
    assert_close(&gpu, &expected, k, "cm2 un-demoted epilogue");
}

/// Poisoned-output coverage: C prefilled with NaN, non-accumulating
/// dispatches (plain and with an epilogue reading a clean D) must
/// overwrite every element without ever reading C — on a ragged shape
/// whose edge tiles exercise the clamped store.  A NaN leak would fail
/// `assert_close` loudly.
#[test]
#[ignore]
fn cm2_poisoned_c_is_fully_overwritten() {
    if !cm2_ready() {
        return;
    }
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::F16wCm2);
    let (m, n, k) = (257_u32, 130_u32, 100_u32);
    let (a, b, host_a, host_b) = setup_cm2_case(&ctx, &exec, &[m, k], &[k, n], 14_000, 14_001);
    let (d, host_d) = upload_det(&ctx, &exec, &[m, n], 14_002);
    let poison = vec![f32::NAN; (m * n) as usize];
    let plain = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);

    // Plain store.
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    exec.upload(&poison, &c).unwrap();
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
    assert_close(&gpu, &plain, k, "cm2 poisoned C plain");

    // Epilogue store (per-element callback path).
    exec.upload(&poison, &c).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            activation: Activation::Silu,
            binary: EpilogueBinary::AddScaled { d: &d, beta: 0.5 },
            ..Epilogue::NONE
        },
    )])
    .unwrap();
    exec.download(&c, &mut gpu).unwrap();
    let expected: Vec<f32> = plain
        .iter()
        .zip(&host_d)
        .map(|(&v, &d)| silu(v) + 0.5 * d)
        .collect();
    assert_close(&gpu, &expected, k, "cm2 poisoned C epilogue");
}
