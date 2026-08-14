use crate::common::*;

use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

#[test]
#[ignore]
fn outer_kernel_covers_interior_edge_and_general_k() {
    struct OuterCase<'a> {
        shape_a: &'a [u32],
        shape_b: &'a [u32],
        shape_c: &'a [u32],
        alpha: f32,
        accumulate: bool,
    }

    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::OuterBda);
    let cases = [
        // Interior vec4 path: M % 16 == 0 and N % 128 == 0.
        OuterCase {
            shape_a: &[32, 1],
            shape_b: &[1, 256],
            shape_c: &[32, 256],
            alpha: 1.0,
            accumulate: false,
        },
        // Interior path with alpha scaling and accumulation.
        OuterCase {
            shape_a: &[16, 1],
            shape_b: &[1, 128],
            shape_c: &[16, 128],
            alpha: 0.5,
            accumulate: true,
        },
        // Edge dispatch mixing full workgroups with guarded ones.
        OuterCase {
            shape_a: &[33, 1],
            shape_b: &[1, 261],
            shape_c: &[33, 261],
            alpha: 0.75,
            accumulate: false,
        },
        // Tiny all-edge shape with accumulation.
        OuterCase {
            shape_a: &[3, 1],
            shape_b: &[1, 5],
            shape_c: &[3, 5],
            alpha: 1.0,
            accumulate: true,
        },
        // A broadcast across batches.
        OuterCase {
            shape_a: &[1, 20, 1],
            shape_b: &[3, 1, 40],
            shape_c: &[3, 20, 40],
            alpha: 1.0,
            accumulate: false,
        },
        // B broadcast across batches.
        OuterCase {
            shape_a: &[4, 12, 1],
            shape_b: &[1, 1, 36],
            shape_c: &[4, 12, 36],
            alpha: 1.0,
            accumulate: false,
        },
        // K > 1 stays correct when the kernel is selected explicitly.
        OuterCase {
            shape_a: &[7, 5],
            shape_b: &[5, 9],
            shape_c: &[7, 9],
            alpha: 1.0,
            accumulate: false,
        },
        // K > 1 on the interior vec4 path.
        OuterCase {
            shape_a: &[2, 16, 3],
            shape_b: &[2, 3, 128],
            shape_c: &[2, 16, 128],
            alpha: 0.25,
            accumulate: false,
        },
    ];

    for (index, case) in cases.iter().enumerate() {
        let initial = case
            .accumulate
            .then(|| vec![0.125; Tensor::numel(case.shape_c) as usize]);
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            case.shape_a,
            case.shape_b,
            case.shape_c,
            case.alpha,
            case.accumulate,
            901 + index as u64,
            1001 + index as u64,
            initial.as_deref(),
        );
        let k = *case.shape_a.last().unwrap();
        let (error, error_index) = max_abs_err(&gpu, &cpu);
        assert!(
            error <= tolerance(k),
            "outer case {index} error {error:.3e} at {error_index}: got {}, expected {}",
            gpu[error_index],
            cpu[error_index]
        );
    }
}

#[test]
#[ignore]
fn outer_kernel_supports_batched_fused_epilogue() {
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::OuterBda);
    // Interior shape so the epilogue runs through the vec4 store path.
    let (batch, m, n, k) = (2_u32, 16_u32, 128_u32, 1_u32);
    let a = Tensor::uninit_device(&ctx, &[batch, m, k]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[batch, k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let bias = Tensor::uninit_device(&ctx, &[batch, n]).unwrap();
    let residual = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let mut host_a = vec![0.0; Tensor::numel(a.shape()) as usize];
    let mut host_b = vec![0.0; Tensor::numel(b.shape()) as usize];
    let mut host_c = vec![0.0; Tensor::numel(c.shape()) as usize];
    let mut host_bias = vec![0.0; Tensor::numel(bias.shape()) as usize];
    let mut host_residual = vec![0.0; Tensor::numel(residual.shape()) as usize];
    fill_det(&mut host_a, 1101);
    fill_det(&mut host_b, 1102);
    fill_det(&mut host_c, 1103);
    fill_det(&mut host_bias, 1104);
    fill_det(&mut host_residual, 1105);
    exec.upload(&host_a, &a).unwrap();
    exec.upload(&host_b, &b).unwrap();
    exec.upload(&host_c, &c).unwrap();
    exec.upload(&host_bias, &bias).unwrap();
    exec.upload(&host_residual, &residual).unwrap();

    let alpha = 0.75;
    let beta = 0.25;
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha,
            accumulate: true,
        },
        Epilogue {
            bias: Some(&bias),
            activation: Activation::Relu,
            binary: EpilogueBinary::AddScaled { d: &residual, beta },
        },
    )])
    .unwrap();

    let mut actual = vec![0.0; host_c.len()];
    exec.download(&c, &mut actual).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, Some(&host_c), batch, m, n, k, alpha, true);
    for batch_index in 0..batch as usize {
        for row in 0..m as usize {
            for col in 0..n as usize {
                let index = (batch_index * m as usize + row) * n as usize + col;
                let with_bias = expected[index] + host_bias[batch_index * n as usize + col];
                expected[index] = with_bias.max(0.0) + beta * host_residual[index];
            }
        }
    }

    let (error, index) = max_abs_err(&actual, &expected);
    assert!(
        error <= tolerance(k),
        "outer fused epilogue error {error:.3e} at {index}: got {}, expected {}",
        actual[index],
        expected[index]
    );
}

#[test]
#[ignore]
fn auto_routes_k1_to_outer_after_gemv_precedence() {
    let (ctx, exec) = make_setup(2, 8);
    // The GEMV/K=1 routing block requires bufferDeviceAddress.
    if !ctx.buffer_device_address_enabled {
        return;
    }
    // Odd dimensions make a pre-existing tuned-store entry for these
    // integration-only shapes exceedingly unlikely (see splitk2.rs).
    // K=1 with both dimensions real routes to the outer product.
    assert_eq!(exec.dispatch_info(1, 37, 53, 1).kernel, "outer_bda");
    assert_eq!(exec.dispatch_info(7, 501, 483, 1).kernel, "outer_bda");
    // The GEMV routes keep precedence over the K=1 rule.
    assert_eq!(exec.dispatch_info(1, 1, 53, 1).kernel, "row_bda");
    assert_eq!(exec.dispatch_info(1, 37, 1, 1).kernel, "col_bda");
}
