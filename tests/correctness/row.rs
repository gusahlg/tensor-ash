use crate::common::*;

use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KernelSelection, MatmulCall, MatmulOp, Tensor,
};

#[test]
#[ignore]
fn row_kernel_handles_tails_broadcasting_and_multiple_rows() {
    struct RowCase<'a> {
        shape_a: &'a [u32],
        shape_b: &'a [u32],
        shape_c: &'a [u32],
        alpha: f32,
        accumulate: bool,
    }

    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::RowBda);
    let cases = [
        RowCase {
            shape_a: &[3, 1, 203],
            shape_b: &[3, 203, 30],
            shape_c: &[3, 1, 30],
            alpha: 1.0,
            accumulate: false,
        },
        RowCase {
            shape_a: &[2, 1, 1010],
            shape_b: &[2, 1010, 32],
            shape_c: &[2, 1, 32],
            alpha: 1.0,
            accumulate: false,
        },
        RowCase {
            shape_a: &[2, 3, 17],
            shape_b: &[2, 17, 37],
            shape_c: &[2, 3, 37],
            alpha: 0.75,
            accumulate: false,
        },
        RowCase {
            shape_a: &[2, 2, 17],
            shape_b: &[2, 17, 33],
            shape_c: &[2, 2, 33],
            alpha: 0.5,
            accumulate: true,
        },
        RowCase {
            shape_a: &[1, 1, 19],
            shape_b: &[4, 19, 35],
            shape_c: &[4, 1, 35],
            alpha: 1.0,
            accumulate: false,
        },
        RowCase {
            shape_a: &[4, 1, 19],
            shape_b: &[1, 19, 35],
            shape_c: &[4, 1, 35],
            alpha: 1.0,
            accumulate: false,
        },
        RowCase {
            shape_a: &[1, 2, 19],
            shape_b: &[1, 19, 35],
            shape_c: &[4, 2, 35],
            alpha: 1.0,
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
            301 + index as u64,
            401 + index as u64,
            initial.as_deref(),
        );
        let k = *case.shape_a.last().unwrap();
        let (error, error_index) = max_abs_err(&gpu, &cpu);
        assert!(
            error <= tolerance(k),
            "row case {index} error {error:.3e} at {error_index}: got {}, expected {}",
            gpu[error_index],
            cpu[error_index]
        );
    }
}

#[test]
#[ignore]
fn row_kernel_supports_accumulation_and_batched_fused_epilogue() {
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::RowBda);
    let (batch, m, n, k) = (3_u32, 1_u32, 35_u32, 19_u32);
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
    fill_det(&mut host_a, 501);
    fill_det(&mut host_b, 502);
    fill_det(&mut host_c, 503);
    fill_det(&mut host_bias, 504);
    fill_det(&mut host_residual, 505);
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
        for col in 0..n as usize {
            let index = batch_index * n as usize + col;
            expected[index] =
                (expected[index] + host_bias[index]).max(0.0) + beta * host_residual[index];
        }
    }

    let (error, index) = max_abs_err(&actual, &expected);
    assert!(
        error <= tolerance(k),
        "row fused epilogue error {error:.3e} at {index}: got {}, expected {}",
        actual[index],
        expected[index]
    );
}
