use crate::common::*;

use tensor_ash::{KernelSelection, MatmulCall, Tensor};

#[test]
#[ignore]
fn cross_kernel_equivalence() {
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 128, 128, 128),
        (1, 256, 256, 256),
        (1, 512, 512, 64),
        (2, 256, 256, 256),
        (1, 384, 256, 192),
    ];

    let (ctx_l, exec_l) = make_setup_with_kernel(2, 8, KernelSelection::Large);
    let (ctx_s, exec_s) = make_setup_with_kernel(2, 8, KernelSelection::Small);

    for &(b, m, n, k) in cases {
        let shape_a = if b == 1 { vec![m, k] } else { vec![b, m, k] };
        let shape_b = if b == 1 { vec![k, n] } else { vec![b, k, n] };
        let shape_c = if b == 1 { vec![m, n] } else { vec![b, m, n] };

        let na = Tensor::numel(&shape_a) as usize;
        let nb = Tensor::numel(&shape_b) as usize;
        let nc = Tensor::numel(&shape_c) as usize;
        let mut ha = vec![0.0f32; na];
        let mut hb = vec![0.0f32; nb];
        fill_det(&mut ha, 0xABC + (b * m * n * k) as u64);
        fill_det(&mut hb, 0xDEF + (b * m * n * k) as u64);

        let a_l = Tensor::uninit_device(&ctx_l, &shape_a).unwrap();
        let b_l = Tensor::uninit_device(&ctx_l, &shape_b).unwrap();
        let c_l = Tensor::uninit_device(&ctx_l, &shape_c).unwrap();
        exec_l.upload(&ha, &a_l).unwrap();
        exec_l.upload(&hb, &b_l).unwrap();
        exec_l
            .run_matmuls(&[MatmulCall {
                a: &a_l,
                b: &b_l,
                c: &c_l,
                alpha: 1.0,
                accumulate: false,
            }])
            .unwrap();
        let mut large_out = vec![0.0f32; nc];
        exec_l.download(&c_l, &mut large_out).unwrap();

        let a_s = Tensor::uninit_device(&ctx_s, &shape_a).unwrap();
        let b_s = Tensor::uninit_device(&ctx_s, &shape_b).unwrap();
        let c_s = Tensor::uninit_device(&ctx_s, &shape_c).unwrap();
        exec_s.upload(&ha, &a_s).unwrap();
        exec_s.upload(&hb, &b_s).unwrap();
        exec_s
            .run_matmuls(&[MatmulCall {
                a: &a_s,
                b: &b_s,
                c: &c_s,
                alpha: 1.0,
                accumulate: false,
            }])
            .unwrap();
        let mut small_out = vec![0.0f32; nc];
        exec_s.download(&c_s, &mut small_out).unwrap();

        let (e, idx) = max_abs_err(&large_out, &small_out);
        let tol = 4.0 * f32::EPSILON * large_out[idx].abs().max(1.0);
        assert!(
            e <= tol,
            "kernel mismatch B={b} M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: large={:.6} small={:.6}",
            large_out[idx],
            small_out[idx],
        );
    }
}
