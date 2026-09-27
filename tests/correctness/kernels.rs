use crate::common::*;

use tensor_ash::{
    Activation, Epilogue, EpilogueBinary, KERNEL_SPECS, KernelSelection, MatmulCall, MatmulOp,
    Tensor,
};

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

/// The transposed-A v4t tile (`large_bda_v4t`, the f32 Large route) vs
/// the CPU reference: interior tiles, batched, odd M/N/K, K-tails,
/// alpha+accumulate, and a fused bias+ReLU epilogue.  Selected
/// explicitly; a device that gates the tile (32,768 B, shouldn't
/// happen) skips it instead of failing.  (exp/v4t also covers three
/// v4t tiles that are not shipped.)
#[test]
#[ignore]
fn v4t_kernels_match_reference() {
    let kernels = [KernelSelection::LargeBdaV4t];
    let shapes: &[(u32, u32, u32, u32)] = &[
        (1, 128, 128, 128),
        (1, 256, 256, 256),
        (2, 256, 256, 256),
        (1, 384, 256, 192),
        (1, 255, 257, 263),
        (1, 513, 515, 517),
        (1, 1000, 1000, 1000),
        (3, 65, 130, 33),
    ];
    for sel in kernels {
        let Some((ctx, exec)) = make_setup_with_kernel_if_fits(2, 8, sel) else {
            continue;
        };
        let name = KERNEL_SPECS[sel.index().expect("concrete")].name;
        for &(b, m, n, k) in shapes {
            let shape = |rows: u32, cols: u32| -> Vec<u32> {
                if b == 1 {
                    vec![rows, cols]
                } else {
                    vec![b, rows, cols]
                }
            };
            let (gpu, cpu) = run_one(
                &ctx,
                &exec,
                &shape(m, k),
                &shape(k, n),
                &shape(m, n),
                1.0,
                false,
                0xABC + (b as u64).wrapping_mul(m as u64 * n as u64 * k as u64),
                0xDEF + (b as u64).wrapping_mul(m as u64 * n as u64 * k as u64),
                None,
            );
            assert_close(&gpu, &cpu, k, &format!("{name} B={b} {m}x{n}x{k}"));
        }

        let (m, n, k) = (256u32, 256u32, 256u32);
        let mut c0 = vec![0.0f32; (m * n) as usize];
        fill_det(&mut c0, 73);
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            0.5,
            true,
            71,
            72,
            Some(&c0),
        );
        assert_close(&gpu, &cpu, k, &format!("{name} alpha=0.5 accumulate"));

        let (a, ha) = upload_det(&ctx, &exec, &[m, k], 81);
        let (b, hb) = upload_det(&ctx, &exec, &[k, n], 82);
        let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
        let (bias, hbias) = upload_det(&ctx, &exec, &[n], 84);
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
                activation: Activation::Relu,
                binary: EpilogueBinary::None,
            },
        )])
        .unwrap();
        let mut gpu = vec![0.0f32; (m * n) as usize];
        exec.download(&c, &mut gpu).unwrap();
        let mut expect = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
        for (i, v) in expect.iter_mut().enumerate() {
            *v = (*v + hbias[i % n as usize]).max(0.0);
        }
        assert_close(&gpu, &expect, k, &format!("{name} bias+relu"));
    }
}
