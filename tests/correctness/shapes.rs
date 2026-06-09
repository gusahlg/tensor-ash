use crate::common::*;

use tensor_ash::KernelSelection;

#[test]
#[ignore]
fn shape_sweep_rank2() {
    let (ctx, exec) = make_setup(2, 16);
    let cases: &[(u32, u32, u32)] = &[
        (1, 1, 1),
        (1, 16, 16),
        (16, 1, 16),
        (16, 16, 1),
        (8, 7, 5),
        (17, 19, 23),
        (31, 33, 47),
        (64, 64, 64),
        (128, 128, 128),
        (129, 130, 131),
        (256, 256, 256),
        (200, 300, 100),
    ];
    for &(m, n, k) in cases {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            11,
            13,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "rank2 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}

#[test]
#[ignore]
fn large_tile_kernel_path() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::Large);
    let (m, n, k) = (640u32, 640u32, 64u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[m, k],
        &[k, n],
        &[m, n],
        1.0,
        false,
        17,
        19,
        None,
    );
    let (e, idx) = max_abs_err(&gpu, &cpu);
    let tol = tolerance(k);
    assert!(
        e <= tol,
        "large-tile M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
         at idx {idx}: gpu={:.6} cpu={:.6}",
        gpu[idx],
        cpu[idx],
    );
}

#[test]
#[ignore]
fn manual_large_kernel_handles_partial_tiles() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::Large);
    let (m, n, k) = (129u32, 130u32, 31u32);
    let (gpu, cpu) = run_one(
        &ctx,
        &exec,
        &[m, k],
        &[k, n],
        &[m, n],
        1.0,
        false,
        23,
        29,
        None,
    );
    let (e, idx) = max_abs_err(&gpu, &cpu);
    let tol = tolerance(k);
    assert!(
        e <= tol,
        "manual large partial tile M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
         at idx {idx}: gpu={:.6} cpu={:.6}",
        gpu[idx],
        cpu[idx],
    );
}
