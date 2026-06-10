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

#[test]
#[ignore]
fn manual_m64n128_kernel_handles_wide_and_partial_tiles() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::M64N128);
    for (m, n, k, seed_a, seed_b) in [
        (128u32, 1024u32, 512u32, 31u64, 37u64),
        (70u32, 190u32, 45u32, 41u64, 43u64),
    ] {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            seed_a,
            seed_b,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "m64n128 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}

#[test]
#[ignore]
fn manual_m128n64_kernel_handles_tall_and_partial_tiles() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::M128N64);
    for (m, n, k, seed_a, seed_b) in [
        (1024u32, 128u32, 512u32, 47u64, 53u64),
        (190u32, 70u32, 45u32, 59u64, 61u64),
    ] {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            seed_a,
            seed_b,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "m128n64 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}

#[test]
#[ignore]
fn manual_m128n64k64_kernel_handles_deep_k_and_tail() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::M128N64K64);
    for (m, n, k, seed_a, seed_b) in [
        (512u32, 512u32, 512u32, 54u64, 56u64),
        (190u32, 70u32, 95u32, 58u64, 60u64),
    ] {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            seed_a,
            seed_b,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "m128n64k64 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}

#[test]
#[ignore]
fn manual_m64n32_kernel_handles_near_square_partial_tiles() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::M64N32);
    for (m, n, k, seed_a, seed_b) in [
        (255u32, 257u32, 263u32, 62u64, 64u64),
        (130u32, 97u32, 95u32, 66u64, 68u64),
    ] {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            seed_a,
            seed_b,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "m64n32 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}

#[test]
#[ignore]
fn manual_k64_kernel_handles_small_k_and_tail() {
    let (ctx, exec) = make_setup_with_kernel(1, 4, KernelSelection::K64);
    for (m, n, k, seed_a, seed_b) in [
        (1024u32, 1024u32, 64u32, 67u64, 71u64),
        (130u32, 130u32, 95u32, 73u64, 79u64),
    ] {
        let (gpu, cpu) = run_one(
            &ctx,
            &exec,
            &[m, k],
            &[k, n],
            &[m, n],
            1.0,
            false,
            seed_a,
            seed_b,
            None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "k64 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e} \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx],
            cpu[idx],
        );
    }
}
