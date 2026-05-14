//! End-to-end correctness suite.  Every test compares GPU output against
//! an f64-accumulated CPU reference.
//!
//! All tests are `#[ignore]`d so `cargo test` works on a GPU-less host.
//! Run with:
//!
//!     cargo test --release --test correctness -- --ignored --test-threads=1
//!
//! `--test-threads=1` keeps to one Vulkan instance at a time, which
//! avoids flaky driver behavior on certain platforms.

mod common;
use common::*;

use ml_project::{MatmulCall, Tensor};

// --- 1. Shape sweep -------------------------------------------------------

#[test] #[ignore]
fn shape_sweep_rank2() {
    let (ctx, exec) = make_setup(2, 16);
    let cases: &[(u32, u32, u32)] = &[
        (1,    1,   1),    // scalar
        (1,   16,  16),    // single row × full col
        (16,   1,  16),    // full row × single col
        (16,  16,   1),    // K = 1
        (8,    7,   5),    // tiny
        (17,  19,  23),    // primes, no tile alignment
        (31,  33,  47),    // off-tile odd
        (64,  64,  64),    // half-tile
        (128, 128, 128),   // one full tile
        (129, 130, 131),   // 1-over-tile (edge tile is partial)
        (256, 256, 256),
        (200, 300, 100),
    ];
    for &(m, n, k) in cases {
        let (gpu, cpu) = run_one(
            &ctx, &exec,
            &[m, k], &[k, n], &[m, n],
            1.0, false, 11, 13, None,
        );
        let (e, idx) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(
            e <= tol,
            "rank2 M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e}  \
             at idx {idx}: gpu={:.6} cpu={:.6}",
            gpu[idx], cpu[idx],
        );
    }
}

// --- 2. Batched matmul, no broadcasting -----------------------------------

#[test] #[ignore]
fn batched_no_broadcast() {
    let (ctx, exec) = make_setup(2, 16);
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 8, 8, 8),
        (4, 32, 32, 32),
        (3, 47, 53, 41),
        (8, 128, 128, 128),
        (12, 256, 192, 64),
    ];
    for &(bsz, m, n, k) in cases {
        let (gpu, cpu) = run_one(
            &ctx, &exec,
            &[bsz, m, k], &[bsz, k, n], &[bsz, m, n],
            1.0, false, 21, 23, None,
        );
        let (e, _) = max_abs_err(&gpu, &cpu);
        let tol = tolerance(k);
        assert!(e <= tol, "B={bsz} M={m} N={n} K={k}: err={e:.3e} > tol={tol:.3e}");
    }
}

// --- 3. Batch broadcasting -------------------------------------------------

#[test] #[ignore]
fn batched_broadcast_a() {
    let (ctx, exec) = make_setup(2, 8);
    let bsz = 5u32;
    let (m, n, k) = (32u32, 48u32, 24u32);
    let (gpu, cpu) = run_one(
        &ctx, &exec,
        &[1, m, k], &[bsz, k, n], &[bsz, m, n],
        1.0, false, 31, 37, None,
    );
    let (e, _) = max_abs_err(&gpu, &cpu);
    assert!(e <= tolerance(k), "broadcast A err={e:.3e}");
}

#[test] #[ignore]
fn batched_broadcast_b() {
    let (ctx, exec) = make_setup(2, 8);
    let bsz = 5u32;
    let (m, n, k) = (32u32, 48u32, 24u32);
    let (gpu, cpu) = run_one(
        &ctx, &exec,
        &[bsz, m, k], &[1, k, n], &[bsz, m, n],
        1.0, false, 41, 43, None,
    );
    let (e, _) = max_abs_err(&gpu, &cpu);
    assert!(e <= tolerance(k), "broadcast B err={e:.3e}");
}

// --- 4. alpha + accumulate ------------------------------------------------

#[test] #[ignore]
fn alpha_and_accumulate() {
    let (ctx, exec) = make_setup(2, 8);
    let (m, n, k) = (37u32, 41u32, 29u32);
    let mut c_init = vec![0.0f32; (m * n) as usize];
    fill_det(&mut c_init, 99);
    let alpha = 0.5;

    let (gpu, cpu) = run_one(
        &ctx, &exec,
        &[m, k], &[k, n], &[m, n],
        alpha, true, 51, 53, Some(&c_init),
    );
    let (e, idx) = max_abs_err(&gpu, &cpu);
    let tol = tolerance(k) + alpha.abs() * 1e-5;
    assert!(
        e <= tol,
        "α+accumulate err={e:.3e} > tol={tol:.3e}  \
         at idx {idx}: gpu={:.6} cpu={:.6}",
        gpu[idx], cpu[idx],
    );
}

// --- 5. Identity sanity check: I @ X == X ---------------------------------

#[test] #[ignore]
fn identity_matmul() {
    let (ctx, exec) = make_setup(2, 4);
    let n = 64u32;
    let mut id = vec![0.0f32; (n * n) as usize];
    for i in 0..n as usize { id[i * n as usize + i] = 1.0; }
    let mut x = vec![0.0f32; (n * n) as usize];
    fill_det(&mut x, 71);

    let a = Tensor::zeros_device(&ctx, &[n, n]).unwrap();
    let b = Tensor::zeros_device(&ctx, &[n, n]).unwrap();
    let c = Tensor::zeros_device(&ctx, &[n, n]).unwrap();
    exec.upload(&id, &a).unwrap();
    exec.upload(&x,  &b).unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false,
    }]).unwrap();
    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();

    let (e, _) = max_abs_err(&got, &x);
    assert!(e == 0.0, "identity matmul should be exact, err={e}");
}

// --- 6. Many independent matmuls in one submit ----------------------------

#[test] #[ignore]
fn many_calls_one_submit() {
    let (ctx, exec) = make_setup(2, 32);
    let n_calls = 16u32;
    let m = 64u32; let k = 96u32; let nn = 80u32;

    let mut tensors = Vec::new();
    let mut hosts = Vec::new();
    for s in 0..n_calls {
        let a = Tensor::zeros_device(&ctx, &[m, k]).unwrap();
        let b = Tensor::zeros_device(&ctx, &[k, nn]).unwrap();
        let c = Tensor::zeros_device(&ctx, &[m, nn]).unwrap();
        let mut ha = vec![0.0f32; (m*k) as usize];
        let mut hb = vec![0.0f32; (k*nn) as usize];
        fill_det(&mut ha, 1000 + s as u64);
        fill_det(&mut hb, 2000 + s as u64);
        exec.upload(&ha, &a).unwrap();
        exec.upload(&hb, &b).unwrap();
        tensors.push((a, b, c));
        hosts.push((ha, hb));
    }
    let calls: Vec<MatmulCall> = tensors.iter()
        .map(|(a, b, c)| MatmulCall {
            a, b, c, alpha: 1.0, accumulate: false,
        }).collect();
    exec.run_matmuls(&calls).unwrap();

    for (i, (_, _, c)) in tensors.iter().enumerate() {
        let mut got = vec![0.0f32; (m*nn) as usize];
        exec.download(c, &mut got).unwrap();
        let cpu = cpu_bmm(&hosts[i].0, &hosts[i].1, None,
                          1, m, nn, k, 1.0, false);
        let (e, _) = max_abs_err(&got, &cpu);
        assert!(e <= tolerance(k), "call {i}: err {e:.3e}");
    }
}

// --- 7. Concurrent run_matmuls from multiple host threads -----------------

#[test] #[ignore]
fn concurrent_submitters() {
    let (ctx, exec) = make_setup(/*n_slots=*/4, 8);
    let exec = std::sync::Arc::new(exec);
    let (m, n, k) = (64u32, 64u32, 64u32);

    let mut threads = Vec::new();
    for t in 0..8u32 {
        let exec = exec.clone();
        let ctx  = ctx.clone();
        threads.push(std::thread::spawn(move || {
            let a = Tensor::zeros_device(&ctx, &[m, k]).unwrap();
            let b = Tensor::zeros_device(&ctx, &[k, n]).unwrap();
            let c = Tensor::zeros_device(&ctx, &[m, n]).unwrap();
            let mut ha = vec![0.0f32; (m*k) as usize];
            let mut hb = vec![0.0f32; (k*n) as usize];
            fill_det(&mut ha, 10_000 + t as u64);
            fill_det(&mut hb, 20_000 + t as u64);
            exec.upload(&ha, &a).unwrap();
            exec.upload(&hb, &b).unwrap();
            exec.run_matmuls(&[MatmulCall {
                a: &a, b: &b, c: &c, alpha: 1.0, accumulate: false,
            }]).unwrap();
            let mut got = vec![0.0f32; (m*n) as usize];
            exec.download(&c, &mut got).unwrap();
            let cpu = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
            let (e, _) = max_abs_err(&got, &cpu);
            assert!(e <= tolerance(k), "thread {t}: err {e:.3e}");
        }));
    }
    for th in threads { th.join().unwrap(); }
}

// --- 8. Empty submit must succeed -----------------------------------------

#[test] #[ignore]
fn empty_submit() {
    let (_ctx, exec) = make_setup(1, 1);
    let stats = exec.run_matmuls(&[]).unwrap();
    assert_eq!(stats.n_calls, 0);
    assert_eq!(stats.total_flops, 0);
}
