use crate::common::*;

use tensor_ash::{MatmulCall, Tensor};

#[test]
#[ignore]
fn many_calls_one_submit() {
    let (ctx, exec) = make_setup(2, 32);
    let n_calls = 16u32;
    let m = 64u32;
    let k = 96u32;
    let nn = 80u32;

    let mut tensors = Vec::new();
    let mut hosts = Vec::new();
    for s in 0..n_calls {
        let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
        let b = Tensor::uninit_device(&ctx, &[k, nn]).unwrap();
        let c = Tensor::uninit_device(&ctx, &[m, nn]).unwrap();
        let mut ha = vec![0.0f32; (m * k) as usize];
        let mut hb = vec![0.0f32; (k * nn) as usize];
        fill_det(&mut ha, 1000 + s as u64);
        fill_det(&mut hb, 2000 + s as u64);
        exec.upload(&ha, &a).unwrap();
        exec.upload(&hb, &b).unwrap();
        tensors.push((a, b, c));
        hosts.push((ha, hb));
    }
    let calls: Vec<MatmulCall> = tensors
        .iter()
        .map(|(a, b, c)| MatmulCall {
            a,
            b,
            c,
            alpha: 1.0,
            accumulate: false,
        })
        .collect();
    exec.run_matmuls(&calls).unwrap();

    for (i, (_, _, c)) in tensors.iter().enumerate() {
        let mut got = vec![0.0f32; (m * nn) as usize];
        exec.download(c, &mut got).unwrap();
        let cpu = cpu_bmm(&hosts[i].0, &hosts[i].1, None, 1, m, nn, k, 1.0, false);
        let (e, _) = max_abs_err(&got, &cpu);
        assert!(e <= tolerance(k), "call {i}: err {e:.3e}");
    }
}

#[test]
#[ignore]
fn concurrent_submitters() {
    let (ctx, exec) = make_setup(4, 8);
    let exec = std::sync::Arc::new(exec);
    let (m, n, k) = (64u32, 64u32, 64u32);

    let mut threads = Vec::new();
    for t in 0..8u32 {
        let exec = exec.clone();
        let ctx = ctx.clone();
        threads.push(std::thread::spawn(move || {
            let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
            let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
            let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
            let mut ha = vec![0.0f32; (m * k) as usize];
            let mut hb = vec![0.0f32; (k * n) as usize];
            fill_det(&mut ha, 10_000 + t as u64);
            fill_det(&mut hb, 20_000 + t as u64);
            exec.upload(&ha, &a).unwrap();
            exec.upload(&hb, &b).unwrap();
            exec.run_matmuls(&[MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            }])
            .unwrap();
            let mut got = vec![0.0f32; (m * n) as usize];
            exec.download(&c, &mut got).unwrap();
            let cpu = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
            let (e, _) = max_abs_err(&got, &cpu);
            assert!(e <= tolerance(k), "thread {t}: err {e:.3e}");
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
}

#[test]
#[ignore]
fn empty_submit() {
    let (_ctx, exec) = make_setup(1, 1);
    let stats = exec.run_matmuls(&[]).unwrap();
    assert_eq!(stats.n_calls, 0);
    assert_eq!(stats.total_flops, 0);
}

#[test]
#[ignore]
fn chained_run_matmuls_observe_dependencies() {
    let (ctx, exec) = make_setup(2, 4);
    let n = 64u32;
    let mut ha = vec![0.0f32; (n * n) as usize];
    let mut hb = vec![0.0f32; (n * n) as usize];
    fill_det(&mut ha, 111);
    fill_det(&mut hb, 222);

    let a = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let t1 = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let t2 = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let bb = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let abb = Tensor::uninit_device(&ctx, &[n, n]).unwrap();

    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();

    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &t1,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &t1,
        b: &b,
        c: &t2,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();

    exec.run_matmuls(&[MatmulCall {
        a: &b,
        b: &b,
        c: &bb,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();
    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &bb,
        c: &abb,
        alpha: 1.0,
        accumulate: false,
    }])
    .unwrap();

    let mut left = vec![0.0f32; (n * n) as usize];
    let mut right = vec![0.0f32; (n * n) as usize];
    exec.download(&t2, &mut left).unwrap();
    exec.download(&abb, &mut right).unwrap();

    let (e, idx) = max_abs_err(&left, &right);
    let tol = 16.0 * (n as f32) * f32::EPSILON * left[idx].abs().max(1.0);
    assert!(
        e <= tol,
        "(A@B)@B != A@(B@B) within tolerance: err={e:.3e} tol={tol:.3e} \
         at idx {idx}: left={:.6} right={:.6}",
        left[idx],
        right[idx],
    );
}

#[test]
#[ignore]
fn many_consecutive_submits() {
    let (ctx, exec) = make_setup(2, 4);
    let n = 64u32;
    let mut ha = vec![0.0f32; (n * n) as usize];
    let mut hb = vec![0.0f32; (n * n) as usize];
    fill_det(&mut ha, 91);
    fill_det(&mut hb, 92);
    let a = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[n, n]).unwrap();
    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();

    for _ in 0..200 {
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap();
    }

    let mut got = vec![0.0f32; (n * n) as usize];
    exec.download(&c, &mut got).unwrap();
    let cpu = cpu_bmm(&ha, &hb, None, 1, n, n, n, 1.0, false);
    let (e, _) = max_abs_err(&got, &cpu);
    assert!(e <= tolerance(n), "200-iter drift err={e:.3e}");
}

#[test]
#[ignore]
fn concurrent_heavy_load() {
    let (ctx, exec) = make_setup(4, 4);
    let exec = std::sync::Arc::new(exec);
    let (m, n, k) = (96u32, 96u32, 96u32);

    let mut threads = Vec::new();
    for t in 0..16u32 {
        let exec = exec.clone();
        let ctx = ctx.clone();
        threads.push(std::thread::spawn(move || {
            let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
            let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
            let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
            let mut ha = vec![0.0f32; (m * k) as usize];
            let mut hb = vec![0.0f32; (k * n) as usize];
            fill_det(&mut ha, 50_000 + t as u64);
            fill_det(&mut hb, 60_000 + t as u64);
            exec.upload(&ha, &a).unwrap();
            exec.upload(&hb, &b).unwrap();

            for _ in 0..50 {
                exec.run_matmuls(&[MatmulCall {
                    a: &a,
                    b: &b,
                    c: &c,
                    alpha: 1.0,
                    accumulate: false,
                }])
                .unwrap();
            }
            let mut got = vec![0.0f32; (m * n) as usize];
            exec.download(&c, &mut got).unwrap();
            let cpu = cpu_bmm(&ha, &hb, None, 1, m, n, k, 1.0, false);
            let (e, _) = max_abs_err(&got, &cpu);
            assert!(e <= tolerance(k), "thread {t}: err {e:.3e}");
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
}

#[test]
#[ignore]
fn run_stats_reports_correct_counts() {
    let (ctx, exec) = make_setup(1, 4);
    let m = 32u32;
    let n = 48u32;
    let k = 16u32;
    let a = Tensor::uninit_device(&ctx, &[m, k]).unwrap();
    let b = Tensor::uninit_device(&ctx, &[k, n]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let mut ha = vec![0.0f32; (m * k) as usize];
    let mut hb = vec![0.0f32; (k * n) as usize];
    fill_det(&mut ha, 1);
    fill_det(&mut hb, 2);
    exec.upload(&ha, &a).unwrap();
    exec.upload(&hb, &b).unwrap();

    let calls = [MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    }; 3];
    let stats = exec.run_matmuls(&calls).unwrap();
    assert_eq!(stats.n_calls, 3);
    assert_eq!(
        stats.total_flops,
        3 * 2 * (m as u64) * (n as u64) * (k as u64)
    );
}
