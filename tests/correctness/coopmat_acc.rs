//! f16-accumulator coopmat1 GEMMs (`CoopmatAcc::F16` / `F16Promote`).
//!
//! These kernels are a *numerics class*, not exact-f32 GEMMs: every
//! 16-deep MMA rounds the running sum (F16) or a 128-K chunk partial
//! (F16Promote) to f16.  So the check is per element against an f64
//! reference with the error budget the class promises (the same
//! budget `ml_bench` validates with): `2 tolerance(k)` + the f16 C
//! store's rounding + one f16 rounding of the product l2 norm
//! `sqrt(sum_k (a_k b_k)^2)` per MMA (F16) or 8 per chunk grown with
//! sqrt(K / 128) (F16Promote).  Measured errors sit 2-3x inside it
//! (`benchmarks/campaign/f16acc.md` section 3); a wrong tile, a
//! dropped K-tile or garbage output is of order l2 itself, ~1000x over.
//!
//! The tests also pin that the policy actually reaches the f16
//! accumulator: the auto route names the `_acc16` / `_acc16p` twin, and
//! the f16 classes are measurably *less* exact than the f32 route on
//! the same inputs (a silent fallback to f32 would pass a pure
//! tolerance check).

use std::sync::Arc;

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    CoopmatAcc, ExecOp, Executor, ExecutorConfig, MatmulCall, MatmulOp, MatmulPipeline, Tensor,
    VulkanContext,
};

/// One context + pipeline, one executor per accumulator policy.
fn acc_setup() -> Option<(Arc<VulkanContext>, [Executor; 3])> {
    let ctx = VulkanContext::new(false).expect("Vulkan init");
    if !(ctx.coopmat_enabled
        && ctx.f16_storage_enabled
        && ctx.buffer_device_address_enabled
        && ctx.coopmat_f16_acc_enabled)
    {
        eprintln!("skipping: no f16-accumulator cooperative-matrix configuration");
        return None;
    }
    let pipe = Arc::new(MatmulPipeline::new(&ctx).expect("pipeline"));
    let make = |acc| {
        Executor::new_with_config(
            ctx.clone(),
            pipe.clone(),
            ExecutorConfig {
                n_slots: 2,
                max_calls_per_submit: 8,
                tune: false,
                coopmat_acc: acc,
            },
        )
        .expect("executor")
    };
    let execs = [
        make(CoopmatAcc::F32),
        make(CoopmatAcc::F16),
        make(CoopmatAcc::F16Promote),
    ];
    Some((ctx, execs))
}

/// Per-element budget of an accumulator class (mirrors
/// `ml_bench`'s `acc16_budget`).
fn acc_budget(acc: CoopmatAcc, k: u32, l2: f64) -> f64 {
    let u = 2.0f64.powi(-11) * l2;
    match acc {
        CoopmatAcc::F32 => 0.0,
        CoopmatAcc::F16 => f64::from(k) / 16.0 * u,
        CoopmatAcc::F16Promote => 8.0 * u * (f64::from(k) / 128.0).sqrt(),
    }
}

struct ErrorStats {
    /// Largest `error / budget` over the checked elements (<= 1 passes).
    worst_ratio: f64,
    worst_index: usize,
    /// RMS of `gpu - ref` over the checked elements.
    rms: f64,
}

/// Compare `gpu` against an f64 reference over `rows` (all columns of
/// each), `c_old` the prior C for accumulate (already f16-rounded when
/// C is f16).
#[allow(clippy::too_many_arguments)]
fn check_rows(
    gpu: &[f32],
    a: &[f32],
    b: &[f32],
    c_old: Option<&[f32]>,
    (batch, m, n, k): (u32, u32, u32, u32),
    alpha: f32,
    c_f16: bool,
    acc: CoopmatAcc,
    rows: &[u32],
) -> ErrorStats {
    let (m_, n_, k_) = (m as usize, n as usize, k as usize);
    let mut worst_ratio = 0.0f64;
    let mut worst_index = 0;
    let (mut sum_sq, mut count) = (0.0f64, 0usize);
    for bi in 0..batch as usize {
        for &row in rows {
            let row = row as usize;
            for col in 0..n_ {
                let (mut dot, mut l2) = (0.0f64, 0.0f64);
                for inner in 0..k_ {
                    let p = f64::from(a[(bi * m_ + row) * k_ + inner])
                        * f64::from(b[(bi * k_ + inner) * n_ + col]);
                    dot += p;
                    l2 += p * p;
                }
                let index = (bi * m_ + row) * n_ + col;
                let alpha = f64::from(alpha);
                let expected = alpha * dot + c_old.map_or(0.0, |c| f64::from(c[index]));
                let store = if c_f16 {
                    expected.abs() * 2.0f64.powi(-10)
                } else {
                    0.0
                };
                let budget = 2.0 * f64::from(tolerance(k))
                    + store
                    + acc_budget(acc, k, alpha.abs() * l2.sqrt());
                let error = (f64::from(gpu[index]) - expected).abs();
                let ratio = if error.is_finite() {
                    error / budget
                } else {
                    f64::INFINITY
                };
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_index = index;
                }
                sum_sq += error * error;
                count += 1;
            }
        }
    }
    ErrorStats {
        worst_ratio,
        worst_index,
        rms: (sum_sq / count as f64).sqrt(),
    }
}

/// 16 rows spread over M (first, last, and in between).
fn sample_rows(m: u32) -> Vec<u32> {
    let count = m.min(16);
    (0..count)
        .map(|i| {
            if count == 1 {
                0
            } else {
                i * (m - 1) / (count - 1)
            }
        })
        .collect()
}

fn upload_f16(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    shape: &[u32],
    seed: u64,
) -> (Tensor, Vec<f32>) {
    let mut host = vec![0.0; Tensor::numel(shape) as usize];
    fill_det(&mut host, seed);
    let tensor = Tensor::uninit_device_f16(ctx, shape).unwrap();
    exec.upload(&host, &tensor).unwrap();
    for value in &mut host {
        *value = round_f32_via_f16(*value);
    }
    (tensor, host)
}

fn shape(batch: u32, rows: u32, cols: u32) -> Vec<u32> {
    if batch == 1 {
        vec![rows, cols]
    } else {
        vec![batch, rows, cols]
    }
}

#[test]
#[ignore]
fn coopmat_acc_policy_routes_to_the_f16_accumulator_twins() {
    let Some((_ctx, [f32_exec, f16_exec, promote_exec])) = acc_setup() else {
        return;
    };
    assert_eq!(f32_exec.coopmat_acc(), CoopmatAcc::F32);
    assert_eq!(f16_exec.coopmat_acc(), CoopmatAcc::F16);
    assert_eq!(promote_exec.coopmat_acc(), CoopmatAcc::F16Promote);
    // The accumulator-aware tile rule on the llama prefill shapes (a16)
    // and a large square; f32 keeps its own (f32-class) tile.
    for (m, n, k, tile) in [
        (512, 2048, 2048, "m64n128"),
        (512, 5632, 2048, "m64n128"),
        (512, 2048, 5632, "m64n128"),
        (4096, 4096, 4096, "aligned"),
    ] {
        let base = f32_exec.dispatch_info_a16(1, m, n, k).kernel;
        assert!(
            base.contains("coopmat") && !base.contains("acc16"),
            "{m}x{n}x{k} f32 a16 route: {base}"
        );
        let (tile_name, tile_suffix) = if tile == "aligned" {
            ("f16w_a16_coopmat", "")
        } else {
            ("f16w_a16_coopmat_", tile)
        };
        let f16 = f16_exec.dispatch_info_a16(1, m, n, k).kernel;
        assert_eq!(
            f16,
            format!("{tile_name}{tile_suffix}_acc16"),
            "{m}x{n}x{k}"
        );
        let promote = promote_exec.dispatch_info_a16(1, m, n, k).kernel;
        assert_eq!(
            promote,
            format!("{tile_name}{tile_suffix}_acc16p"),
            "{m}x{n}x{k}"
        );
    }
    // The f32-activation f16-weight route remaps the same way wherever
    // it lands on coopmat; f32-B routes never change.
    let f32_route = f32_exec.dispatch_info_for(1, 512, 2048, 2048, true).kernel;
    if f32_route.contains("coopmat") {
        let f16 = f16_exec.dispatch_info_for(1, 512, 2048, 2048, true).kernel;
        assert_eq!(f16, "f16w_coopmat_m64n128_acc16");
        let promote = promote_exec
            .dispatch_info_for(1, 512, 2048, 2048, true)
            .kernel;
        assert_eq!(promote, "f16w_coopmat_m64n128_acc16p");
    }
    assert_eq!(
        f16_exec.dispatch_info(1, 1024, 1024, 1024).kernel,
        f32_exec.dispatch_info(1, 1024, 1024, 1024).kernel,
        "f32-B routes ignore the accumulator policy"
    );
}

#[test]
#[ignore]
fn coopmat_acc_a16_gemms_stay_within_the_class_budget() {
    let Some((ctx, execs)) = acc_setup() else {
        return;
    };
    // Every accumulator-rule tile: 64x128 (M=512 prefill, K=2048 and
    // the deep K=5632 down projection), 64x64 (64-only alignment),
    // 128x64, batched, and a K that is not a PROMOTE chunk multiple.
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 512, 256, 2048),
        (1, 128, 256, 5632),
        (1, 192, 320, 64),
        (1, 256, 320, 256),
        (2, 128, 256, 96),
        (1, 256, 384, 416),
    ];
    for &(batch, m, n, k) in cases {
        let exec = &execs[0];
        let (a, host_a) = upload_f16(&ctx, exec, &shape(batch, m, k), 31_000 + u64::from(k));
        let (b, host_b) = upload_f16(&ctx, exec, &shape(batch, k, n), 31_100 + u64::from(n));
        let rows = sample_rows(m);
        let mut rms = [0.0f64; 3];
        for (slot, exec) in execs.iter().enumerate() {
            let acc = exec.coopmat_acc();
            let c = Tensor::uninit_device_f16(&ctx, &shape(batch, m, n)).unwrap();
            exec.run_matmuls(&[MatmulCall {
                a: &a,
                b: &b,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            }])
            .unwrap();
            let mut gpu = vec![0.0; Tensor::numel(&shape(batch, m, n)) as usize];
            exec.download(&c, &mut gpu).unwrap();
            let route = exec.dispatch_info_a16(batch, m, n, k).kernel;
            let stats = check_rows(
                &gpu,
                &host_a,
                &host_b,
                None,
                (batch, m, n, k),
                1.0,
                true,
                acc,
                &rows,
            );
            assert!(
                stats.worst_ratio <= 1.0,
                "B={batch} {m}x{n}x{k} {acc:?} ({route}): error {:.2}x the class budget at {}",
                stats.worst_ratio,
                stats.worst_index
            );
            rms[slot] = stats.rms;
        }
        // The f16 classes must really accumulate in f16: on deep K the
        // accumulation error dominates the f16 C store's rounding.
        if k >= 2048 {
            assert!(
                rms[1] > 3.0 * rms[0] && rms[2] > 1.5 * rms[0] && rms[2] < rms[1],
                "B={batch} {m}x{n}x{k}: rms f32 {:.3e} f16 {:.3e} promote {:.3e} \
                 (expected f32 < promote < f16)",
                rms[0],
                rms[1],
                rms[2]
            );
        }
    }
}

#[test]
#[ignore]
fn coopmat_acc_f32_activation_route_stays_within_the_class_budget() {
    let Some((ctx, execs)) = acc_setup() else {
        return;
    };
    // f32 A (converted while staging), f16 B, f32 C.
    let (batch, m, n, k) = (1u32, 512u32, 2048u32, 2048u32);
    if !execs[0]
        .dispatch_info_for(batch, m, n, k, true)
        .kernel
        .contains("coopmat")
    {
        eprintln!("skipping: the f16w route does not take coopmat here");
        return;
    }
    let (a, host_a) = upload_det(&ctx, &execs[0], &[m, k], 32_000);
    // The kernels convert A to f16 while staging.
    let host_a: Vec<f32> = host_a.iter().map(|&v| round_f32_via_f16(v)).collect();
    let (b, host_b) = upload_f16(&ctx, &execs[0], &[k, n], 32_001);
    let rows = sample_rows(m);
    let mut rms = [0.0f64; 3];
    for (slot, exec) in execs.iter().enumerate() {
        let acc = exec.coopmat_acc();
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
        let stats = check_rows(
            &gpu,
            &host_a,
            &host_b,
            None,
            (batch, m, n, k),
            1.0,
            false,
            acc,
            &rows,
        );
        let route = exec.dispatch_info_for(batch, m, n, k, true).kernel;
        assert!(
            stats.worst_ratio <= 1.0,
            "{m}x{n}x{k} f16w {acc:?} ({route}): error {:.2}x the class budget at {}",
            stats.worst_ratio,
            stats.worst_index
        );
        rms[slot] = stats.rms;
    }
    // f32 C: the f32 route is ~1e-6 relative, the f16 classes 1e-4..1e-3.
    assert!(
        rms[1] > 10.0 * rms[0] && rms[2] > 10.0 * rms[0] && rms[2] < rms[1],
        "rms f32 {:.3e} f16 {:.3e} promote {:.3e}",
        rms[0],
        rms[1],
        rms[2]
    );
}

#[test]
#[ignore]
fn coopmat_acc_alpha_accumulate_and_graph_path_stay_within_budget() {
    let Some((ctx, execs)) = acc_setup() else {
        return;
    };
    let (m, n, k) = (256u32, 256u32, 1024u32);
    let (a, host_a) = upload_f16(&ctx, &execs[0], &[m, k], 33_000);
    let (b, host_b) = upload_f16(&ctx, &execs[0], &[k, n], 33_001);
    let rows = sample_rows(m);
    for exec in &execs[1..] {
        let acc = exec.coopmat_acc();
        // alpha + accumulate onto an existing f16 C.
        let (c, host_c) = upload_f16(&ctx, exec, &[m, n], 33_002);
        let alpha = 0.75;
        exec.run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha,
            accumulate: true,
        }])
        .unwrap();
        let mut gpu = vec![0.0; (m * n) as usize];
        exec.download(&c, &mut gpu).unwrap();
        let stats = check_rows(
            &gpu,
            &host_a,
            &host_b,
            Some(&host_c),
            (1, m, n, k),
            alpha,
            true,
            acc,
            &rows,
        );
        assert!(
            stats.worst_ratio <= 1.0,
            "{acc:?} alpha+accumulate: {:.2}x the class budget at {}",
            stats.worst_ratio,
            stats.worst_index
        );
        // The mixed-graph path (what llama prefill records).
        let c2 = Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap();
        exec.run_exec_ops(&[ExecOp::Matmul(MatmulOp::new(MatmulCall {
            a: &a,
            b: &b,
            c: &c2,
            alpha: 1.0,
            accumulate: false,
        }))])
        .unwrap();
        let mut gpu2 = vec![0.0; (m * n) as usize];
        exec.download(&c2, &mut gpu2).unwrap();
        let stats = check_rows(
            &gpu2,
            &host_a,
            &host_b,
            None,
            (1, m, n, k),
            1.0,
            true,
            acc,
            &rows,
        );
        assert!(
            stats.worst_ratio <= 1.0,
            "{acc:?} graph path: {:.2}x the class budget at {}",
            stats.worst_ratio,
            stats.worst_index
        );
    }
}
