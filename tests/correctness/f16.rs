//! f16-storage-B (half-precision weights) correctness.
//!
//! The CPU reference rounds B through f16 exactly like the upload path,
//! so the GPU result differs only by f32 summation order and the usual
//! `tolerance(k)` applies.

use crate::common::*;

use tensor_ash::dtype::round_f32_via_f16;
use tensor_ash::{
    Activation, CopyDesc, Epilogue, EpilogueBinary, ExecOp, KernelSelection, MatmulCall, MatmulOp,
    Tensor, f16w_row_tile_n, pack_f16w_row_tiles,
};

fn f16_available(ctx: &std::sync::Arc<tensor_ash::VulkanContext>) -> bool {
    ctx.f16_storage_enabled && ctx.buffer_device_address_enabled
}

/// Upload helper: fill A (f32) and B (f16) deterministically, return
/// the host copies with B already rounded to what the GPU stores.
fn setup_f16_case(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    a_shape: &[u32],
    b_shape: &[u32],
    seed_a: u64,
    seed_b: u64,
) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let (a, host_a) = upload_det(ctx, exec, a_shape, seed_a);
    let b = Tensor::uninit_device_f16(ctx, b_shape).unwrap();
    let mut host_b = vec![0.0; Tensor::numel(b_shape) as usize];
    fill_det(&mut host_b, seed_b);
    exec.upload(&host_b, &b).unwrap();
    for value in &mut host_b {
        *value = round_f32_via_f16(*value);
    }
    (a, b, host_a, host_b)
}

#[test]
#[ignore]
fn f16_upload_download_roundtrip() {
    let (ctx, exec) = make_setup(2, 8);
    if !ctx.f16_storage_enabled {
        eprintln!("skipping: no f16 storage support");
        return;
    }
    let t = Tensor::uninit_device_f16(&ctx, &[37, 53]).unwrap();
    let mut host = vec![0.0; 37 * 53];
    fill_det(&mut host, 4242);
    exec.upload(&host, &t).unwrap();
    let mut back = vec![0.0; 37 * 53];
    exec.download(&t, &mut back).unwrap();
    for (i, (&sent, &got)) in host.iter().zip(&back).enumerate() {
        assert_eq!(
            round_f32_via_f16(sent).to_bits(),
            got.to_bits(),
            "element {i}: sent {sent}, got {got}"
        );
    }
}

#[test]
#[ignore]
fn f16_weights_match_rounded_reference_across_routes() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    // Shapes chosen to hit every f16w tile class: large (512^3),
    // k64-class (384), small/batched, interior and ragged edges.
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 512, 512, 512),
        (1, 384, 384, 384),
        (1, 127, 129, 65),
        (1, 64, 64, 64),
        (4, 96, 112, 80),
        (1, 1023, 1025, 127),
    ];
    for &(batch, m, n, k) in cases {
        let shape = |rows: u32, cols: u32| -> Vec<u32> {
            if batch == 1 {
                vec![rows, cols]
            } else {
                vec![batch, rows, cols]
            }
        };
        let (a, b, mut host_a, host_b) = setup_f16_case(
            &ctx,
            &exec,
            &shape(m, k),
            &shape(k, n),
            9000 + k as u64,
            9100 + n as u64,
        );
        // The tensor-core route quantizes A to f16 while staging;
        // mirror it in the reference so products stay exact.
        if exec
            .dispatch_info_for(batch, m, n, k, true)
            .kernel
            .contains("coopmat")
        {
            for value in &mut host_a {
                *value = round_f32_via_f16(*value);
            }
            exec.upload(&host_a, &a).unwrap();
        }
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
        assert_close(&gpu, &cpu, k, &format!("B={batch} {m}x{n}x{k} f16w"));
    }
}

#[test]
#[ignore]
fn f16_routes_pick_f16w_kernels_and_skip_splitk2() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    // Big aligned shapes take the tensor cores when available; the
    // SIMT large tile handles the unaligned siblings.  Route names are
    // asserted budget-aware: the exact measured winner wherever the
    // shared-memory gate admits its slot, and the documented in-budget
    // demotion (`in_budget_index`) where it stays empty.
    let large = exec.dispatch_info_for(1, 2048, 2048, 2048, true);
    if ctx.coopmat_enabled {
        assert_eq!(
            large.kernel, "f16w_coopmat_aligned",
            "large route: {large:?}"
        );
    } else if kernel_in_budget(&ctx, KernelSelection::F16wLargeBdaV4) {
        assert_eq!(large.kernel, "f16w_large_bda_v4", "large route: {large:?}");
    } else {
        // Sub-floor budget (software Vulkan): the family's 64x64 tile.
        assert_eq!(large.kernel, "f16w_small_bda_v4", "large route: {large:?}");
    }
    let unaligned = exec.dispatch_info_for(1, 2048, 2040, 2048, true);
    assert!(
        unaligned.kernel.starts_with("f16w_") && !unaligned.kernel.contains("coopmat"),
        "unaligned route: {unaligned:?}"
    );
    // 1024^3: tensor cores when available, else the f32 route class
    // mirror (m128n64k64) — a 49,664 B BK=64 tile the shared-memory
    // gate empties on 48 KiB devices (e.g. NVK on Turing), where the
    // route demotes to the family's 64x64 fallback.
    let mid = exec.dispatch_info_for(1, 1024, 1024, 1024, true);
    if ctx.coopmat_enabled {
        assert_eq!(mid.kernel, "f16w_coopmat_m64n64", "mid route: {mid:?}");
    } else if kernel_in_budget(&ctx, KernelSelection::F16wM128N64K64BdaV4) {
        assert_eq!(mid.kernel, "f16w_m128n64k64_bda_v4", "mid route: {mid:?}");
    } else {
        assert_eq!(mid.kernel, "f16w_small_bda_v4", "mid route: {mid:?}");
    }
    let row = exec.dispatch_info_for(1, 1, 4096, 4096, true);
    assert_eq!(row.kernel, "f16w_row_bda_k16", "row route: {row:?}");
    let row_shallow = exec.dispatch_info_for(1, 1, 4096, 2048, true);
    assert_eq!(
        row_shallow.kernel, "f16w_row_bda_k16_v2",
        "row route: {row_shallow:?}"
    );
    // Deep-K would tempt split-K2 on f32; f16 must stay data-parallel
    // on an f16w kernel.
    let deep = exec.dispatch_info_for(1, 37, 41, 4096, true);
    assert_eq!(deep.split_k2_splits, None, "deep-K route: {deep:?}");
    assert!(deep.kernel.starts_with("f16w_"), "deep-K kernel: {deep:?}");
    // And the same shapes with f32 B keep their f32 routes.
    let f32_large = exec.dispatch_info_for(1, 1024, 1024, 1024, false);
    assert!(!f32_large.kernel.starts_with("f16w_"), "{f32_large:?}");
}

#[test]
#[ignore]
fn f16_row_gemv_and_batched_epilogue() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    // Lone-row decode-style GEMV.
    let (m, n, k) = (1_u32, 2048_u32, 1024_u32);
    let (a, b, host_a, host_b) = setup_f16_case(&ctx, &exec, &[m, k], &[k, n], 9500, 9501);
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
    let cpu = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    assert_close(&gpu, &cpu, k, "gemv");

    // Batched broadcast-B with bias + SiLU + scaled residual: the
    // epilogue operands stay f32 while B is f16.
    let (batch, m, n, k) = (3_u32, 33_u32, 47_u32, 65_u32);
    let (a, b, host_a, host_b) =
        setup_f16_case(&ctx, &exec, &[batch, m, k], &[1, k, n], 9600, 9601);
    let c = Tensor::uninit_device(&ctx, &[batch, m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 9602);
    let (residual, host_residual) = upload_det(&ctx, &exec, &[batch, m, n], 9603);
    let beta = 0.5;
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
    let mut gpu = vec![0.0; (batch * m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let col = index % n as usize;
        let with_bias = *value + host_bias[col];
        let act = with_bias / (1.0 + (-with_bias).exp());
        *value = act + beta * host_residual[index];
    }
    assert_close(&gpu, &expected, k, "f16w epilogue");
}

#[test]
#[ignore]
fn f16_b_with_explicit_f32_kernel_is_rejected() {
    let (ctx, exec) = make_setup_with_kernel(2, 8, KernelSelection::LargeBdaV4);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    let a = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let b = Tensor::uninit_device_f16(&ctx, &[64, 64]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[64, 64]).unwrap();
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("expects f32 B storage"),
        "unexpected error: {err}"
    );

    // And f16 A with an f32 C is rejected (a16 needs f16 end to end).
    let a16 = Tensor::uninit_device_f16(&ctx, &[64, 64]).unwrap();
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a16,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires f16 C"), "unexpected error: {err}");
}

/// Upload helper for the f16-activations route: A, B, and C all f16.
fn setup_a16_case(
    ctx: &std::sync::Arc<tensor_ash::VulkanContext>,
    exec: &tensor_ash::Executor,
    a_shape: &[u32],
    b_shape: &[u32],
    seed_a: u64,
    seed_b: u64,
) -> (Tensor, Tensor, Vec<f32>, Vec<f32>) {
    let mut host_a = vec![0.0; Tensor::numel(a_shape) as usize];
    fill_det(&mut host_a, seed_a);
    let a = Tensor::uninit_device_f16(ctx, a_shape).unwrap();
    exec.upload(&host_a, &a).unwrap();
    let mut host_b = vec![0.0; Tensor::numel(b_shape) as usize];
    fill_det(&mut host_b, seed_b);
    let b = Tensor::uninit_device_f16(ctx, b_shape).unwrap();
    exec.upload(&host_b, &b).unwrap();
    for value in host_a.iter_mut().chain(host_b.iter_mut()) {
        *value = round_f32_via_f16(*value);
    }
    (a, b, host_a, host_b)
}

/// f16-C tolerance: the K-scaled GEMM bound plus the store's RNE
/// narrowing (half an f16 ulp of the largest reference magnitude).
fn a16_tolerance(cpu: &[f32], k: u32) -> f32 {
    let max_mag = cpu.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    tolerance(k) + max_mag * (0.5 / 1024.0)
}

#[test]
#[ignore]
fn a16_activations_coopmat_matches_rounded_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Aligned shapes across the prefill projection classes (batched
    // included); dual-rounded inputs make every product exact in f32,
    // so only the accumulation order and the f16 C store differ.
    let cases: &[(u32, u32, u32, u32)] = &[
        (1, 256, 384, 256),
        (1, 128, 128, 64),
        (1, 512, 256, 2048),
        (2, 128, 256, 96),
        // 64-tile-aligned but not 128-aligned: only the wave-fill kernel.
        (1, 192, 320, 64),
    ];
    for &(batch, m, n, k) in cases {
        let shape = |rows: u32, cols: u32| -> Vec<u32> {
            if batch == 1 {
                vec![rows, cols]
            } else {
                vec![batch, rows, cols]
            }
        };
        let (a, b, host_a, host_b) = setup_a16_case(
            &ctx,
            &exec,
            &shape(m, k),
            &shape(k, n),
            9900 + k as u64,
            9910 + n as u64,
        );
        let c = Tensor::uninit_device_f16(&ctx, &shape(m, n)).unwrap();
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
        assert_close_tol(
            &gpu,
            &cpu,
            a16_tolerance(&cpu, k),
            &format!("B={batch} {m}x{n}x{k} a16"),
        );
    }
}

#[test]
#[ignore]
fn a16_alpha_accumulate_and_graph_path_match() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Alpha + accumulate onto an existing f16 C.
    let (m, n, k) = (128_u32, 256_u32, 128_u32);
    let (a, b, host_a, host_b) = setup_a16_case(&ctx, &exec, &[m, k], &[k, n], 9920, 9921);
    let c = Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap();
    let mut host_c = vec![0.0; (m * n) as usize];
    fill_det(&mut host_c, 9922);
    exec.upload(&host_c, &c).unwrap();
    for value in &mut host_c {
        *value = round_f32_via_f16(*value);
    }
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
    let cpu = cpu_bmm(&host_a, &host_b, Some(&host_c), 1, m, n, k, alpha, true);
    assert_close_tol(&gpu, &cpu, a16_tolerance(&cpu, k), "a16 alpha+accumulate");

    // The same op through the mixed-graph path (what prefill uses).
    let c2 = Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap();
    exec.run_exec_ops(&[tensor_ash::ExecOp::Matmul(MatmulOp::new(MatmulCall {
        a: &a,
        b: &b,
        c: &c2,
        alpha: 1.0,
        accumulate: false,
    }))])
    .unwrap();
    let mut gpu2 = vec![0.0; (m * n) as usize];
    exec.download(&c2, &mut gpu2).unwrap();
    let cpu2 = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    assert_close_tol(&gpu2, &cpu2, a16_tolerance(&cpu2, k), "a16 graph path");
}

/// Prefill QKV concat: one wide GEMM + row-splits must match three
/// independent projections (and the CPU reference) on an a16 shape
/// that is 64-aligned but not 128-aligned, so it hits the wave-fill
/// tile the llama path uses at T=128/512.
#[test]
#[ignore]
fn a16_qkv_concat_matches_three_gemms() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    let (t, k, n_q, n_kv) = (192_u32, 256_u32, 256_u32, 64_u32);
    let n_qkv = n_q + 2 * n_kv;
    let (x, wq, host_x, host_wq) = setup_a16_case(&ctx, &exec, &[t, k], &[k, n_q], 11_000, 11_001);
    let (wk, host_wk) = {
        let mut host = vec![0.0; (k * n_kv) as usize];
        fill_det(&mut host, 11_002);
        let w = Tensor::uninit_device_f16(&ctx, &[k, n_kv]).unwrap();
        exec.upload(&host, &w).unwrap();
        for v in &mut host {
            *v = round_f32_via_f16(*v);
        }
        (w, host)
    };
    let (wv, host_wv) = {
        let mut host = vec![0.0; (k * n_kv) as usize];
        fill_det(&mut host, 11_003);
        let w = Tensor::uninit_device_f16(&ctx, &[k, n_kv]).unwrap();
        exec.upload(&host, &w).unwrap();
        for v in &mut host {
            *v = round_f32_via_f16(*v);
        }
        (w, host)
    };
    let mut host_qkv = vec![0.0; (k * n_qkv) as usize];
    for row in 0..k as usize {
        let dst = row * n_qkv as usize;
        let q = row * n_q as usize;
        host_qkv[dst..dst + n_q as usize].copy_from_slice(&host_wq[q..q + n_q as usize]);
        let kv = row * n_kv as usize;
        host_qkv[dst + n_q as usize..dst + n_q as usize + n_kv as usize]
            .copy_from_slice(&host_wk[kv..kv + n_kv as usize]);
        host_qkv[dst + n_q as usize + n_kv as usize..dst + n_qkv as usize]
            .copy_from_slice(&host_wv[kv..kv + n_kv as usize]);
    }
    let w_qkv = Tensor::uninit_device_f16(&ctx, &[k, n_qkv]).unwrap();
    exec.upload(&host_qkv, &w_qkv).unwrap();

    let qkv = Tensor::uninit_device_f16(&ctx, &[t, n_qkv]).unwrap();
    let q = Tensor::uninit_device_f16(&ctx, &[t, n_q]).unwrap();
    let k_out = Tensor::uninit_device_f16(&ctx, &[t, n_kv]).unwrap();
    let v = Tensor::uninit_device_f16(&ctx, &[t, n_kv]).unwrap();
    exec.run_exec_ops(&[
        ExecOp::Matmul(MatmulOp::new(MatmulCall {
            a: &x,
            b: &w_qkv,
            c: &qkv,
            alpha: 1.0,
            accumulate: false,
        })),
        ExecOp::CopyStrided {
            src: &qkv,
            dst: &q,
            desc: CopyDesc {
                extent: [n_q, t, 1],
                src_offset: 0,
                src_strides: [1, n_qkv, 0],
                dst_offset: 0,
                dst_strides: [1, n_q, 0],
                ..Default::default()
            },
        },
        ExecOp::CopyStrided {
            src: &qkv,
            dst: &k_out,
            desc: CopyDesc {
                extent: [n_kv, t, 1],
                src_offset: n_q,
                src_strides: [1, n_qkv, 0],
                dst_offset: 0,
                dst_strides: [1, n_kv, 0],
                ..Default::default()
            },
        },
        ExecOp::CopyStrided {
            src: &qkv,
            dst: &v,
            desc: CopyDesc {
                extent: [n_kv, t, 1],
                src_offset: n_q + n_kv,
                src_strides: [1, n_qkv, 0],
                dst_offset: 0,
                dst_strides: [1, n_kv, 0],
                ..Default::default()
            },
        },
    ])
    .unwrap();

    let mut gpu_q = vec![0.0; (t * n_q) as usize];
    let mut gpu_k = vec![0.0; (t * n_kv) as usize];
    let mut gpu_v = vec![0.0; (t * n_kv) as usize];
    exec.download(&q, &mut gpu_q).unwrap();
    exec.download(&k_out, &mut gpu_k).unwrap();
    exec.download(&v, &mut gpu_v).unwrap();

    let cpu_q = cpu_bmm(&host_x, &host_wq, None, 1, t, n_q, k, 1.0, false);
    let cpu_k = cpu_bmm(&host_x, &host_wk, None, 1, t, n_kv, k, 1.0, false);
    let cpu_v = cpu_bmm(&host_x, &host_wv, None, 1, t, n_kv, k, 1.0, false);
    assert_close_tol(&gpu_q, &cpu_q, a16_tolerance(&cpu_q, k), "qkv concat q");
    assert_close_tol(&gpu_k, &cpu_k, a16_tolerance(&cpu_k, k), "qkv concat k");
    assert_close_tol(&gpu_v, &cpu_v, a16_tolerance(&cpu_v, k), "qkv concat v");

    // Separate projections on the same inputs must match the split.
    let q2 = Tensor::uninit_device_f16(&ctx, &[t, n_q]).unwrap();
    let k2 = Tensor::uninit_device_f16(&ctx, &[t, n_kv]).unwrap();
    let v2 = Tensor::uninit_device_f16(&ctx, &[t, n_kv]).unwrap();
    exec.run_matmuls(&[
        MatmulCall {
            a: &x,
            b: &wq,
            c: &q2,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &x,
            b: &wk,
            c: &k2,
            alpha: 1.0,
            accumulate: false,
        },
        MatmulCall {
            a: &x,
            b: &wv,
            c: &v2,
            alpha: 1.0,
            accumulate: false,
        },
    ])
    .unwrap();
    let mut sep_q = vec![0.0; (t * n_q) as usize];
    let mut sep_k = vec![0.0; (t * n_kv) as usize];
    let mut sep_v = vec![0.0; (t * n_kv) as usize];
    exec.download(&q2, &mut sep_q).unwrap();
    exec.download(&k2, &mut sep_k).unwrap();
    exec.download(&v2, &mut sep_v).unwrap();
    assert_close_tol(&gpu_q, &sep_q, a16_tolerance(&sep_q, k), "concat vs 3x q");
    assert_close_tol(&gpu_k, &sep_k, a16_tolerance(&sep_k, k), "concat vs 3x k");
    assert_close_tol(&gpu_v, &sep_v, a16_tolerance(&sep_v, k), "concat vs 3x v");
}

/// Packed-B row GEMV (decode layout) vs the unpacked kernel and a CPU
/// reference.  Covers both tile_n=32 (narrow N) and tile_n=64 (wide N).
#[test]
#[ignore]
fn packed_row_gemv_matches_unpacked_and_cpu() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) {
        eprintln!("skipping: no f16/BDA support");
        return;
    }
    let cases: &[(u32, u32)] = &[
        (2048, 256),  // k/v: k16 tile 32
        (2048, 2048), // q/o: k16 tile 32
        (2048, 5632), // gate/up: k16_v2 tile 64
    ];
    for &(k, n) in cases {
        let tile = f16w_row_tile_n(k, n);
        let (a, _) = upload_det(&ctx, &exec, &[1, k], 12_000 + n as u64);
        let mut host_b = vec![0.0; (k * n) as usize];
        fill_det(&mut host_b, 12_100 + n as u64);
        let b = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
        exec.upload(&host_b, &b).unwrap();
        for v in &mut host_b {
            *v = round_f32_via_f16(*v);
        }
        let packed_host = pack_f16w_row_tiles(&host_b, k as usize, n as usize, tile as usize);
        let b_p = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
        exec.upload(&packed_host, &b_p).unwrap();

        let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
        let c_p = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
        exec.run_ops(&[MatmulOp::new(MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        })])
        .unwrap();
        exec.run_ops(&[MatmulOp::new(MatmulCall {
            a: &a,
            b: &b_p,
            c: &c_p,
            alpha: 1.0,
            accumulate: false,
        })
        .with_packed_b()])
            .unwrap();
        let mut gpu = vec![0.0; n as usize];
        let mut gpu_p = vec![0.0; n as usize];
        exec.download(&c, &mut gpu).unwrap();
        exec.download(&c_p, &mut gpu_p).unwrap();
        let mut host_a = vec![0.0; k as usize];
        exec.download(&a, &mut host_a).unwrap();
        let cpu = cpu_bmm(&host_a, &host_b, None, 1, 1, n, k, 1.0, false);
        assert_close(&gpu_p, &cpu, k, &format!("packed {k}x{n} vs cpu"));
        assert_close(&gpu_p, &gpu, k, &format!("packed vs unpacked {k}x{n}"));
    }

    // Decode path: packed + folded RMSNorm vs f64 RMS then GEMM.
    let (k, n) = (2048_u32, 256_u32);
    let tile = f16w_row_tile_n(k, n);
    let (a, host_a) = upload_det(&ctx, &exec, &[1, k], 13_001);
    let (w, host_w) = upload_det(&ctx, &exec, &[k], 13_002);
    let mut host_b = vec![0.0; (k * n) as usize];
    fill_det(&mut host_b, 13_003);
    let packed_host = {
        let mut b = host_b.clone();
        for v in &mut b {
            *v = round_f32_via_f16(*v);
        }
        host_b = b.clone();
        pack_f16w_row_tiles(&b, k as usize, n as usize, tile as usize)
    };
    let b_p = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
    exec.upload(&packed_host, &b_p).unwrap();
    let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let eps = 1e-5_f32;
    exec.run_ops(&[MatmulOp::new(MatmulCall {
        a: &a,
        b: &b_p,
        c: &c,
        alpha: 1.0,
        accumulate: false,
    })
    .with_packed_b()
    .with_normed_a(&w, eps)])
        .unwrap();
    let mut gpu = vec![0.0; n as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mean = host_a.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / k as f64;
    let inv = 1.0 / (mean + eps as f64).sqrt();
    let xn: Vec<f32> = host_a
        .iter()
        .zip(&host_w)
        .map(|(&v, &ww)| (v as f64 * inv * ww as f64) as f32)
        .collect();
    let cpu = cpu_bmm(&xn, &host_b, None, 1, 1, n, k, 1.0, false);
    assert_close(&gpu, &cpu, k, "packed + normed-A vs f64 RMS+GEMM");

    // Gate-shaped packed GEMV with Silu+Mul epilogue (decode MLP).
    let (k, n) = (2048_u32, 5632_u32);
    let tile = f16w_row_tile_n(k, n);
    let (a, host_a) = upload_det(&ctx, &exec, &[1, k], 14_001);
    let (d, host_d) = upload_det(&ctx, &exec, &[1, n], 14_002);
    let mut host_b = vec![0.0; (k * n) as usize];
    fill_det(&mut host_b, 14_003);
    for v in &mut host_b {
        *v = round_f32_via_f16(*v);
    }
    let packed = pack_f16w_row_tiles(&host_b, k as usize, n as usize, tile as usize);
    let b_p = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
    exec.upload(&packed, &b_p).unwrap();
    let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    exec.run_ops(&[MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b_p,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        Epilogue {
            bias: None,
            activation: Activation::Silu,
            binary: EpilogueBinary::Mul { d: &d },
        },
    )
    .with_packed_b()])
        .unwrap();
    let mut gpu = vec![0.0; n as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut cpu = cpu_bmm(&host_a, &host_b, None, 1, 1, n, k, 1.0, false);
    for (value, &dv) in cpu.iter_mut().zip(&host_d) {
        *value = (*value / (1.0 + (-*value).exp())) * dv;
    }
    assert_close(&gpu, &cpu, k, "packed silu*mul epilogue");
}

#[test]
#[ignore]
fn a16_invalid_combinations_are_rejected() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    let a = Tensor::uninit_device_f16(&ctx, &[128, 128]).unwrap();
    let b32 = Tensor::uninit_device(&ctx, &[128, 128]).unwrap();
    let c = Tensor::uninit_device_f16(&ctx, &[128, 128]).unwrap();
    // f16 A with f32 B.
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a,
            b: &b32,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires f16 B"), "unexpected error: {err}");
    // Misaligned shape has no a16 route.
    let a_ragged = Tensor::uninit_device_f16(&ctx, &[100, 128]).unwrap();
    let b16 = Tensor::uninit_device_f16(&ctx, &[128, 128]).unwrap();
    let c_ragged = Tensor::uninit_device_f16(&ctx, &[100, 128]).unwrap();
    let err = exec
        .run_matmuls(&[MatmulCall {
            a: &a_ragged,
            b: &b16,
            c: &c_ragged,
            alpha: 1.0,
            accumulate: false,
        }])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("multiples of (64, 64, 32)"),
        "unexpected error: {err}"
    );
    // Fusions cannot ride the a16 route.
    let err = exec
        .run_ops(&[MatmulOp::with_epilogue(
            MatmulCall {
                a: &a,
                b: &b16,
                c: &c,
                alpha: 1.0,
                accumulate: false,
            },
            Epilogue {
                bias: None,
                activation: Activation::Silu,
                binary: EpilogueBinary::None,
            },
        )])
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot fuse"), "unexpected error: {err}");
}

#[test]
#[ignore]
fn coopmat_routes_and_matches_dual_rounded_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Aligned + big => tensor cores; small or ragged stays SIMT.
    let big = exec.dispatch_info_for(1, 1024, 1024, 1024, true);
    assert_eq!(big.kernel, "f16w_coopmat_m64n64", "{big:?}");
    let small = exec.dispatch_info_for(1, 128, 128, 128, true);
    assert_eq!(small.kernel, "f16w_coopmat_m64n64", "{small:?}");
    let ragged = exec.dispatch_info_for(1, 1024, 1000, 1024, true);
    assert_ne!(ragged.kernel, "f16w_coopmat_aligned", "{ragged:?}");

    // Batched + alpha + accumulate against a dual-rounded reference:
    // f16 x f16 products are exact in f32, so tolerance(k) holds.
    let (batch, m, n, k) = (2_u32, 256_u32, 384_u32, 256_u32);
    let (a, b, mut host_a, host_b) =
        setup_f16_case(&ctx, &exec, &[batch, m, k], &[batch, k, n], 9700, 9701);
    for value in &mut host_a {
        *value = round_f32_via_f16(*value);
    }
    exec.upload(&host_a, &a).unwrap();
    let (c, host_c) = upload_det(&ctx, &exec, &[batch, m, n], 9702);
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
    assert_close(&gpu, &cpu, k, "coopmat");
}

/// The asymmetric wave-fill tiles (64x128 / 128x64) are explicit-only
/// until a heuristic A/B lands; this pins they match the dual-rounded
/// reference on a shape both tiles cover.
#[test]
#[ignore]
fn coopmat_asymmetric_tiles_match_dual_rounded_reference() {
    let selections = [
        KernelSelection::F16wCoopmatM64N128,
        KernelSelection::F16wCoopmatM128N64,
        KernelSelection::F16wA16CoopmatM64N128,
        KernelSelection::F16wA16CoopmatM128N64,
    ];
    let (batch, m, n, k) = (1_u32, 256, 384, 256);
    for selection in selections {
        let Some((ctx, exec)) = make_setup_with_kernel_if_fits(2, 8, selection) else {
            continue;
        };
        if !f16_available(&ctx) || !ctx.coopmat_enabled {
            continue;
        }
        let a16 = selection_is_a16(selection);
        let (a, b, host_a, host_b) = if a16 {
            setup_a16_case(&ctx, &exec, &[m, k], &[k, n], 9800, 9801)
        } else {
            let (a, b, mut host_a, host_b) =
                setup_f16_case(&ctx, &exec, &[m, k], &[k, n], 9800, 9801);
            for value in &mut host_a {
                *value = round_f32_via_f16(*value);
            }
            exec.upload(&host_a, &a).unwrap();
            (a, b, host_a, host_b)
        };
        let c = if a16 {
            Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap()
        } else {
            Tensor::uninit_device(&ctx, &[m, n]).unwrap()
        };
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
        let cpu = cpu_bmm(&host_a, &host_b, None, batch, m, n, k, 1.0, false);
        if a16 {
            assert_close_tol(
                &gpu,
                &cpu,
                a16_tolerance(&cpu, k),
                &format!("asymmetric {}", kernel_sel_name(selection)),
            );
        } else {
            assert_close(
                &gpu,
                &cpu,
                k,
                &format!("asymmetric {}", kernel_sel_name(selection)),
            );
        }
    }
}

fn selection_is_a16(selection: KernelSelection) -> bool {
    selection
        .index()
        .is_some_and(|i| tensor_ash::KERNEL_SPECS[i].a_f16())
}

fn kernel_sel_name(selection: KernelSelection) -> &'static str {
    selection
        .index()
        .map(|i| tensor_ash::KERNEL_SPECS[i].name)
        .unwrap_or("?")
}

/// Regression: TinyLlama prefill surfaced that aligned f16 shapes
/// route to the coopmat kernel, which cannot fuse epilogues — fused
/// ops must demote to the SIMT sibling (or, with coopmat2, reroute to
/// the CM2 tensor-core GEMM) while plain ops keep the tensor cores,
/// and explicit selections keep their loud failure.
#[test]
#[ignore]
fn f16_epilogue_on_coopmat_shape_demotes_and_matches() {
    let (ctx, exec) = make_setup(2, 8);
    if !f16_available(&ctx) || !ctx.coopmat_enabled {
        eprintln!("skipping: no coopmat support");
        return;
    }
    // Plain route for this aligned shape is the tensor-core kernel.
    let (m, n, k) = (256_u32, 384_u32, 256_u32);
    let plain = exec.dispatch_info_for(1, m, n, k, true);
    assert_eq!(plain.kernel, "f16w_coopmat_m64n64", "{plain:?}");

    // A fused bias+SiLU+gate op on the same shape must run (demoted)
    // and match the reference computed from f16-rounded inputs.
    let (a, b, mut host_a, host_b) = setup_f16_case(&ctx, &exec, &[m, k], &[k, n], 9800, 9801);
    // Without coopmat2 the op demotes to a SIMT kernel that reads A as
    // f32 (no rounding).  With coopmat2 it reroutes to the CM2
    // tensor-core GEMM instead, which quantizes A to f16 at load —
    // mirror that in the reference.
    if ctx.coopmat2_enabled {
        for value in &mut host_a {
            *value = round_f32_via_f16(*value);
        }
        exec.upload(&host_a, &a).unwrap();
    }
    let c = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let (bias, host_bias) = upload_det(&ctx, &exec, &[n], 9802);
    let (gate, host_gate) = upload_det(&ctx, &exec, &[m, n], 9803);
    exec.run_ops(&[tensor_ash::MatmulOp::with_epilogue(
        MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha: 1.0,
            accumulate: false,
        },
        tensor_ash::Epilogue {
            bias: Some(&bias),
            activation: tensor_ash::Activation::Silu,
            binary: tensor_ash::EpilogueBinary::Mul { d: &gate },
        },
    )])
    .unwrap();
    let mut gpu = vec![0.0; (m * n) as usize];
    exec.download(&c, &mut gpu).unwrap();
    let mut expected = cpu_bmm(&host_a, &host_b, None, 1, m, n, k, 1.0, false);
    for (index, value) in expected.iter_mut().enumerate() {
        let with_bias = *value + host_bias[index % n as usize];
        let act = with_bias / (1.0 + (-with_bias).exp());
        *value = act * host_gate[index];
    }
    assert_close(&gpu, &expected, k, "demoted epilogue on coopmat shape");
}
