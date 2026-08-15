//! Fused store-epilogue correctness: the M=1 f16-weights row GEMVs
//! rotating (RoPE) and/or scattering their output row into KV-cache
//! layouts at store time.  Every fused result is compared BITWISE
//! against the composed reference (same GEMV route, then the
//! standalone rope / copy / rope-scatter op): the reduce order and the
//! rotation fma order are identical, so even f16 narrowing rounds
//! identically.  The bitwise form holds only where the driver compiles
//! that identical arithmetic identically across the STORE_MODE
//! pipeline variants ([`VulkanContext::fused_store_bit_reproducible`]);
//! on drivers that do not (Mesa NVK today) the fused-vs-composed
//! comparison relaxes to the suite's standard k-scaled tolerance.
//! Same-pipeline comparisons (pos indirection) stay bitwise
//! everywhere.

use crate::common::*;

use tensor_ash::{
    Activation, CopyDesc, Epilogue, EpilogueBinary, ExecOp, Executor, KernelSelection, MatmulCall,
    MatmulOp, MatmulStoreDesc, RopeDesc, RopeScatterDesc, Tensor, VulkanContext,
};

fn store_available(ctx: &std::sync::Arc<VulkanContext>) -> bool {
    ctx.buffer_device_address_enabled && ctx.f16_storage_enabled
}

/// Upload one store-GEMV case: f32 A `[1, k]`, f16 B `[k, n]`, f32
/// rope table `[t_max, head_dim/2, 2]`.
fn setup_case(
    ctx: &std::sync::Arc<VulkanContext>,
    exec: &Executor,
    k: u32,
    n: u32,
    head_dim: u32,
    t_max: u32,
    seed: u64,
) -> (Tensor, Tensor, Tensor) {
    let (a, _) = upload_det(ctx, exec, &[1, k], seed);
    let mut host_b = vec![0.0; (k * n) as usize];
    fill_det(&mut host_b, seed + 1);
    let b = Tensor::uninit_device_f16(ctx, &[k, n]).unwrap();
    exec.upload(&host_b, &b).unwrap();
    let (table, _) = upload_det(ctx, exec, &[t_max, head_dim / 2, 2], seed + 2);
    (a, b, table)
}

fn mm<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
    MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }
}

fn assert_bits_eq(expected: &[f32], actual: &[f32], label: &str) {
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "{label}: element {index}: ref {want}, fused {got}"
        );
    }
}

/// Fused-vs-composed comparison: bitwise wherever the driver keeps
/// spec-constant pipeline variants bit-reproducible (see module doc),
/// the standard `tolerance(k)` where it does not.  Only the assertion
/// form changes — both branches compare the same fused result against
/// the same composed reference.
fn assert_fused_matches(
    ctx: &std::sync::Arc<VulkanContext>,
    expected: &[f32],
    actual: &[f32],
    k: u32,
    label: &str,
) {
    if ctx.fused_store_bit_reproducible() {
        assert_bits_eq(expected, actual, label);
    } else {
        eprintln!(
            "{label}: driver lacks cross-variant bit reproducibility; \
             comparing within tolerance instead of bitwise"
        );
        assert_close(actual, expected, k, label);
    }
}

/// Fused rope-at-store vs the same GEMV followed by the standalone
/// in-place rope, across the k16 (VCOLS=1, partner via shared
/// partials) and k16_v2 (VCOLS=2, lane-local partner) kernels,
/// interior and ragged-N.
#[test]
#[ignore]
fn store_rope_matches_gemv_plus_standalone_rope() {
    // (kernel, k, n, head_dim): n % 32 != 0 (k16) and n % 64 != 0
    // (v2) exercise the bounds-checked variants.
    let cases: &[(KernelSelection, u32, u32, u32)] = &[
        (KernelSelection::F16wRowBdaK16, 96, 128, 32),
        (KernelSelection::F16wRowBdaK16, 96, 48, 16),
        (KernelSelection::F16wRowBdaK16V2, 96, 128, 32),
        (KernelSelection::F16wRowBdaK16V2, 96, 160, 32),
    ];
    for &(kernel, k, n, head_dim) in cases {
        let (ctx, exec) = make_setup_with_kernel(2, 8, kernel);
        if !store_available(&ctx) {
            eprintln!("skipping: no BDA/f16 support");
            return;
        }
        let (t_max, pos, eps) = (16_u32, 5_u32, 1e-5_f32);
        let (a, b, table) = setup_case(&ctx, &exec, k, n, head_dim, t_max, 61_000 + n as u64);
        let (w, _) = upload_det(&ctx, &exec, &[k], 61_900 + n as u64);
        let c_ref = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
        let c_fused = Tensor::uninit_device(&ctx, &[1, n]).unwrap();

        exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_ref)).with_normed_a(&w, eps)])
            .unwrap();
        exec.run_rope(
            &c_ref,
            &table,
            &c_ref,
            RopeDesc {
                heads: n / head_dim,
                head_dim,
                rot_dim: head_dim,
                pos_base: pos,
                ..Default::default()
            },
        )
        .unwrap();

        exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_fused))
            .with_normed_a(&w, eps)
            .with_store_rope(
                &table,
                MatmulStoreDesc {
                    head_dim,
                    pos_base: pos,
                    ..Default::default()
                },
            )])
        .unwrap();

        let mut expected = vec![0.0; n as usize];
        let mut actual = vec![0.0; n as usize];
        exec.download(&c_ref, &mut expected).unwrap();
        exec.download(&c_fused, &mut actual).unwrap();
        assert_fused_matches(
            &ctx,
            &expected,
            &actual,
            k,
            &format!("store rope {kernel:?} n={n}"),
        );
    }
}

/// Fused V-append scatter vs GEMV + strided copy into a poisoned
/// `[heads, t_max, dh]` cache, f32 and f16 storage.
fn store_scatter_case(kv_f16: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !store_available(&ctx) {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (k, heads, dh, t_max, pos) = (96_u32, 4_u32, 32_u32, 16_u32, 5_u32);
    let n = heads * dh;
    let (a, b, _table) = setup_case(&ctx, &exec, k, n, dh, t_max, 62_000);
    let cache_shape = [heads, t_max, dh];
    let poison = vec![7.0_f32; Tensor::numel(&cache_shape) as usize];
    let make_cache = || {
        let cache = if kv_f16 {
            Tensor::uninit_device_f16(&ctx, &cache_shape).unwrap()
        } else {
            Tensor::uninit_device(&ctx, &cache_shape).unwrap()
        };
        exec.upload(&poison, &cache).unwrap();
        cache
    };
    let cache_ref = make_cache();
    let cache_fused = make_cache();
    let c_ref = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_scratch = Tensor::uninit_device(&ctx, &[1, n]).unwrap();

    exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_ref))]).unwrap();
    exec.run_copy_strided(
        &c_ref,
        &cache_ref,
        CopyDesc {
            extent: [dh, heads, 1],
            src_offset: 0,
            src_strides: [1, dh, n],
            dst_offset: pos * dh,
            dst_strides: [1, t_max * dh, dh],
            ..Default::default()
        },
    )
    .unwrap();

    exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_scratch)).with_store_scatter(
        &cache_fused,
        MatmulStoreDesc {
            head_dim: dh,
            pos_base: pos,
            pos_scale: dh,
            stride_head: t_max * dh,
            stride_dim: 1,
            ..Default::default()
        },
    )])
    .unwrap();

    let mut expected = vec![0.0; poison.len()];
    let mut actual = vec![0.0; poison.len()];
    exec.download(&cache_ref, &mut expected).unwrap();
    exec.download(&cache_fused, &mut actual).unwrap();
    assert_fused_matches(
        &ctx,
        &expected,
        &actual,
        k,
        &format!("store scatter f16={kv_f16}"),
    );
}

#[test]
#[ignore]
fn store_scatter_matches_gemv_plus_copy_f32() {
    store_scatter_case(false);
}

#[test]
#[ignore]
fn store_scatter_matches_gemv_plus_copy_f16() {
    store_scatter_case(true);
}

/// Fused k rope-scatter vs GEMV + the standalone rope-scatter into a
/// poisoned `[heads, dh, t_max]` Kt cache, f32 and f16 storage.
fn store_rope_scatter_case(kv_f16: bool) {
    let (ctx, exec) = make_setup(2, 8);
    if !store_available(&ctx) {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (k, heads, dh, t_max, pos, eps) = (96_u32, 4_u32, 32_u32, 16_u32, 5_u32, 1e-5_f32);
    let n = heads * dh;
    let (a, b, table) = setup_case(&ctx, &exec, k, n, dh, t_max, 63_000);
    let (w, _) = upload_det(&ctx, &exec, &[k], 63_900);
    let cache_shape = [heads, dh, t_max];
    let poison = vec![7.0_f32; Tensor::numel(&cache_shape) as usize];
    let make_cache = || {
        let cache = if kv_f16 {
            Tensor::uninit_device_f16(&ctx, &cache_shape).unwrap()
        } else {
            Tensor::uninit_device(&ctx, &cache_shape).unwrap()
        };
        exec.upload(&poison, &cache).unwrap();
        cache
    };
    let cache_ref = make_cache();
    let cache_fused = make_cache();
    let c_ref = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_scratch = Tensor::uninit_device(&ctx, &[1, n]).unwrap();

    exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_ref)).with_normed_a(&w, eps)])
        .unwrap();
    exec.run_rope_scatter(
        &c_ref,
        &table,
        &cache_ref,
        RopeDesc {
            heads,
            head_dim: dh,
            rot_dim: dh,
            pos_base: pos,
            ..Default::default()
        },
        RopeScatterDesc {
            dst_offset: pos,
            dst_strides: [1, dh * t_max, t_max],
            ..Default::default()
        },
    )
    .unwrap();

    exec.run_ops(&[MatmulOp::new(mm(&a, &b, &c_scratch))
        .with_normed_a(&w, eps)
        .with_store_rope_scatter(
            &table,
            &cache_fused,
            MatmulStoreDesc {
                head_dim: dh,
                pos_base: pos,
                pos_scale: 1,
                stride_head: dh * t_max,
                stride_dim: t_max,
                ..Default::default()
            },
        )])
    .unwrap();

    let mut expected = vec![0.0; poison.len()];
    let mut actual = vec![0.0; poison.len()];
    exec.download(&cache_ref, &mut expected).unwrap();
    exec.download(&cache_fused, &mut actual).unwrap();
    assert_fused_matches(
        &ctx,
        &expected,
        &actual,
        k,
        &format!("store rope-scatter f16={kv_f16}"),
    );
}

#[test]
#[ignore]
fn store_rope_scatter_matches_gemv_plus_rope_scatter_f32() {
    store_rope_scatter_case(false);
}

#[test]
#[ignore]
fn store_rope_scatter_matches_gemv_plus_rope_scatter_f16() {
    store_rope_scatter_case(true);
}

/// The pos-cell indirection: a store op recorded with `pos_base = b`
/// and a position buffer holding `p` must land exactly where
/// `pos_base = b + p` without the buffer lands.
#[test]
#[ignore]
fn store_pos_indirection_offsets_position() {
    let (ctx, exec) = make_setup(2, 8);
    if !store_available(&ctx) {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (k, heads, dh, t_max) = (64_u32, 2_u32, 16_u32, 12_u32);
    let n = heads * dh;
    let (base, p) = (2_u32, 3_u32);
    let (a, b, table) = setup_case(&ctx, &exec, k, n, dh, t_max, 64_000);
    let cache_shape = [heads, dh, t_max];
    let zeros = vec![0.0_f32; Tensor::numel(&cache_shape) as usize];
    let make_cache = || {
        let cache = Tensor::uninit_device(&ctx, &cache_shape).unwrap();
        exec.upload(&zeros, &cache).unwrap();
        cache
    };
    let cache_direct = make_cache();
    let cache_indirect = make_cache();
    let c_scratch = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let desc = |pos_base, pos_addr| MatmulStoreDesc {
        head_dim: dh,
        pos_base,
        pos_scale: 1,
        stride_head: dh * t_max,
        stride_dim: t_max,
        pos_addr,
    };

    exec.run_ops(&[
        MatmulOp::new(mm(&a, &b, &c_scratch)).with_store_rope_scatter(
            &table,
            &cache_direct,
            desc(base + p, 0),
        ),
    ])
    .unwrap();

    let pos_buf = exec.create_pos_buffer().unwrap();
    pos_buf.set(p).unwrap();
    exec.run_exec_ops(&[ExecOp::Matmul(
        MatmulOp::new(mm(&a, &b, &c_scratch)).with_store_rope_scatter(
            &table,
            &cache_indirect,
            desc(base, pos_buf.device_address()),
        ),
    )])
    .unwrap();

    let mut expected = vec![0.0; zeros.len()];
    let mut actual = vec![0.0; zeros.len()];
    exec.download(&cache_direct, &mut expected).unwrap();
    exec.download(&cache_indirect, &mut actual).unwrap();
    assert_bits_eq(&expected, &actual, "store pos indirection");
}

#[test]
#[ignore]
fn store_rejects_conflicting_forms() {
    let (ctx, exec) = make_setup(2, 8);
    if !store_available(&ctx) {
        eprintln!("skipping: no BDA/f16 support");
        return;
    }
    let (k, heads, dh, t_max) = (64_u32, 2_u32, 16_u32, 8_u32);
    let n = heads * dh;
    let (a, b, table) = setup_case(&ctx, &exec, k, n, dh, t_max, 65_000);
    let (a_wide, _) = upload_det(&ctx, &exec, &[4, k], 65_100);
    let (b_f32, _) = upload_det(&ctx, &exec, &[k, n], 65_101);
    let (d, _) = upload_det(&ctx, &exec, &[1, n], 65_102);
    let cache = Tensor::uninit_device(&ctx, &[heads, dh, t_max]).unwrap();
    let c = Tensor::uninit_device(&ctx, &[1, n]).unwrap();
    let c_wide = Tensor::uninit_device(&ctx, &[4, n]).unwrap();
    let desc = MatmulStoreDesc {
        head_dim: dh,
        pos_base: 0,
        pos_scale: 1,
        stride_head: dh * t_max,
        stride_dim: t_max,
        ..Default::default()
    };
    let rope = |call| MatmulOp::new(call).with_store_rope(&table, desc);

    let err = exec
        .run_ops(&[rope(mm(&a_wide, &b, &c_wide))])
        .unwrap_err()
        .to_string();
    assert!(err.contains("M == 1"), "unexpected: {err}");

    let err = exec
        .run_ops(&[rope(mm(&a, &b_f32, &c))])
        .unwrap_err()
        .to_string();
    assert!(err.contains("f16 weights"), "unexpected: {err}");

    let err = exec
        .run_ops(&[rope(MatmulCall {
            accumulate: true,
            ..mm(&a, &b, &c)
        })])
        .unwrap_err()
        .to_string();
    assert!(err.contains("accumulate"), "unexpected: {err}");

    let err = exec
        .run_ops(&[MatmulOp::with_epilogue(
            mm(&a, &b, &c),
            Epilogue {
                bias: None,
                activation: Activation::Silu,
                binary: EpilogueBinary::Mul { d: &d },
            },
        )
        .with_store_rope(&table, desc)])
        .unwrap_err()
        .to_string();
    assert!(err.contains("fused epilogue"), "unexpected: {err}");

    let err = exec
        .run_ops(&[MatmulOp::new(mm(&a, &b, &c)).with_store_rope(
            &table,
            MatmulStoreDesc {
                head_dim: dh + 1,
                ..desc
            },
        )])
        .unwrap_err()
        .to_string();
    assert!(err.contains("head_dim"), "unexpected: {err}");

    // Position past the cache extent must fail the bounds check
    // (scatter form, so the table-coverage check cannot fire first).
    let err = exec
        .run_ops(&[MatmulOp::new(mm(&a, &b, &c)).with_store_scatter(
            &cache,
            MatmulStoreDesc {
                pos_base: t_max,
                ..desc
            },
        )])
        .unwrap_err()
        .to_string();
    assert!(err.contains("out of bounds"), "unexpected: {err}");

    // An explicitly selected non-row f16w kernel must fail loudly at
    // record time rather than silently skip the rope/scatter.
    let (ctx2, exec2) = make_setup_with_kernel(2, 8, KernelSelection::F16wK64BdaV4);
    if store_available(&ctx2) {
        let (a2, b2, table2) = setup_case(&ctx2, &exec2, k, n, dh, t_max, 65_200);
        let c2 = Tensor::uninit_device(&ctx2, &[1, n]).unwrap();
        let err = exec2
            .run_ops(&[MatmulOp::new(mm(&a2, &b2, &c2)).with_store_rope(&table2, desc)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not implement the fused store epilogue"),
            "unexpected: {err}"
        );
    }
}
