use super::*;

#[test]
fn resolves_rank2_matmul() {
    let dims = ResolvedMatmul::from_shapes(&[2, 3], &[3, 4], &[2, 4]).unwrap();
    assert_eq!(
        dims,
        ResolvedMatmul {
            m: 2,
            n: 4,
            k: 3,
            batch: 1,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_c: 0,
            total_flops: 48,
            b_f16: false,
            a_f16: false,
            packed_b: false,
        }
    );
}

#[test]
fn resolves_batched_matmul() {
    let dims = ResolvedMatmul::from_shapes(&[5, 2, 3], &[5, 3, 4], &[5, 2, 4]).unwrap();
    assert_eq!(dims.batch, 5);
    assert_eq!(
        (
            dims.batch_stride_a,
            dims.batch_stride_b,
            dims.batch_stride_c
        ),
        (6, 12, 8)
    );
    assert_eq!(dims.total_flops, 240);
}

#[test]
fn broadcasts_either_or_both_inputs_to_output_batch() {
    let a = ResolvedMatmul::from_shapes(&[1, 2, 3], &[7, 3, 4], &[7, 2, 4]).unwrap();
    let b = ResolvedMatmul::from_shapes(&[7, 2, 3], &[1, 3, 4], &[7, 2, 4]).unwrap();
    let both = ResolvedMatmul::from_shapes(&[1, 2, 3], &[1, 3, 4], &[7, 2, 4]).unwrap();

    assert_eq!((a.batch_stride_a, a.batch_stride_b), (0, 12));
    assert_eq!((b.batch_stride_a, b.batch_stride_b), (6, 0));
    assert_eq!((both.batch_stride_a, both.batch_stride_b), (0, 0));
    assert_eq!(both.batch_stride_c, 8);
    assert_eq!(both.total_flops, 336);
}

#[test]
fn rejects_mismatched_inner_dimension() {
    let err = ResolvedMatmul::from_shapes(&[2, 3], &[5, 4], &[2, 4])
        .unwrap_err()
        .to_string();
    assert!(err.contains("A.K=3"));
    assert!(err.contains("B.K=5"));
}

#[test]
fn rejects_output_batch_that_cannot_hold_inputs() {
    let err = ResolvedMatmul::from_shapes(&[7, 2, 3], &[7, 3, 4], &[1, 2, 4])
        .unwrap_err()
        .to_string();
    assert!(err.contains("incompatible batch dims"));
    assert!(err.contains("A.B=7"));
}

#[test]
fn rejects_zero_dimensions() {
    let err = MatrixShape::from_tensor_shape(&[0, 3])
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-zero"));
}

#[test]
fn rejects_batch_stride_overflow() {
    let err = ResolvedMatmul::from_shapes(
        &[2, u32::MAX, u32::MAX],
        &[1, u32::MAX, 1],
        &[2, u32::MAX, 1],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("A batch stride"));
    assert!(err.contains("overflows"));
}

#[test]
fn accepts_the_largest_shader_addressable_layout() {
    ResolvedMatmul::from_shapes(&[65_536, 65_536], &[65_536, 1], &[65_536, 1]).unwrap();
    ResolvedMatmul::from_shapes(&[2, 32_768, 65_536], &[65_536, 1], &[2, 32_768, 1]).unwrap();
}

#[test]
fn rejects_layouts_beyond_shader_u32_indexing() {
    type LayoutCase<'a> = (&'a str, &'a [u32], &'a [u32], &'a [u32]);
    let cases: [LayoutCase<'_>; 6] = [
        ("A", &[65_537, 65_536], &[65_536, 1], &[65_537, 1]),
        ("B", &[1, 65_537], &[65_537, 65_536], &[1, 65_536]),
        ("C", &[65_537, 1], &[1, 65_536], &[65_537, 65_536]),
        ("A", &[3, 32_768, 65_536], &[65_536, 1], &[3, 32_768, 1]),
        ("B", &[3, 1, 32_768], &[3, 32_768, 65_536], &[3, 1, 65_536]),
        ("C", &[3, 32_768, 1], &[1, 65_536], &[3, 32_768, 65_536]),
    ];

    for (label, a, b, c) in cases {
        let err = ResolvedMatmul::from_shapes(a, b, c)
            .unwrap_err()
            .to_string();
        assert!(err.contains(label), "unexpected error: {err}");
        assert!(
            err.contains("shader u32 indexing"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn rejects_flop_count_overflow() {
    let err = ResolvedMatmul::from_shapes(&[1, u32::MAX], &[u32::MAX, 1], &[u32::MAX, 1, 1])
        .unwrap_err()
        .to_string();
    assert!(err.contains("FLOP count overflows"));
}

#[test]
fn validates_bias_shape_not_just_element_count() {
    let dims = ResolvedMatmul::from_shapes(&[3, 2, 4], &[3, 4, 5], &[3, 2, 5]).unwrap();
    assert!(dims.validate_bias_shape(&[5]).is_ok());
    assert!(dims.validate_bias_shape(&[3, 5]).is_ok());

    let err = dims.validate_bias_shape(&[5, 3]).unwrap_err().to_string();
    assert!(err.contains("bias shape"));
}

#[test]
fn builds_push_constants_from_resolved_shape_and_call_flags() {
    let dims = ResolvedMatmul::from_shapes(&[2, 2, 3], &[1, 3, 4], &[2, 2, 4]).unwrap();
    let pc = dims.push_constants(0.5, true, 0xAA00, 0xBB00, 0xCC00);

    assert_eq!((pc.m, pc.n, pc.k), (2, 4, 3));
    assert_eq!(
        (pc.batch_stride_a, pc.batch_stride_b, pc.batch_stride_c),
        (6, 0, 8)
    );
    assert_eq!(pc.flags, 1);
    assert_eq!(pc.alpha, 0.5);
    assert_eq!((pc.a_ptr, pc.b_ptr, pc.c_ptr), (0xAA00, 0xBB00, 0xCC00));
}

#[test]
fn stats_tflops_require_nonzero_gpu_time() {
    let stats = |gpu_time_ns| RunStats {
        gpu_time_ns,
        n_calls: 1,
        total_flops: 2_000_000_000,
    };

    assert_eq!(stats(Some(1_000_000)).tflops(), Some(2.0));
    assert_eq!(stats(Some(0)).tflops(), None);
    assert_eq!(stats(None).tflops(), None);
}
