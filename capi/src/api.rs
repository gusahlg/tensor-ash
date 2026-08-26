mod diagnostics;
mod elementwise;
mod lifecycle;
mod matmul;
mod ops;
mod prepared;
mod tensor;

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use super::diagnostics::{ta_dispatch_info_for, ta_dispatch_info_packed, ta_tune_shape};
    use super::elementwise::{
        ta_copy_strided, ta_layer_norm, ta_rms_norm, ta_rope, ta_softmax_rows,
    };
    use super::lifecycle::{
        ta_context_destroy, ta_context_supports_bda, ta_context_supports_f16, ta_executor_destroy,
        ta_last_error, ta_version,
    };
    use super::ops::{ta_run_op_graph, ta_run_ops};
    use super::prepared::{
        ta_prepared_create, ta_prepared_destroy, ta_prepared_run, ta_prepared_submit,
        ta_prepared_wait,
    };
    use super::tensor::{
        ta_f16w_row_tile_n, ta_pack_f16w_row_tiles, ta_tensor_destroy, ta_tensor_is_f16, ta_upload,
    };

    fn last_error() -> &'static str {
        unsafe { CStr::from_ptr(ta_last_error()) }.to_str().unwrap()
    }

    #[test]
    fn version_is_nul_terminated() {
        let version = unsafe { CStr::from_ptr(ta_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn null_upload_reports_error() {
        let rc = unsafe { ta_upload(ptr::null(), ptr::null(), ptr::null(), 0) };
        assert_eq!(rc, -1);
        let err = unsafe { CStr::from_ptr(ta_last_error()) }.to_str().unwrap();
        assert!(err.contains("exec is null"));
    }

    #[test]
    fn destroy_null_handles_are_noops() {
        unsafe {
            ta_context_destroy(ptr::null_mut());
            ta_executor_destroy(ptr::null_mut());
            ta_tensor_destroy(ptr::null_mut());
            ta_prepared_destroy(ptr::null_mut());
        }
    }

    #[test]
    fn null_capability_queries_return_zero() {
        unsafe {
            assert_eq!(ta_context_supports_f16(ptr::null()), 0);
            assert_eq!(ta_context_supports_bda(ptr::null()), 0);
            assert_eq!(ta_tensor_is_f16(ptr::null()), 0);
        }
    }

    #[test]
    fn null_op_entry_points_report_errors() {
        unsafe {
            assert_eq!(ta_run_ops(ptr::null(), ptr::null(), 0, ptr::null_mut()), -1);
            assert!(last_error().contains("ta_run_ops: exec is null"));

            assert_eq!(
                ta_run_op_graph(ptr::null(), ptr::null(), 0, ptr::null_mut()),
                -1
            );
            assert!(last_error().contains("ta_run_op_graph: exec is null"));

            assert!(ta_prepared_create(ptr::null(), ptr::null(), 0, 0).is_null());
            assert!(last_error().contains("ta_prepared_create: exec is null"));

            assert_eq!(ta_prepared_run(ptr::null_mut(), ptr::null_mut()), -1);
            assert!(last_error().contains("ta_prepared_run: prepared is null"));
            assert_eq!(ta_prepared_submit(ptr::null_mut()), -1);
            assert!(last_error().contains("ta_prepared_submit: prepared is null"));
            assert_eq!(ta_prepared_wait(ptr::null_mut(), ptr::null_mut()), -1);
            assert!(last_error().contains("ta_prepared_wait: prepared is null"));
        }
    }

    #[test]
    fn null_elementwise_entry_points_report_errors() {
        unsafe {
            assert_eq!(
                ta_softmax_rows(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    1.0,
                    0,
                    0,
                    0,
                    ptr::null_mut()
                ),
                -1
            );
            assert!(last_error().contains("ta_softmax_rows: exec is null"));

            assert_eq!(
                ta_rms_norm(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    1e-5,
                    ptr::null_mut()
                ),
                -1
            );
            assert!(last_error().contains("ta_rms_norm: exec is null"));

            assert_eq!(
                ta_layer_norm(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    1e-5,
                    ptr::null_mut()
                ),
                -1
            );
            assert!(last_error().contains("ta_layer_norm: exec is null"));

            assert_eq!(
                ta_rope(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut()
                ),
                -1
            );
            assert!(last_error().contains("ta_rope: exec is null"));

            assert_eq!(
                ta_copy_strided(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut()
                ),
                -1
            );
            assert!(last_error().contains("ta_copy_strided: exec is null"));
        }
    }

    #[test]
    fn null_diagnostics_report_errors() {
        unsafe {
            assert_eq!(
                ta_dispatch_info_for(ptr::null(), 1, 1, 1, 1, 0, ptr::null_mut()),
                -1
            );
            assert!(last_error().contains("ta_dispatch_info_for: exec is null"));
            assert_eq!(ta_tune_shape(ptr::null(), 1, 64, 64, 64), -1);
            assert!(last_error().contains("ta_tune_shape: exec is null"));
            assert_eq!(
                ta_dispatch_info_packed(ptr::null(), 64, 64, ptr::null_mut()),
                -1
            );
            assert!(last_error().contains("ta_dispatch_info_packed: exec is null"));
        }
    }

    #[test]
    fn pack_f16w_row_tiles_matches_rust_helper() {
        let (k, n) = (4u32, 4u32);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        let rc =
            unsafe { ta_pack_f16w_row_tiles(src.as_ptr(), k, n, 2, dst.as_mut_ptr(), dst.len()) };
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(
            dst,
            vec![
                0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0,
                15.0
            ]
        );
        let rust = tensor_ash_core::pack_f16w_row_tiles(&src, k as usize, n as usize, 2);
        assert_eq!(dst, rust);
    }

    #[test]
    fn pack_f16w_row_tiles_rejects_bad_args() {
        let src = [1.0f32; 4];
        let mut dst = [0.0f32; 4];
        unsafe {
            assert_eq!(
                ta_pack_f16w_row_tiles(ptr::null(), 2, 2, 2, dst.as_mut_ptr(), 4),
                -1
            );
            assert!(last_error().contains("src is null"));

            assert_eq!(
                ta_pack_f16w_row_tiles(src.as_ptr(), 2, 2, 2, ptr::null_mut(), 4),
                -1
            );
            assert!(last_error().contains("dst is null"));

            assert_eq!(
                ta_pack_f16w_row_tiles(src.as_ptr(), 2, 2, 2, dst.as_mut_ptr(), 3),
                -1
            );
            assert!(last_error().contains("len 3"));

            assert_eq!(
                ta_pack_f16w_row_tiles(src.as_ptr(), 2, 2, 3, dst.as_mut_ptr(), 4),
                -1
            );
            assert!(last_error().contains("not a multiple of tile_n=3"));
        }
    }

    #[test]
    fn f16w_row_tile_n_matches_rust_helper() {
        assert_eq!(ta_f16w_row_tile_n(2048, 256), 32);
        assert_eq!(ta_f16w_row_tile_n(2048, 5632), 64);
        assert_eq!(ta_f16w_row_tile_n(2048, 32000), 64);
        assert_eq!(
            ta_f16w_row_tile_n(2048, 256),
            tensor_ash_core::f16w_row_tile_n(2048, 256)
        );
        assert_eq!(
            ta_f16w_row_tile_n(2048, 5632),
            tensor_ash_core::f16w_row_tile_n(2048, 5632)
        );
    }
}
