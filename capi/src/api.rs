mod diagnostics;
mod lifecycle;
mod matmul;
mod ops;
mod prepared;
mod tensor;

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use super::diagnostics::{ta_dispatch_info_for, ta_tune_shape};
    use super::lifecycle::{
        ta_context_destroy, ta_context_supports_bda, ta_context_supports_f16, ta_executor_destroy,
        ta_last_error, ta_version,
    };
    use super::ops::{ta_run_op_graph, ta_run_ops};
    use super::prepared::{
        ta_prepared_create, ta_prepared_destroy, ta_prepared_run, ta_prepared_submit,
        ta_prepared_wait,
    };
    use super::tensor::{ta_tensor_destroy, ta_tensor_is_f16, ta_upload};

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
    fn null_diagnostics_report_errors() {
        unsafe {
            assert_eq!(
                ta_dispatch_info_for(ptr::null(), 1, 1, 1, 1, 0, ptr::null_mut()),
                -1
            );
            assert!(last_error().contains("ta_dispatch_info_for: exec is null"));
            assert_eq!(ta_tune_shape(ptr::null(), 1, 64, 64, 64), -1);
            assert!(last_error().contains("ta_tune_shape: exec is null"));
        }
    }
}
