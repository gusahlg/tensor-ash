mod lifecycle;
mod matmul;
mod tensor;

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use super::lifecycle::{ta_context_destroy, ta_executor_destroy, ta_last_error, ta_version};
    use super::tensor::{ta_tensor_destroy, ta_upload};

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
        }
    }
}
