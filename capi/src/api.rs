use std::ffi::{c_char, c_int};
use std::sync::Arc;

use tensor_ash_core::{
    DevicePreference, Executor, KernelSelection, MatmulCall, MatmulPipeline, Tensor, VulkanContext,
};

use crate::error::{
    checked_ref, checked_slice, checked_slice_mut, ffi_create, ffi_status, last_error_ptr,
    parse_optional_cstr,
};
use crate::handles::{ta_context, ta_executor, ta_tensor};
use crate::types::{ta_matmul_call, ta_run_stats};

const TA_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

#[unsafe(no_mangle)]
pub extern "C" fn ta_version() -> *const c_char {
    TA_VERSION.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn ta_last_error() -> *const c_char {
    last_error_ptr()
}

/// Create a Vulkan context.
///
/// # Safety
///
/// `device_preference` may be null; otherwise it must point to a valid
/// NUL-terminated UTF-8 string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_context_create(
    enable_validation: u32,
    device_preference: *const c_char,
) -> *mut ta_context {
    ffi_create(|| {
        let preference = parse_optional_cstr(device_preference, "auto")?;
        let preference = DevicePreference::parse(&preference)?;
        let ctx = VulkanContext::new_with_device_preference(enable_validation != 0, preference)?;
        Ok(Box::into_raw(Box::new(ta_context { ctx })))
    })
}

/// Destroy a context returned by `ta_context_create`.
///
/// # Safety
///
/// `ctx` may be null. Otherwise it must be a pointer returned by
/// `ta_context_create` that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_context_destroy(ctx: *mut ta_context) {
    if !ctx.is_null() {
        unsafe {
            drop(Box::from_raw(ctx));
        }
    }
}

/// Create an executor and its matmul pipeline.
///
/// # Safety
///
/// `ctx` must be a live context returned by `ta_context_create`.
/// `kernel_selection` may be null; otherwise it must point to a valid
/// NUL-terminated UTF-8 string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_executor_create(
    ctx: *const ta_context,
    n_slots: usize,
    max_calls_per_submit: u32,
    kernel_selection: *const c_char,
) -> *mut ta_executor {
    ffi_create(|| {
        let ctx = checked_ref(ctx, "ta_executor_create: ctx is null")?;
        let selection = parse_optional_cstr(kernel_selection, "auto")?;
        let selection = KernelSelection::parse(&selection)?;
        let pipeline = Arc::new(MatmulPipeline::new_with_kernel_selection(
            &ctx.ctx, selection,
        )?);
        let exec = Executor::new(
            Arc::clone(&ctx.ctx),
            Arc::clone(&pipeline),
            n_slots,
            max_calls_per_submit,
        )?;
        Ok(Box::into_raw(Box::new(ta_executor {
            ctx: Arc::clone(&ctx.ctx),
            _pipeline: pipeline,
            exec,
        })))
    })
}

/// Destroy an executor returned by `ta_executor_create`.
///
/// # Safety
///
/// `exec` may be null. Otherwise it must be a pointer returned by
/// `ta_executor_create` that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_executor_destroy(exec: *mut ta_executor) {
    if !exec.is_null() {
        unsafe {
            drop(Box::from_raw(exec));
        }
    }
}

/// Allocate a device-local tensor from a context.
///
/// # Safety
///
/// `ctx` must be a live context. If `rank > 0`, `shape` must point to at least
/// `rank` `uint32_t` dimensions for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_create(
    ctx: *const ta_context,
    shape: *const u32,
    rank: usize,
) -> *mut ta_tensor {
    ffi_create(|| {
        let ctx = checked_ref(ctx, "ta_tensor_create: ctx is null")?;
        let shape = checked_slice(shape, rank, "ta_tensor_create: shape is null")?;
        let tensor = Tensor::uninit_device(&ctx.ctx, shape)?;
        Ok(Box::into_raw(Box::new(ta_tensor { tensor })))
    })
}

/// Allocate a device-local tensor using the context held by an executor.
///
/// # Safety
///
/// `exec` must be a live executor. If `rank > 0`, `shape` must point to at
/// least `rank` `uint32_t` dimensions for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_create_on_executor(
    exec: *const ta_executor,
    shape: *const u32,
    rank: usize,
) -> *mut ta_tensor {
    ffi_create(|| {
        let exec = checked_ref(exec, "ta_tensor_create_on_executor: exec is null")?;
        let shape = checked_slice(shape, rank, "ta_tensor_create_on_executor: shape is null")?;
        let tensor = Tensor::uninit_device(&exec.ctx, shape)?;
        Ok(Box::into_raw(Box::new(ta_tensor { tensor })))
    })
}

/// Destroy a tensor returned by a tensor creation function.
///
/// # Safety
///
/// `tensor` may be null. Otherwise it must be a pointer returned by this API
/// that has not already been destroyed and is not in use by an active call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_destroy(tensor: *mut ta_tensor) {
    if !tensor.is_null() {
        unsafe {
            drop(Box::from_raw(tensor));
        }
    }
}

/// Return the tensor element count, or zero for null.
///
/// # Safety
///
/// `tensor` may be null. Otherwise it must point to a live tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_len(tensor: *const ta_tensor) -> u64 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).tensor.len() }
}

/// Return the tensor byte size, or zero for null.
///
/// # Safety
///
/// `tensor` may be null. Otherwise it must point to a live tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_size_bytes(tensor: *const ta_tensor) -> u64 {
    if tensor.is_null() {
        return 0;
    }
    unsafe { (*tensor).tensor.size_bytes() }
}

/// Upload `len` f32 values into a tensor.
///
/// # Safety
///
/// `exec` and `dst` must be live handles. If `len > 0`, `src` must point to at
/// least `len` `float` values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_upload(
    exec: *const ta_executor,
    dst: *const ta_tensor,
    src: *const f32,
    len: usize,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_upload: exec is null")?;
        let dst = checked_ref(dst, "ta_upload: dst is null")?;
        let src = checked_slice(src, len, "ta_upload: src is null")?;
        exec.exec.upload(src, &dst.tensor)
    })
}

/// Download `len` f32 values from a tensor.
///
/// # Safety
///
/// `exec` and `src` must be live handles. If `len > 0`, `dst` must point to at
/// least `len` writable `float` values for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_download(
    exec: *const ta_executor,
    src: *const ta_tensor,
    dst: *mut f32,
    len: usize,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_download: exec is null")?;
        let src = checked_ref(src, "ta_download: src is null")?;
        let dst = checked_slice_mut(dst, len, "ta_download: dst is null")?;
        exec.exec.download(&src.tensor, dst)
    })
}

/// Run one synchronous GEMM.
///
/// # Safety
///
/// `exec`, `a`, `b`, and `c` must be live handles. `stats` may be null;
/// otherwise it must point to writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_matmul(
    exec: *const ta_executor,
    a: *const ta_tensor,
    b: *const ta_tensor,
    c: *const ta_tensor,
    alpha: f32,
    accumulate: u32,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_matmul: exec is null")?;
        let a = checked_ref(a, "ta_matmul: a is null")?;
        let b = checked_ref(b, "ta_matmul: b is null")?;
        let c = checked_ref(c, "ta_matmul: c is null")?;
        let call = MatmulCall {
            a: &a.tensor,
            b: &b.tensor,
            c: &c.tensor,
            alpha,
            accumulate: accumulate != 0,
        };
        let run_stats = exec.exec.run_matmuls(&[call])?;
        write_stats(stats, run_stats);
        Ok(())
    })
}

/// Run a synchronous batch of GEMMs in one submit.
///
/// # Safety
///
/// `exec` must be a live executor. If `n_calls > 0`, `calls` must point to at
/// least `n_calls` valid `ta_matmul_call` entries, and every tensor pointer
/// inside those entries must be a live tensor. `stats` may be null; otherwise
/// it must point to writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_matmul_batch(
    exec: *const ta_executor,
    calls: *const ta_matmul_call,
    n_calls: usize,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_matmul_batch: exec is null")?;
        let raw_calls = checked_slice(calls, n_calls, "ta_matmul_batch: calls is null")?;
        let mut calls = Vec::with_capacity(raw_calls.len());
        for (index, raw) in raw_calls.iter().enumerate() {
            let a = checked_ref(raw.a, &format!("ta_matmul_batch: call {index} a is null"))?;
            let b = checked_ref(raw.b, &format!("ta_matmul_batch: call {index} b is null"))?;
            let c = checked_ref(raw.c, &format!("ta_matmul_batch: call {index} c is null"))?;
            calls.push(MatmulCall {
                a: &a.tensor,
                b: &b.tensor,
                c: &c.tensor,
                alpha: raw.alpha,
                accumulate: raw.accumulate != 0,
            });
        }
        let run_stats = exec.exec.run_matmuls(&calls)?;
        write_stats(stats, run_stats);
        Ok(())
    })
}

fn write_stats(stats: *mut ta_run_stats, run_stats: tensor_ash_core::RunStats) {
    if stats.is_null() {
        return;
    }
    let (has_gpu_time, gpu_time_ns) = match run_stats.gpu_time_ns {
        Some(ns) => (1, ns),
        None => (0, 0),
    };
    unsafe {
        *stats = ta_run_stats {
            has_gpu_time,
            gpu_time_ns,
            n_calls: run_stats.n_calls,
            total_flops: run_stats.total_flops,
            tflops: run_stats.tflops().unwrap_or(f64::NAN),
        };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ptr;

    use super::*;

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
