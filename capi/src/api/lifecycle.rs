use std::ffi::c_char;
use std::sync::Arc;

use tensor_ash_core::{DevicePreference, Executor, KernelSelection, MatmulPipeline, VulkanContext};

use crate::error::{checked_ref, destroy_handle, ffi_create, last_error_ptr, parse_optional_cstr};
use crate::handles::{ta_context, ta_executor};

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
    unsafe { destroy_handle(ctx) }
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
    unsafe { destroy_handle(exec) }
}
