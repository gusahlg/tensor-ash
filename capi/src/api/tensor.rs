use std::ffi::c_int;
use std::sync::Arc;

use tensor_ash_core::{DType, Tensor, VulkanContext};

use crate::error::{
    checked_ref, checked_slice, checked_slice_mut, destroy_handle, ffi_create, ffi_status,
};
use crate::handles::{ta_context, ta_executor, ta_tensor};

/// Shared body of the four `ta_tensor_create*` exports.
fn tensor_create(
    ctx: &Arc<VulkanContext>,
    shape: *const u32,
    rank: usize,
    dtype: DType,
    operation: &str,
) -> anyhow::Result<*mut ta_tensor> {
    let shape = checked_slice(shape, rank, &format!("{operation}: shape is null"))?;
    let tensor = match dtype {
        DType::F16 => Tensor::uninit_device_f16(ctx, shape)?,
        DType::F32 => Tensor::uninit_device(ctx, shape)?,
    };
    Ok(Box::into_raw(Box::new(ta_tensor {
        ctx: Arc::clone(ctx),
        tensor,
    })))
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
        tensor_create(&ctx.ctx, shape, rank, DType::F32, "ta_tensor_create")
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
        tensor_create(
            &exec.ctx,
            shape,
            rank,
            DType::F32,
            "ta_tensor_create_on_executor",
        )
    })
}

/// Allocate a device-local f16-storage tensor from a context.
///
/// Fails when the device lacks f16 storage support (check
/// `ta_context_supports_f16`). Kernels read the tensor as half
/// precision and accumulate in f32; `ta_upload`/`ta_download` keep the
/// `float` host API and convert automatically (round-to-nearest half on
/// upload, widen on download).
///
/// # Safety
///
/// `ctx` must be a live context. If `rank > 0`, `shape` must point to at
/// least `rank` `uint32_t` dimensions for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_create_f16(
    ctx: *const ta_context,
    shape: *const u32,
    rank: usize,
) -> *mut ta_tensor {
    ffi_create(|| {
        let ctx = checked_ref(ctx, "ta_tensor_create_f16: ctx is null")?;
        tensor_create(&ctx.ctx, shape, rank, DType::F16, "ta_tensor_create_f16")
    })
}

/// `ta_tensor_create_f16` using the context held by an executor.
///
/// # Safety
///
/// `exec` must be a live executor. If `rank > 0`, `shape` must point to
/// at least `rank` `uint32_t` dimensions for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_create_f16_on_executor(
    exec: *const ta_executor,
    shape: *const u32,
    rank: usize,
) -> *mut ta_tensor {
    ffi_create(|| {
        let exec = checked_ref(exec, "ta_tensor_create_f16_on_executor: exec is null")?;
        tensor_create(
            &exec.ctx,
            shape,
            rank,
            DType::F16,
            "ta_tensor_create_f16_on_executor",
        )
    })
}

/// Return 1 if the tensor stores f16 elements, 0 for f32 or null.
///
/// # Safety
///
/// `tensor` may be null. Otherwise it must point to a live tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_is_f16(tensor: *const ta_tensor) -> u32 {
    if tensor.is_null() {
        return 0;
    }
    (unsafe { (*tensor).tensor.dtype() } == DType::F16) as u32
}

/// Destroy a tensor returned by a tensor creation function.
///
/// # Safety
///
/// `tensor` may be null. Otherwise it must be a pointer returned by this API
/// that has not already been destroyed and is not in use by an active call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tensor_destroy(tensor: *mut ta_tensor) {
    unsafe { destroy_handle(tensor) }
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
/// The host side is always `float`; values headed for an f16 tensor are
/// rounded to the nearest half automatically during staging.
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
        let dst = exec.checked_tensor(dst, "ta_upload", "dst")?;
        let src = checked_slice(src, len, "ta_upload: src is null")?;
        exec.exec.upload(src, &dst.tensor)
    })
}

/// Download `len` f32 values from a tensor.
///
/// The host side is always `float`; f16 tensors are widened to f32
/// automatically during staging.
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
        let src = exec.checked_tensor(src, "ta_download", "src")?;
        let dst = checked_slice_mut(dst, len, "ta_download: dst is null")?;
        exec.exec.download(&src.tensor, dst)
    })
}
