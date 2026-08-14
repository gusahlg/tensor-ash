//! Route diagnostics and shape pre-tuning.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_int};
use std::sync::{Mutex, OnceLock};

use anyhow::Context;

use crate::error::{checked_ref, ffi_status};
use crate::handles::ta_executor;
use crate::types::ta_dispatch_info;

/// Kernel names in `DispatchInfo` are `&'static str` (not
/// NUL-terminated), so intern each one as a leaked `CString` the first
/// time it is seen. The set is bounded by the kernel catalog, and the
/// returned pointers stay valid for the process lifetime.
fn intern_kernel_name(name: &'static str) -> anyhow::Result<*const c_char> {
    static NAMES: OnceLock<Mutex<HashMap<&'static str, &'static std::ffi::CStr>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let interned = match names.get(name) {
        Some(&cstr) => cstr,
        None => {
            let cstr: &'static std::ffi::CStr = Box::leak(
                CString::new(name)
                    .context("kernel name contains NUL")?
                    .into_boxed_c_str(),
            );
            names.insert(name, cstr);
            cstr
        }
    };
    Ok(interned.as_ptr())
}

/// Report the kernel and optional two-stage split-K route a plain,
/// non-accumulating matmul of the given shape would use. `b_f16 != 0`
/// reports the route for an f16-weights matmul. `out->kernel` points to
/// an interned name valid for the process lifetime.
///
/// # Safety
///
/// `exec` must be a live executor and `out` must point to writable
/// `ta_dispatch_info` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_dispatch_info_for(
    exec: *const ta_executor,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    b_f16: u32,
    out: *mut ta_dispatch_info,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_dispatch_info_for: exec is null")?;
        if out.is_null() {
            anyhow::bail!("ta_dispatch_info_for: out is null");
        }
        let info = exec.exec.dispatch_info_for(batch, m, n, k, b_f16 != 0);
        let kernel = intern_kernel_name(info.kernel)?;
        unsafe {
            *out = ta_dispatch_info {
                kernel,
                tile: info.tile,
                has_split_k2: info.split_k2_splits.is_some() as u32,
                split_k2_splits: info.split_k2_splits.unwrap_or(0),
            };
        }
        Ok(())
    })
}

/// Measure every eligible kernel for one GEMM shape against scratch
/// tensors and persist the winner in the tuning store. Useful to
/// pre-warm shapes an inference workload will hit without paying the
/// measurement cost on the first real call. No-op if the shape is
/// already tuned; fails on devices without timestamp support.
///
/// # Safety
///
/// `exec` must be a live executor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_tune_shape(
    exec: *const ta_executor,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_tune_shape: exec is null")?;
        exec.exec.tune_shape(batch, m, n, k)
    })
}
