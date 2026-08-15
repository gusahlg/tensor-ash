use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub(crate) fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

pub(crate) fn ffi_status<F>(f: F) -> c_int
where
    F: FnOnce() -> Result<()>,
{
    clear_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(err)) => {
            set_error(err);
            -1
        }
        Err(_) => {
            set_error(anyhow!("panic crossed tensor-ash C ABI boundary"));
            -1
        }
    }
}

pub(crate) fn ffi_create<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> Result<*mut T>,
{
    clear_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(err)) => {
            set_error(err);
            ptr::null_mut()
        }
        Err(_) => {
            set_error(anyhow!("panic crossed tensor-ash C ABI boundary"));
            ptr::null_mut()
        }
    }
}

pub(crate) fn parse_optional_cstr(ptr: *const c_char, default: &str) -> Result<String> {
    if ptr.is_null() {
        return Ok(default.into());
    }
    let value = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .context("C string is not valid UTF-8")?;
    Ok(value.into())
}

pub(crate) fn checked_ref<'a, T>(ptr: *const T, message: &str) -> Result<&'a T> {
    if ptr.is_null() {
        bail!("{message}");
    }
    Ok(unsafe { &*ptr })
}

pub(crate) fn checked_mut<'a, T>(ptr: *mut T, message: &str) -> Result<&'a mut T> {
    if ptr.is_null() {
        bail!("{message}");
    }
    Ok(unsafe { &mut *ptr })
}

pub(crate) fn checked_slice<'a, T>(ptr: *const T, len: usize, message: &str) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        bail!("{message}");
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub(crate) fn checked_slice_mut<'a, T>(
    ptr: *mut T,
    len: usize,
    message: &str,
) -> Result<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        bail!("{message}");
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

pub(crate) unsafe fn destroy_handle<T>(ptr: *mut T) {
    if ptr.is_null() {
        return;
    }
    clear_error();
    if catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(ptr)) })).is_err() {
        set_error(anyhow!("panic while destroying tensor-ash C ABI handle"));
    }
}

fn clear_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new("").unwrap();
    });
}

fn set_error(err: anyhow::Error) {
    let msg = err.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(msg).unwrap();
    });
}
