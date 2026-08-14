//! Record-once, replay-many submission: `ta_prepared_*`.

use std::ffi::c_int;
use std::sync::Arc;

use crate::error::{checked_ref, destroy_handle, ffi_create, ffi_status};
use crate::handles::{ta_executor, ta_prepared};
use crate::types::{ta_matmul_op, ta_run_stats};

use super::matmul::write_stats;
use super::ops::checked_ops;

/// Validate, route, and record `n_ops` GEMM+epilogue ops once for
/// repeated replay. `as_graph != 0` records hazard barriers between
/// dependent ops (like `ta_run_op_graph`); `as_graph == 0` requires the
/// ops to be independent (like `ta_run_ops`). Requires a device with
/// `bufferDeviceAddress` (check `ta_context_supports_bda`).
///
/// # Safety
///
/// `exec` must be a live executor and `ops` must point to `n_ops` valid
/// entries whose non-null tensor pointers are live tensors.
///
/// Lifetime contract (undefined behavior if violated): the executor and
/// every tensor referenced by `ops` must stay alive until the returned
/// handle has been passed to `ta_prepared_destroy` — the recorded
/// command buffer bakes their device addresses in. Destroying a tensor
/// or the executor while a `ta_prepared` that references it is alive is
/// undefined behavior. `ta_prepared_destroy` (or `ta_prepared_wait`)
/// waits out any in-flight submission; a handle that is never destroyed
/// before process exit leaks Vulkan objects but must at minimum not
/// have work in flight when the executor is destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_prepared_create(
    exec: *const ta_executor,
    ops: *const ta_matmul_op,
    n_ops: usize,
    as_graph: u32,
) -> *mut ta_prepared {
    ffi_create(|| {
        // The free lifetime on `checked_ref` / `checked_ops` is
        // instantiated at 'static here; see the SAFETY note on
        // `ta_prepared` for why the documented C contract makes that
        // sound.
        let exec: &'static ta_executor = checked_ref(exec, "ta_prepared_create: exec is null")?;
        let ops = checked_ops(exec, ops, n_ops, "ta_prepared_create")?;
        let prepared = if as_graph != 0 {
            exec.exec.prepare_op_graph(&ops)?
        } else {
            exec.exec.prepare_ops(&ops)?
        };
        Ok(Box::into_raw(Box::new(ta_prepared {
            prepared,
            _ctx: Arc::clone(&exec.ctx),
        })))
    })
}

/// Submit the prepared batch and block until GPU completion.
///
/// # Safety
///
/// `prepared` must be a live handle whose executor and tensors are
/// still alive. `stats` may be null; otherwise it must point to
/// writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_prepared_run(
    prepared: *mut ta_prepared,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        if prepared.is_null() {
            anyhow::bail!("ta_prepared_run: prepared is null");
        }
        let prepared = unsafe { &mut *prepared };
        write_stats(stats, prepared.prepared.run()?);
        Ok(())
    })
}

/// Submit the prepared batch without waiting. Call `ta_prepared_wait`
/// before submitting this handle again; overlap is achieved by
/// ping-ponging two prepared handles. Fails if a submission from this
/// handle is already in flight.
///
/// # Safety
///
/// `prepared` must be a live handle. After this returns, GPU work is in
/// flight against device addresses baked into the recorded command
/// buffer: the executor and every referenced tensor must stay alive,
/// and the handle must reach `ta_prepared_wait` or
/// `ta_prepared_destroy` (both fence-wait) before any of them is
/// destroyed. Exiting the process without destroying the handle while
/// work is in flight is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_prepared_submit(prepared: *mut ta_prepared) -> c_int {
    ffi_status(|| {
        if prepared.is_null() {
            anyhow::bail!("ta_prepared_submit: prepared is null");
        }
        let prepared = unsafe { &mut *prepared };
        // SAFETY: the C contract documented on `ta_prepared_create` /
        // `ta_prepared` translates Rust's leak-soundness requirement:
        // the handle cannot be leaked past its tensors because
        // destroying tensors/executor before the handle is already UB,
        // and destroy fence-waits.
        unsafe { prepared.prepared.submit() }
    })
}

/// Block until the in-flight submission from `ta_prepared_submit`
/// completes and report its stats. Fails if nothing is in flight.
///
/// # Safety
///
/// `prepared` must be a live handle whose executor and tensors are
/// still alive. `stats` may be null; otherwise it must point to
/// writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_prepared_wait(
    prepared: *mut ta_prepared,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        if prepared.is_null() {
            anyhow::bail!("ta_prepared_wait: prepared is null");
        }
        let prepared = unsafe { &mut *prepared };
        write_stats(stats, prepared.prepared.wait()?);
        Ok(())
    })
}

/// Destroy a prepared handle. Waits for any in-flight submission
/// before releasing its Vulkan objects.
///
/// # Safety
///
/// `prepared` may be null. Otherwise it must be a pointer returned by
/// `ta_prepared_create` that has not already been destroyed, and the
/// executor and tensors it references must still be alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_prepared_destroy(prepared: *mut ta_prepared) {
    unsafe { destroy_handle(prepared) }
}
