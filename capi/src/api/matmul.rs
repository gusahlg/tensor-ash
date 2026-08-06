use std::ffi::c_int;

use anyhow::Result;
use tensor_ash_core::{MatmulCall, RunStats};

use crate::error::{checked_ref, checked_slice, ffi_status};
use crate::handles::{ta_executor, ta_tensor};
use crate::types::{ta_matmul_call, ta_run_stats};

fn checked_call<'a>(
    exec: &ta_executor,
    raw: &ta_matmul_call,
    operation: &str,
) -> Result<MatmulCall<'a>> {
    let a = exec.checked_tensor(raw.a, operation, "a")?;
    let b = exec.checked_tensor(raw.b, operation, "b")?;
    let c = exec.checked_tensor(raw.c, operation, "c")?;
    Ok(MatmulCall {
        a: &a.tensor,
        b: &b.tensor,
        c: &c.tensor,
        alpha: raw.alpha,
        accumulate: raw.accumulate != 0,
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
        let raw = ta_matmul_call {
            a,
            b,
            c,
            alpha,
            accumulate,
        };
        let call = checked_call(exec, &raw, "ta_matmul")?;
        write_stats(stats, exec.exec.run_matmuls(&[call])?);
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
        let calls = raw_calls
            .iter()
            .enumerate()
            .map(|(index, raw)| checked_call(exec, raw, &format!("ta_matmul_batch: call {index}")))
            .collect::<Result<Vec<_>>>()?;
        write_stats(stats, exec.exec.run_matmuls(&calls)?);
        Ok(())
    })
}

fn write_stats(stats: *mut ta_run_stats, run_stats: RunStats) {
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
