//! Fused-epilogue op batches: `ta_run_ops` / `ta_run_op_graph`.

use std::ffi::c_int;

use anyhow::{Result, bail};
use tensor_ash_core::{Activation, Epilogue, EpilogueBinary, MatmulOp};

use crate::error::{checked_ref, checked_slice, ffi_status};
use crate::handles::ta_executor;
use crate::types::{ta_matmul_op, ta_run_stats};

use super::matmul::{checked_call, write_stats};

/// Validate one C op and translate it to the Rust `MatmulOp`.
pub(crate) fn checked_op<'a>(
    exec: &ta_executor,
    raw: &ta_matmul_op,
    operation: &str,
) -> Result<MatmulOp<'a>> {
    let call = checked_call(exec, &raw.call, operation)?;
    let ep = &raw.epilogue;
    let bias = if ep.bias.is_null() {
        None
    } else {
        Some(
            &exec
                .checked_tensor(ep.bias, operation, "epilogue.bias")?
                .tensor,
        )
    };
    let activation = match ep.activation {
        0 => Activation::None,
        1 => Activation::Relu,
        2 => Activation::Silu,
        3 => Activation::Gelu,
        other => bail!(
            "{operation}: epilogue.activation {other} is invalid (0=none 1=relu 2=silu 3=gelu)"
        ),
    };
    let binary = match ep.binary_kind {
        0 => {
            if !ep.d.is_null() {
                bail!("{operation}: epilogue.d is set but binary_kind is 0 (none)");
            }
            EpilogueBinary::None
        }
        1 => EpilogueBinary::AddScaled {
            d: &exec.checked_tensor(ep.d, operation, "epilogue.d")?.tensor,
            beta: ep.beta,
        },
        2 => EpilogueBinary::Mul {
            d: &exec.checked_tensor(ep.d, operation, "epilogue.d")?.tensor,
        },
        other => bail!(
            "{operation}: epilogue.binary_kind {other} is invalid (0=none 1=add_scaled 2=mul)"
        ),
    };
    Ok(MatmulOp {
        call,
        epilogue: Epilogue {
            bias,
            activation,
            binary,
        },
    })
}

pub(crate) fn checked_ops<'a>(
    exec: &ta_executor,
    ops: *const ta_matmul_op,
    n_ops: usize,
    operation: &str,
) -> Result<Vec<MatmulOp<'a>>> {
    let raw_ops = checked_slice(ops, n_ops, &format!("{operation}: ops is null"))?;
    raw_ops
        .iter()
        .enumerate()
        .map(|(index, raw)| checked_op(exec, raw, &format!("{operation}: op {index}")))
        .collect()
}

/// Run a synchronous batch of independent GEMM+epilogue ops in one
/// submit. No inter-op barriers are inserted; for dependent chains use
/// `ta_run_op_graph`.
///
/// # Safety
///
/// `exec` must be a live executor. If `n_ops > 0`, `ops` must point to
/// at least `n_ops` valid `ta_matmul_op` entries, and every non-null
/// tensor pointer inside those entries must be a live tensor. `stats`
/// may be null; otherwise it must point to writable `ta_run_stats`
/// storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_run_ops(
    exec: *const ta_executor,
    ops: *const ta_matmul_op,
    n_ops: usize,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_run_ops: exec is null")?;
        let ops = checked_ops(exec, ops, n_ops, "ta_run_ops")?;
        write_stats(stats, exec.exec.run_ops(&ops)?);
        Ok(())
    })
}

/// `ta_run_ops` for dependent chains: hazards between ops (including
/// epilogue bias / D operands) are resolved with automatic barriers, so
/// op N may read outputs of ops 0..N-1 within the same submission.
///
/// # Safety
///
/// Same contract as `ta_run_ops`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_run_op_graph(
    exec: *const ta_executor,
    ops: *const ta_matmul_op,
    n_ops: usize,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_run_op_graph: exec is null")?;
        let ops = checked_ops(exec, ops, n_ops, "ta_run_op_graph")?;
        write_stats(stats, exec.exec.run_op_graph(&ops)?);
        Ok(())
    })
}
