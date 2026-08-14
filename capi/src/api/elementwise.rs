//! Non-GEMM model ops: `ta_softmax_rows`, `ta_rms_norm`,
//! `ta_layer_norm`, `ta_rope`, `ta_copy_strided`.

use std::ffi::c_int;

use anyhow::bail;
use tensor_ash_core::{CopyDesc, RopeDesc, SoftmaxMask};

use crate::error::{checked_ref, ffi_status};
use crate::handles::{ta_executor, ta_tensor};
use crate::types::{ta_copy_desc, ta_rope_desc, ta_run_stats};

use super::matmul::write_stats;

/// Numerically stable softmax over the last dimension. `scale`
/// multiplies inputs before the max/exp passes. `mask_kind`: 0 = full
/// (`valid_or_prefix` / `rows_per_group` ignored), 1 = prefix (columns
/// `>= valid_or_prefix` store exact `0.0`), 2 = causal (row `i` of each
/// `rows_per_group` block sees `valid_or_prefix + (i % rows_per_group)
/// + 1` columns). In-place (`input == output`) is allowed.
///
/// # Safety
///
/// `exec`, `input`, and `output` must be live handles. `stats` may be
/// null; otherwise it must point to writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_softmax_rows(
    exec: *const ta_executor,
    input: *const ta_tensor,
    output: *const ta_tensor,
    scale: f32,
    mask_kind: u32,
    valid_or_prefix: u32,
    rows_per_group: u32,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_softmax_rows: exec is null")?;
        let input = exec.checked_tensor(input, "ta_softmax_rows", "input")?;
        let output = exec.checked_tensor(output, "ta_softmax_rows", "output")?;
        let mask = match mask_kind {
            0 => SoftmaxMask::Full,
            1 => SoftmaxMask::Prefix {
                valid: valid_or_prefix,
            },
            2 => SoftmaxMask::Causal {
                prefix: valid_or_prefix,
                rows_per_group,
            },
            other => {
                bail!("ta_softmax_rows: mask_kind {other} is invalid (0=full 1=prefix 2=causal)")
            }
        };
        write_stats(
            stats,
            exec.exec
                .run_softmax_rows(&input.tensor, &output.tensor, scale, mask)?,
        );
        Ok(())
    })
}

/// RMSNorm over the last dimension: `out = x * w / sqrt(mean(x^2) +
/// eps)`. `weight` has the row length. In-place is allowed.
///
/// # Safety
///
/// `exec`, `input`, `weight`, and `output` must be live handles.
/// `stats` may be null; otherwise it must point to writable
/// `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_rms_norm(
    exec: *const ta_executor,
    input: *const ta_tensor,
    weight: *const ta_tensor,
    output: *const ta_tensor,
    eps: f32,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_rms_norm: exec is null")?;
        let input = exec.checked_tensor(input, "ta_rms_norm", "input")?;
        let weight = exec.checked_tensor(weight, "ta_rms_norm", "weight")?;
        let output = exec.checked_tensor(output, "ta_rms_norm", "output")?;
        write_stats(
            stats,
            exec.exec
                .run_rms_norm(&input.tensor, &weight.tensor, &output.tensor, eps)?,
        );
        Ok(())
    })
}

/// LayerNorm over the last dimension:
/// `out = (x - mean) * w / sqrt(var + eps) + b`. `weight` and `bias`
/// have the row length. In-place is allowed.
///
/// # Safety
///
/// `exec`, `input`, `weight`, `bias`, and `output` must be live
/// handles. `stats` may be null; otherwise it must point to writable
/// `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_layer_norm(
    exec: *const ta_executor,
    input: *const ta_tensor,
    weight: *const ta_tensor,
    bias: *const ta_tensor,
    output: *const ta_tensor,
    eps: f32,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_layer_norm: exec is null")?;
        let input = exec.checked_tensor(input, "ta_layer_norm", "input")?;
        let weight = exec.checked_tensor(weight, "ta_layer_norm", "weight")?;
        let bias = exec.checked_tensor(bias, "ta_layer_norm", "bias")?;
        let output = exec.checked_tensor(output, "ta_layer_norm", "output")?;
        write_stats(
            stats,
            exec.exec.run_layer_norm(
                &input.tensor,
                &weight.tensor,
                &bias.tensor,
                &output.tensor,
                eps,
            )?,
        );
        Ok(())
    })
}

/// Rotary position embedding over `[T, heads, head_dim]` activations
/// (any tensor whose element count is `T * heads * head_dim`). `table`
/// is the precomputed `[T_max, rot_dim/2, 2]` (cos, sin) tensor.
/// In-place is allowed.
///
/// # Safety
///
/// `exec`, `input`, `table`, and `output` must be live handles. `desc`
/// must point to a valid `ta_rope_desc`. `stats` may be null; otherwise
/// it must point to writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_rope(
    exec: *const ta_executor,
    input: *const ta_tensor,
    table: *const ta_tensor,
    output: *const ta_tensor,
    desc: *const ta_rope_desc,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_rope: exec is null")?;
        let input = exec.checked_tensor(input, "ta_rope", "input")?;
        let table = exec.checked_tensor(table, "ta_rope", "table")?;
        let output = exec.checked_tensor(output, "ta_rope", "output")?;
        let desc = checked_ref(desc, "ta_rope: desc is null")?;
        write_stats(
            stats,
            exec.exec.run_rope(
                &input.tensor,
                &table.tensor,
                &output.tensor,
                RopeDesc {
                    heads: desc.heads,
                    head_dim: desc.head_dim,
                    rot_dim: desc.rot_dim,
                    pos_base: desc.pos_base,
                    ..Default::default()
                },
            )?,
        );
        Ok(())
    })
}

/// Strided 3D copy: for every `(x, y, z)` in `desc->extent`, element
/// `src_offset + x*src_strides[0] + y*src_strides[1] +
/// z*src_strides[2]` of `src` is copied to the equivalent destination
/// index. Covers transpose/permute, KV-cache append, head reshaping,
/// and sub-matrix extraction. `src` and `dst` must be different tensors
/// (invocations are unordered); passing the same tensor fails with
/// `-1`.
///
/// # Safety
///
/// `exec`, `src`, and `dst` must be live handles. `desc` must point to
/// a valid `ta_copy_desc`. `stats` may be null; otherwise it must point
/// to writable `ta_run_stats` storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ta_copy_strided(
    exec: *const ta_executor,
    src: *const ta_tensor,
    dst: *const ta_tensor,
    desc: *const ta_copy_desc,
    stats: *mut ta_run_stats,
) -> c_int {
    ffi_status(|| {
        let exec = checked_ref(exec, "ta_copy_strided: exec is null")?;
        let src = exec.checked_tensor(src, "ta_copy_strided", "src")?;
        let dst = exec.checked_tensor(dst, "ta_copy_strided", "dst")?;
        let desc = checked_ref(desc, "ta_copy_strided: desc is null")?;
        write_stats(
            stats,
            exec.exec.run_copy_strided(
                &src.tensor,
                &dst.tensor,
                CopyDesc {
                    extent: desc.extent,
                    src_offset: desc.src_offset,
                    src_strides: desc.src_strides,
                    dst_offset: desc.dst_offset,
                    dst_strides: desc.dst_strides,
                    ..Default::default()
                },
            )?,
        );
        Ok(())
    })
}
