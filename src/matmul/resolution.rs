use anyhow::{Result, anyhow, bail};

use super::{MatmulCall, MatmulOp};
use crate::pipeline::MatmulPushConstants;
use crate::tensor::Tensor;

/// A matrix or batched-matrix shape normalized to `[B, rows, cols]`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MatrixShape {
    pub batch: u32,
    pub rows: u32,
    pub cols: u32,
}

impl MatrixShape {
    /// Convert rank-2 `[rows, cols]` or rank-3 `[batch, rows, cols]` shapes.
    pub fn from_tensor_shape(shape: &[u32]) -> Result<Self> {
        let normalized = match shape {
            [rows, cols] => Self {
                batch: 1,
                rows: *rows,
                cols: *cols,
            },
            [batch, rows, cols] => Self {
                batch: *batch,
                rows: *rows,
                cols: *cols,
            },
            _ => bail!(
                "tensor rank {} not supported by matmul (must be 2 or 3)",
                shape.len()
            ),
        };

        if normalized.batch == 0 || normalized.rows == 0 || normalized.cols == 0 {
            bail!(
                "matmul dimensions must be non-zero, got B={} rows={} cols={}",
                normalized.batch,
                normalized.rows,
                normalized.cols
            );
        }
        Ok(normalized)
    }

    fn from_tensor(tensor: &Tensor) -> Result<Self> {
        Self::from_tensor_shape(tensor.shape())
    }

    fn batch_stride(self, label: &'static str) -> Result<u32> {
        let plane_elements = checked_product([self.rows as u64, self.cols as u64])
            .ok_or_else(|| anyhow!("{label} matrix size overflows u64"))?;
        let stride = if self.batch == 1 {
            0
        } else {
            u32::try_from(plane_elements)
                .map_err(|_| anyhow!("{label} batch stride overflows u32"))?
        };

        // Shader address expressions are `uint`. A contiguous layout is
        // addressable when its element count is at most 2^32, because the
        // largest zero-based offset is then exactly `u32::MAX`.
        let addressable = checked_product([self.batch as u64, plane_elements])
            .map(|elements| elements <= u32::MAX as u64 + 1)
            .unwrap_or(false);
        if !addressable {
            bail!(
                "{label} layout exceeds shader u32 indexing: B={} rows={} cols={}",
                self.batch,
                self.rows,
                self.cols
            );
        }
        Ok(stride)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct ResolvedMatmul {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_c: u32,
    pub total_flops: u64,
}

pub(crate) enum ResolvedMatmulBatch {
    One([ResolvedMatmul; 1]),
    Many(Vec<ResolvedMatmul>),
}

impl ResolvedMatmulBatch {
    pub(crate) fn from_ops(ops: &[MatmulOp<'_>]) -> Result<Self> {
        if let [op] = ops {
            return Ok(Self::One([ResolvedMatmul::from_op(op)?]));
        }
        ops.iter()
            .map(ResolvedMatmul::from_op)
            .collect::<Result<Vec<_>>>()
            .map(Self::Many)
    }

    pub(crate) fn as_slice(&self) -> &[ResolvedMatmul] {
        match self {
            Self::One(resolved) => resolved,
            Self::Many(resolved) => resolved,
        }
    }
}

impl ResolvedMatmul {
    pub(crate) fn from_call(call: &MatmulCall<'_>) -> Result<Self> {
        Self::from_matrix_shapes(
            MatrixShape::from_tensor(call.a)?,
            MatrixShape::from_tensor(call.b)?,
            MatrixShape::from_tensor(call.c)?,
        )
    }

    pub(crate) fn from_op(op: &MatmulOp<'_>) -> Result<Self> {
        let resolved = Self::from_call(&op.call)?;
        resolved.validate_epilogue(op)?;
        Ok(resolved)
    }

    fn validate_epilogue(&self, op: &MatmulOp<'_>) -> Result<()> {
        if let Some(bias) = op.epilogue.bias {
            self.validate_bias_shape(bias.shape())?;
        }
        if let Some(d) = op.epilogue.d_tensor()
            && d.shape() != op.call.c.shape()
        {
            bail!(
                "epilogue D shape {:?} must equal C shape {:?}",
                d.shape(),
                op.call.c.shape()
            );
        }
        Ok(())
    }

    pub(super) fn validate_bias_shape(&self, shape: &[u32]) -> Result<()> {
        let shared = [self.n];
        let batched = [self.batch, self.n];
        if shape != shared && shape != batched {
            bail!(
                "epilogue bias shape {shape:?}, expected {:?} or {:?}",
                shared,
                batched
            );
        }
        Ok(())
    }

    pub(crate) fn push_constants(
        &self,
        alpha: f32,
        accumulate: bool,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
    ) -> MatmulPushConstants {
        MatmulPushConstants {
            m: self.m,
            n: self.n,
            k: self.k,
            batch_stride_a: self.batch_stride_a,
            batch_stride_b: self.batch_stride_b,
            batch_stride_c: self.batch_stride_c,
            flags: u32::from(accumulate),
            alpha,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr: 0,
            bias_ptr: 0,
            beta: 0.0,
            bias_batch_stride: 0,
        }
    }

    /// Build push constants for the split-K kernel. Bits 16..31 of `flags`
    /// carry `num_k_splits`; split-K never sets the legacy accumulate bit.
    pub(crate) fn split_k_push_constants(
        &self,
        alpha: f32,
        a_ptr: u64,
        b_ptr: u64,
        c_ptr: u64,
        num_k_splits: u32,
    ) -> MatmulPushConstants {
        debug_assert!((1..=0xFFFF).contains(&num_k_splits));
        MatmulPushConstants {
            m: self.m,
            n: self.n,
            k: self.k,
            batch_stride_a: self.batch_stride_a,
            batch_stride_b: self.batch_stride_b,
            batch_stride_c: self.batch_stride_c,
            flags: num_k_splits << 16,
            alpha,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr: 0,
            bias_ptr: 0,
            beta: 0.0,
            bias_batch_stride: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn from_shapes(a: &[u32], b: &[u32], c: &[u32]) -> Result<Self> {
        Self::from_matrix_shapes(
            MatrixShape::from_tensor_shape(a)?,
            MatrixShape::from_tensor_shape(b)?,
            MatrixShape::from_tensor_shape(c)?,
        )
    }

    fn from_matrix_shapes(a: MatrixShape, b: MatrixShape, c: MatrixShape) -> Result<Self> {
        ensure_dim_eq("A.K", a.cols, "B.K", b.rows)?;
        ensure_dim_eq("A.M", a.rows, "C.M", c.rows)?;
        ensure_dim_eq("B.N", b.cols, "C.N", c.cols)?;
        ensure_broadcastable("A", a.batch, c.batch)?;
        ensure_broadcastable("B", b.batch, c.batch)?;

        Ok(Self {
            m: a.rows,
            n: b.cols,
            k: a.cols,
            batch: c.batch,
            batch_stride_a: a.batch_stride("A")?,
            batch_stride_b: b.batch_stride("B")?,
            batch_stride_c: c.batch_stride("C")?,
            total_flops: checked_flops(c.batch, a.rows, b.cols, a.cols)?,
        })
    }
}

pub(crate) fn total_flops(resolved: &[ResolvedMatmul]) -> Result<u64> {
    resolved.iter().try_fold(0u64, |acc, matmul| {
        acc.checked_add(matmul.total_flops)
            .ok_or_else(|| anyhow!("total FLOP count overflows u64"))
    })
}

fn ensure_dim_eq(left_label: &str, left: u32, right_label: &str, right: u32) -> Result<()> {
    if left != right {
        bail!("matmul shape mismatch: {left_label}={left} vs {right_label}={right}");
    }
    Ok(())
}

fn ensure_broadcastable(label: &str, input_batch: u32, output_batch: u32) -> Result<()> {
    if input_batch != 1 && input_batch != output_batch {
        bail!(
            "matmul incompatible batch dims: {label}.B={input_batch} cannot broadcast to C.B={output_batch}"
        );
    }
    Ok(())
}

fn checked_flops(batch: u32, m: u32, n: u32, k: u32) -> Result<u64> {
    checked_product([2, batch as u64, m as u64, n as u64, k as u64])
        .ok_or_else(|| anyhow!("matmul FLOP count overflows u64"))
}

fn checked_product(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1u64, |acc, value| acc.checked_mul(value))
}
