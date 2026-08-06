//! Matmul API types and shape resolution.
//!
//! This module keeps the CPU-side contract for the GEMM shader in one place:
//! rank handling, batch broadcasting, push constants, dispatch dimensions, and
//! FLOP accounting.

use anyhow::{Result, anyhow, bail};

use crate::pipeline::MatmulPushConstants;
use crate::tensor::Tensor;

macro_rules! ensure_dim_eq {
    ($left_label:literal, $left:expr, $right_label:literal, $right:expr) => {{
        if $left != $right {
            bail!(
                "matmul shape mismatch: {}={} vs {}={}",
                $left_label,
                $left,
                $right_label,
                $right
            );
        }
    }};
}

/// One GEMM problem: `C <- alpha * A @ B  (+ C if accumulate)`.
#[derive(Copy, Clone)]
pub struct MatmulCall<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub c: &'a Tensor,
    pub alpha: f32,
    pub accumulate: bool,
}

/// Elementwise activation applied in the fused GEMM epilogue.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Activation {
    #[default]
    None,
    Relu,
    /// `x * sigmoid(x)` (a.k.a. swish).
    Silu,
    /// tanh-approximated GELU (matches PyTorch `approximate="tanh"`).
    Gelu,
}

impl Activation {
    #[inline]
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Relu => 1,
            Self::Silu => 2,
            Self::Gelu => 3,
        }
    }
}

/// Optional second elementwise operand applied after the activation.
/// `d` must have exactly C's shape (batch included); it is indexed with
/// C's layout.
#[derive(Copy, Clone, Default)]
pub enum EpilogueBinary<'a> {
    #[default]
    None,
    /// `out = act_result + beta * d` — residual connections.
    AddScaled { d: &'a Tensor, beta: f32 },
    /// `out = act_result * d` — SwiGLU-style gating
    /// (`silu(x@W_gate) * up`).
    Mul { d: &'a Tensor },
}

/// Fused epilogue applied while the output tile is still in registers,
/// in order: `+bias` → activation → binary op with `D`.  `bias` may be
/// broadcast `[N]` or batched `[B, N]`. Avoids a
/// full write+read+write round-trip over C for bias/activation/residual
/// chains.
///
/// Requires a BDA kernel (any device with `bufferDeviceAddress`, i.e.
/// every kernel the auto-selector picks there).  Descriptor-bound
/// kernels reject epilogues.
#[derive(Copy, Clone, Default)]
pub struct Epilogue<'a> {
    /// A shared length-N vector or B-by-N rows added before activation.
    pub bias: Option<&'a Tensor>,
    pub activation: Activation,
    pub binary: EpilogueBinary<'a>,
}

impl Epilogue<'_> {
    pub const NONE: Epilogue<'static> = Epilogue {
        bias: None,
        activation: Activation::None,
        binary: EpilogueBinary::None,
    };

    #[inline]
    pub fn is_none(&self) -> bool {
        self.bias.is_none()
            && self.activation == Activation::None
            && matches!(self.binary, EpilogueBinary::None)
    }

    pub(crate) fn key(&self) -> crate::pipeline::EpilogueKey {
        crate::pipeline::EpilogueKey {
            bias: self.bias.is_some(),
            activation: self.activation.code(),
            binary: match self.binary {
                EpilogueBinary::None => 0,
                EpilogueBinary::AddScaled { .. } => 1,
                EpilogueBinary::Mul { .. } => 2,
            },
        }
    }

    /// The `D` operand tensor, if any.
    #[inline]
    pub(crate) fn d_tensor(&self) -> Option<&Tensor> {
        match self.binary {
            EpilogueBinary::None => None,
            EpilogueBinary::AddScaled { d, .. } | EpilogueBinary::Mul { d } => Some(d),
        }
    }

    #[inline]
    pub(crate) fn beta(&self) -> f32 {
        match self.binary {
            EpilogueBinary::AddScaled { beta, .. } => beta,
            _ => 0.0,
        }
    }
}

/// A GEMM call plus its fused epilogue.  The plain `run_matmuls` /
/// `run_matmul_graph` entry points wrap each `MatmulCall` in a
/// `MatmulOp` with `Epilogue::NONE`.
#[derive(Copy, Clone)]
pub struct MatmulOp<'a> {
    pub call: MatmulCall<'a>,
    pub epilogue: Epilogue<'a>,
}

impl<'a> MatmulOp<'a> {
    #[inline]
    pub fn new(call: MatmulCall<'a>) -> Self {
        Self {
            call,
            epilogue: Epilogue::NONE,
        }
    }

    #[inline]
    pub fn with_epilogue(call: MatmulCall<'a>, epilogue: Epilogue<'a>) -> Self {
        Self { call, epilogue }
    }
}

impl<'a> From<MatmulCall<'a>> for MatmulOp<'a> {
    #[inline]
    fn from(call: MatmulCall<'a>) -> Self {
        Self::new(call)
    }
}

/// Per-run statistics. `gpu_time_ns` is the on-device GPU time measured
/// via timestamp queries, or `None` if the device does not support them.
#[derive(Debug, Copy, Clone)]
pub struct RunStats {
    pub gpu_time_ns: Option<u64>,
    pub n_calls: usize,
    pub total_flops: u64,
}

impl RunStats {
    /// GPU TFLOPS if GPU time was measured, else `None`.
    pub fn tflops(&self) -> Option<f64> {
        self.gpu_time_ns
            .filter(|&ns| ns > 0)
            .map(|ns| self.total_flops as f64 / ns as f64 * 1e-3)
    }
}

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

    pub(crate) fn from_tensor(tensor: &Tensor) -> Result<Self> {
        Self::from_tensor_shape(&tensor.shape)
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
        let a = MatrixShape::from_tensor(call.a)?;
        let b = MatrixShape::from_tensor(call.b)?;
        let c = MatrixShape::from_tensor(call.c)?;
        Self::from_matrix_shapes(a, b, c)
    }

    pub(crate) fn from_op(op: &MatmulOp<'_>) -> Result<Self> {
        let resolved = Self::from_call(&op.call)?;
        resolved.validate_epilogue(op)?;
        Ok(resolved)
    }

    /// Shape checks for the fused epilogue: bias must contain either N
    /// shared elements or B*N per-batch elements; D must match C's shape
    /// exactly (it is indexed with C's layout, batch stride included).
    fn validate_epilogue(&self, op: &MatmulOp<'_>) -> Result<()> {
        if let Some(bias) = op.epilogue.bias {
            let numel = bias.len();
            let shared = self.n as u64;
            let batched = self.batch as u64 * shared;
            if numel != shared && numel != batched {
                bail!(
                    "epilogue bias has {numel} elements, expected N={} or B*N={batched}",
                    self.n
                );
            }
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
            flags: if accumulate { 1 } else { 0 },
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

    /// Build push constants for the split-K kernel.  Same layout as
    /// `push_constants()` (so we don't disturb the bit-stable
    /// `MatmulPushConstants` block), but the `flags` field is encoded
    /// differently:
    ///
    ///   * bits 0..15 : reserved (kept zero — split-K never sets the
    ///     legacy `accumulate` bit; correctness comes from the
    ///     external zero-fill of C).
    ///   * bits 16..31: `num_k_splits` (the kernel reads this with
    ///     `pc.flags >> 16u`).
    ///
    /// `num_k_splits` must be in `[1, 65535]`; the executor guards the
    /// range before calling this.
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
    fn from_shapes(a: &[u32], b: &[u32], c: &[u32]) -> Result<Self> {
        Self::from_matrix_shapes(
            MatrixShape::from_tensor_shape(a)?,
            MatrixShape::from_tensor_shape(b)?,
            MatrixShape::from_tensor_shape(c)?,
        )
    }

    fn from_matrix_shapes(a: MatrixShape, b: MatrixShape, c: MatrixShape) -> Result<Self> {
        ensure_dim_eq!("A.K", a.cols, "B.K", b.rows);
        ensure_dim_eq!("A.M", a.rows, "C.M", c.rows);
        ensure_dim_eq!("B.N", b.cols, "C.N", c.cols);

        if !(a.batch == c.batch || a.batch == 1) {
            bail!(
                "matmul incompatible batch dims A.B={} B.B={} C.B={}",
                a.batch,
                b.batch,
                c.batch
            );
        }
        if !(b.batch == c.batch || b.batch == 1) {
            bail!(
                "matmul incompatible batch dims A.B={} B.B={} C.B={}",
                a.batch,
                b.batch,
                c.batch
            );
        }

        let batch_stride_a = if a.batch == 1 {
            0
        } else {
            checked_mul_u32("A batch stride", a.rows, a.cols)?
        };
        let batch_stride_b = if b.batch == 1 {
            0
        } else {
            checked_mul_u32("B batch stride", b.rows, b.cols)?
        };
        let batch_stride_c = if c.batch == 1 {
            0
        } else {
            checked_mul_u32("C batch stride", c.rows, c.cols)?
        };

        Ok(Self {
            m: a.rows,
            n: b.cols,
            k: a.cols,
            batch: c.batch,
            batch_stride_a,
            batch_stride_b,
            batch_stride_c,
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

fn checked_mul_u32(label: &'static str, lhs: u32, rhs: u32) -> Result<u32> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| anyhow!("{label} overflows u32: {lhs} * {rhs}"))
}

fn checked_flops(batch: u32, m: u32, n: u32, k: u32) -> Result<u64> {
    [2u64, batch as u64, m as u64, n as u64, k as u64]
        .into_iter()
        .try_fold(1u64, |acc, value| {
            acc.checked_mul(value)
                .ok_or_else(|| anyhow!("matmul FLOP count overflows u64"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_rank2_matmul() {
        let dims = ResolvedMatmul::from_shapes(&[2, 3], &[3, 4], &[2, 4]).unwrap();

        assert_eq!(dims.m, 2);
        assert_eq!(dims.n, 4);
        assert_eq!(dims.k, 3);
        assert_eq!(dims.batch, 1);
        assert_eq!(dims.batch_stride_a, 0);
        assert_eq!(dims.batch_stride_b, 0);
        assert_eq!(dims.batch_stride_c, 0);
        assert_eq!(dims.total_flops, 48);
    }

    #[test]
    fn resolves_batched_matmul() {
        let dims = ResolvedMatmul::from_shapes(&[5, 2, 3], &[5, 3, 4], &[5, 2, 4]).unwrap();

        assert_eq!(dims.batch, 5);
        assert_eq!(dims.batch_stride_a, 6);
        assert_eq!(dims.batch_stride_b, 12);
        assert_eq!(dims.batch_stride_c, 8);
        assert_eq!(dims.total_flops, 240);
    }

    #[test]
    fn broadcasts_either_or_both_inputs_to_output_batch() {
        let a = ResolvedMatmul::from_shapes(&[1, 2, 3], &[7, 3, 4], &[7, 2, 4]).unwrap();
        let b = ResolvedMatmul::from_shapes(&[7, 2, 3], &[1, 3, 4], &[7, 2, 4]).unwrap();
        let both = ResolvedMatmul::from_shapes(&[1, 2, 3], &[1, 3, 4], &[7, 2, 4]).unwrap();

        assert_eq!(a.batch_stride_a, 0);
        assert_eq!(a.batch_stride_b, 12);
        assert_eq!(b.batch_stride_a, 6);
        assert_eq!(b.batch_stride_b, 0);
        assert_eq!(both.batch_stride_a, 0);
        assert_eq!(both.batch_stride_b, 0);
        assert_eq!(both.batch_stride_c, 8);
        assert_eq!(both.total_flops, 336);
    }

    #[test]
    fn rejects_mismatched_inner_dimension() {
        let err = ResolvedMatmul::from_shapes(&[2, 3], &[5, 4], &[2, 4])
            .unwrap_err()
            .to_string();

        assert!(err.contains("A.K=3"));
        assert!(err.contains("B.K=5"));
    }

    #[test]
    fn rejects_output_batch_that_cannot_hold_inputs() {
        let err = ResolvedMatmul::from_shapes(&[7, 2, 3], &[7, 3, 4], &[1, 2, 4])
            .unwrap_err()
            .to_string();

        assert!(err.contains("incompatible batch dims"));
    }

    #[test]
    fn rejects_zero_dimensions() {
        let err = MatrixShape::from_tensor_shape(&[0, 3])
            .unwrap_err()
            .to_string();

        assert!(err.contains("non-zero"));
    }

    #[test]
    fn rejects_batch_stride_overflow() {
        let err = ResolvedMatmul::from_shapes(
            &[2, u32::MAX, u32::MAX],
            &[1, u32::MAX, 1],
            &[2, u32::MAX, 1],
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("A batch stride"));
        assert!(err.contains("overflows"));
    }

    #[test]
    fn rejects_flop_count_overflow() {
        let err = ResolvedMatmul::from_shapes(
            &[u32::MAX, u32::MAX],
            &[u32::MAX, u32::MAX],
            &[u32::MAX, u32::MAX],
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("FLOP count overflows"));
    }

    #[test]
    fn builds_push_constants_from_resolved_shape_and_call_flags() {
        let dims = ResolvedMatmul::from_shapes(&[2, 2, 3], &[1, 3, 4], &[2, 2, 4]).unwrap();
        let pc = dims.push_constants(0.5, true, 0xAA00, 0xBB00, 0xCC00);

        assert_eq!(pc.m, 2);
        assert_eq!(pc.n, 4);
        assert_eq!(pc.k, 3);
        assert_eq!(pc.batch_stride_a, 6);
        assert_eq!(pc.batch_stride_b, 0);
        assert_eq!(pc.batch_stride_c, 8);
        assert_eq!(pc.flags, 1);
        assert_eq!(pc.alpha, 0.5);
        assert_eq!(pc.a_ptr, 0xAA00);
        assert_eq!(pc.b_ptr, 0xBB00);
        assert_eq!(pc.c_ptr, 0xCC00);
    }

    #[test]
    fn stats_tflops_use_gpu_time_when_available() {
        let stats = RunStats {
            gpu_time_ns: Some(1_000_000),
            n_calls: 1,
            total_flops: 2_000_000_000,
        };

        assert_eq!(stats.tflops(), Some(2.0));
    }

    #[test]
    fn stats_tflops_are_absent_for_zero_time() {
        let stats = RunStats {
            gpu_time_ns: Some(0),
            n_calls: 1,
            total_flops: 2_000_000_000,
        };

        assert_eq!(stats.tflops(), None);
    }
}
