use crate::tensor::Tensor;

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
    /// `out = act_result * d` — SwiGLU-style gating.
    Mul { d: &'a Tensor },
}

/// Fused epilogue applied while the output tile is still in registers,
/// in order: `+bias` → activation → binary op with `D`. `bias` may be
/// broadcast `[N]` or batched `[B, N]`.
///
/// Requires a BDA kernel. Descriptor-bound kernels reject epilogues.
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
            ..Default::default()
        }
    }

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

/// Geometry for the fused store epilogue (see [`MatmulStore`]).
///
/// The effective position is `P = pos_base + *pos_addr`; the scatter
/// element address for output column `c` is
/// `P * pos_scale + (c / head_dim) * stride_head +
///  (c % head_dim) * stride_dim`.  RoPE rotates the interleaved
/// `(even, odd)` column pairs of each head by the angle at row
/// `P * head_dim/2 + (c % head_dim)/2` of a `[T_max, head_dim/2, 2]`
/// (cos, sin) f32 table — full rotary only (`rot_dim == head_dim`).
#[derive(Copy, Clone, Debug, Default)]
pub struct MatmulStoreDesc {
    /// Head vector length; even, must divide N (heads = N / head_dim).
    pub head_dim: u32,
    /// Absolute position (decode step) of the produced row.
    pub pos_base: u32,
    /// Destination element advance per position (scatter modes).
    pub pos_scale: u32,
    /// Destination element stride per head (scatter modes).
    pub stride_head: u32,
    /// Destination element stride per head lane (scatter modes).
    pub stride_dim: u32,
    /// Optional device address of a `PosBuffer` u32 cell (0 = none):
    /// the effective position becomes `pos_base + *cell` at execution
    /// time, so a recorded dispatch replays per token.  Bounds and
    /// table-coverage validation only see `pos_base`; the caller keeps
    /// the runtime position in range.
    pub pos_addr: u64,
}

/// Fused store-time destination for the M=1 f16-weights row GEMVs: the
/// output row is RoPE-rotated and/or scattered into a strided KV-cache
/// layout as it leaves the kernel, replacing the separate rope / append
/// dispatch (and its hazard barrier) that would otherwise follow.
///
/// Requires an `f16w_*row_bda*` route (M=1, f16 B); excludes
/// `accumulate` and every [`Epilogue`] form (the host validates).
/// `dst` may be f32 or f16 (RNE-narrowing) storage; the scatter modes
/// never touch C.
#[derive(Copy, Clone, Default)]
pub enum MatmulStore<'a> {
    #[default]
    None,
    /// Rotate adjacent output pairs and store to C (decode q, in place).
    Rope {
        table: &'a Tensor,
        desc: MatmulStoreDesc,
    },
    /// Scatter the row into `dst` (decode v -> `[H, T_max, dh]` cache).
    Scatter {
        dst: &'a Tensor,
        desc: MatmulStoreDesc,
    },
    /// Rotate, then scatter (decode k -> `[H, dh, T_max]` Kt cache).
    RopeScatter {
        table: &'a Tensor,
        dst: &'a Tensor,
        desc: MatmulStoreDesc,
    },
}

impl<'a> MatmulStore<'a> {
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Shader `STORE_MODE` value (spec constant 7).
    #[inline]
    pub(crate) fn mode(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Rope { .. } => 1,
            Self::Scatter { .. } => 2,
            Self::RopeScatter { .. } => 3,
        }
    }

    #[inline]
    pub(crate) fn table(&self) -> Option<&'a Tensor> {
        match self {
            Self::Rope { table, .. } | Self::RopeScatter { table, .. } => Some(table),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn dst(&self) -> Option<&'a Tensor> {
        match self {
            Self::Scatter { dst, .. } | Self::RopeScatter { dst, .. } => Some(dst),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn desc(&self) -> MatmulStoreDesc {
        match self {
            Self::None => MatmulStoreDesc::default(),
            Self::Rope { desc, .. }
            | Self::Scatter { desc, .. }
            | Self::RopeScatter { desc, .. } => *desc,
        }
    }
}

/// A GEMM call plus its fused epilogue.
#[derive(Copy, Clone)]
pub struct MatmulOp<'a> {
    pub call: MatmulCall<'a>,
    pub epilogue: Epilogue<'a>,
    /// `(weight, eps)`: RMS-normalize each A row in the kernel before
    /// the product, i.e. compute `C = rms_norm(A, weight, eps) @ B`.
    /// Only the row-GEMV (`*row_bda*`) kernels implement this, so it is
    /// restricted to M=1 decode-style calls; the norm weight rides the
    /// unused `bias_ptr` slot and `eps` rides `beta`, so it cannot
    /// combine with an epilogue bias or the `AddScaled` binary.
    pub normed_a: Option<(&'a Tensor, f32)>,
    /// Fused store-time RoPE / KV-cache scatter (see [`MatmulStore`]).
    pub store: MatmulStore<'a>,
}

impl<'a> MatmulOp<'a> {
    #[inline]
    pub fn new(call: MatmulCall<'a>) -> Self {
        Self {
            call,
            epilogue: Epilogue::NONE,
            normed_a: None,
            store: MatmulStore::None,
        }
    }

    #[inline]
    pub fn with_epilogue(call: MatmulCall<'a>, epilogue: Epilogue<'a>) -> Self {
        Self {
            call,
            epilogue,
            normed_a: None,
            store: MatmulStore::None,
        }
    }

    /// Fold an RMSNorm of A into the kernel (see [`MatmulOp::normed_a`]):
    /// `C = rms_norm(A, weight, eps) @ B`.  Composable with
    /// [`with_epilogue`](Self::with_epilogue) as long as the epilogue
    /// carries no bias and no `AddScaled` binary.
    #[inline]
    pub fn with_normed_a(mut self, weight: &'a Tensor, eps: f32) -> Self {
        self.normed_a = Some((weight, eps));
        self
    }

    /// RoPE-rotate the output row in place at store time (decode q).
    #[inline]
    pub fn with_store_rope(mut self, table: &'a Tensor, desc: MatmulStoreDesc) -> Self {
        self.store = MatmulStore::Rope { table, desc };
        self
    }

    /// Scatter the output row into `dst` at store time (decode v
    /// append); C is never written.
    #[inline]
    pub fn with_store_scatter(mut self, dst: &'a Tensor, desc: MatmulStoreDesc) -> Self {
        self.store = MatmulStore::Scatter { dst, desc };
        self
    }

    /// RoPE-rotate, then scatter into `dst` (decode k -> Kt-cache
    /// append); C is never written.
    #[inline]
    pub fn with_store_rope_scatter(
        mut self,
        table: &'a Tensor,
        dst: &'a Tensor,
        desc: MatmulStoreDesc,
    ) -> Self {
        self.store = MatmulStore::RopeScatter { table, dst, desc };
        self
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
