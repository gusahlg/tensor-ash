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
}

impl<'a> MatmulOp<'a> {
    #[inline]
    pub fn new(call: MatmulCall<'a>) -> Self {
        Self {
            call,
            epilogue: Epilogue::NONE,
            normed_a: None,
        }
    }

    #[inline]
    pub fn with_epilogue(call: MatmulCall<'a>, epilogue: Epilogue<'a>) -> Self {
        Self {
            call,
            epilogue,
            normed_a: None,
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
