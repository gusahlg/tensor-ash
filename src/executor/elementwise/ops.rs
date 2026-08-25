//! Softmax, RMS/LayerNorm, RoPE, strided copy, binary, argmax, embed-gather.

use anyhow::{Result, bail};

use crate::dtype::DType;
use crate::matmul::RunStats;
use crate::tensor::Tensor;

use super::pc::{
    ArgmaxPc, BinaryPc, CopyPc, EmbedGatherPc, NormPc, RopePc, RopeScatterPc, SoftmaxPc,
};
use super::{
    BinaryOp, CopyDesc, ElementwiseDispatch, Executor, Op, RopeDesc, RopeScatterDesc, SoftmaxMask,
    WG, rows_cols,
};
use crate::executor::{HostU32Buffer, TokenIdBuffer};

impl Executor {
    /// Numerically stable softmax over the last dimension, with
    /// optional valid-length masking (see [`SoftmaxMask`]).  `scale`
    /// multiplies inputs before the max/exp passes (pass `1.0`, or
    /// `1/sqrt(dh)` to fold in attention scaling).  In-place
    /// (`input == output`) is safe.
    pub fn run_softmax_rows(
        &self,
        input: &Tensor,
        output: &Tensor,
        scale: f32,
        mask: SoftmaxMask,
    ) -> Result<RunStats> {
        let dispatch = self.plan_softmax_rows(input, output, scale, mask)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_softmax_rows(
        &self,
        input: &Tensor,
        output: &Tensor,
        scale: f32,
        mask: SoftmaxMask,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(input, "run_softmax_rows", "input")?;
        self.ensure_f32(output, "run_softmax_rows", "output")?;
        Self::ensure_same_shape("run_softmax_rows", input, output)?;
        let (rows, cols) = rows_cols(input, "run_softmax_rows")?;
        let (valid_base, rows_per_group, causal) = match mask {
            SoftmaxMask::Full => (cols, 1, 0),
            SoftmaxMask::Prefix { valid } => (valid.min(cols), 1, 0),
            SoftmaxMask::Causal {
                prefix,
                rows_per_group,
            } => {
                if rows_per_group == 0 || !rows.is_multiple_of(rows_per_group) {
                    bail!(
                        "run_softmax_rows: rows_per_group {rows_per_group} must divide rows {rows}"
                    );
                }
                (prefix, rows_per_group, 1)
            }
        };
        let pipeline = self.elementwise()?.pipeline(Op::Softmax);
        self.plan_elementwise(
            pipeline,
            &SoftmaxPc {
                rows,
                cols,
                valid_base,
                rows_per_group,
                causal,
                scale,
                in_ptr: input.device_address(),
                out_ptr: output.device_address(),
            },
            rows,
            1,
        )
    }

    /// RMSNorm over the last dimension: `out = x * w / sqrt(mean(x^2)
    /// + eps)`.  `weight` has the row length.  In-place is safe.
    pub fn run_rms_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        output: &Tensor,
        eps: f32,
    ) -> Result<RunStats> {
        let dispatch = self.plan_norm("run_rms_norm", input, weight, None, output, eps)?;
        self.submit_one_elementwise(dispatch)
    }

    /// LayerNorm over the last dimension: `out = (x - mean) * w /
    /// sqrt(var + eps) + b`.  In-place is safe.
    pub fn run_layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        output: &Tensor,
        eps: f32,
    ) -> Result<RunStats> {
        let dispatch = self.plan_norm("run_layer_norm", input, weight, Some(bias), output, eps)?;
        self.submit_one_elementwise(dispatch)
    }

    /// Validate and plan one norm dispatch: RMSNorm when `bias` is
    /// `None`, LayerNorm when `Some` (the two specializations of the
    /// same shader).
    pub(in crate::executor) fn plan_norm(
        &self,
        op: &str,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &Tensor,
        eps: f32,
    ) -> Result<ElementwiseDispatch> {
        // Activations may be f16 (RMSNorm only); the weight (and
        // LayerNorm bias) stay f32 in every variant.
        let io_f16 = self.io_dtype_f16(op, input, output)?;
        self.ensure_f32(weight, op, "weight")?;
        if let Some(bias) = bias {
            self.ensure_f32(bias, op, "bias")?;
        }
        if io_f16 && bias.is_some() {
            bail!("{op}: f16 activations are supported for RMSNorm only (no f16 LayerNorm kernel)");
        }
        Self::ensure_same_shape(op, input, output)?;
        let (rows, cols) = rows_cols(input, op)?;
        if weight.len() != cols as u64 {
            bail!("{op}: weight length {} != row length {cols}", weight.len());
        }
        if let Some(bias) = bias
            && bias.len() != cols as u64
        {
            bail!("{op}: bias length {} != row length {cols}", bias.len());
        }
        let kernel = match (bias.is_some(), io_f16) {
            (true, _) => Op::LayerNorm,
            (false, false) => Op::RmsNorm,
            (false, true) => Op::RmsNormF16,
        };
        self.plan_elementwise(
            self.elementwise()?.pipeline(kernel),
            &NormPc {
                rows,
                cols,
                eps,
                _pad: 0,
                in_ptr: input.device_address(),
                out_ptr: output.device_address(),
                weight_ptr: weight.device_address(),
                bias_ptr: bias.map_or(0, |b| b.device_address()),
            },
            rows,
            1,
        )
    }

    /// Rotary position embedding over `[T, H, dh]` activations (any
    /// tensor whose element count is `T * heads * head_dim`).  `table`
    /// is the precomputed `[T_max, rot_dim/2, 2]` (cos, sin) tensor.
    /// In-place is safe.
    pub fn run_rope(
        &self,
        input: &Tensor,
        table: &Tensor,
        output: &Tensor,
        desc: RopeDesc,
    ) -> Result<RunStats> {
        let dispatch = self.plan_rope(input, table, output, desc)?;
        self.submit_one_elementwise(dispatch)
    }

    /// Shared rope-geometry validation: returns the token count for
    /// `input` under `desc`, checking rot_dim and table coverage.
    fn rope_tokens(op: &str, input: &Tensor, table: &Tensor, desc: RopeDesc) -> Result<u32> {
        if desc.rot_dim < 2 || !desc.rot_dim.is_multiple_of(2) || desc.rot_dim > desc.head_dim {
            bail!(
                "{op}: rot_dim {} must be even, >= 2, and <= head_dim {}",
                desc.rot_dim,
                desc.head_dim
            );
        }
        let vec_elems = desc.heads as u64 * desc.head_dim as u64;
        if vec_elems == 0 || !input.len().is_multiple_of(vec_elems) {
            bail!(
                "{op}: input length {} is not a multiple of heads*head_dim {}",
                input.len(),
                vec_elems
            );
        }
        let tokens = u32::try_from(input.len() / vec_elems)
            .map_err(|_| anyhow::anyhow!("{op}: token count exceeds u32"))?;
        let needed = (desc.pos_base as u64 + tokens as u64) * desc.rot_dim as u64;
        if table.len() < needed {
            bail!(
                "{op}: table length {} < required {} (pos_base {} + {} tokens, rot_dim {})",
                table.len(),
                needed,
                desc.pos_base,
                tokens,
                desc.rot_dim
            );
        }
        Ok(tokens)
    }

    pub(in crate::executor) fn plan_rope(
        &self,
        input: &Tensor,
        table: &Tensor,
        output: &Tensor,
        desc: RopeDesc,
    ) -> Result<ElementwiseDispatch> {
        // f16 activations rotate through the IO_F16 variant (f32
        // math, RNE store); the cos/sin table stays f32 either way.
        let io_f16 = self.io_dtype_f16("run_rope", input, output)?;
        self.ensure_f32(table, "run_rope", "table")?;
        Self::ensure_same_shape("run_rope", input, output)?;
        let tokens = Self::rope_tokens("run_rope", input, table, desc)?;
        let pairs = tokens * desc.heads * (desc.rot_dim / 2);
        let kernel = if io_f16 { Op::RopeF16 } else { Op::Rope };
        let pipeline = self.elementwise()?.pipeline(kernel);
        self.plan_elementwise(
            pipeline,
            &RopePc {
                tokens,
                heads: desc.heads,
                head_dim: desc.head_dim,
                rot_dim: desc.rot_dim,
                pos_base: desc.pos_base,
                _pad: 0,
                in_ptr: input.device_address(),
                out_ptr: output.device_address(),
                table_ptr: table.device_address(),
                pos_ptr: desc.pos_addr,
            },
            pairs.div_ceil(WG),
            1,
        )
    }

    /// Fused RoPE + strided scatter: rotate `[T, H, dh]` activations
    /// (exactly like [`run_rope`](Self::run_rope)) but write every
    /// element straight into `dst` at the strided location given by
    /// [`RopeScatterDesc`] — one dispatch covers the decode k-rope
    /// plus KV-cache append.  `dst` may be f32 or f16 storage (f16
    /// narrows with RNE like the strided-copy path) and must be a
    /// different tensor from `input`.
    pub fn run_rope_scatter(
        &self,
        input: &Tensor,
        table: &Tensor,
        dst: &Tensor,
        desc: RopeDesc,
        scatter: RopeScatterDesc,
    ) -> Result<RunStats> {
        let dispatch = self.plan_rope_scatter(input, table, dst, desc, scatter)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_rope_scatter(
        &self,
        input: &Tensor,
        table: &Tensor,
        dst: &Tensor,
        desc: RopeDesc,
        scatter: RopeScatterDesc,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(input, "run_rope_scatter", "input")?;
        self.ensure_f32(table, "run_rope_scatter", "table")?;
        self.validate_tensor_context(dst, "dst")?;
        if input.raw_buffer() == dst.raw_buffer() {
            bail!("run_rope_scatter: input and dst must be different tensors");
        }
        let tokens = Self::rope_tokens("run_rope_scatter", input, table, desc)?;
        // Every (token, head, dim) element is written (rotated lanes
        // and pass-through lanes alike); bound the farthest one.
        let max_index = scatter.dst_offset as u64
            + (tokens as u64 - 1) * scatter.dst_strides[0] as u64
            + (desc.heads as u64 - 1) * scatter.dst_strides[1] as u64
            + (desc.head_dim as u64 - 1) * scatter.dst_strides[2] as u64;
        if max_index >= dst.len() {
            bail!(
                "run_rope_scatter: destination access reaches element {max_index} but dst has {}",
                dst.len()
            );
        }
        let pairs = tokens * desc.heads * (desc.rot_dim / 2);
        let pipeline = self.elementwise()?.pipeline(match dst.dtype() {
            DType::F32 => Op::RopeScatter,
            DType::F16 => Op::RopeScatterToF16,
        });
        self.plan_elementwise(
            pipeline,
            &RopeScatterPc {
                tokens,
                heads: desc.heads,
                head_dim: desc.head_dim,
                rot_dim: desc.rot_dim,
                pos_base: desc.pos_base,
                dst_offset: scatter.dst_offset,
                dst_strides: scatter.dst_strides,
                pos_scale: scatter.pos_scale,
                in_ptr: input.device_address(),
                dst_ptr: dst.device_address(),
                table_ptr: table.device_address(),
                pos_ptr: desc.pos_addr,
            },
            pairs.div_ceil(WG),
            1,
        )
    }

    /// Binary elementwise combine (see [`BinaryOp`]).  Shapes must
    /// have identical element counts; `out` may alias either input.
    pub fn run_binary(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: &Tensor,
        op: BinaryOp,
    ) -> Result<RunStats> {
        let dispatch = self.plan_binary(a, b, out, op)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_binary(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: &Tensor,
        op: BinaryOp,
    ) -> Result<ElementwiseDispatch> {
        // All three operands share a storage type; f16 combines run
        // through the IO_F16 variant (f32 math, RNE store).
        let io_f16 = self.io_dtype_f16("run_binary", a, out)?;
        self.validate_tensor_context(b, "b")?;
        if b.dtype() != a.dtype() {
            bail!(
                "run_binary: b ({}) must match a ({}) storage",
                b.dtype().name(),
                a.dtype().name()
            );
        }
        if a.len() != b.len() || a.len() != out.len() {
            bail!(
                "run_binary: element counts differ (a {}, b {}, out {})",
                a.len(),
                b.len(),
                out.len()
            );
        }
        let n = u32::try_from(a.len()).map_err(|_| anyhow::anyhow!("run_binary: too large"))?;
        let (mode, beta) = match op {
            BinaryOp::AddScaled { beta } => (0, beta),
            BinaryOp::SiluMul => (1, 0.0),
        };
        let kernel = if io_f16 { Op::BinaryF16 } else { Op::Binary };
        let pipeline = self.elementwise()?.pipeline(kernel);
        self.plan_elementwise(
            pipeline,
            &BinaryPc {
                n,
                mode,
                beta,
                _pad: 0,
                a_ptr: a.device_address(),
                b_ptr: b.device_address(),
                out_ptr: out.device_address(),
            },
            n.div_ceil(4 * WG),
            1,
        )
    }

    /// Strided 3D copy (see [`CopyDesc`]): covers transpose/permute,
    /// KV-cache append, head reshaping, and sub-matrix extraction.
    /// `src` and `dst` must be different tensors — invocations are
    /// unordered, so overlapping in-place copies would race.
    pub fn run_copy_strided(&self, src: &Tensor, dst: &Tensor, desc: CopyDesc) -> Result<RunStats> {
        let dispatch = self.plan_copy_strided(src, dst, desc)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_copy_strided(
        &self,
        src: &Tensor,
        dst: &Tensor,
        desc: CopyDesc,
    ) -> Result<ElementwiseDispatch> {
        self.validate_tensor_context(src, "src")?;
        self.validate_tensor_context(dst, "dst")?;
        // Four storage pairings, one variant each: f16 destinations
        // narrow (RNE), f16 sources widen (exactly); strides and
        // offsets are in elements either way.
        if src.raw_buffer() == dst.raw_buffer() {
            bail!("run_copy_strided: src and dst must be different tensors");
        }
        let extent_product = desc.extent.iter().map(|&e| e as u64).product::<u64>();
        if desc.extent.contains(&0) {
            bail!("run_copy_strided: extent {:?} has a zero axis", desc.extent);
        }
        let max_index = |offset: u32, strides: [u32; 3]| -> u64 {
            offset as u64
                + desc
                    .extent
                    .iter()
                    .zip(strides)
                    .map(|(&extent, stride)| (extent as u64 - 1) * stride as u64)
                    .sum::<u64>()
        };
        if max_index(desc.src_offset, desc.src_strides) >= src.len() {
            bail!(
                "run_copy_strided: source access reaches element {} but src has {}",
                max_index(desc.src_offset, desc.src_strides),
                src.len()
            );
        }
        if max_index(desc.dst_offset, desc.dst_strides) >= dst.len() {
            bail!(
                "run_copy_strided: destination access reaches element {} but dst has {}",
                max_index(desc.dst_offset, desc.dst_strides),
                dst.len()
            );
        }
        let total = u32::try_from(extent_product)
            .map_err(|_| anyhow::anyhow!("run_copy_strided: extent product exceeds u32"))?;
        let pipeline = self
            .elementwise()?
            .pipeline(match (src.dtype(), dst.dtype()) {
                (DType::F32, DType::F32) => Op::Copy,
                (DType::F32, DType::F16) => Op::CopyToF16,
                (DType::F16, DType::F16) => Op::CopyF16,
                (DType::F16, DType::F32) => Op::CopyF16ToF32,
            });
        self.plan_elementwise(
            pipeline,
            &CopyPc {
                extent: desc.extent,
                src_offset: desc.src_offset,
                src_strides: desc.src_strides,
                dst_offset: desc.dst_offset,
                dst_strides: desc.dst_strides,
                pos_scale: desc.pos_scale,
                src_ptr: src.device_address(),
                dst_ptr: dst.device_address(),
                pos_ptr: desc.pos_addr,
            },
            total.div_ceil(WG),
            1,
        )
    }

    /// Greedy-sampling argmax: writes the index of the largest element
    /// of `input` (any shape, treated flat) into `result` as one u32.
    /// Ties resolve to the LARGEST index — exactly Rust's
    /// `Iterator::max_by(f32::total_cmp)`, which keeps the last
    /// maximum, so a GPU-argmaxed decode loop reproduces a CPU greedy
    /// sampler token-for-token (finite inputs assumed; see the shader
    /// header for the NaN / signed-zero caveats).  The result cell is
    /// host-readable: [`HostU32Buffer::read`] after the submission
    /// completes returns the token without touching the logits.
    ///
    /// [`HostU32Buffer::read`]: super::HostU32Buffer::read
    pub fn run_argmax(&self, input: &Tensor, result: &HostU32Buffer) -> Result<RunStats> {
        let dispatch = self.plan_argmax(input, result)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_argmax(
        &self,
        input: &Tensor,
        result: &HostU32Buffer,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(input, "run_argmax", "input")?;
        if !result.buffer().belongs_to(&self.ctx) {
            bail!("run_argmax: result buffer belongs to a different VulkanContext");
        }
        let n = u32::try_from(input.len())
            .map_err(|_| anyhow::anyhow!("run_argmax: input length exceeds u32"))?;
        if n == 0 {
            bail!("run_argmax: input is empty");
        }
        let pipeline = self.elementwise()?.pipeline(Op::Argmax);
        // ONE workgroup: 256 threads stride the n elements, then a
        // shared tree reduce — at n = 32000 the whole op is ~4 us.
        self.plan_elementwise(
            pipeline,
            &ArgmaxPc {
                n,
                _pad: 0,
                in_ptr: input.device_address(),
                result_ptr: result.device_address(),
            },
            1,
            1,
        )
    }

    /// Embedding-row gather with a device-side token id: copies row
    /// `token` of the `[vocab, embd]` `table` (f16 or f32; f16 widens
    /// exactly) into the f32 `out` (`embd` elements).  `token` is read
    /// from the u32 cell at execution time — chain it after
    /// [`run_argmax`](Self::run_argmax) and the decode loop feeds
    /// itself on the GPU, no host trip.  Out-of-range ids clamp to
    /// `vocab - 1`.
    pub fn run_embed_gather(
        &self,
        token: &HostU32Buffer,
        table: &Tensor,
        out: &Tensor,
    ) -> Result<RunStats> {
        let dispatch =
            self.plan_embed_gather(token.device_address(), token.buffer(), table, out)?;
        self.submit_one_elementwise(dispatch)
    }

    /// Gather `tokens[0..n]` into `out` shaped `[n, embd]`.
    pub fn run_embed_gather_rows(
        &self,
        tokens: &TokenIdBuffer,
        table: &Tensor,
        out: &Tensor,
    ) -> Result<RunStats> {
        let dispatch =
            self.plan_embed_gather(tokens.device_address(), tokens.buffer(), table, out)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(in crate::executor) fn plan_embed_gather(
        &self,
        token_addr: u64,
        token_buf: &crate::buffer::Buffer,
        table: &Tensor,
        out: &Tensor,
    ) -> Result<ElementwiseDispatch> {
        self.validate_tensor_context(table, "table")?;
        self.validate_tensor_context(out, "out")?;
        if !token_buf.belongs_to(&self.ctx) {
            bail!("run_embed_gather: token buffer belongs to a different VulkanContext");
        }
        if !matches!(out.dtype(), DType::F32 | DType::F16) {
            bail!(
                "run_embed_gather: out must be f32 or f16 (got {})",
                out.dtype().name()
            );
        }
        let [vocab, embd] = *table.shape() else {
            bail!(
                "run_embed_gather: table must be [vocab, embd], got {:?}",
                table.shape()
            );
        };
        if vocab == 0 || embd == 0 {
            bail!(
                "run_embed_gather: table {:?} has a zero axis",
                table.shape()
            );
        }
        let n_tokens = match *out.shape() {
            [e] if e == embd => 1,
            [t, e] if e == embd && t > 0 => t,
            _ => bail!(
                "run_embed_gather: out shape {:?} must be [embd] or [t, embd] with embd={embd}",
                out.shape()
            ),
        };
        let need = n_tokens as u64 * 4;
        if token_buf.size_bytes() < need {
            bail!(
                "run_embed_gather: token buffer {} B < {need} B for {n_tokens} ids",
                token_buf.size_bytes()
            );
        }
        let pipeline = self.elementwise()?.pipeline(Op::EmbedGather);
        self.plan_elementwise(
            pipeline,
            &EmbedGatherPc {
                embd,
                vocab,
                table_f16: (table.dtype() == DType::F16) as u32,
                n_tokens,
                out_f16: (out.dtype() == DType::F16) as u32,
                _pad: 0,
                token_ptr: token_addr,
                table_ptr: table.device_address(),
                out_ptr: out.device_address(),
            },
            embd.div_ceil(4 * WG),
            n_tokens,
        )
    }
}
