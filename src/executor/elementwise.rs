//! Non-GEMM model ops: row softmax, RMS/LayerNorm, RoPE, strided copy.
//!
//! The minimal kernel family that closes the gap between epilogue-fused
//! batched GEMM and a full transformer decoder block.  All four are
//! memory-bound kernels (a couple of passes over one operand), so naive
//! BDA implementations already run near bandwidth; there is
//! deliberately no tuner surface here.
//!
//! Structured like `SplitK2Pipeline`: a lazily-built sibling pipeline
//! in a `OnceLock` on the executor, push-constant-only layouts, and
//! dispatch through the shared slot + `submit_timed` machinery.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::context::VulkanContext;
use crate::dtype::DType;
use crate::matmul::RunStats;
use crate::tensor::Tensor;

use super::Executor;
use super::splitk2::create_pc_only_layout;

const SPIRV_SOFTMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/op_softmax_f32_row.spv"));
const SPIRV_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/op_rmsnorm_f32.spv"));
const SPIRV_ROPE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/op_rope_f32.spv"));
const SPIRV_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/op_copy_strided_f32.spv"));
const SPIRV_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/op_binary_f32.spv"));
const SPIRV_COPY_TO_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/op_copy_strided_f32_to_f16.spv"));
const SPIRV_FLASH_KV16_DH64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/op_flash_attention_kv16_dh64.spv"));
const SPIRV_FLASH_KV16_DH128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/op_flash_attention_kv16_dh128.spv"));
const SPIRV_FLASH_DH64: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/op_flash_attention_dh64.spv"));
const SPIRV_FLASH_DH128: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/op_flash_attention_dh128.spv"));

/// Threads per workgroup in every op shader.
const WG: u32 = 256;

/// Row-softmax masking mode.  Masked columns store exactly `0.0`, so
/// attention over a zero-padded KV cache stays correct end to end: the
/// padded tail gets zero probability and a downstream P@V matmul over
/// the padded extent is exact.
#[derive(Copy, Clone, Debug)]
pub enum SoftmaxMask {
    /// Every column participates.
    Full,
    /// Columns `>= valid` are masked (decode: `valid` = KV length).
    Prefix { valid: u32 },
    /// Row `i` of each `rows_per_group` block sees
    /// `prefix + (i % rows_per_group) + 1` columns — causal prefill
    /// over rows flattened as `[heads, rows_per_group]`, with `prefix`
    /// already-cached positions.
    Causal { prefix: u32, rows_per_group: u32 },
}

/// Rotary-embedding geometry for [`Executor::run_rope`].
#[derive(Copy, Clone, Debug)]
pub struct RopeDesc {
    pub heads: u32,
    pub head_dim: u32,
    /// Rotated lanes per head vector (even, `<= head_dim`); lanes past
    /// it pass through (partial rotary).
    pub rot_dim: u32,
    /// Absolute position of the first token in the input.
    pub pos_base: u32,
}

/// Fused causal-attention geometry for
/// [`Executor::run_flash_attention`].  Query row `i` attends to
/// positions `< min(kv_len, pos_base + i + 1)` — the same semantics as
/// [`SoftmaxMask::Causal`] in the composed path.
#[derive(Copy, Clone, Debug)]
pub struct FlashAttentionDesc {
    /// Valid K/V positions in the caches (for a from-scratch prefill
    /// of T tokens this is T; with a warm cache, `pos_base + t_q`).
    pub kv_len: u32,
    /// Absolute position of query row 0.
    pub pos_base: u32,
    /// `1/sqrt(dh)` for standard attention.
    pub scale: f32,
}

/// Standalone binary elementwise operator for
/// [`Executor::run_binary`]: lets a large matmul keep its tensor-core
/// route (which cannot fuse epilogues) and apply the combination as
/// one cheap bandwidth pass.  In-place safe for either operand.
#[derive(Copy, Clone, Debug)]
pub enum BinaryOp {
    /// `out = a + beta * b` (residual add).
    AddScaled { beta: f32 },
    /// `out = silu(a) * b` (SwiGLU gating).
    SiluMul,
}

/// Strided-copy geometry for [`Executor::run_copy_strided`]: for every
/// `(x, y, z)` in `extent`, element `src_offset + x*src_strides[0] +
/// y*src_strides[1] + z*src_strides[2]` is copied to the equivalent
/// destination index.  Strides and offsets are in elements.
#[derive(Copy, Clone, Debug)]
pub struct CopyDesc {
    pub extent: [u32; 3],
    pub src_offset: u32,
    pub src_strides: [u32; 3],
    pub dst_offset: u32,
    pub dst_strides: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SoftmaxPc {
    rows: u32,
    cols: u32,
    valid_base: u32,
    rows_per_group: u32,
    causal: u32,
    scale: f32,
    in_ptr: u64,
    out_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct NormPc {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
    in_ptr: u64,
    out_ptr: u64,
    weight_ptr: u64,
    bias_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RopePc {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    rot_dim: u32,
    pos_base: u32,
    _pad: u32,
    in_ptr: u64,
    out_ptr: u64,
    table_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct CopyPc {
    extent: [u32; 3],
    src_offset: u32,
    src_strides: [u32; 3],
    dst_offset: u32,
    dst_strides: [u32; 3],
    _pad: u32,
    src_ptr: u64,
    dst_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BinaryPc {
    n: u32,
    mode: u32,
    beta: f32,
    _pad: u32,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct FlashPc {
    t_q: u32,
    t_max: u32,
    kv_len: u32,
    pos_base: u32,
    group_size: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    q_ptr: u64,
    kt_ptr: u64,
    v_ptr: u64,
    out_ptr: u64,
}

/// A fully planned elementwise dispatch: pipeline, push constants,
/// and grid — ready to record into any command buffer.
pub(super) struct ElementwiseDispatch {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push: [u8; 64],
    push_len: usize,
    groups: (u32, u32),
}

pub(super) struct OpKernel {
    module: vk::ShaderModule,
    pub(super) pipeline: vk::Pipeline,
}

pub(super) struct ElementwisePipeline {
    ctx: Arc<VulkanContext>,
    layout: vk::PipelineLayout,
    softmax: OpKernel,
    pub(super) rmsnorm: OpKernel,
    pub(super) layernorm: OpKernel,
    rope: OpKernel,
    copy: OpKernel,
    pub(super) binary: OpKernel,
    copy_to_f16: OpKernel,
    flash_dh64: OpKernel,
    flash_dh128: OpKernel,
    flash_kv16_dh64: OpKernel,
    flash_kv16_dh128: OpKernel,
}

impl ElementwisePipeline {
    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        // One PC-only layout sized for the largest block (CopyPc);
        // a range larger than a shader's declared block is valid.
        let layout = unsafe { create_pc_only_layout(ctx, std::mem::size_of::<CopyPc>() as u32) }
            .context("elementwise pipeline layout")?;
        let layout_guard = scopeguard::guard(layout, |l| unsafe {
            ctx.device.destroy_pipeline_layout(l, None);
        });
        let built: Vec<OpKernel> = Vec::with_capacity(11);
        let mut built_guard = scopeguard::guard(built, |kernels| {
            for kernel in kernels {
                unsafe {
                    ctx.device.destroy_pipeline(kernel.pipeline, None);
                    ctx.device.destroy_shader_module(kernel.module, None);
                }
            }
        });
        for (spv, spec, label) in [
            (SPIRV_SOFTMAX, &[][..], "op softmax"),
            (SPIRV_NORM, &[0u32][..], "op rmsnorm"),
            (SPIRV_NORM, &[1u32][..], "op layernorm"),
            (SPIRV_ROPE, &[][..], "op rope"),
            (SPIRV_COPY, &[][..], "op copy_strided"),
            (SPIRV_BINARY, &[][..], "op binary"),
            (SPIRV_COPY_TO_F16, &[][..], "op copy_strided f32->f16"),
            (SPIRV_FLASH_DH64, &[][..], "op flash_attention dh64"),
            (SPIRV_FLASH_DH128, &[][..], "op flash_attention dh128"),
            (SPIRV_FLASH_KV16_DH64, &[][..], "op flash_attention kv16 dh64"),
            (SPIRV_FLASH_KV16_DH128, &[][..], "op flash_attention kv16 dh128"),
        ] {
            let (module, pipeline) =
                crate::pipeline::build_compute_pipeline(ctx, layout, spec, spv, label)?;
            built_guard.push(OpKernel { module, pipeline });
        }
        let built = scopeguard::ScopeGuard::into_inner(built_guard);
        let mut it = built.into_iter();
        #[rustfmt::skip]
        let (softmax, rmsnorm, layernorm, rope, copy, binary, copy_to_f16,
            flash_dh64, flash_dh128, flash_kv16_dh64, flash_kv16_dh128) = (
            it.next().unwrap(), it.next().unwrap(), it.next().unwrap(),
            it.next().unwrap(), it.next().unwrap(), it.next().unwrap(),
            it.next().unwrap(), it.next().unwrap(), it.next().unwrap(),
            it.next().unwrap(), it.next().unwrap(),
        );
        Ok(Self {
            ctx: Arc::clone(ctx),
            layout: scopeguard::ScopeGuard::into_inner(layout_guard),
            softmax,
            rmsnorm,
            layernorm,
            rope,
            copy,
            binary,
            copy_to_f16,
            flash_dh64,
            flash_dh128,
            flash_kv16_dh64,
            flash_kv16_dh128,
        })
    }
}

impl Drop for ElementwisePipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [
                &self.softmax,
                &self.rmsnorm,
                &self.layernorm,
                &self.rope,
                &self.copy,
                &self.binary,
                &self.copy_to_f16,
                &self.flash_dh64,
                &self.flash_dh128,
                &self.flash_kv16_dh64,
                &self.flash_kv16_dh128,
            ] {
                self.ctx.device.destroy_pipeline(kernel.pipeline, None);
                self.ctx.device.destroy_shader_module(kernel.module, None);
            }
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// `(rows, cols)` of a tensor treated as a stack of rows over its last
/// dimension.
fn rows_cols(tensor: &Tensor, label: &str) -> Result<(u32, u32)> {
    let shape = tensor.shape();
    let cols = *shape.last().expect("tensor rank >= 1");
    let rows = tensor.len() / cols as u64;
    u32::try_from(rows)
        .map(|rows| (rows, cols))
        .map_err(|_| anyhow::anyhow!("{label}: row count {rows} exceeds u32"))
}

impl Executor {
    fn elementwise(&self) -> Result<&ElementwisePipeline> {
        if !self.ctx.buffer_device_address_enabled {
            bail!("elementwise ops require bufferDeviceAddress, which this device lacks");
        }
        if let Some(pipeline) = self.elementwise.get() {
            return Ok(pipeline);
        }
        let built = ElementwisePipeline::new(&self.ctx)?;
        Ok(self.elementwise.get_or_init(|| built))
    }

    fn ensure_f32(&self, tensor: &Tensor, op: &str, operand: &str) -> Result<()> {
        self.validate_tensor_context(tensor, operand)?;
        if tensor.dtype() != DType::F32 {
            bail!("{op}: {operand} must be f32 storage");
        }
        Ok(())
    }

    /// Dispatch one elementwise op: bind, push, dispatch, spin-wait.
    fn plan_elementwise<T: bytemuck::Pod>(
        &self,
        pipeline: vk::Pipeline,
        pc: &T,
        groups_x: u32,
        groups_y: u32,
    ) -> Result<ElementwiseDispatch> {
        let max = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count;
        if groups_x > max[0] || groups_y > max[1] {
            bail!(
                "elementwise dispatch ({groups_x}, {groups_y}) exceeds device limits ({}, {})",
                max[0],
                max[1]
            );
        }
        let bytes = bytemuck::bytes_of(pc);
        let mut push = [0u8; 64];
        push[..bytes.len()].copy_from_slice(bytes);
        Ok(ElementwiseDispatch {
            pipeline,
            layout: self.elementwise()?.layout,
            push,
            push_len: bytes.len(),
            groups: (groups_x, groups_y),
        })
    }

    /// Record one planned elementwise dispatch into `cb` (no barriers).
    pub(super) fn record_elementwise(&self, cb: vk::CommandBuffer, dispatch: &ElementwiseDispatch) {
        let dev = &self.ctx.device;
        unsafe {
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, dispatch.pipeline);
            dev.cmd_push_constants(
                cb,
                dispatch.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &dispatch.push[..dispatch.push_len],
            );
            dev.cmd_dispatch(cb, dispatch.groups.0, dispatch.groups.1, 1);
        }
    }

    fn submit_one_elementwise(&self, dispatch: ElementwiseDispatch) -> Result<RunStats> {
        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (elementwise)",
                |_dev, cb, _slot| {
                    self.record_elementwise(cb, &dispatch);
                    Ok(())
                },
            )
        }?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: 0,
        })
    }

    fn run_elementwise_2d<T: bytemuck::Pod>(
        &self,
        pipeline: vk::Pipeline,
        pc: &T,
        groups_x: u32,
        groups_y: u32,
    ) -> Result<RunStats> {
        let dispatch = self.plan_elementwise(pipeline, pc, groups_x, groups_y)?;
        self.submit_one_elementwise(dispatch)
    }

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

    pub(super) fn plan_softmax_rows(
        &self,
        input: &Tensor,
        output: &Tensor,
        scale: f32,
        mask: SoftmaxMask,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(input, "run_softmax_rows", "input")?;
        self.ensure_f32(output, "run_softmax_rows", "output")?;
        if input.shape() != output.shape() {
            bail!(
                "run_softmax_rows: output shape {:?} must equal input shape {:?}",
                output.shape(),
                input.shape()
            );
        }
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
        let pipeline = self.elementwise()?.softmax.pipeline;
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
        let pipeline = self
            .norm_common("run_rms_norm", input, weight, None, output)?
            .rmsnorm
            .pipeline;
        let dispatch = self.plan_norm(pipeline, input, weight, None, output, eps)?;
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
        let pipeline = self
            .norm_common("run_layer_norm", input, weight, Some(bias), output)?
            .layernorm
            .pipeline;
        let dispatch = self.plan_norm(pipeline, input, weight, Some(bias), output, eps)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(super) fn norm_common(
        &self,
        op: &str,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &Tensor,
    ) -> Result<&ElementwisePipeline> {
        self.ensure_f32(input, op, "input")?;
        self.ensure_f32(weight, op, "weight")?;
        self.ensure_f32(output, op, "output")?;
        if let Some(bias) = bias {
            self.ensure_f32(bias, op, "bias")?;
        }
        if input.shape() != output.shape() {
            bail!(
                "{op}: output shape {:?} must equal input shape {:?}",
                output.shape(),
                input.shape()
            );
        }
        let (_, cols) = rows_cols(input, op)?;
        if weight.len() != cols as u64 {
            bail!("{op}: weight length {} != row length {cols}", weight.len());
        }
        if let Some(bias) = bias
            && bias.len() != cols as u64
        {
            bail!("{op}: bias length {} != row length {cols}", bias.len());
        }
        self.elementwise()
    }

    pub(super) fn plan_norm(
        &self,
        pipeline: vk::Pipeline,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &Tensor,
        eps: f32,
    ) -> Result<ElementwiseDispatch> {
        let (rows, cols) = rows_cols(input, "norm")?;
        self.plan_elementwise(
            pipeline,
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

    pub(super) fn plan_rope(
        &self,
        input: &Tensor,
        table: &Tensor,
        output: &Tensor,
        desc: RopeDesc,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(input, "run_rope", "input")?;
        self.ensure_f32(table, "run_rope", "table")?;
        self.ensure_f32(output, "run_rope", "output")?;
        if input.shape() != output.shape() {
            bail!(
                "run_rope: output shape {:?} must equal input shape {:?}",
                output.shape(),
                input.shape()
            );
        }
        if desc.rot_dim < 2 || !desc.rot_dim.is_multiple_of(2) || desc.rot_dim > desc.head_dim {
            bail!(
                "run_rope: rot_dim {} must be even, >= 2, and <= head_dim {}",
                desc.rot_dim,
                desc.head_dim
            );
        }
        let vec_elems = desc.heads as u64 * desc.head_dim as u64;
        if vec_elems == 0 || !input.len().is_multiple_of(vec_elems) {
            bail!(
                "run_rope: input length {} is not a multiple of heads*head_dim {}",
                input.len(),
                vec_elems
            );
        }
        let tokens = u32::try_from(input.len() / vec_elems)
            .map_err(|_| anyhow::anyhow!("run_rope: token count exceeds u32"))?;
        let needed = (desc.pos_base as u64 + tokens as u64) * desc.rot_dim as u64;
        if table.len() < needed {
            bail!(
                "run_rope: table length {} < required {} (pos_base {} + {} tokens, rot_dim {})",
                table.len(),
                needed,
                desc.pos_base,
                tokens,
                desc.rot_dim
            );
        }
        let pairs = tokens * desc.heads * (desc.rot_dim / 2);
        let pipeline = self.elementwise()?.rope.pipeline;
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
            },
            pairs.div_ceil(WG),
            1,
        )
    }

    /// Fused causal prefill attention (FlashAttention pattern):
    /// `out = softmax_causal(q @ K^T * scale) @ V` in one dispatch with
    /// online softmax — score tiles never touch global memory, and
    /// tiles above the causal frontier are skipped.  Layouts match the
    /// composed path: `q`/`out` are `[H, T_q, dh]`, `kt` is
    /// `[H_kv, dh, T_max]`, `v` is `[H_kv, T_max, dh]`; GQA works when
    /// `H` is a multiple of `H_kv`.  `dh` must be 64 or 128 (the
    /// compiled head-dimension variants).
    pub fn run_flash_attention(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
    ) -> Result<RunStats> {
        self.ensure_f32(q, "run_flash_attention", "q")?;
        self.validate_tensor_context(kt, "kt")?;
        self.validate_tensor_context(v, "v")?;
        self.ensure_f32(out, "run_flash_attention", "out")?;
        if kt.dtype() != v.dtype() {
            bail!(
                "run_flash_attention: kt ({}) and v ({}) must share a storage type",
                kt.dtype().name(),
                v.dtype().name()
            );
        }
        let kv_f16 = kt.dtype() == DType::F16;
        let [heads, t_q, dh] = *q.shape() else {
            bail!(
                "run_flash_attention: q must be [H, T_q, dh], got {:?}",
                q.shape()
            );
        };
        if out.shape() != q.shape() {
            bail!(
                "run_flash_attention: out shape {:?} must equal q shape {:?}",
                out.shape(),
                q.shape()
            );
        }
        let [kv_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_flash_attention: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_flash_attention: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        if kt_dh != dh || v_dh != dh || v_t != t_max || v_heads != kv_heads {
            bail!(
                "run_flash_attention: inconsistent shapes q {:?}, kt {:?}, v {:?}",
                q.shape(),
                kt.shape(),
                v.shape()
            );
        }
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            bail!("run_flash_attention: H={heads} must be a multiple of H_kv={kv_heads}");
        }
        if desc.kv_len > t_max {
            bail!(
                "run_flash_attention: kv_len {} exceeds cache T_max {t_max}",
                desc.kv_len
            );
        }
        let pipes = self.elementwise()?;
        let (pipeline, rows_per_wg) = match (dh, kv_f16) {
            (64, false) => (pipes.flash_dh64.pipeline, 128),
            (128, false) => (pipes.flash_dh128.pipeline, 128),
            (64, true) => (pipes.flash_kv16_dh64.pipeline, 128),
            (128, true) => (pipes.flash_kv16_dh128.pipeline, 128),
            (other, _) => bail!(
                "run_flash_attention: head dimension {other} unsupported (compiled variants: 64, 128)"
            ),
        };
        self.run_elementwise_2d(
            pipeline,
            &FlashPc {
                t_q,
                t_max,
                kv_len: desc.kv_len,
                pos_base: desc.pos_base,
                group_size: heads / kv_heads,
                scale: desc.scale,
                _pad0: 0,
                _pad1: 0,
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                out_ptr: out.device_address(),
            },
            t_q.div_ceil(rows_per_wg),
            heads,
        )
    }

    /// Strided 3D copy (see [`CopyDesc`]): covers transpose/permute,
    /// KV-cache append, head reshaping, and sub-matrix extraction.
    /// `src` and `dst` must be different tensors — invocations are
    /// unordered, so overlapping in-place copies would race.
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

    pub(super) fn plan_binary(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: &Tensor,
        op: BinaryOp,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(a, "run_binary", "a")?;
        self.ensure_f32(b, "run_binary", "b")?;
        self.ensure_f32(out, "run_binary", "out")?;
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
        let pipeline = self.elementwise()?.binary.pipeline;
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
            n.div_ceil(WG),
            1,
        )
    }

    pub fn run_copy_strided(&self, src: &Tensor, dst: &Tensor, desc: CopyDesc) -> Result<RunStats> {
        let dispatch = self.plan_copy_strided(src, dst, desc)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(super) fn plan_copy_strided(
        &self,
        src: &Tensor,
        dst: &Tensor,
        desc: CopyDesc,
    ) -> Result<ElementwiseDispatch> {
        self.ensure_f32(src, "run_copy_strided", "src")?;
        self.validate_tensor_context(dst, "dst")?;
        // f16 destinations narrow through a dedicated variant; strides
        // and offsets are in elements either way.
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
        let pipeline = match dst.dtype() {
            DType::F32 => self.elementwise()?.copy.pipeline,
            DType::F16 => self.elementwise()?.copy_to_f16.pipeline,
        };
        self.plan_elementwise(
            pipeline,
            &CopyPc {
                extent: desc.extent,
                src_offset: desc.src_offset,
                src_strides: desc.src_strides,
                dst_offset: desc.dst_offset,
                dst_strides: desc.dst_strides,
                _pad: 0,
                src_ptr: src.device_address(),
                dst_ptr: dst.device_address(),
            },
            total.div_ceil(WG),
            1,
        )
    }
}

// SAFETY: plain-old-data push-constant mirrors of the GLSL blocks.
unsafe impl bytemuck::Pod for SoftmaxPc {}
unsafe impl bytemuck::Zeroable for SoftmaxPc {}
unsafe impl bytemuck::Pod for NormPc {}
unsafe impl bytemuck::Zeroable for NormPc {}
unsafe impl bytemuck::Pod for RopePc {}
unsafe impl bytemuck::Zeroable for RopePc {}
unsafe impl bytemuck::Pod for CopyPc {}
unsafe impl bytemuck::Zeroable for CopyPc {}
unsafe impl bytemuck::Pod for FlashPc {}
unsafe impl bytemuck::Zeroable for FlashPc {}
unsafe impl bytemuck::Pod for BinaryPc {}
unsafe impl bytemuck::Zeroable for BinaryPc {}
