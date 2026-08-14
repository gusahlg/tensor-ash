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

struct OpKernel {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

pub(super) struct ElementwisePipeline {
    ctx: Arc<VulkanContext>,
    layout: vk::PipelineLayout,
    softmax: OpKernel,
    rmsnorm: OpKernel,
    layernorm: OpKernel,
    rope: OpKernel,
    copy: OpKernel,
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
        let built: Vec<OpKernel> = Vec::with_capacity(5);
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
        ] {
            let (module, pipeline) =
                crate::pipeline::build_compute_pipeline(ctx, layout, spec, spv, label)?;
            built_guard.push(OpKernel { module, pipeline });
        }
        let built = scopeguard::ScopeGuard::into_inner(built_guard);
        let mut it = built.into_iter();
        let (softmax, rmsnorm, layernorm, rope, copy) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        Ok(Self {
            ctx: Arc::clone(ctx),
            layout: scopeguard::ScopeGuard::into_inner(layout_guard),
            softmax,
            rmsnorm,
            layernorm,
            rope,
            copy,
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
    fn run_elementwise<T: bytemuck::Pod>(
        &self,
        pipeline: vk::Pipeline,
        pc: &T,
        groups_x: u32,
    ) -> Result<RunStats> {
        let max = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count[0];
        if groups_x > max {
            bail!("elementwise dispatch ({groups_x} groups) exceeds device limit {max}");
        }
        let layout = self.elementwise()?.layout;
        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (elementwise)",
                |dev, cb, _slot| {
                    dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.cmd_push_constants(
                        cb,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(pc),
                    );
                    dev.cmd_dispatch(cb, groups_x, 1, 1);
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
        self.run_elementwise(
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
        self.run_norm(pipeline, input, weight, None, output, eps)
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
        self.run_norm(pipeline, input, weight, Some(bias), output, eps)
    }

    fn norm_common(
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

    fn run_norm(
        &self,
        pipeline: vk::Pipeline,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        output: &Tensor,
        eps: f32,
    ) -> Result<RunStats> {
        let (rows, cols) = rows_cols(input, "norm")?;
        self.run_elementwise(
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
        self.run_elementwise(
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
        )
    }

    /// Strided 3D copy (see [`CopyDesc`]): covers transpose/permute,
    /// KV-cache append, head reshaping, and sub-matrix extraction.
    /// `src` and `dst` must be different tensors — invocations are
    /// unordered, so overlapping in-place copies would race.
    pub fn run_copy_strided(&self, src: &Tensor, dst: &Tensor, desc: CopyDesc) -> Result<RunStats> {
        self.ensure_f32(src, "run_copy_strided", "src")?;
        self.ensure_f32(dst, "run_copy_strided", "dst")?;
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
        let pipeline = self.elementwise()?.copy.pipeline;
        self.run_elementwise(
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
