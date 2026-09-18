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

use crate::buffer::Buffer;
use crate::context::VulkanContext;
use crate::dtype::DType;
use crate::matmul::RunStats;
use crate::tensor::Tensor;

use super::Executor;

mod pc;
use pc::ELEMENTWISE_PC_BYTES;

/// Declares one elementwise kernel set ONCE: an index enum and the
/// same-order `(spirv, spec constants, debug label)` build table.
/// Adding a kernel is one line here plus its dispatch site.
macro_rules! op_kernels {
    ($op:ident / $table:ident: $($variant:ident => ($file:literal, $spec:expr, $label:literal)),+ $(,)?) => {
        /// Index of one compiled elementwise kernel (see the paired
        /// build table).
        #[derive(Copy, Clone)]
        pub(super) enum $op { $($variant),+ }

        impl $op {
            const COUNT: usize = [$($op::$variant),+].len();
        }

        /// Build inputs per index variant, in declaration order.
        const $table: [(&[u8], &[u32], &str); $op::COUNT] = [
            $((
                include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".spv")),
                $spec,
                $label,
            )),+
        ];
    };
}

op_kernels! { Op / KERNELS:
    Softmax => ("op_softmax_f32_row", &[], "op softmax"),
    RmsNorm => ("op_rmsnorm_f32", &[0], "op rmsnorm"),
    LayerNorm => ("op_rmsnorm_f32", &[1], "op layernorm"),
    RmsNormF16 => ("op_rmsnorm_f16", &[0], "op rmsnorm f16"),
    Rope => ("op_rope_f32", &[], "op rope"),
    RopeF16 => ("op_rope_f16", &[], "op rope f16"),
    RopeScatter => ("op_rope_scatter_f32", &[], "op rope_scatter"),
    RopeScatterToF16 => ("op_rope_scatter_f32_to_f16", &[], "op rope_scatter f32->f16"),
    Copy => ("op_copy_strided_f32", &[], "op copy_strided"),
    Binary => ("op_binary_f32", &[], "op binary"),
    BinaryF16 => ("op_binary_f16", &[], "op binary f16"),
    CopyToF16 => ("op_copy_strided_f32_to_f16", &[], "op copy_strided f32->f16"),
    CopyF16 => ("op_copy_strided_f16", &[], "op copy_strided f16->f16"),
    CopyF16ToF32 => ("op_copy_strided_f16_to_f32", &[], "op copy_strided f16->f32"),
    FlashDh64 => ("op_flash_attention_dh64", &[], "op flash_attention dh64"),
    FlashDh128 => ("op_flash_attention_dh128", &[], "op flash_attention dh128"),
    FlashKv16Dh64 => ("op_flash_attention_kv16_dh64", &[], "op flash_attention kv16 dh64"),
    FlashKv16Dh128 => ("op_flash_attention_kv16_dh128", &[], "op flash_attention kv16 dh128"),
    AttnDecodeDh64 => ("op_attn_decode_dh64", &[], "op attn_decode dh64"),
    AttnDecodeKv16Dh64 => ("op_attn_decode_kv16_dh64", &[], "op attn_decode kv16 dh64"),
    AttnDecodeCombine => ("op_attn_decode_combine", &[], "op attn_decode combine"),
    Argmax => ("op_argmax_f32", &[], "op argmax"),
    EmbedGather => ("op_embed_gather", &[], "op embed_gather"),
    PrefillQkvPack => ("op_prefill_qkv_pack", &[], "op prefill_qkv_pack"),
}

// Tensor-core flash-attention kernels (`VK_NV_cooperative_matrix2`),
// built only when [`VulkanContext::coopmat2_enabled`] — their SPIR-V
// 1.6 modules only validate where the extension is live.  Same push
// constants and semantics as the SIMT flash variants; Br=64 query rows
// per workgroup instead of 128.
op_kernels! { Cm2Op / CM2_KERNELS:
    FlashDh64 => ("attention_flash_cm2_dh64", &[], "op flash_attention cm2 dh64"),
    FlashDh128 => ("attention_flash_cm2_dh128", &[], "op flash_attention cm2 dh128"),
    FlashKv16Dh64 => ("attention_flash_cm2_kv16_dh64", &[], "op flash_attention cm2 kv16 dh64"),
    FlashKv16Dh128 => ("attention_flash_cm2_kv16_dh128", &[], "op flash_attention cm2 kv16 dh128"),
    FlashKv16Io16Dh64 => ("attention_flash_cm2_kv16_io16_dh64", &[],
        "op flash_attention cm2 kv16 io16 dh64"),
}

/// Threads per workgroup in every op shader.
pub(super) const WG: u32 = 256;

/// Maximum jobs in one [`super::ExecOp::GemvChain`] dispatch.
pub const GEMV_CHAIN_MAX_JOBS: usize = 8;

/// Cap on the persistent grid.  64 × 16-warp blocks fill a 46-SM
/// Ampere (and still fit a 24-SM Ada occupancy budget); leftover
/// tiles are looped.  Launching only 32 WGs left 14 SMs idle on the
/// 3070 and serialized ~3 tiles/WG on the 88-tile gate/up GEMVs.
pub(super) const GEMV_CHAIN_MAX_WG: u32 = 64;

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
#[derive(Copy, Clone, Debug, Default)]
pub struct RopeDesc {
    pub heads: u32,
    pub head_dim: u32,
    /// Rotated lanes per head vector (even, `<= head_dim`); lanes past
    /// it pass through (partial rotary).
    pub rot_dim: u32,
    /// Absolute position of the first token in the input.
    pub pos_base: u32,
    /// Optional device address of a [`PosBuffer`](super::PosBuffer)
    /// (0 = none).  When set, the shader reads position `p` from it at
    /// execution time and the effective base becomes `pos_base + p` —
    /// the indirection that lets a recorded command buffer replay at a
    /// new position each token.  The buffer must stay alive for every
    /// execution; table-coverage validation only sees `pos_base`, so
    /// the caller keeps `pos_base + p + tokens` within the table.
    pub pos_addr: u64,
}

/// Destination geometry for [`Executor::run_rope_scatter`]: the fused
/// RoPE + strided-scatter op writes each rotated (or pass-through)
/// element `(token, head, dim)` to `dst_offset + token*dst_strides[0]
/// + head*dst_strides[1] + dim*dst_strides[2]` (elements).  Covers the
/// decode k-rope + Kt-cache append as one dispatch: for a
/// `[H_kv, dh, T_max]` Kt cache at position `pos`, use
/// `dst_offset = pos` and `dst_strides = [1, dh * t_max, t_max]`.
#[derive(Copy, Clone, Debug, Default)]
pub struct RopeScatterDesc {
    pub dst_offset: u32,
    /// Element strides per (token, head, dim).
    pub dst_strides: [u32; 3],
    /// `dst_offset` advance per indirect position: when the paired
    /// [`RopeDesc::pos_addr`] is set, the effective destination offset
    /// is `dst_offset + p * pos_scale` (Kt-column append: 1).
    pub pos_scale: u32,
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
    /// `Some(H)` means `q` is token-major: rank-2 `[T, H*dh]` or
    /// rank-3 `[T, H, dh]`.  `None` is the default head-major `[H, T, dh]`.
    /// `out` follows `q` unless [`out_token_major_heads`](Self::out_token_major_heads)
    /// is set.
    pub token_major_heads: Option<u32>,
    /// Independent `out` layout.  `Some(H)` writes token-major
    /// `[T, H*dh]` even when `q` is head-major — the wide-prefill
    /// path (contiguous Q loads, Wo-ready O, no permutes).
    pub out_token_major_heads: Option<u32>,
}

/// Geometry for [`Executor::run_prefill_qkv_pack`]: RoPE Q into
/// head-major `[H, T, dh]` (contiguous query rows for flash),
/// RoPE-scatter K into `[H_kv, dh, T_max]`, and copy V into
/// `[H_kv, T_max, dh]`, all from one concatenated
/// `[T, H*dh + 2*H_kv*dh]` QKV row.
#[derive(Copy, Clone, Debug)]
pub struct PrefillQkvPackDesc {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Even, `<= head_dim`; lanes past it pass through (partial rotary).
    pub rot_dim: u32,
    /// Absolute position of token 0.
    pub pos_base: u32,
}

/// Fused split-K decode-attention geometry for
/// [`Executor::run_attn_decode`]: ONE query row per head attends to
/// the `kv_len` valid cache positions (prefix mask, scale applied
/// before the max — identical semantics to the composed
/// scores/softmax/PV trio, but only the valid prefix is ever read).
#[derive(Copy, Clone, Debug, Default)]
pub struct AttnDecodeDesc {
    /// Valid K/V positions in the caches (`>= 1`).
    pub kv_len: u32,
    /// `1/sqrt(dh)` for standard attention.
    pub scale: f32,
    /// Optional device address of a [`PosBuffer`](super::PosBuffer)
    /// (0 = none).  When set, the effective KV length is
    /// `kv_len + p` (record with `kv_len = 1` for a decode step at
    /// position `p`), and BOTH stages dispatch the fixed
    /// [`ATTN_DECODE_MAX_CHUNKS`] grid — chunks past the effective
    /// length write neutral partials that merge exactly, so one
    /// recorded grid covers every position.
    pub pos_addr: u64,
}

/// Upper bound on split-K chunks in decode attention; sizes the
/// caller-provided scratch: `kv_heads * MAX_CHUNKS * group * (dh+2)`
/// f32 elements covers every `kv_len`.
pub const ATTN_DECODE_MAX_CHUNKS: u32 = 32;

/// Sequence chunks for one decode-attention dispatch: enough
/// workgroups to fill the device (grid = num_chunks * kv_heads)
/// without shrinking chunks below the merge overhead.  Measured on
/// GA104 at kv_heads=4 (TinyLlama, kv_len ~640): 8 chunks 715 us/22
/// layers, 16 -> 565, 20 -> 526, 32 -> 523; short contexts favor the
/// floor of 8 and deep ones the cap of 32, so target ~32 positions
/// per chunk inside [8, 32].
pub(super) fn attn_decode_num_chunks(kv_len: u32) -> u32 {
    kv_len.div_ceil(32).clamp(8, ATTN_DECODE_MAX_CHUNKS)
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
#[derive(Copy, Clone, Debug, Default)]
pub struct CopyDesc {
    pub extent: [u32; 3],
    pub src_offset: u32,
    pub src_strides: [u32; 3],
    pub dst_offset: u32,
    pub dst_strides: [u32; 3],
    /// Optional device address of a [`PosBuffer`](super::PosBuffer)
    /// (0 = none): the effective destination offset becomes
    /// `dst_offset + p * pos_scale` at execution time, so a recorded
    /// KV-append replays at each new position.  Bounds validation only
    /// sees `dst_offset`; the caller keeps the runtime offset in range.
    pub pos_addr: u64,
    /// `dst_offset` advance per indirect position (V-row append: dh).
    pub pos_scale: u32,
}

/// A fully planned elementwise dispatch: pipeline, push constants,
/// and grid — ready to record into any command buffer.
pub(super) struct ElementwiseDispatch {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push: [u8; ELEMENTWISE_PC_BYTES],
    push_len: usize,
    groups: (u32, u32),
    /// Side allocation the recorded push constants point at (GEMV-chain
    /// job table + quorum counters).  Held so prepared replay cannot
    /// drop the buffer out from under a baked device address.
    pub(super) retain: Option<Arc<Buffer>>,
}

struct OpKernel {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

pub(super) struct ElementwisePipeline {
    ctx: Arc<VulkanContext>,
    layout: vk::PipelineLayout,
    kernels: [OpKernel; Op::COUNT],
    /// `Some` iff [`VulkanContext::coopmat2_enabled`].
    cm2_kernels: Option<[OpKernel; Cm2Op::COUNT]>,
    /// Persistent GEMV-chain kernel; `Some` iff the device enabled
    /// the Vulkan memory model with device scope (required for the
    /// quorum atomics).
    gemv_chain: Option<OpKernel>,
}

impl ElementwisePipeline {
    fn pipeline(&self, op: Op) -> vk::Pipeline {
        self.kernels[op as usize].pipeline
    }

    fn cm2_pipeline(&self, op: Cm2Op) -> Option<vk::Pipeline> {
        self.cm2_kernels
            .as_ref()
            .map(|kernels| kernels[op as usize].pipeline)
    }

    pub(super) fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        // One PC-only layout sized for the largest block (see
        // ELEMENTWISE_PC_BYTES); a range larger than a shader's
        // declared block is valid.
        let layout = crate::pipeline::create_pc_only_layout(ctx, ELEMENTWISE_PC_BYTES as u32)
            .context("elementwise pipeline layout")?;
        let layout_guard = scopeguard::guard(layout, |l| unsafe {
            ctx.device.destroy_pipeline_layout(l, None);
        });
        let built: Vec<OpKernel> = Vec::with_capacity(Op::COUNT);
        let mut built_guard = scopeguard::guard(built, |kernels| {
            for kernel in kernels {
                unsafe {
                    ctx.device.destroy_pipeline(kernel.pipeline, None);
                    ctx.device.destroy_shader_module(kernel.module, None);
                }
            }
        });
        for (spv, spec, label) in KERNELS {
            let (module, pipeline) =
                crate::pipeline::build_compute_pipeline(ctx, layout, spec, spv, label)?;
            built_guard.push(OpKernel { module, pipeline });
        }
        if ctx.coopmat2_enabled {
            for (spv, spec, label) in CM2_KERNELS {
                let (module, pipeline) =
                    crate::pipeline::build_compute_pipeline(ctx, layout, spec, spv, label)?;
                built_guard.push(OpKernel { module, pipeline });
            }
        }
        let mut built = scopeguard::ScopeGuard::into_inner(built_guard).into_iter();
        let kernels = std::array::from_fn(|_| built.next().expect("KERNELS has Op::COUNT entries"));
        let cm2_kernels = ctx.coopmat2_enabled.then(|| {
            std::array::from_fn(|_| built.next().expect("CM2_KERNELS has Cm2Op::COUNT entries"))
        });
        let gemv_chain = if ctx.memory_model_device_scope_enabled {
            let (module, pipeline) = crate::pipeline::build_compute_pipeline(
                ctx,
                layout,
                &[],
                include_bytes!(concat!(env!("OUT_DIR"), "/matmul_gemv_chain.spv")),
                "op gemv_chain",
            )?;
            Some(OpKernel { module, pipeline })
        } else {
            None
        };
        Ok(Self {
            ctx: Arc::clone(ctx),
            layout: scopeguard::ScopeGuard::into_inner(layout_guard),
            kernels,
            cm2_kernels,
            gemv_chain,
        })
    }
}

impl Drop for ElementwisePipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in self
                .kernels
                .iter()
                .chain(self.cm2_kernels.iter().flatten())
                .chain(self.gemv_chain.iter())
            {
                self.ctx.device.destroy_pipeline(kernel.pipeline, None);
                self.ctx.device.destroy_shader_module(kernel.module, None);
            }
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// `(rows, cols)` of a tensor treated as a stack of rows over its last
/// dimension.
pub(super) fn rows_cols(tensor: &Tensor, label: &str) -> Result<(u32, u32)> {
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

    /// Shared f16-capable IO check: `input` and `output` (validated
    /// against this context) must share a storage type; returns `true`
    /// when both are f16.  For the ops with an IO_F16 kernel sibling —
    /// mixed pairs must go through the strided-copy converters.
    fn io_dtype_f16(&self, op: &str, input: &Tensor, output: &Tensor) -> Result<bool> {
        self.validate_tensor_context(input, "input")?;
        self.validate_tensor_context(output, "output")?;
        if input.dtype() != output.dtype() {
            bail!(
                "{op}: input ({}) and output ({}) must share a storage type",
                input.dtype().name(),
                output.dtype().name()
            );
        }
        Ok(input.dtype() == DType::F16)
    }

    /// Shared in-place-capable shape check: `output` must match `input`.
    fn ensure_same_shape(op: &str, input: &Tensor, output: &Tensor) -> Result<()> {
        if input.shape() != output.shape() {
            bail!(
                "{op}: output shape {:?} must equal input shape {:?}",
                output.shape(),
                input.shape()
            );
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
        let mut push = [0u8; ELEMENTWISE_PC_BYTES];
        push[..bytes.len()].copy_from_slice(bytes);
        Ok(ElementwiseDispatch {
            pipeline,
            layout: self.elementwise()?.layout,
            push,
            push_len: bytes.len(),
            groups: (groups_x, groups_y),
            retain: None,
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
}

mod attn;
mod flash;
mod gemv_chain;
mod ops;
