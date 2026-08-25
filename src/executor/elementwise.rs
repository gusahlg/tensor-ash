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

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::dtype::DType;
use crate::matmul::{Activation, EpilogueBinary, MatmulOp, ResolvedMatmul, RunStats};
use crate::tensor::Tensor;

use super::Executor;
use super::splitk2::create_pc_only_layout;

/// Declares one elementwise kernel set ONCE: an index enum and the
/// same-order `(spirv, spec constants, debug label)` build table.
/// Adding a kernel is one line here plus its dispatch site.
macro_rules! op_kernels {
    ($op:ident / $table:ident: $($variant:ident => ($file:literal, $spec:expr, $label:literal)),+ $(,)?) => {
        /// Index of one compiled elementwise kernel (see the paired
        /// build table).
        #[derive(Copy, Clone)]
        enum $op { $($variant),+ }

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
const WG: u32 = 256;

/// Maximum jobs in one [`super::ExecOp::GemvChain`] dispatch.
pub const GEMV_CHAIN_MAX_JOBS: usize = 8;

/// Cap on the persistent grid.  64 × 16-warp blocks fill a 46-SM
/// Ampere (and still fit a 24-SM Ada occupancy budget); leftover
/// tiles are looped.  Launching only 32 WGs left 14 SMs idle on the
/// 3070 and serialized ~3 tiles/WG on the 88-tile gate/up GEMVs.
const GEMV_CHAIN_MAX_WG: u32 = 64;

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
fn attn_decode_num_chunks(kv_len: u32) -> u32 {
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
    pos_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct RopeScatterPc {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    rot_dim: u32,
    pos_base: u32,
    dst_offset: u32,
    dst_strides: [u32; 3],
    pos_scale: u32,
    in_ptr: u64,
    dst_ptr: u64,
    table_ptr: u64,
    pos_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct CopyPc {
    extent: [u32; 3],
    src_offset: u32,
    src_strides: [u32; 3],
    dst_offset: u32,
    dst_strides: [u32; 3],
    pos_scale: u32,
    src_ptr: u64,
    dst_ptr: u64,
    pos_ptr: u64,
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
    q_head_stride: u32,
    q_row_stride: u32,
    o_head_stride: u32,
    o_row_stride: u32,
    q_ptr: u64,
    kt_ptr: u64,
    v_ptr: u64,
    out_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PrefillQkvPackPc {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rot_dim: u32,
    pos_base: u32,
    qkv_stride: u32,
    t_max: u32,
    k_offset: u32,
    v_offset: u32,
    qkv_ptr: u64,
    q_ptr: u64,
    kt_ptr: u64,
    v_ptr: u64,
    table_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct AttnDecodePc {
    kv_len: u32,
    num_chunks: u32,
    group: u32,
    t_max: u32,
    scale: f32,
    _pad0: u32,
    q_ptr: u64,
    kt_ptr: u64,
    v_ptr: u64,
    scratch_ptr: u64,
    pos_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ArgmaxPc {
    n: u32,
    _pad: u32,
    in_ptr: u64,
    result_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct EmbedGatherPc {
    embd: u32,
    vocab: u32,
    table_f16: u32,
    n_tokens: u32,
    out_f16: u32,
    _pad: u32,
    token_ptr: u64,
    table_ptr: u64,
    out_ptr: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GemvChainPc {
    jobs_ptr: u64,
    sync_ptr: u64,
    n_jobs: u32,
    n_wg: u32,
}

/// Must stay 80 bytes / 16-aligned to match the GLSL `GemvJob`.
#[repr(C)]
#[derive(Copy, Clone)]
struct GemvJob {
    n: u32,
    k: u32,
    flags: u32,
    vcols: u32,
    alpha: f32,
    beta: f32,
    pad0: u32,
    pad1: u32,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    d_ptr: u64,
    bias_ptr: u64,
    pad2: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct AttnCombinePc {
    num_chunks: u32,
    group: u32,
    dh: u32,
    _pad0: u32,
    scratch_ptr: u64,
    out_ptr: u64,
}

/// Push-constant budget of the shared elementwise layout: the largest
/// block (`CopyPc` / `RopeScatterPc`, 72 bytes) rounded up for slack;
/// well under the 128-byte device minimum.
const ELEMENTWISE_PC_BYTES: usize = 80;

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
        let layout = unsafe { create_pc_only_layout(ctx, ELEMENTWISE_PC_BYTES as u32) }
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
    pub(super) fn plan_norm(
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

    pub(super) fn plan_rope(
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

    pub(super) fn plan_rope_scatter(
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

    /// Fused causal prefill attention (FlashAttention pattern):
    /// `out = softmax_causal(q @ K^T * scale) @ V` in one dispatch with
    /// online softmax — score tiles never touch global memory, and
    /// tiles above the causal frontier are skipped.  Layouts match the
    /// composed path: `q`/`out` are `[H, T_q, dh]`, `kt` is
    /// `[H_kv, dh, T_max]`, `v` is `[H_kv, T_max, dh]`; GQA works when
    /// `H` is a multiple of `H_kv`.  `dh` must be 64 or 128 (the
    /// compiled head-dimension variants).
    ///
    /// On devices where [`VulkanContext::coopmat2_enabled`] is set the
    /// dispatch routes to the tensor-core `NV_cooperative_matrix2`
    /// kernels (f16 operands, f32 accumulate — expect f16-level
    /// rounding vs the f32 SIMT path); otherwise the SIMT kernels run.
    pub fn run_flash_attention(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
    ) -> Result<RunStats> {
        self.run_flash_attention_impl(q, kt, v, out, desc, false)
    }

    /// [`Self::run_flash_attention`] pinned to the SIMT kernels even
    /// when the `NV_cooperative_matrix2` path is available.  A/B hook
    /// for the GPU validation tests; not part of the stable API.
    #[doc(hidden)]
    pub fn run_flash_attention_simt(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
    ) -> Result<RunStats> {
        self.run_flash_attention_impl(q, kt, v, out, desc, true)
    }

    fn run_flash_attention_impl(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
        force_simt: bool,
    ) -> Result<RunStats> {
        let dispatch = self.plan_flash_attention(q, kt, v, out, desc, force_simt)?;
        self.submit_one_elementwise(dispatch)
    }

    /// Validate and plan one flash-attention dispatch (CM2-first
    /// routing, exactly like [`Self::run_flash_attention`]).
    pub(super) fn plan_flash_attention(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        out: &Tensor,
        desc: FlashAttentionDesc,
        force_simt: bool,
    ) -> Result<ElementwiseDispatch> {
        // q/out may both be f16 (f16 activations): the CM2 kv16 io16
        // kernel loads halves directly and narrows the output store.
        let io_f16 = self.io_dtype_f16("run_flash_attention", q, out)?;
        self.validate_tensor_context(kt, "kt")?;
        self.validate_tensor_context(v, "v")?;
        if kt.dtype() != v.dtype() {
            bail!(
                "run_flash_attention: kt ({}) and v ({}) must share a storage type",
                kt.dtype().name(),
                v.dtype().name()
            );
        }
        let kv_f16 = kt.dtype() == DType::F16;
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
        if v_t != t_max || v_heads != kv_heads || v_dh != kt_dh {
            bail!(
                "run_flash_attention: inconsistent cache shapes kt {:?}, v {:?}",
                kt.shape(),
                v.shape()
            );
        }
        let dh = kt_dh;
        let (heads, t_q, q_head_stride, q_row_stride) = if let Some(heads) = desc.token_major_heads
        {
            if heads == 0 {
                bail!("run_flash_attention: token_major_heads must be nonzero");
            }
            let t_q = q.shape().first().copied().unwrap_or(0);
            let q_elems = q.len();
            let want = t_q as u64 * heads as u64 * dh as u64;
            if t_q == 0 || q_elems != want {
                bail!(
                    "run_flash_attention: token-major q {:?} does not hold [T={t_q}, H={heads}, dh={dh}]",
                    q.shape()
                );
            }
            // [T, H, dh] or rank-2 [T, H*dh]: head stride is dh, row stride is H*dh.
            (heads, t_q, dh, heads * dh)
        } else {
            let [heads, t_q, q_dh] = *q.shape() else {
                bail!(
                    "run_flash_attention: q must be [H, T_q, dh], got {:?}",
                    q.shape()
                );
            };
            if q_dh != dh {
                bail!(
                    "run_flash_attention: q dh {q_dh} != cache dh {dh} (q {:?}, kt {:?})",
                    q.shape(),
                    kt.shape()
                );
            }
            (heads, t_q, t_q * dh, dh)
        };
        let (o_head_stride, o_row_stride) = if let Some(oh) = desc.out_token_major_heads {
            if oh == 0 {
                bail!("run_flash_attention: out_token_major_heads must be nonzero");
            }
            if oh != heads {
                bail!("run_flash_attention: out_token_major_heads {oh} != q heads {heads}");
            }
            let want = t_q as u64 * oh as u64 * dh as u64;
            if out.len() != want {
                bail!(
                    "run_flash_attention: token-major out {:?} does not hold [T={t_q}, H={oh}, dh={dh}]",
                    out.shape()
                );
            }
            (dh, oh * dh)
        } else if desc.token_major_heads.is_some() {
            if out.len() != q.len() {
                bail!(
                    "run_flash_attention: out shape {:?} must match token-major q {:?}",
                    out.shape(),
                    q.shape()
                );
            }
            (q_head_stride, q_row_stride)
        } else if out.shape() != q.shape() || out.len() != q.len() {
            bail!(
                "run_flash_attention: out shape {:?} must equal q shape {:?}",
                out.shape(),
                q.shape()
            );
        } else {
            (q_head_stride, q_row_stride)
        };
        if v_dh != dh {
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
        // Query rows per workgroup: the cm2 kernels tile Br=64 rows
        // across 128 threads; the SIMT kernels take 128 rows.
        let (pipeline, rows_per_wg) = if io_f16 {
            // No SIMT sibling reads f16 q — the compiled variant is
            // the CM2 kv16 io16 dh64 kernel only.
            if kv_f16 && dh == 64 && !force_simt {
                match pipes.cm2_pipeline(Cm2Op::FlashKv16Io16Dh64) {
                    Some(pipeline) => (pipeline, 64),
                    None => bail!(
                        "run_flash_attention: f16 q/out require NV_cooperative_matrix2 \
                         support, which this device lacks"
                    ),
                }
            } else {
                bail!(
                    "run_flash_attention: f16 q/out are compiled for f16 KV caches with \
                     head dimension 64 on the CM2 route only (kv {}, dh {dh})",
                    kt.dtype().name()
                )
            }
        } else {
            let (simt_kernel, cm2_kernel) = match (dh, kv_f16) {
                (64, false) => (Op::FlashDh64, Cm2Op::FlashDh64),
                (128, false) => (Op::FlashDh128, Cm2Op::FlashDh128),
                (64, true) => (Op::FlashKv16Dh64, Cm2Op::FlashKv16Dh64),
                (128, true) => (Op::FlashKv16Dh128, Cm2Op::FlashKv16Dh128),
                (other, _) => bail!(
                    "run_flash_attention: head dimension {other} unsupported (compiled variants: 64, 128)"
                ),
            };
            let cm2 = if force_simt {
                None
            } else {
                pipes.cm2_pipeline(cm2_kernel)
            };
            match cm2 {
                Some(pipeline) => (pipeline, 64),
                None => (pipes.pipeline(simt_kernel), 128),
            }
        };
        self.plan_elementwise(
            pipeline,
            &FlashPc {
                t_q,
                t_max,
                kv_len: desc.kv_len,
                pos_base: desc.pos_base,
                group_size: heads / kv_heads,
                scale: desc.scale,
                q_head_stride,
                q_row_stride,
                o_head_stride,
                o_row_stride,
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                out_ptr: out.device_address(),
            },
            t_q.div_ceil(rows_per_wg),
            heads,
        )
    }

    /// Fused wide-prefill QKV pack: RoPE-rotate Q into head-major
    /// `[H, T, dh]`, RoPE-scatter K into the Kt cache, and copy V into
    /// the V cache, all from one concatenated `[T, n_qkv]` row.  f16
    /// activations and f16 caches only — the T≥128 a16 prefill path.
    pub fn run_prefill_qkv_pack(
        &self,
        qkv: &Tensor,
        table: &Tensor,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        desc: PrefillQkvPackDesc,
    ) -> Result<RunStats> {
        let dispatch = self.plan_prefill_qkv_pack(qkv, table, q, kt, v, desc)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(super) fn plan_prefill_qkv_pack(
        &self,
        qkv: &Tensor,
        table: &Tensor,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        desc: PrefillQkvPackDesc,
    ) -> Result<ElementwiseDispatch> {
        for (tensor, name) in [(qkv, "qkv"), (q, "q"), (kt, "kt"), (v, "v")] {
            self.validate_tensor_context(tensor, name)?;
            if tensor.dtype() != DType::F16 {
                bail!(
                    "run_prefill_qkv_pack: {name} must be f16 storage, got {}",
                    tensor.dtype().name()
                );
            }
        }
        self.ensure_f32(table, "run_prefill_qkv_pack", "table")?;
        if desc.heads == 0
            || desc.kv_heads == 0
            || desc.head_dim == 0
            || desc.rot_dim == 0
            || !desc.rot_dim.is_multiple_of(2)
            || desc.rot_dim > desc.head_dim
        {
            bail!(
                "run_prefill_qkv_pack: invalid heads/kv_heads/dh/rot_dim \
                 ({}/{}/{}/{})",
                desc.heads,
                desc.kv_heads,
                desc.head_dim,
                desc.rot_dim
            );
        }
        let [tokens, qkv_stride] = *qkv.shape() else {
            bail!(
                "run_prefill_qkv_pack: qkv must be [T, n_qkv], got {:?}",
                qkv.shape()
            );
        };
        let embd = desc.heads * desc.head_dim;
        let kv_dim = desc.kv_heads * desc.head_dim;
        let want_stride = embd + 2 * kv_dim;
        if qkv_stride != want_stride {
            bail!("run_prefill_qkv_pack: qkv N {qkv_stride} != H*dh + 2*H_kv*dh ({want_stride})");
        }
        if *q.shape() != [desc.heads, tokens, desc.head_dim] {
            bail!(
                "run_prefill_qkv_pack: q must be head-major [H={}, T={tokens}, dh={}], got {:?}",
                desc.heads,
                desc.head_dim,
                q.shape()
            );
        }
        let [kt_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_prefill_qkv_pack: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_prefill_qkv_pack: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        if kt_heads != desc.kv_heads
            || v_heads != desc.kv_heads
            || kt_dh != desc.head_dim
            || v_dh != desc.head_dim
            || v_t != t_max
        {
            bail!(
                "run_prefill_qkv_pack: cache shapes kt {:?} v {:?} vs H_kv={} dh={}",
                kt.shape(),
                v.shape(),
                desc.kv_heads,
                desc.head_dim
            );
        }
        if desc.pos_base.saturating_add(tokens) > t_max {
            bail!(
                "run_prefill_qkv_pack: pos_base {} + T {tokens} exceeds T_max {t_max}",
                desc.pos_base
            );
        }
        let needed = (desc.pos_base as u64 + tokens as u64) * desc.rot_dim as u64;
        if table.len() < needed {
            bail!(
                "run_prefill_qkv_pack: table length {} < required {needed} \
                 (pos_base {} + {tokens} tokens, rot_dim {})",
                table.len(),
                desc.pos_base,
                desc.rot_dim
            );
        }
        if !desc.head_dim.is_multiple_of(4) {
            bail!(
                "run_prefill_qkv_pack: head_dim {} must be a multiple of 4",
                desc.head_dim
            );
        }
        let q_pairs = tokens * desc.heads * (desc.rot_dim / 2);
        let k_pairs = tokens * desc.kv_heads * (desc.rot_dim / 2);
        let v_vec4s = tokens * desc.kv_heads * (desc.head_dim / 4);
        let threads = q_pairs.saturating_add(k_pairs).saturating_add(v_vec4s);
        let pipeline = self.elementwise()?.pipeline(Op::PrefillQkvPack);
        self.plan_elementwise(
            pipeline,
            &PrefillQkvPackPc {
                tokens,
                heads: desc.heads,
                kv_heads: desc.kv_heads,
                head_dim: desc.head_dim,
                rot_dim: desc.rot_dim,
                pos_base: desc.pos_base,
                qkv_stride,
                t_max,
                k_offset: embd,
                v_offset: embd + kv_dim,
                qkv_ptr: qkv.device_address(),
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                table_ptr: table.device_address(),
            },
            threads.div_ceil(WG),
            1,
        )
    }

    /// Fused split-K decode attention: `out = softmax(q @ K^T * scale)
    /// @ V` for ONE query row per head, reading only the `kv_len`
    /// valid cache prefix.  Two dispatches in one submission: stage 1
    /// writes per-chunk online-softmax partials to `scratch`
    /// (`[kv_heads, num_chunks, group, dh+2]` f32, at least
    /// `kv_heads * ATTN_DECODE_MAX_CHUNKS * group * (dh+2)` elements),
    /// stage 2 merges them exactly.  Layouts match the composed path:
    /// `q`/`out` are `[kv_heads, group, dh]` (or any contiguous
    /// reshape ending in `dh`), `kt` is `[H_kv, dh, T_max]`, `v` is
    /// `[H_kv, T_max, dh]`, f32 or f16 caches (matching).  `dh` must
    /// be 64 (the compiled variant) and `group <= 8`.
    pub fn run_attn_decode(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        scratch: &Tensor,
        out: &Tensor,
        desc: AttnDecodeDesc,
    ) -> Result<RunStats> {
        let (stage1, combine) = self.plan_attn_decode(q, kt, v, scratch, out, desc)?;
        let mut slot = self.checkout_slot();
        let gpu_time_ns = unsafe {
            self.submit_timed(
                &mut slot,
                "get_query_pool_results (attn_decode)",
                |_dev, cb, _slot| {
                    self.record_elementwise(cb, &stage1);
                    super::recording::record_compute_to_compute_barrier(&self.ctx, cb);
                    self.record_elementwise(cb, &combine);
                    Ok(())
                },
            )
        }?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 2,
            total_flops: 0,
        })
    }

    /// Validate and plan both decode-attention dispatches.  The caller
    /// must place a compute barrier between them (stage 2 reads the
    /// scratch stage 1 writes).
    pub(super) fn plan_attn_decode(
        &self,
        q: &Tensor,
        kt: &Tensor,
        v: &Tensor,
        scratch: &Tensor,
        out: &Tensor,
        desc: AttnDecodeDesc,
    ) -> Result<(ElementwiseDispatch, ElementwiseDispatch)> {
        self.ensure_f32(q, "run_attn_decode", "q")?;
        self.validate_tensor_context(kt, "kt")?;
        self.validate_tensor_context(v, "v")?;
        self.ensure_f32(scratch, "run_attn_decode", "scratch")?;
        self.ensure_f32(out, "run_attn_decode", "out")?;
        if kt.dtype() != v.dtype() {
            bail!(
                "run_attn_decode: kt ({}) and v ({}) must share a storage type",
                kt.dtype().name(),
                v.dtype().name()
            );
        }
        let kv_f16 = kt.dtype() == DType::F16;
        let [kv_heads, kt_dh, t_max] = *kt.shape() else {
            bail!(
                "run_attn_decode: kt must be [H_kv, dh, T_max], got {:?}",
                kt.shape()
            );
        };
        let [v_heads, v_t, v_dh] = *v.shape() else {
            bail!(
                "run_attn_decode: v must be [H_kv, T_max, dh], got {:?}",
                v.shape()
            );
        };
        let dh = kt_dh;
        if v_dh != dh || v_t != t_max || v_heads != kv_heads || kv_heads == 0 {
            bail!(
                "run_attn_decode: inconsistent caches kt {:?}, v {:?}",
                kt.shape(),
                v.shape()
            );
        }
        if dh != 64 {
            bail!("run_attn_decode: head dimension {dh} unsupported (compiled variant: 64)");
        }
        let heads_elems = kv_heads as u64 * dh as u64;
        if q.is_empty() || !q.len().is_multiple_of(heads_elems) {
            bail!(
                "run_attn_decode: q length {} must be kv_heads*group*dh (kv_heads {kv_heads}, dh {dh})",
                q.len()
            );
        }
        let group = u32::try_from(q.len() / heads_elems)
            .map_err(|_| anyhow::anyhow!("run_attn_decode: group exceeds u32"))?;
        if group == 0 || group > 8 {
            bail!("run_attn_decode: GQA group {group} unsupported (1..=8)");
        }
        if out.len() != q.len() {
            bail!(
                "run_attn_decode: out length {} must equal q length {}",
                out.len(),
                q.len()
            );
        }
        if desc.kv_len == 0 || desc.kv_len > t_max {
            bail!(
                "run_attn_decode: kv_len {} out of range 1..={t_max}",
                desc.kv_len
            );
        }
        // Position-driven dispatches cannot size the grid per token, so
        // they always use the fixed MAX_CHUNKS decomposition; chunks
        // past the effective kv_len write neutral partials.
        let num_chunks = if desc.pos_addr != 0 {
            ATTN_DECODE_MAX_CHUNKS
        } else {
            attn_decode_num_chunks(desc.kv_len)
        };
        let needed = kv_heads as u64 * num_chunks as u64 * group as u64 * (dh as u64 + 2);
        if scratch.len() < needed {
            bail!(
                "run_attn_decode: scratch length {} < required {needed} \
                 (kv_heads {kv_heads} * chunks {num_chunks} * group {group} * (dh+2))",
                scratch.len()
            );
        }
        let pipes = self.elementwise()?;
        let stage1_kernel = if kv_f16 {
            Op::AttnDecodeKv16Dh64
        } else {
            Op::AttnDecodeDh64
        };
        let stage1 = self.plan_elementwise(
            pipes.pipeline(stage1_kernel),
            &AttnDecodePc {
                kv_len: desc.kv_len,
                num_chunks,
                group,
                t_max,
                scale: desc.scale,
                _pad0: 0,
                q_ptr: q.device_address(),
                kt_ptr: kt.device_address(),
                v_ptr: v.device_address(),
                scratch_ptr: scratch.device_address(),
                pos_ptr: desc.pos_addr,
            },
            num_chunks,
            kv_heads,
        )?;
        let combine = self.plan_elementwise(
            pipes.pipeline(Op::AttnDecodeCombine),
            &AttnCombinePc {
                num_chunks,
                group,
                dh,
                _pad0: 0,
                scratch_ptr: scratch.device_address(),
                out_ptr: out.device_address(),
            },
            kv_heads * group,
            1,
        )?;
        Ok((stage1, combine))
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

    pub(super) fn plan_binary(
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

    pub(super) fn plan_copy_strided(
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
    pub fn run_argmax(&self, input: &Tensor, result: &super::HostU32Buffer) -> Result<RunStats> {
        let dispatch = self.plan_argmax(input, result)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(super) fn plan_argmax(
        &self,
        input: &Tensor,
        result: &super::HostU32Buffer,
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
        token: &super::HostU32Buffer,
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
        tokens: &super::TokenIdBuffer,
        table: &Tensor,
        out: &Tensor,
    ) -> Result<RunStats> {
        let dispatch =
            self.plan_embed_gather(tokens.device_address(), tokens.buffer(), table, out)?;
        self.submit_one_elementwise(dispatch)
    }

    pub(super) fn plan_embed_gather(
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

    /// Plan a persistent GEMV chain (see [`super::ExecOp::GemvChain`]).
    pub(super) fn plan_gemv_chain(&self, jobs: &[MatmulOp<'_>]) -> Result<ElementwiseDispatch> {
        if !self.ctx.memory_model_device_scope_enabled {
            bail!(
                "gemv_chain: device lacks vulkanMemoryModelDeviceScope \
                 (required for the quorum barrier)"
            );
        }
        if jobs.is_empty() {
            bail!("gemv_chain: empty job list");
        }
        if jobs.len() > GEMV_CHAIN_MAX_JOBS {
            bail!(
                "gemv_chain: {} jobs exceeds max {GEMV_CHAIN_MAX_JOBS}",
                jobs.len()
            );
        }
        let pipeline = self
            .elementwise()?
            .gemv_chain
            .as_ref()
            .map(|k| k.pipeline)
            .ok_or_else(|| anyhow::anyhow!("gemv_chain: kernel was not built"))?;

        let mut packed = Vec::with_capacity(jobs.len());
        for (i, op) in jobs.iter().enumerate() {
            packed.push(self.pack_gemv_job(op, i)?);
        }
        // SYNC_AFTER on job i means "barrier after the group that
        // *ends* at i".  Independent neighbours (no RAW/WAW/WAR)
        // share a group so they occupy one flattened tile space.
        for i in 0..packed.len().saturating_sub(1) {
            if gemv_jobs_hazard(&jobs[..=i], &jobs[i + 1]) {
                packed[i].flags |= GEMV_FLAG_SYNC_AFTER;
            }
        }

        // Header: arrived[2] + phase + pad, then 80-byte jobs.  Device-local
        // so the quorum atomics stay on the GPU; a host-visible sync
        // cell would bounce every arrival over PCIe and lose the
        // 7.7 µs pipeline-barrier win.
        let header = [0u32; 4];
        let nbytes = std::mem::size_of_val(&header) + packed.len() * std::mem::size_of::<GemvJob>();
        let buf = Buffer::new(
            &self.ctx,
            nbytes as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferLocation::Device,
        )
        .context("gemv_chain job buffer")?;
        let job_bytes = bytemuck::cast_slice::<GemvJob, u8>(&packed);
        let mut all = vec![0u8; nbytes];
        all[..16].copy_from_slice(bytemuck::bytes_of(&header));
        all[16..].copy_from_slice(job_bytes);
        self.upload_bytes_to_buffer(&buf, &all)?;

        let max_tiles = packed
            .iter()
            .map(|j| {
                let vcols = if j.vcols == 2 { 2 } else { 1 };
                j.n.div_ceil(32 * vcols)
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let n_wg = max_tiles.min(GEMV_CHAIN_MAX_WG);
        let pc = GemvChainPc {
            jobs_ptr: buf.device_address() + 16,
            sync_ptr: buf.device_address(),
            n_jobs: packed.len() as u32,
            n_wg,
        };
        let bytes = bytemuck::bytes_of(&pc);
        let mut push = [0u8; ELEMENTWISE_PC_BYTES];
        push[..bytes.len()].copy_from_slice(bytes);
        Ok(ElementwiseDispatch {
            pipeline,
            layout: self.elementwise()?.layout,
            push,
            push_len: bytes.len(),
            groups: (n_wg, 1),
            retain: Some(Arc::new(buf)),
        })
    }

    fn pack_gemv_job(&self, op: &MatmulOp<'_>, index: usize) -> Result<GemvJob> {
        self.validate_op_context(op)?;
        if !matches!(op.store, crate::matmul::MatmulStore::None) {
            bail!("gemv_chain[{index}]: fused store epilogues are not supported");
        }
        if op.call.accumulate {
            bail!("gemv_chain[{index}]: accumulate is not supported");
        }
        if op.call.a.dtype() != DType::F32 || op.call.c.dtype() != DType::F32 {
            bail!("gemv_chain[{index}]: A and C must be f32 storage");
        }
        if op.call.b.dtype() != DType::F16 {
            bail!("gemv_chain[{index}]: B must be f16 storage");
        }
        if op.packed_b {
            bail!("gemv_chain[{index}]: packed-B layout is not supported");
        }
        if op.epilogue.bias.is_some() {
            bail!("gemv_chain[{index}]: bias epilogue is not supported");
        }
        let dims = ResolvedMatmul::from_op(op)?;
        if dims.batch != 1 || dims.m != 1 {
            bail!(
                "gemv_chain[{index}]: requires batch=1 M=1 (got B={} M={})",
                dims.batch,
                dims.m
            );
        }
        if dims.a_f16 {
            bail!("gemv_chain[{index}]: f16 activations are not supported");
        }
        let (epi_bin, epi_beta) = match op.epilogue.binary {
            EpilogueBinary::None => (0u32, 0.0f32),
            EpilogueBinary::AddScaled { beta, .. } => (1, beta),
            EpilogueBinary::Mul { .. } => (2, 0.0),
        };
        let epi_act = match op.epilogue.activation {
            Activation::None => 0u32,
            Activation::Silu => 2,
            other => bail!("gemv_chain[{index}]: unsupported activation {other:?}"),
        };
        if op.normed_a.is_some() && epi_bin == 1 {
            bail!("gemv_chain[{index}]: NORM_A cannot combine with AddScaled");
        }
        let mut flags = 0u32;
        if op.normed_a.is_some() {
            flags |= GEMV_FLAG_NORM_A;
        }
        flags |= (epi_bin & 3) << 8;
        flags |= (epi_act & 3) << 16;
        let beta = if let Some((_, eps)) = op.normed_a {
            eps
        } else {
            epi_beta
        };
        let vcols = gemv_chain_vcols(dims.k, dims.n);
        let d_ptr = op.epilogue.d_tensor().map_or(0, Tensor::device_address);
        let bias_ptr = op.normed_a.map(|(w, _)| w.device_address()).unwrap_or(0);
        Ok(GemvJob {
            n: dims.n,
            k: dims.k,
            flags,
            vcols,
            alpha: op.call.alpha,
            beta,
            pad0: 0,
            pad1: 0,
            a_ptr: op.call.a.device_address(),
            b_ptr: op.call.b.device_address(),
            c_ptr: op.call.c.device_address(),
            d_ptr,
            bias_ptr,
            pad2: 0,
        })
    }

    /// Run a GEMV chain as its own submission (tests / microbench).
    pub fn run_gemv_chain(&self, jobs: &[MatmulOp<'_>]) -> Result<RunStats> {
        let dispatch = self.plan_gemv_chain(jobs)?;
        let total_flops = jobs
            .iter()
            .map(|op| ResolvedMatmul::from_op(op).map(|d| d.total_flops))
            .sum::<Result<u64>>()?;
        let stats = self.submit_one_elementwise(dispatch)?;
        Ok(RunStats {
            gpu_time_ns: stats.gpu_time_ns,
            n_calls: jobs.len(),
            total_flops,
        })
    }
}

/// VCOLS pick matching the f16w row-kernel heuristic.
fn gemv_chain_vcols(k: u32, n: u32) -> u32 {
    if k >= 4096 || n <= 512 || (k >= 2048 && n <= 2048) {
        1
    } else {
        2
    }
}

const GEMV_FLAG_NORM_A: u32 = 1;
const GEMV_FLAG_SYNC_AFTER: u32 = 1 << 24;

fn gemv_jobs_hazard(prior: &[MatmulOp<'_>], next: &MatmulOp<'_>) -> bool {
    let next_reads = gemv_job_reads(next);
    let next_writes = gemv_job_writes(next);
    for prev in prior {
        let writes = gemv_job_writes(prev);
        let reads = gemv_job_reads(prev);
        if next_reads.iter().any(|b| writes.contains(b))
            || next_writes
                .iter()
                .any(|b| writes.contains(b) || reads.contains(b))
        {
            return true;
        }
    }
    false
}

fn gemv_job_reads(op: &MatmulOp<'_>) -> Vec<vk::Buffer> {
    let mut v = vec![op.call.a.raw_buffer(), op.call.b.raw_buffer()];
    if let Some(d) = op.epilogue.d_tensor() {
        v.push(d.raw_buffer());
    }
    if let Some((w, _)) = op.normed_a {
        v.push(w.raw_buffer());
    }
    if op.call.accumulate {
        v.push(op.call.c.raw_buffer());
    }
    v
}

fn gemv_job_writes(op: &MatmulOp<'_>) -> Vec<vk::Buffer> {
    vec![op.call.c.raw_buffer()]
}

// SAFETY: plain-old-data push-constant mirrors of the GLSL blocks.
unsafe impl bytemuck::Pod for SoftmaxPc {}
unsafe impl bytemuck::Zeroable for SoftmaxPc {}
unsafe impl bytemuck::Pod for NormPc {}
unsafe impl bytemuck::Zeroable for NormPc {}
unsafe impl bytemuck::Pod for RopePc {}
unsafe impl bytemuck::Zeroable for RopePc {}
unsafe impl bytemuck::Pod for RopeScatterPc {}
unsafe impl bytemuck::Zeroable for RopeScatterPc {}
unsafe impl bytemuck::Pod for CopyPc {}
unsafe impl bytemuck::Zeroable for CopyPc {}
unsafe impl bytemuck::Pod for FlashPc {}
unsafe impl bytemuck::Zeroable for FlashPc {}
unsafe impl bytemuck::Pod for AttnDecodePc {}
unsafe impl bytemuck::Zeroable for AttnDecodePc {}
unsafe impl bytemuck::Pod for AttnCombinePc {}
unsafe impl bytemuck::Zeroable for AttnCombinePc {}
unsafe impl bytemuck::Pod for ArgmaxPc {}
unsafe impl bytemuck::Zeroable for ArgmaxPc {}
unsafe impl bytemuck::Pod for EmbedGatherPc {}
unsafe impl bytemuck::Zeroable for EmbedGatherPc {}
unsafe impl bytemuck::Pod for PrefillQkvPackPc {}
unsafe impl bytemuck::Zeroable for PrefillQkvPackPc {}
unsafe impl bytemuck::Pod for BinaryPc {}
unsafe impl bytemuck::Zeroable for BinaryPc {}
unsafe impl bytemuck::Pod for GemvChainPc {}
unsafe impl bytemuck::Zeroable for GemvChainPc {}
unsafe impl bytemuck::Pod for GemvJob {}
unsafe impl bytemuck::Zeroable for GemvJob {}

const _: () = assert!(std::mem::size_of::<GemvJob>() == 80);
const _: () = assert!(std::mem::size_of::<PrefillQkvPackPc>() == 80);
const _: () = assert!(std::mem::size_of::<PrefillQkvPackPc>() <= ELEMENTWISE_PC_BYTES);
