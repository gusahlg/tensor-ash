//! Llama-family model built on tensor-ash ops: GGUF weight loading
//! (with the GGML->tensor-ash transpose), KV caches, and the
//! prefill / decode forward passes.
//!
//! The per-layer decode step mirrors tests/correctness/decoder.rs;
//! prefill swaps the composed attention for run_flash_attention.

use std::sync::Arc;

use anyhow::{Context, Result};
use tensor_ash::{
    ATTN_DECODE_MAX_CHUNKS, Activation, Epilogue, EpilogueBinary, Executor, HostU32Buffer,
    MatmulCall, PosBuffer, PreparedOps, Tensor, TokenIdBuffer, VulkanContext,
};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layers: u32,
    pub embd: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub dh: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub rms_eps: f32,
    pub rope_base: f32,
    /// KV-cache capacity in tokens.
    pub t_max: u32,
}

struct Layer {
    /// Prefill concat of Q|K|V along N: `[embd, embd+2*kv]`.  One
    /// GEMM replaces the three projections (wide T fills the device;
    /// T=6 rides the same weights so we do not keep a second copy).
    w_qkv: Tensor,
    wo: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    /// Packed `[N/tile][K][tile]` copies for decode GEMVs that win on
    /// the packed K-walk (q/k/v have no unpacked sibling; gate/up are
    /// the wide-N v2 shapes).  o/down stay on the unpacked tensors —
    /// packing them was measured-neutral.
    wq_p: Tensor,
    wk_p: Tensor,
    wv_p: Tensor,
    w_gate_p: Tensor,
    w_up_p: Tensor,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    /// Kt cache [kv_heads, dh, t_max], f16 by default
    /// (`LLAMA_ASH_KV=f32` opts out).
    kt_cache: Tensor,
    /// V cache [kv_heads, t_max, dh], same storage as `kt_cache`.
    v_cache: Tensor,
}

/// Scratch activations for one forward width `t`, reused across layers.
struct Scratch {
    t: u32,
    x_a: Tensor,
    x_b: Tensor,
    xn: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// Prefill only: concatenated QKV row `[t, embd + 2*kv_dim]`.
    qkv: Option<Tensor>,
    /// Prefill: q permuted to [heads, t, dh].  Decode: q reshaped to
    /// [kv_heads, group, dh].
    q_heads: Tensor,
    /// Same shape story as `q_heads`, holds the attention output.
    attn_heads: Tensor,
    /// Decode only: [kv_heads, group, t_max] score rows.
    scores: Option<Tensor>,
    /// Decode only: split-K partials for the fused attention op,
    /// [kv_heads * MAX_CHUNKS * group * (dh+2)] f32.
    attn_partials: Option<Tensor>,
    /// Decode only: same memory order as `q_heads`/`attn_heads` but
    /// shaped [heads, 1, dh] for the flash-decode variant.
    q_flash: Option<Tensor>,
    attn_flash: Option<Tensor>,
    /// Decode only: free contiguous reshapes of `q` / `attn_flat` to
    /// [kv_heads, group, dh] for the fused GQA attention op.
    q_gqa: Option<Tensor>,
    attn_gqa: Option<Tensor>,
    attn_flat: Tensor,
    o: Tensor,
    on: Tensor,
    up: Tensor,
    gate: Tensor,
    last: Tensor,
    last_n: Tensor,
    logits: Tensor,
}

impl Scratch {
    /// `decode` selects the GQA-batched shapes and split-K partials;
    /// it cannot be inferred from `t` — a 1-token *prefill* still needs
    /// the `[heads, t, dh]` flash layout, not the decode aliases.
    ///
    /// `act_f16` stores the layer-loop activations (x ping-pong, xn,
    /// q/k/v, head-permuted q/attn, o/on, up/gate) as f16 — the
    /// prefill-only fast path where every GEMM takes the a16 coopmat
    /// route and attention the CM2 io16 kernel.  The LM-head tail
    /// (`last`/`last_n`/`logits`) stays f32: post-norm LM-head inputs
    /// and logits can exceed the activations' comfortable f16 range,
    /// and the final matmul is one row.
    fn new(
        ctx: &Arc<VulkanContext>,
        cfg: &Config,
        t: u32,
        decode: bool,
        act_f16: bool,
    ) -> Result<Self> {
        debug_assert!(!decode || t == 1, "decode scratch is single-token");
        debug_assert!(!(decode && act_f16), "decode scratch stays f32");
        let (h, f, kv) = (cfg.embd, cfg.ffn, cfg.kv_heads * cfg.dh);
        let dev = |shape: &[u32]| {
            if act_f16 {
                Tensor::uninit_device_f16(ctx, shape)
            } else {
                Tensor::uninit_device(ctx, shape)
            }
        };
        let dev32 = |shape: &[u32]| Tensor::uninit_device(ctx, shape);
        let q = dev(&[t, h])?;
        let attn_flat = dev(&[t, h])?;
        let (q_gqa, attn_gqa) = if decode {
            let group = cfg.heads / cfg.kv_heads;
            (
                Some(q.alias_with_shape(&[cfg.kv_heads, group, cfg.dh])?),
                Some(attn_flat.alias_with_shape(&[cfg.kv_heads, group, cfg.dh])?),
            )
        } else {
            (None, None)
        };
        let (q_heads, attn_heads, scores, attn_partials, q_flash, attn_flash) = if decode {
            let group = cfg.heads / cfg.kv_heads;
            let partials_len = cfg.kv_heads * ATTN_DECODE_MAX_CHUNKS * group * (cfg.dh + 2);
            (
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                dev(&[cfg.kv_heads, group, cfg.dh])?,
                Some(dev(&[cfg.kv_heads, group, cfg.t_max])?),
                Some(dev(&[partials_len])?),
                Some(dev(&[cfg.heads, 1, cfg.dh])?),
                Some(dev(&[cfg.heads, 1, cfg.dh])?),
            )
        } else {
            (
                dev(&[cfg.heads, t, cfg.dh])?,
                dev(&[cfg.heads, t, cfg.dh])?,
                None,
                None,
                None,
                None,
            )
        };
        Ok(Self {
            t,
            x_a: dev(&[t, h])?,
            x_b: dev(&[t, h])?,
            xn: dev(&[t, h])?,
            q,
            k: dev(&[t, kv])?,
            v: dev(&[t, kv])?,
            qkv: if decode {
                None
            } else {
                Some(dev(&[t, h + 2 * kv])?)
            },
            q_heads,
            attn_heads,
            scores,
            attn_partials,
            q_flash,
            attn_flash,
            q_gqa,
            attn_gqa,
            attn_flat,
            o: dev(&[t, h])?,
            on: dev(&[t, h])?,
            up: dev(&[t, f])?,
            gate: dev(&[t, f])?,
            last: dev32(&[1, h])?,
            last_n: dev32(&[1, h])?,
            logits: dev32(&[1, cfg.vocab])?,
        })
    }
}

pub struct Model {
    ctx: Arc<VulkanContext>,
    exec: Arc<Executor>,
    pub cfg: Config,
    layers: Vec<Layer>,
    /// Token embeddings on the CPU, row-major [vocab][embd] f32.
    /// Decode still seeds the first replay row from here; prefill
    /// gathers on-device from [`embd_gpu`].
    embd_cpu: Vec<f32>,
    /// The same table on the DEVICE ([vocab, embd], in the GGUF's
    /// storage precision) for the decode loop's on-GPU row gather.
    embd_gpu: Tensor,
    output_norm: Tensor,
    lm_head: Tensor,
    lm_head_p: Tensor,
    rope_table: Tensor,
    /// Tokens currently in the KV caches.
    pub pos: u32,
    decode_scratch: Scratch,
    prefill_scratch: Option<Scratch>,
    /// The 4-byte device-readable position cell the prepared decode
    /// graph reads; the host bumps it between replays.
    pos_buf: PosBuffer,
    /// The 4-byte host-readable token cell the prepared graph's GPU
    /// argmax writes; per token the host reads this ONE u32 instead of
    /// downloading the 32000-float logits.
    token_buf: HostU32Buffer,
    /// Prefill prompt ids (host-visible, gathered on-device).
    token_ids: TokenIdBuffer,
    /// Decode strategy.  `LLAMA_ASH_DECODE=prepared|graph|perop|flash`
    /// overrides the default (prepared: record the token's command
    /// buffer once, replay it per token via the position buffer).
    decode_mode: DecodeMode,
    /// Fused split-K decode attention (default when dh == 64).
    /// `LLAMA_ASH_ATTN=composed` restores the scores/softmax/PV trio.
    fused_attn: bool,
    /// The model/device half of the f16-prefill-activations gate:
    /// every layer-loop op has an f16 route (a16 coopmat GEMMs need
    /// embd/kv_dim/ffn % 64, the CM2 io16 flash kernel needs dh 64 +
    /// f16 KV caches + coopmat2).  The per-call half lives in
    /// [`prefill`](Self::prefill): `t >= 128` (the plain-matmul MLP
    /// branch) and `t % 64 == 0` (a16 M alignment for the 64-tile
    /// wave-fill route; 128-tile shapes still prefer 128x128).
    /// `LLAMA_ASH_ACT=f32` opts out.  Decode always stays f32.
    prefill_act_f16: bool,
    /// Per-op-class GPU nanoseconds for the current perop decode step
    /// (diagnostics; filled when `LLAMA_ASH_BREAKDOWN=1`).
    pub breakdown: std::cell::RefCell<Vec<(&'static str, u64)>>,
    /// Record-once prefill graph for the last `t` at `pos_base == 0`.
    /// Dropped before scratch/weights (see [`Drop`]).
    prefill_prepared: Option<PrefillPrepared>,
    /// Pos-relative single-token decode graph (generate / short `n`).
    decode_prepared: Option<PreparedOps<'static, 'static>>,
    /// One-submit unroll of `n` baked-position decode steps, keyed by
    /// `(start_pos, n)`.  The bench warmup records it; the timed tg
    /// run replays so 128 tokens pay one `vkQueueSubmit`.
    decode_unrolled: Option<(u32, u32, PreparedOps<'static, 'static>)>,
}

/// Prepared prefill command buffer with erased lifetimes.  Device
/// addresses baked in belong to [`Model`]'s scratch and weights; the
/// cache is cleared whenever those tensors are recreated and in
/// [`Drop`].
struct PrefillPrepared {
    t: u32,
    prepared: PreparedOps<'static, 'static>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecodeMode {
    /// Record the whole token ONCE, replay per token (default): the
    /// position-dependent values are read from a device-side position
    /// buffer, so a decode step costs one `vkQueueSubmit` with zero
    /// host-side re-recording.  Requires the fused decode attention.
    Prepared,
    /// One `run_exec_ops` submission per token, re-recorded each call.
    Graph,
    /// The original per-op path: one submit + wait per dispatch.
    PerOp,
    /// Per-op with the prefill flash kernel for attention.  Kept as a
    /// measured comparison switch only — it loses to the fused split-K
    /// decode attention (`AttnDecode`) at every probed context length
    /// (see benchmarks/experiment-branch.md).
    Flash,
}

/// Programmatic overrides for the environment knobs [`Model::load`]
/// reads (`LLAMA_ASH_DECODE`, `LLAMA_ASH_KV`).  `None` keeps the
/// environment-derived behaviour; `Some` wins over the environment.
/// The thesis harness uses this to sweep decode modes and KV dtypes
/// in one process without mutating the environment.
#[derive(Copy, Clone, Debug, Default)]
pub struct LoadOverrides {
    /// Decode strategy (see [`DecodeMode`]).
    pub decode_mode: Option<DecodeMode>,
    /// `true` = f32 KV caches (the `LLAMA_ASH_KV=f32` opt-out),
    /// `false` = f16 caches (the default).
    pub kv_f32: Option<bool>,
}

/// Dispatch/barrier census of one recorded graph (see
/// [`tensor_ash::Executor::exec_ops_barrier_count`]): how many
/// dispatches one submission records and how many full compute
/// barriers the hazard tracker emits between them.
#[derive(Copy, Clone, Debug)]
pub struct GraphStats {
    pub dispatches: usize,
    pub barriers: usize,
}

/// Greedy sampling: index of the largest logit.
fn argmax(logits: &[f32]) -> Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .context("empty logits")
}

/// The plain matmul call every projection uses: `c = a @ b`.
fn mm<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> MatmulCall<'t> {
    MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }
}

/// Bias-free fused epilogue (llama has no linear biases).
fn epi<'t>(activation: Activation, binary: EpilogueBinary<'t>) -> Epilogue<'t> {
    Epilogue {
        bias: None,
        activation,
        binary,
    }
}

/// GGML row-major [n_out][n_in] -> tensor-ash [n_in][n_out].
fn transpose(src: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    assert_eq!(src.len(), n_out * n_in);
    let mut dst = vec![0.0_f32; src.len()];
    for r in 0..n_out {
        for c in 0..n_in {
            dst[c * n_out + r] = src[r * n_in + c];
        }
    }
    dst
}

mod forward;
mod load;

impl Model {
    /// Re-zeroes the KV caches and rewinds to position 0.
    pub fn reset(&mut self) -> Result<()> {
        let zeros = vec![0.0_f32; (self.cfg.kv_heads * self.cfg.dh * self.cfg.t_max) as usize];
        for layer in &self.layers {
            self.exec.upload(&zeros, &layer.kt_cache)?;
            self.exec.upload(&zeros, &layer.v_cache)?;
        }
        self.pos = 0;
        Ok(())
    }

    /// Drop prepared CBs whose baked addresses belong to scratch /
    /// weights.  Call before those tensors are replaced.
    fn drop_prepared(&mut self) {
        self.prefill_prepared = None;
        self.decode_prepared = None;
        self.decode_unrolled = None;
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Release the erased prepared recording while tensors/exec
        // are still alive (field drop order is not enough).
        self.drop_prepared();
    }
}
