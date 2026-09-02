//! GLSL push-constant mirrors for the elementwise kernel family.

/// Push-constant budget of the shared elementwise layout: the largest
/// block (`CopyPc` / `RopeScatterPc`, 72 bytes) rounded up for slack;
/// well under the 128-byte device minimum.
pub(in crate::executor::elementwise) const ELEMENTWISE_PC_BYTES: usize = 80;

macro_rules! pc_block {
    ($(
        $(#[$meta:meta])*
        $name:ident { $($field:ident: $ty:ty),+ $(,)? }
    )+) => {
        $(
            $(#[$meta])*
            #[repr(C)]
            #[derive(Copy, Clone)]
            pub(in crate::executor::elementwise) struct $name {
                $(pub $field: $ty),+
            }
            unsafe impl bytemuck::Pod for $name {}
            unsafe impl bytemuck::Zeroable for $name {}
        )+
    };
}

pc_block! {
    SoftmaxPc {
        rows: u32,
        cols: u32,
        valid_base: u32,
        rows_per_group: u32,
        causal: u32,
        scale: f32,
        in_ptr: u64,
        out_ptr: u64,
    }
    NormPc {
        rows: u32,
        cols: u32,
        eps: f32,
        _pad: u32,
        in_ptr: u64,
        out_ptr: u64,
        weight_ptr: u64,
        bias_ptr: u64,
    }
    RopePc {
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
    RopeScatterPc {
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
    CopyPc {
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
    BinaryPc {
        n: u32,
        mode: u32,
        beta: f32,
        _pad: u32,
        a_ptr: u64,
        b_ptr: u64,
        out_ptr: u64,
    }
    FlashPc {
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
    PrefillQkvPackPc {
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
    AttnDecodePc {
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
    ArgmaxPc {
        n: u32,
        _pad: u32,
        in_ptr: u64,
        result_ptr: u64,
    }
    EmbedGatherPc {
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
    GemvChainPc {
        jobs_ptr: u64,
        sync_ptr: u64,
        n_jobs: u32,
        n_wg: u32,
    }
    /// Must stay 80 bytes / 16-aligned to match the GLSL `GemvJob`.
    GemvJob {
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
    AttnCombinePc {
        num_chunks: u32,
        group: u32,
        dh: u32,
        _pad0: u32,
        scratch_ptr: u64,
        out_ptr: u64,
    }
}

const _: () = assert!(std::mem::size_of::<GemvJob>() == 80);
const _: () = assert!(std::mem::size_of::<PrefillQkvPackPc>() == 80);
const _: () = assert!(std::mem::size_of::<PrefillQkvPackPc>() <= ELEMENTWISE_PC_BYTES);
