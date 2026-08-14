use std::ffi::c_char;

use crate::handles::ta_tensor;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_run_stats {
    pub has_gpu_time: u32,
    pub gpu_time_ns: u64,
    pub n_calls: usize,
    pub total_flops: u64,
    pub tflops: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_matmul_call {
    pub a: *const ta_tensor,
    pub b: *const ta_tensor,
    pub c: *const ta_tensor,
    pub alpha: f32,
    pub accumulate: u32,
}

/// Fused epilogue description. An all-zero value means "no epilogue".
///
/// `activation`: 0 = none, 1 = relu, 2 = silu, 3 = gelu (tanh approx).
/// `binary_kind`: 0 = none, 1 = add_scaled (`out += beta * D`), 2 = mul
/// (`out *= D`). `bias` may be null (absent); `d` must be non-null iff
/// `binary_kind != 0`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_epilogue {
    pub bias: *const ta_tensor,
    pub activation: u32,
    pub binary_kind: u32,
    pub d: *const ta_tensor,
    pub beta: f32,
}

/// One GEMM plus its fused epilogue.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_matmul_op {
    pub call: ta_matmul_call,
    pub epilogue: ta_epilogue,
}

/// Rotary-embedding geometry for `ta_rope`. `rot_dim` is the rotated
/// lane count per head vector (even, `>= 2`, `<= head_dim`); lanes
/// past it pass through (partial rotary). `pos_base` is the absolute
/// position of the first token in the input.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_rope_desc {
    pub heads: u32,
    pub head_dim: u32,
    pub rot_dim: u32,
    pub pos_base: u32,
}

/// Strided-copy geometry for `ta_copy_strided`. Strides and offsets
/// are in elements (f32 lanes), not bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_copy_desc {
    pub extent: [u32; 3],
    pub src_offset: u32,
    pub src_strides: [u32; 3],
    pub dst_offset: u32,
    pub dst_strides: [u32; 3],
}

/// Route description for one matmul shape. `kernel` points to an
/// interned NUL-terminated name valid for the process lifetime.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ta_dispatch_info {
    pub kernel: *const c_char,
    pub tile: [u32; 3],
    pub has_split_k2: u32,
    pub split_k2_splits: u32,
}
