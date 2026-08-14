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
