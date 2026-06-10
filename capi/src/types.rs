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
