//! Shared test helpers.
use std::sync::Arc;

use tensor_ash::{Executor, KernelSelection, MatmulPipeline, Tensor, VulkanContext};

pub fn make_setup(n_slots: usize, max_calls: u32) -> (Arc<VulkanContext>, Executor) {
    let ctx = VulkanContext::new(false).expect("Vulkan init");
    let pipe = Arc::new(MatmulPipeline::new(&ctx).expect("pipeline"));
    let exec = Executor::new(ctx.clone(), pipe, n_slots, max_calls).expect("executor");
    (ctx, exec)
}

pub fn make_setup_with_kernel(
    n_slots: usize,
    max_calls: u32,
    selection: KernelSelection,
) -> (Arc<VulkanContext>, Executor) {
    let ctx = VulkanContext::new(false).expect("Vulkan init");
    let pipe =
        Arc::new(MatmulPipeline::new_with_kernel_selection(&ctx, selection).expect("pipeline"));
    let exec = Executor::new(ctx.clone(), pipe, n_slots, max_calls).expect("executor");
    (ctx, exec)
}

/// Deterministic pseudo-random fill in [-1, 1).
pub fn fill_det(out: &mut [f32], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    for v in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (s >> 40) as u32;
        *v = (bits as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0;
    }
}

/// Reference batched matmul on CPU with `alpha` and optional accumulate.
#[allow(clippy::too_many_arguments)]
pub fn cpu_bmm(
    a: &[f32],
    b: &[f32],
    c_init: Option<&[f32]>,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    accumulate: bool,
) -> Vec<f32> {
    let m_ = m as usize;
    let n_ = n as usize;
    let k_ = k as usize;
    let mk = m_ * k_;
    let kn = k_ * n_;
    let mn = m_ * n_;
    let mut out = vec![0.0f32; batch as usize * mn];
    if let Some(c0) = c_init {
        out.copy_from_slice(c0);
    }
    for bi in 0..batch as usize {
        let a = &a[bi * mk..(bi + 1) * mk];
        let b = &b[bi * kn..(bi + 1) * kn];
        let c = &mut out[bi * mn..(bi + 1) * mn];
        for i in 0..m_ {
            for j in 0..n_ {
                let mut acc = 0.0f64;
                for kk in 0..k_ {
                    acc += (a[i * k_ + kk] as f64) * (b[kk * n_ + j] as f64);
                }
                let scaled = alpha * (acc as f32);
                c[i * n_ + j] = if accumulate {
                    c[i * n_ + j] + scaled
                } else {
                    scaled
                };
            }
        }
    }
    out
}

/// Per-shape tolerance for f32 sums of length K.
pub fn tolerance(k: u32) -> f32 {
    8.0 * (k as f32) * f32::EPSILON
}

pub fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > m {
            m = e;
            idx = i;
        }
    }
    (m, idx)
}

/// Convenience: run one matmul and produce (gpu_result, cpu_reference).
#[allow(clippy::too_many_arguments)]
pub fn run_one(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    shape_a: &[u32],
    shape_b: &[u32],
    shape_c: &[u32],
    alpha: f32,
    accumulate: bool,
    seed_a: u64,
    seed_b: u64,
    initial_c: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    use tensor_ash::MatmulCall;
    let a = Tensor::zeros_device(ctx, shape_a).unwrap();
    let b = Tensor::zeros_device(ctx, shape_b).unwrap();
    let c = Tensor::zeros_device(ctx, shape_c).unwrap();

    let mut h_a = vec![0.0f32; Tensor::numel(shape_a) as usize];
    let mut h_b = vec![0.0f32; Tensor::numel(shape_b) as usize];
    fill_det(&mut h_a, seed_a);
    fill_det(&mut h_b, seed_b);
    exec.upload(&h_a, &a).unwrap();
    exec.upload(&h_b, &b).unwrap();
    if let Some(c0) = initial_c {
        exec.upload(c0, &c).unwrap();
    }

    exec.run_matmuls(&[MatmulCall {
        a: &a,
        b: &b,
        c: &c,
        alpha,
        accumulate,
    }])
    .unwrap();

    let mut h_c = vec![0.0f32; Tensor::numel(shape_c) as usize];
    exec.download(&c, &mut h_c).unwrap();

    let (ba, m, k) = match shape_a.len() {
        2 => (1u32, shape_a[0], shape_a[1]),
        3 => (shape_a[0], shape_a[1], shape_a[2]),
        _ => unreachable!(),
    };
    let (bb, _, n) = match shape_b.len() {
        2 => (1u32, shape_b[0], shape_b[1]),
        3 => (shape_b[0], shape_b[1], shape_b[2]),
        _ => unreachable!(),
    };
    let batch = match shape_c.len() {
        2 => 1u32,
        3 => shape_c[0],
        _ => unreachable!(),
    };
    let h_a_x = if ba == 1 && batch > 1 {
        let mut v = Vec::with_capacity(batch as usize * h_a.len());
        for _ in 0..batch {
            v.extend_from_slice(&h_a);
        }
        v
    } else {
        h_a
    };
    let h_b_x = if bb == 1 && batch > 1 {
        let mut v = Vec::with_capacity(batch as usize * h_b.len());
        for _ in 0..batch {
            v.extend_from_slice(&h_b);
        }
        v
    } else {
        h_b
    };
    let ref_c = cpu_bmm(&h_a_x, &h_b_x, initial_c, batch, m, n, k, alpha, accumulate);
    (h_c, ref_c)
}
