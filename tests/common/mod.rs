//! Shared test helpers.
use std::sync::Arc;

use tensor_ash::{
    Executor, ExecutorConfig, KernelSelection, MatmulPipeline, Tensor, VulkanContext,
};

pub use tensor_ash_test_support::{LcgRng, cpu_bmm, fill_det, max_abs_err, tolerance};

/// `ML_VALIDATE=1` opts the test suite into the Vulkan validation
/// layers.  Off by default so the suite is fast on CI machines that
/// don't have validation layers installed.
fn validate_from_env() -> bool {
    std::env::var("ML_VALIDATE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Tests construct executors with tuning explicitly disabled so a suite
/// run under `ML_TUNE=1` cannot persist winners for the odd shapes whose
/// heuristic routes the tests assert.  Explicit `tune_shape` calls still
/// work — only the implicit first-use tuner is off.
fn executor_config(n_slots: usize, max_calls: u32) -> ExecutorConfig {
    ExecutorConfig {
        n_slots,
        max_calls_per_submit: max_calls,
        tune: false,
    }
}

pub fn make_setup(n_slots: usize, max_calls: u32) -> (Arc<VulkanContext>, Executor) {
    let ctx = VulkanContext::new(validate_from_env()).expect("Vulkan init");
    let pipe = Arc::new(MatmulPipeline::new(&ctx).expect("pipeline"));
    let exec = Executor::new_with_config(ctx.clone(), pipe, executor_config(n_slots, max_calls))
        .expect("executor");
    (ctx, exec)
}

pub fn make_setup_with_kernel(
    n_slots: usize,
    max_calls: u32,
    selection: KernelSelection,
) -> (Arc<VulkanContext>, Executor) {
    let ctx = VulkanContext::new(validate_from_env()).expect("Vulkan init");
    let pipe =
        Arc::new(MatmulPipeline::new_with_kernel_selection(&ctx, selection).expect("pipeline"));
    let exec = Executor::new_with_config(ctx.clone(), pipe, executor_config(n_slots, max_calls))
        .expect("executor");
    (ctx, exec)
}

/// Convenience: run one matmul and produce (gpu_result, cpu_reference).
///
/// Both inputs are filled deterministically from `seed_a`/`seed_b`.  Pass
/// `initial_c` when testing accumulate=true so the CPU reference sees
/// the same starting value as the GPU buffer.
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
    let a = Tensor::uninit_device(ctx, shape_a).unwrap();
    let b = Tensor::uninit_device(ctx, shape_b).unwrap();
    let c = Tensor::uninit_device(ctx, shape_c).unwrap();

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

    let (_, m, k) = match shape_a.len() {
        2 => (1u32, shape_a[0], shape_a[1]),
        3 => (shape_a[0], shape_a[1], shape_a[2]),
        _ => unreachable!(),
    };
    let (_, _, n) = match shape_b.len() {
        2 => (1u32, shape_b[0], shape_b[1]),
        3 => (shape_b[0], shape_b[1], shape_b[2]),
        _ => unreachable!(),
    };
    let batch = match shape_c.len() {
        2 => 1u32,
        3 => shape_c[0],
        _ => unreachable!(),
    };

    // cpu_bmm handles broadcasting natively: pass per-batch or
    // single-batch buffers directly and it picks the right slice.
    let ref_c = cpu_bmm(&h_a, &h_b, initial_c, batch, m, n, k, alpha, accumulate);
    (h_c, ref_c)
}
