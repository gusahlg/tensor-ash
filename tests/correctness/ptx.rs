//! Opt-in PTX path (`VK_NV_cuda_kernel_launch`, see `src/ptx`).  Every
//! test skips unless the context was created with `ML_PTX=1` on a
//! driver that exposes the extension:
//!
//!     ML_PTX=1 cargo test --release --test correctness -- --ignored --test-threads=1 ptx::
//!
//! The integrated a16 route is covered by running the existing a16
//! suite with the route on (f32 accumulation keeps its tolerances):
//!
//!     ML_PTX=1 ML_PTX_GEMM=f32 cargo test --release --test correctness -- --ignored --test-threads=1 f16::a16
use std::sync::Arc;

use ash::vk;
use tensor_ash::dtype::{f16_bits_to_f32, f32_to_f16_bits};
use tensor_ash::ptx::gemm::PtxGemm;
use tensor_ash::ptx::{CudaModule, KernelArg, ProbeAffine, kernels};
use tensor_ash::{Tensor, VulkanContext};

use crate::common::{fill_det, make_setup};

fn skip_without_ptx(ctx: &VulkanContext) -> bool {
    if ctx.cuda_launch.is_none() {
        eprintln!("skipping: needs ML_PTX=1 and VK_NV_cuda_kernel_launch");
        return true;
    }
    false
}

/// Record `record`, submit, wait (30 s guard).
fn run_cb(ctx: &Arc<VulkanContext>, record: impl FnOnce(vk::CommandBuffer)) {
    unsafe {
        let dev = &ctx.device;
        let pool = dev
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.compute_family),
                None,
            )
            .unwrap();
        let cb = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        dev.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .unwrap();
        record(cb);
        compute_barrier(ctx, cb);
        dev.end_command_buffer(cb).unwrap();
        let fence = dev
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let cbs = [cb];
        {
            let q = ctx.queue.lock();
            dev.queue_submit(
                *q,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )
            .unwrap();
        }
        dev.wait_for_fences(&[fence], true, 30_000_000_000).unwrap();
        dev.destroy_fence(fence, None);
        dev.destroy_command_pool(pool, None);
    }
}

fn compute_barrier(ctx: &VulkanContext, cb: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

fn addr(ctx: &VulkanContext, t: &Tensor) -> u64 {
    ctx.buffer_device_address(t.raw_buffer())
}

/// Alternating Vulkan (reverse-walking) / CUDA (forward-walking) affine
/// steps over two ping-pong buffers, compute barriers in between: the
/// final buffer must equal the CPU composition exactly.
#[test]
#[ignore]
fn ptx_launches_order_with_vulkan_dispatches() {
    let (ctx, exec) = make_setup(2, 8);
    if skip_without_ptx(&ctx) {
        return;
    }
    let n = 1u32 << 20;
    let module = CudaModule::from_ptx(&ctx, kernels::PROBE_PTX).unwrap();
    let cu = module.function("probe_affine").unwrap();
    let vk_affine = ProbeAffine::new(&ctx).unwrap();
    let seed: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(2654435761) ^ 0x5bd1e995)
        .collect();
    let x = Tensor::uninit_device(&ctx, &[n]).unwrap();
    let y = Tensor::uninit_device(&ctx, &[n]).unwrap();
    let as_f32: Vec<f32> = seed.iter().map(|&v| f32::from_bits(v)).collect();
    exec.upload(&as_f32, &x).unwrap();
    let (xa, ya) = (addr(&ctx, &x), addr(&ctx, &y));
    let steps = 8u32;
    let groups = n / 256;
    run_cb(&ctx, |cb| {
        for s in 0..steps {
            if s > 0 {
                compute_barrier(&ctx, cb);
            }
            let (src, dst) = if s % 2 == 0 { (xa, ya) } else { (ya, xa) };
            let (mul, add) = (2 * s + 3, s * 977 + 1);
            unsafe {
                if s % 2 == 0 {
                    vk_affine.cmd_dispatch(cb, src, dst, n, mul, add, groups);
                } else {
                    cu.cmd_launch(
                        cb,
                        [groups, 1, 1],
                        [256, 1, 1],
                        0,
                        &[
                            KernelArg::U64(src),
                            KernelArg::U64(dst),
                            KernelArg::U32(n),
                            KernelArg::U32(mul),
                            KernelArg::U32(add),
                        ],
                    );
                }
            }
        }
    });
    let mut out = vec![0f32; n as usize];
    exec.download(&x, &mut out).unwrap();
    let bad = out
        .iter()
        .zip(&seed)
        .filter(|(g, s0)| {
            let want = (0..steps).fold(**s0, |v, s| {
                v.wrapping_mul(2 * s + 3).wrapping_add(s * 977 + 1)
            });
            g.to_bits() != want
        })
        .count();
    assert_eq!(
        bad, 0,
        "{bad} of {n} elements saw stale data across Vulkan<->CUDA barriers"
    );
}

/// Every hgemm entry (plain and split-K) against an f64 reference on an
/// edge-exercising shape: f32-accumulate kernels must meet the a16
/// coopmat error class, f16/f16p kernels their documented classes.
#[test]
#[ignore]
fn ptx_hgemm_kernels_match_reference() {
    let (ctx, exec) = make_setup(2, 8);
    if skip_without_ptx(&ctx) {
        return;
    }
    let gemm = PtxGemm::new(&ctx).unwrap();
    // M, N not multiples of 128 (edge tiles), K a multiple of 8 but not of 32.
    let (m, n, k) = (200u32, 328u32, 488u32);
    let mut ha = vec![0f32; (m * k) as usize];
    let mut hb = vec![0f32; (k * n) as usize];
    fill_det(&mut ha, 41);
    fill_det(&mut hb, 43);
    let a16: Vec<u16> = ha.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let b16: Vec<u16> = hb.iter().map(|&v| f32_to_f16_bits(v * 0.1)).collect();
    let a = Tensor::uninit_device_f16(&ctx, &[m, k]).unwrap();
    let b = Tensor::uninit_device_f16(&ctx, &[k, n]).unwrap();
    exec.upload_f16(&a16, &a).unwrap();
    exec.upload_f16(&b16, &b).unwrap();
    let (fa, fb): (Vec<f64>, Vec<f64>) = (
        a16.iter().map(|&h| f16_bits_to_f32(h) as f64).collect(),
        b16.iter().map(|&h| f16_bits_to_f32(h) as f64).collect(),
    );
    let mut reference = vec![0f64; (m * n) as usize];
    for r in 0..m as usize {
        for kk in 0..k as usize {
            let av = fa[r * k as usize + kk];
            for c in 0..n as usize {
                reference[r * n as usize + c] += av * fb[kk * n as usize + c];
            }
        }
    }
    let rms_ref = (reference.iter().map(|v| v * v).sum::<f64>() / reference.len() as f64).sqrt();
    let c16 = Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap();
    let c32 = Tensor::uninit_device(&ctx, &[m, n]).unwrap();
    let w = Tensor::uninit_device(&ctx, &[4, m, n]).unwrap();
    let (aa, ba, wa) = (addr(&ctx, &a), addr(&ctx, &b), addr(&ctx, &w));
    let rel_rms = |got: &[f32]| {
        let se: f64 = got
            .iter()
            .zip(&reference)
            .map(|(g, r)| (*g as f64 - r).powi(2))
            .sum();
        (se / got.len() as f64).sqrt() / rms_ref
    };
    let gate = |f16_acc: bool, promoted: bool| {
        if f16_acc && !promoted {
            1e-2
        } else if promoted {
            2e-3
        } else {
            1e-3
        }
    };
    let mut checked = 0;
    for (index, (kern, _)) in gemm.hgemm.iter().enumerate() {
        let c = if kern.out_f16 { &c16 } else { &c32 };
        let ca = addr(&ctx, c);
        run_cb(&ctx, |cb| unsafe {
            gemm.cmd_hgemm(cb, index, aa, ba, ca, (m, n, k, 1), 1.0)
        });
        let mut got = vec![0f32; (m * n) as usize];
        exec.download(c, &mut got).unwrap();
        let err = rel_rms(&got);
        assert!(
            err <= gate(kern.f16_acc, kern.promoted()),
            "{}: rel-RMS {err:.3e} over its gate",
            kern.name
        );
        checked += 1;
    }
    for (index, (kern, _)) in gemm.hgemm_splitk.iter().enumerate() {
        for splits in [2u32, 3] {
            run_cb(&ctx, |cb| unsafe {
                gemm.cmd_hgemm_splitk(
                    cb,
                    index,
                    aa,
                    ba,
                    wa,
                    addr(&ctx, &c16),
                    (m, n, k),
                    splits,
                    1.0,
                )
            });
            let mut got = vec![0f32; (m * n) as usize];
            exec.download(&c16, &mut got).unwrap();
            let err = rel_rms(&got);
            assert!(
                err <= gate(kern.f16_acc, kern.promoted()),
                "{}/{splits}: rel-RMS {err:.3e} over its gate",
                kern.name
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} kernels checked");
}
