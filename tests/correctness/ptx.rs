//! PTX path (`VK_NV_cuda_kernel_launch`, see `src/ptx`).  Every test
//! skips unless the context was created with `ML_PTX=1` on a driver that
//! exposes the extension:
//!
//!     ML_PTX=1 cargo test --release --test correctness -- --ignored --test-threads=1 ptx::
//!
//! The integrated a16 route at the f32 accumulator policy (bit-identical
//! to coopmat1, so every a16 tolerance holds) is also covered by running
//! the existing suites with the route serving f32:
//!
//!     ML_PTX=1 ML_PTX_GEMM=all cargo test --release --test correctness -- --ignored --test-threads=1 f16:: graph:: prepared:: replay::
use std::sync::Arc;

use ash::vk;
use tensor_ash::dtype::{f16_bits_to_f32, f32_to_f16_bits};
use tensor_ash::ptx::gemm::PtxGemm;
use tensor_ash::ptx::{CudaModule, KernelArg, ProbeAffine, PtxGemmMode, kernels};
use tensor_ash::{
    BinaryOp, CoopmatAcc, ExecOp, Executor, ExecutorConfig, MatmulCall, MatmulOp, MatmulPipeline,
    Tensor, VulkanContext,
};

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

/// A PTX-capable context plus its (auto-routed) pipeline, or `None`
/// (with a note) when the route is unavailable.
fn ptx_pipeline() -> Option<(Arc<VulkanContext>, Arc<MatmulPipeline>)> {
    let ctx = VulkanContext::new(false).expect("Vulkan init");
    if skip_without_ptx(&ctx) || !ctx.coopmat_enabled || !ctx.f16_storage_enabled {
        return None;
    }
    let pipe = Arc::new(MatmulPipeline::new(&ctx).expect("pipeline"));
    if pipe.ptx_gemm_mode() == PtxGemmMode::Off {
        eprintln!("skipping: PTX route off (ML_PTX_GEMM=off?)");
        return None;
    }
    Some((ctx, pipe))
}

fn executor(ctx: &Arc<VulkanContext>, pipe: &Arc<MatmulPipeline>, acc: CoopmatAcc) -> Executor {
    Executor::new_with_config(
        ctx.clone(),
        pipe.clone(),
        ExecutorConfig {
            n_slots: 2,
            max_calls_per_submit: 16,
            tune: false,
            coopmat_acc: acc,
        },
    )
    .expect("executor")
}

/// f16 tensor of `shape` filled with `scale * U(-1, 1)` (seeded).
fn f16_tensor(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    shape: &[u32],
    seed: u64,
    scale: f32,
) -> Tensor {
    let mut host = vec![0f32; Tensor::numel(shape) as usize];
    fill_det(&mut host, seed);
    for v in &mut host {
        *v *= scale;
    }
    let t = Tensor::uninit_device_f16(ctx, shape).unwrap();
    exec.upload(&host, &t).unwrap();
    t
}

fn read(exec: &Executor, t: &Tensor) -> Vec<f32> {
    let mut out = vec![0f32; t.len() as usize];
    exec.download(t, &mut out).unwrap();
    out
}

/// f16 -> f32 widening is exact and injective, so comparing the f32
/// bit patterns compares the stored f16 bits.
fn assert_bitwise(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let diff = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    let first = got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits());
    assert_eq!(
        diff,
        0,
        "{label}: {diff} of {} elements differ bitwise (first at {first:?})",
        got.len()
    );
}

fn rel_rms(got: &[f32], reference: &[f32]) -> f64 {
    let (mut se, mut sr) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        se += (*g as f64 - *r as f64).powi(2);
        sr += (*r as f64).powi(2);
    }
    (se / sr.max(f64::MIN_POSITIVE)).sqrt()
}

/// The accumulator policy degrades loudly, never silently: without a
/// route serving a class, an executor asking for it runs (and reports)
/// f32; with one, it keeps the class.
#[test]
#[ignore]
fn ptx_policy_is_explicit_and_degrades_to_f32() {
    let ctx = VulkanContext::new(false).expect("Vulkan init");
    let pipe = Arc::new(MatmulPipeline::new(&ctx).expect("pipeline"));
    if ctx.cuda_launch.is_none() {
        // Plain context: no f16-accumulator implementation on this branch.
        for acc in [CoopmatAcc::F16, CoopmatAcc::F16Promote] {
            assert!(!pipe.ptx_serves(acc));
            assert_eq!(executor(&ctx, &pipe, acc).coopmat_acc(), CoopmatAcc::F32);
            assert!(pipe.set_ptx_gemm_mode(PtxGemmMode::All).is_err());
        }
        return;
    }
    if pipe.ptx_gemm_mode() == PtxGemmMode::Off {
        eprintln!("skipping: PTX route off (ML_PTX_GEMM=off?)");
        return;
    }
    assert_eq!(
        executor(&ctx, &pipe, CoopmatAcc::F32).coopmat_acc(),
        CoopmatAcc::F32
    );
    pipe.set_ptx_gemm_mode(PtxGemmMode::Off).unwrap();
    assert_eq!(
        executor(&ctx, &pipe, CoopmatAcc::F16).coopmat_acc(),
        CoopmatAcc::F32
    );
    pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
    for acc in [CoopmatAcc::F16, CoopmatAcc::F16Promote] {
        assert_eq!(executor(&ctx, &pipe, acc).coopmat_acc(), acc);
        let name = pipe.ptx_a16_kernel(acc, 512, 2048, 2048).expect("served");
        assert!(
            name.contains(if acc == CoopmatAcc::F16 {
                "_f16_"
            } else {
                "_f16p_"
            }),
            "{acc:?} -> {name}"
        );
    }
    // Auto keeps f32 on coopmat1; All hands it to an f32-accumulate kernel.
    assert_eq!(pipe.ptx_a16_kernel(CoopmatAcc::F32, 512, 2048, 2048), None);
    pipe.set_ptx_gemm_mode(PtxGemmMode::All).unwrap();
    let name = pipe
        .ptx_a16_kernel(CoopmatAcc::F32, 512, 2048, 2048)
        .expect("served");
    assert!(name.contains("_f32_"), "{name}");
    pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
}

/// At the f32 accumulator policy the PTX kernels are bit-identical to
/// coopmat1: same HMMA.16816.F32 k-order, same alpha scale and f16
/// rounding of C.  Single ops, the graph path and prepared replay.
#[test]
#[ignore]
fn ptx_f32_route_is_bitwise_coopmat1() {
    let Some((ctx, pipe)) = ptx_pipeline() else {
        return;
    };
    let exec = executor(&ctx, &pipe, CoopmatAcc::F32);
    // (M, N, K, alpha): the 64x64 and 128x128 PTX picks (4060: 24 SMs,
    // 3070: 46), a 64-aligned-only shape (coopmat m64n64 tile), a
    // grouped-raster-sized grid, and a non-unit alpha.
    let cases: &[(u32, u32, u32, f32)] = &[
        (128, 256, 128, 1.0),
        (192, 320, 64, 1.0),
        (512, 2048, 2048, 1.0),
        (512, 5632, 2048, 1.0),
        (512, 2048, 5632, 1.0),
        (2048, 2048, 512, 1.0),
        (4096, 4096, 256, 1.0),
        (256, 384, 256, 0.75),
    ];
    for (i, &(m, n, k, alpha)) in cases.iter().enumerate() {
        let a = f16_tensor(&ctx, &exec, &[m, k], 7100 + i as u64, 1.0);
        let b = f16_tensor(
            &ctx,
            &exec,
            &[k, n],
            7200 + i as u64,
            2.0 / (k as f32).sqrt(),
        );
        let c = Tensor::uninit_device_f16(&ctx, &[m, n]).unwrap();
        let call = MatmulCall {
            a: &a,
            b: &b,
            c: &c,
            alpha,
            accumulate: false,
        };
        let label = format!("{m}x{n}x{k} alpha={alpha}");

        pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
        assert_eq!(pipe.ptx_a16_kernel(CoopmatAcc::F32, m, n, k), None);
        exec.run_matmuls(&[call]).unwrap();
        let coopmat = read(&exec, &c);
        assert!(coopmat.iter().all(|v| v.is_finite()), "{label}: non-finite");

        pipe.set_ptx_gemm_mode(PtxGemmMode::All).unwrap();
        let kernel = pipe
            .ptx_a16_kernel(CoopmatAcc::F32, m, n, k)
            .expect("served");
        exec.run_matmuls(&[call]).unwrap();
        assert_bitwise(
            &read(&exec, &c),
            &coopmat,
            &format!("{label} single {kernel}"),
        );

        exec.run_exec_ops(&[ExecOp::Matmul(MatmulOp::new(call))])
            .unwrap();
        assert_bitwise(
            &read(&exec, &c),
            &coopmat,
            &format!("{label} graph {kernel}"),
        );

        let mut prepared = exec.prepare_matmuls(&[call]).unwrap();
        for replay in 0..2 {
            prepared.run().unwrap();
            assert_bitwise(
                &read(&exec, &c),
                &coopmat,
                &format!("{label} prepared replay {replay} {kernel}"),
            );
        }
    }
    pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
}

fn mm<'t>(a: &'t Tensor, b: &'t Tensor, c: &'t Tensor) -> ExecOp<'t> {
    ExecOp::Matmul(MatmulOp::new(MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }))
}

/// A mixed Vulkan/CUDA graph with every hazard kind between PTX
/// launches and Vulkan dispatches (RAW both ways, WAR, WAW, and two
/// independent launches the hazard cursor lets overlap).  At the f32
/// policy it must be bit-identical to the all-Vulkan graph, immediate
/// and prepared; at f16 / f16-promote it must be deterministic across
/// immediate and prepared recording and carry the documented error.
#[test]
#[ignore]
fn ptx_route_in_mixed_graphs_and_prepared_replay() {
    let Some((ctx, pipe)) = ptx_pipeline() else {
        return;
    };
    let (m, d) = (512u32, 2048u32);
    let setup = executor(&ctx, &pipe, CoopmatAcc::F32);
    let w = |seed| f16_tensor(&ctx, &setup, &[d, d], seed, 2.0 / (d as f32).sqrt());
    let (w1, w2, w3, w4) = (w(8101), w(8102), w(8103), w(8104));
    let x = f16_tensor(&ctx, &setup, &[m, d], 8100, 1.0);
    let r0 = f16_tensor(&ctx, &setup, &[m, d], 8105, 1.0);
    let t = || Tensor::uninit_device_f16(&ctx, &[m, d]).unwrap();
    let (h, g, u, y, y2, z, q) = (t(), t(), t(), t(), t(), t(), t());
    let run_graph = |exec: &Executor, prepared: bool| -> Vec<Vec<f32>> {
        let ops = [
            mm(&x, &w1, &h), // CUDA
            mm(&x, &w2, &g), // CUDA, independent of the previous launch
            ExecOp::Binary {
                a: &h,
                b: &g,
                out: &u,
                op: BinaryOp::SiluMul,
            }, // Vulkan reads both launches (RAW CUDA->Vulkan)
            mm(&u, &w3, &y), // RAW Vulkan->CUDA
            ExecOp::Binary {
                a: &y,
                b: &r0,
                out: &y2,
                op: BinaryOp::AddScaled { beta: 1.0 },
            },
            mm(&y2, &w4, &z),
            ExecOp::Binary {
                a: &z,
                b: &h,
                out: &g,
                op: BinaryOp::AddScaled { beta: 0.5 },
            }, // WAR/WAW on g (read by the SiluMul, written by a launch)
            mm(&g, &w1, &q), // RAW on the Vulkan-written g
            mm(&q, &w2, &h), // WAR/WAW on h
        ];
        if prepared {
            let mut p = exec.prepare_exec_ops(&ops, None).unwrap();
            p.run().unwrap();
            p.run().unwrap();
        } else {
            exec.run_exec_ops(&ops).unwrap();
        }
        [&h, &g, &u, &y2, &z, &q]
            .iter()
            .map(|t| read(exec, t))
            .collect()
    };
    let compare = |got: &[Vec<f32>], want: &[Vec<f32>], label: &str| {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_bitwise(g, w, &format!("{label} output {i}"));
        }
    };

    // f32 policy: all-Vulkan reference, then the same graph with the
    // GEMMs on PTX.
    pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
    let exec32 = executor(&ctx, &pipe, CoopmatAcc::F32);
    let reference = run_graph(&exec32, false);
    assert!(
        reference.iter().flatten().all(|v| v.is_finite()),
        "non-finite reference"
    );
    compare(
        &run_graph(&exec32, true),
        &reference,
        "f32 coopmat prepared",
    );
    pipe.set_ptx_gemm_mode(PtxGemmMode::All).unwrap();
    compare(&run_graph(&exec32, false), &reference, "f32 PTX graph");
    compare(&run_graph(&exec32, true), &reference, "f32 PTX prepared");

    // Reduced-precision classes (served in the default mode).
    pipe.set_ptx_gemm_mode(PtxGemmMode::Auto).unwrap();
    let mut errs = Vec::new();
    for (acc, gate) in [(CoopmatAcc::F16Promote, 5e-3), (CoopmatAcc::F16, 3e-2)] {
        let exec = executor(&ctx, &pipe, acc);
        assert_eq!(exec.coopmat_acc(), acc);
        let immediate = run_graph(&exec, false);
        compare(
            &run_graph(&exec, true),
            &immediate,
            &format!("{acc:?} prepared"),
        );
        compare(
            &run_graph(&exec, false),
            &immediate,
            &format!("{acc:?} rerun"),
        );
        let err = immediate
            .iter()
            .zip(&reference)
            .map(|(g, r)| rel_rms(g, r))
            .fold(0f64, f64::max);
        eprintln!("{acc:?}: worst rel-RMS vs the f32 graph {err:.3e}");
        assert!(err <= gate, "{acc:?}: rel-RMS {err:.3e} over {gate:.1e}");
        errs.push(err);
    }
    // Neither class silently ran f32 accumulation, and promotion helps.
    assert!(errs[0] > 0.0 && errs[1] > errs[0], "errors {errs:?}");
}
