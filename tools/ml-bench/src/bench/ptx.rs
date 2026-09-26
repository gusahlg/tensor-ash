//! `ml_bench ptx <mode>` — VK_NV_cuda_kernel_launch campaign harness.
//!
//! Requires `ML_PTX=1` (the context only enables the extension on
//! request).  Modes:
//!
//! * `probe`    — extension/feature report, PTX JIT, launch into a
//!   tensor-ash buffer via buffer-device-address params, module cache
//!   round-trip, cubin/fatbin acceptance (`ML_PTX_CUBINS=a,b,...`)
//!   through both VK_NV_cuda_kernel_launch and VK_NVX_binary_import.
//! * `smem`     — dynamic shared memory above 48 KiB.
//! * `sync`     — Vulkan<->CUDA ordering under several barrier kinds.
//! * `state`    — does a CUDA launch clobber bound Vulkan compute state?
//! * `dump`     — write the driver-JIT cubins (vkGetCudaModuleCacheNV)
//!   of every module to `ML_PTX_DUMP_DIR` for nvdisasm.
//! * `overhead` — launch cost vs a Vulkan dispatch; launch overlap.
//! * `sgemm [MxNxK ...]` — PTX f32 SGEMM vs tensor-ash's f32 route.
//! * `hgemm [MxNxK ...]` — PTX mma.sync GEMM vs tensor-ash a16 coopmat.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use tensor_ash::dtype::{f16_bits_to_f32, f32_to_f16_bits};
use tensor_ash::ptx::gemm::PtxGemm;
use tensor_ash::ptx::{CudaFunction, CudaModule, KernelArg, NvxModule, ProbeAffine, kernels};
use tensor_ash::{Executor, MatmulCall, Tensor, VulkanContext};
use tensor_ash_test_support::fill_det;

use super::env::{env_string, env_u32};

pub(super) fn run(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    mut args: impl Iterator<Item = String>,
) -> Result<()> {
    ensure!(
        ctx.cuda_launch.is_some(),
        "VK_NV_cuda_kernel_launch not enabled: run with ML_PTX=1 on the NVIDIA proprietary driver"
    );
    let mode = args.next().unwrap_or_else(|| "probe".into());
    let rest: Vec<String> = args.collect();
    match mode.as_str() {
        "probe" => probe(ctx, exec),
        "smem" => smem(ctx, exec),
        "sync" => sync(ctx, exec),
        "state" => state(ctx, exec),
        "dump" => dump(ctx),
        "overhead" => overhead(ctx, exec),
        "sgemm" => sgemm(ctx, exec, &rest),
        "hgemm" => hgemm(ctx, exec, &rest),
        _ => bail!("ptx: unknown mode '{mode}' (probe|smem|sync|state|overhead|sgemm|hgemm|dump)"),
    }
}

// ---------------------------------------------------------------------
// Minimal recorder: one command buffer, one fence, a timestamp pool.

struct Rec {
    ctx: Arc<VulkanContext>,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    qpool: vk::QueryPool,
    nq: u32,
}

impl Rec {
    fn new(ctx: &Arc<VulkanContext>, nq: u32) -> Result<Self> {
        unsafe {
            let dev = &ctx.device;
            let pool = dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.compute_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cb = dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let qpool = dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(nq),
                None,
            )?;
            Ok(Self {
                ctx: Arc::clone(ctx),
                pool,
                cb,
                fence,
                qpool,
                nq,
            })
        }
    }

    /// Record with `f`, submit, wait (30 s hang guard).
    fn run(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let dev = &self.ctx.device;
        unsafe {
            dev.reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty())?;
            dev.begin_command_buffer(
                self.cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            dev.cmd_reset_query_pool(self.cb, self.qpool, 0, self.nq);
            f(self.cb);
            barrier(
                dev,
                self.cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );
            dev.end_command_buffer(self.cb)?;
            let cbs = [self.cb];
            let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
            {
                let q = self.ctx.queue.lock();
                dev.queue_submit(*q, &submit, self.fence)
                    .context("queue_submit")?;
            }
            dev.wait_for_fences(&[self.fence], true, 30_000_000_000)
                .context("wait_for_fences (GPU hang or device lost?)")?;
            dev.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    fn ts(&self, cb: vk::CommandBuffer, idx: u32, stage: vk::PipelineStageFlags) {
        unsafe {
            self.ctx
                .device
                .cmd_write_timestamp(cb, stage, self.qpool, idx)
        };
    }

    fn read(&self, n: u32) -> Result<Vec<u64>> {
        let mut v = vec![0u64; n as usize];
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.qpool,
                0,
                &mut v,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(v)
    }

    fn ms(&self, a: u64, b: u64) -> f64 {
        let bits = self.ctx.timestamp_valid_bits;
        let mut d = b.wrapping_sub(a);
        if (1..64).contains(&bits) {
            d &= (1u64 << bits) - 1;
        }
        d as f64 * self.ctx.timestamp_period_ns / 1e6
    }

    /// Record `f` between TOP/BOTTOM timestamps (the executor's
    /// convention) and return the GPU milliseconds.
    fn timed(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<f64> {
        self.run(|cb| {
            self.ts(cb, 0, vk::PipelineStageFlags::TOP_OF_PIPE);
            f(cb);
            self.ts(cb, 1, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        })?;
        let t = self.read(2)?;
        Ok(self.ms(t[0], t[1]))
    }
}

impl Drop for Rec {
    fn drop(&mut self) {
        unsafe {
            let dev = &self.ctx.device;
            let _ = dev.device_wait_idle();
            dev.destroy_query_pool(self.qpool, None);
            dev.destroy_fence(self.fence, None);
            dev.destroy_command_pool(self.pool, None);
        }
    }
}

fn barrier(
    dev: &ash::Device,
    cb: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
    dst: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let mb = [vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)];
    let mbs: &[vk::MemoryBarrier] = if src_access.is_empty() && dst_access.is_empty() {
        &[]
    } else {
        &mb
    };
    unsafe {
        dev.cmd_pipeline_barrier(cb, src, dst, vk::DependencyFlags::empty(), mbs, &[], &[]);
    }
}

fn compute_barrier(dev: &ash::Device, cb: vk::CommandBuffer) {
    barrier(
        dev,
        cb,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
    );
}

fn addr(ctx: &VulkanContext, t: &Tensor) -> u64 {
    ctx.buffer_device_address(t.raw_buffer())
}

fn u32_tensor(ctx: &Arc<VulkanContext>, exec: &Executor, data: &[u32]) -> Result<Tensor> {
    let t = Tensor::uninit_device(ctx, &[data.len() as u32])?;
    exec.upload(bytemuck_f32(data), &t)?;
    Ok(t)
}

fn bytemuck_f32(v: &[u32]) -> &[f32] {
    // u32 and f32 share size/alignment; uploads are byte copies.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<f32>(), v.len()) }
}

fn download_u32(exec: &Executor, t: &Tensor, n: usize) -> Result<Vec<u32>> {
    let mut f = vec![0f32; n];
    exec.download(t, &mut f)?;
    Ok(f.iter().map(|x| x.to_bits()).collect())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        f64::NAN
    } else {
        v[v.len() / 2]
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

// ---------------------------------------------------------------------
// probe

fn fill_check(
    ctx: &Arc<VulkanContext>,
    exec: &Executor,
    rec: &Rec,
    launch: &dyn Fn(vk::CommandBuffer, u64, u32, u32),
    value: u32,
) -> Result<(usize, f64)> {
    let n = 1u32 << 20;
    let t = u32_tensor(ctx, exec, &vec![0u32; n as usize])?;
    let a = addr(ctx, &t);
    let ms = rec.timed(|cb| launch(cb, a, n, value))?;
    let got = download_u32(exec, &t, n as usize)?;
    Ok((got.iter().filter(|&&x| x != value).count(), ms))
}

fn nv_launch(f: &CudaFunction) -> impl Fn(vk::CommandBuffer, u64, u32, u32) + '_ {
    move |cb, a, n, v| unsafe {
        f.cmd_launch(
            cb,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            &[KernelArg::U64(a), KernelArg::U32(n), KernelArg::U32(v)],
        )
    }
}

fn probe(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let rec = Rec::new(ctx, 16)?;
    println!("== ptx probe on {}", ctx.diagnostics());
    println!(
        "cuda_kernel_launch=on compute_capability={:?} nvx_binary_import={}",
        ctx.cuda_compute_capability,
        ctx.nvx_binary_import.is_some()
    );

    let t0 = Instant::now();
    let module = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let jit_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let fill = module.function("probe_fill")?;
    let fn_ms = t0.elapsed().as_secs_f64() * 1e3;
    for name in ["probe_affine", "probe_bump", "probe_smem", "probe_stamp"] {
        module.function(name)?;
    }
    println!(
        "PTX module JIT: {jit_ms:.2} ms ({} B PTX); function lookup {fn_ms:.3} ms",
        kernels::PROBE_PTX.len()
    );
    match module.function("no_such_kernel") {
        Ok(_) => println!("lookup of a missing entry: unexpectedly OK"),
        Err(e) => println!("lookup of a missing entry: error as expected ({e})"),
    }

    let (bad, ms) = fill_check(ctx, exec, &rec, &nv_launch(&fill), 0xC0FF_EE01)?;
    println!(
        "PTX probe_fill into a tensor-ash buffer (BDA param, 1M u32): mismatches={bad} gpu={ms:.4} ms  (param storage freed at record time)"
    );

    // Module cache round trip.
    let cache = module.cache_data()?;
    let head: String = cache.iter().take(16).map(|b| format!("{b:02x}")).collect();
    let dump = env_string("ML_PTX_DUMP", "ptx_probe_module_cache.bin");
    std::fs::write(&dump, &cache).with_context(|| format!("write {dump}"))?;
    println!(
        "vkGetCudaModuleCacheNV: {} bytes, head {head}, written to {dump}",
        cache.len()
    );
    let t0 = Instant::now();
    match CudaModule::from_data(ctx, &cache) {
        Ok(m2) => {
            let ms_create = t0.elapsed().as_secs_f64() * 1e3;
            let f2 = m2.function("probe_fill")?;
            let (bad, _) = fill_check(ctx, exec, &rec, &nv_launch(&f2), 0xCAC4_E002)?;
            println!(
                "module from its own cache blob: create {ms_create:.2} ms, fill mismatches={bad}"
            );
        }
        Err(e) => println!("module from its own cache blob: REJECTED ({e})"),
    }

    // PTX through NVX (expects cubin; check whether PTX works there too).
    if ctx.nvx_binary_import.is_some() {
        let mut ptx = kernels::PROBE_PTX.as_bytes().to_vec();
        ptx.push(0);
        try_nvx(ctx, exec, &rec, "PTX text", &ptx);
        try_nvx(ctx, exec, &rec, "NV module cache blob", &cache);
    }

    // External cubins / fatbins.
    let list = env_string("ML_PTX_CUBINS", "");
    for path in list.split(',').filter(|s| !s.is_empty()) {
        let data = std::fs::read(path).with_context(|| format!("read {path}"))?;
        let head: String = data.iter().take(8).map(|b| format!("{b:02x}")).collect();
        let t0 = Instant::now();
        match CudaModule::from_data(ctx, &data) {
            Ok(m) => {
                let ms_create = t0.elapsed().as_secs_f64() * 1e3;
                match m.function("probe_fill") {
                    Ok(f) => {
                        let (bad, _) = fill_check(ctx, exec, &rec, &nv_launch(&f), 0xB1_0B03)?;
                        println!(
                            "NV  {path} ({} B, head {head}): accepted in {ms_create:.2} ms, fill mismatches={bad}",
                            data.len()
                        );
                    }
                    Err(e) => println!("NV  {path}: module accepted, function lookup failed ({e})"),
                }
            }
            Err(e) => println!("NV  {path} ({} B, head {head}): REJECTED ({e})", data.len()),
        }
        if ctx.nvx_binary_import.is_some() {
            try_nvx(ctx, exec, &rec, path, &data);
        }
    }
    Ok(())
}

fn try_nvx(ctx: &Arc<VulkanContext>, exec: &Executor, rec: &Rec, label: &str, data: &[u8]) {
    let res = (|| -> Result<(usize, f64)> {
        let t0 = Instant::now();
        let m = NvxModule::from_data(ctx, data)?;
        let create = t0.elapsed().as_secs_f64() * 1e3;
        let f = m.function("probe_fill")?;
        let launch = |cb: vk::CommandBuffer, a: u64, n: u32, v: u32| unsafe {
            f.cmd_launch(
                cb,
                [n.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &[KernelArg::U64(a), KernelArg::U32(n), KernelArg::U32(v)],
            )
        };
        let (bad, _) = fill_check(ctx, exec, rec, &launch, 0x0A0B_0C04)?;
        Ok((bad, create))
    })();
    match res {
        Ok((bad, create)) => {
            println!("NVX {label}: accepted ({create:.2} ms), fill mismatches={bad}")
        }
        Err(e) => println!("NVX {label}: FAILED ({e})"),
    }
}

// ---------------------------------------------------------------------
// dump

fn dump(ctx: &Arc<VulkanContext>) -> Result<()> {
    let dir = env_string("ML_PTX_DUMP_DIR", ".");
    std::fs::create_dir_all(&dir)?;
    let cc = ctx.cuda_compute_capability.unwrap_or((0, 0));
    let probe = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let gemm = PtxGemm::new(ctx)?;
    for (name, module) in [
        ("probe", &probe),
        ("sgemm", &gemm.sgemm_module),
        ("hgemm", &gemm.hgemm_module),
    ] {
        let blob = module.cache_data()?;
        let path = format!("{dir}/{name}_jit_sm{}{}.cubin", cc.0, cc.1);
        std::fs::write(&path, &blob)?;
        println!("{path}: {} bytes", blob.len());
    }
    println!(
        "JIT ms: sgemm {:.1}, hgemm {:.1}",
        gemm.jit_ms.0, gemm.jit_ms.1
    );
    Ok(())
}

// ---------------------------------------------------------------------
// smem

fn smem(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let rec = Rec::new(ctx, 4)?;
    let module = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let f = module.function("probe_smem")?;
    let blocks = 48u32;
    let sizes: Vec<u32> = env_string("ML_PTX_SMEM", "16384,32768,49152,65536,98304,101376")
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    for bytes in sizes {
        let words = bytes / 4;
        let out = u32_tensor(ctx, exec, &vec![0u32; blocks as usize])?;
        let a = addr(ctx, &out);
        let res = rec.timed(|cb| unsafe {
            f.cmd_launch(
                cb,
                [blocks, 1, 1],
                [256, 1, 1],
                bytes,
                &[KernelArg::U64(a), KernelArg::U32(words)],
            )
        });
        match res {
            Ok(ms) => {
                let got = download_u32(exec, &out, blocks as usize)?;
                let bad = (0..blocks)
                    .filter(|&b| {
                        let want = (0..words).fold(0u32, |acc, i| {
                            acc.wrapping_add(i.wrapping_mul(2654435761).wrapping_add(b))
                        });
                        got[b as usize] != want
                    })
                    .count();
                println!(
                    "dynamic smem {bytes:>6} B: launched, gpu {ms:.4} ms, bad blocks {bad}/{blocks}"
                );
            }
            Err(e) => {
                println!("dynamic smem {bytes:>6} B: FAILED ({e:#})");
                break;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// sync

#[derive(Copy, Clone, Debug)]
enum BarrierKind {
    /// COMPUTE->COMPUTE, SHADER_WRITE -> SHADER_READ|WRITE
    Compute,
    /// COMPUTE->COMPUTE execution dependency only (no memory barrier)
    ExecOnly,
    /// ALL_COMMANDS->ALL_COMMANDS, MEMORY_WRITE -> MEMORY_READ|WRITE
    AllCommands,
    /// nothing between steps
    None,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Chain {
    /// alternate Vulkan (even steps) and CUDA (odd steps)
    VkCuda,
    CudaOnly,
    /// CUDA only, alternating forward / reverse traversal (has power to
    /// detect a missing CUDA->CUDA dependency; `CudaOnly` does not)
    CudaAlt,
    VkOnly,
}

fn sync(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let n = env_u32("ML_PTX_SYNC_N", 1 << 24);
    let steps = env_u32("ML_PTX_SYNC_STEPS", 32);
    let trials = env_u32("ML_PTX_SYNC_TRIALS", 3);
    let rec = Rec::new(ctx, 4)?;
    let module = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let cu = module.function("probe_affine")?;
    let cu_rev = module.function("probe_affine_rev")?;
    let vk_affine = ProbeAffine::new(ctx)?;
    let dev = &ctx.device;

    let mut seed = vec![0f32; n as usize];
    fill_det(&mut seed, 99);
    let seed: Vec<u32> = seed.iter().map(|x| x.to_bits()).collect();
    let x = u32_tensor(ctx, exec, &seed)?;
    let y = u32_tensor(ctx, exec, &vec![0u32; n as usize])?;
    let (xa, ya) = (addr(ctx, &x), addr(ctx, &y));
    let groups = n.div_ceil(256).min(65535);

    println!(
        "== ptx sync: n={n} u32 ({} MiB/buffer), steps={steps}, trials={trials}",
        n as u64 * 4 >> 20
    );
    println!(
        "{:<10} {:<12} {:>9} {:>12} {:>10}",
        "chain", "barrier", "trial", "mismatches", "gpu_ms"
    );
    let chains = [
        Chain::VkCuda,
        Chain::CudaAlt,
        Chain::CudaOnly,
        Chain::VkOnly,
    ];
    let kinds = [
        BarrierKind::Compute,
        BarrierKind::AllCommands,
        BarrierKind::ExecOnly,
        BarrierKind::None,
    ];
    for chain in chains {
        for kind in kinds {
            for trial in 0..trials {
                exec.upload(bytemuck_f32(&seed), &x)?;
                let (mut m_acc, mut a_acc) = (1u32, 0u32);
                let params: Vec<(u32, u32)> = (0..steps)
                    .map(|s| (2 * (s + trial) + 3, s.wrapping_mul(0x9E37_79B9) ^ trial))
                    .collect();
                for &(m, a) in &params {
                    // F(v) = m*(m_acc*v + a_acc) + a
                    m_acc = m_acc.wrapping_mul(m);
                    a_acc = a_acc.wrapping_mul(m).wrapping_add(a);
                }
                let ms = rec.timed(|cb| {
                    for (s, &(mul, add)) in params.iter().enumerate() {
                        let (src, dst) = if s % 2 == 0 { (xa, ya) } else { (ya, xa) };
                        if s > 0 {
                            match kind {
                                BarrierKind::Compute => compute_barrier(dev, cb),
                                BarrierKind::ExecOnly => barrier(
                                    dev,
                                    cb,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::AccessFlags::empty(),
                                    vk::AccessFlags::empty(),
                                ),
                                BarrierKind::AllCommands => barrier(
                                    dev,
                                    cb,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::PipelineStageFlags::ALL_COMMANDS,
                                    vk::AccessFlags::MEMORY_WRITE,
                                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                                ),
                                BarrierKind::None => {}
                            }
                        }
                        let use_cuda = match chain {
                            Chain::VkCuda => s % 2 == 1,
                            Chain::CudaOnly | Chain::CudaAlt => true,
                            Chain::VkOnly => false,
                        };
                        let func = if chain == Chain::CudaAlt && s % 2 == 1 {
                            &cu_rev
                        } else {
                            &cu
                        };
                        unsafe {
                            if use_cuda {
                                func.cmd_launch(
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
                            } else {
                                vk_affine.cmd_dispatch(cb, src, dst, n, mul, add, groups);
                            }
                        }
                    }
                })?;
                let fin = if steps % 2 == 0 { &x } else { &y };
                let got = download_u32(exec, fin, n as usize)?;
                let bad = got
                    .iter()
                    .zip(&seed)
                    .filter(|(g, s)| **g != s.wrapping_mul(m_acc).wrapping_add(a_acc))
                    .count();
                println!(
                    "{:<10} {:<12} {:>9} {:>12} {:>10.3}",
                    format!("{chain:?}"),
                    format!("{kind:?}"),
                    trial,
                    bad,
                    ms
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// state: Vulkan dispatch, CUDA launch, then a Vulkan dispatch that
// relies on the previously bound pipeline + push constants.

fn state(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let n = 1u32 << 20;
    let rec = Rec::new(ctx, 4)?;
    let module = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let cu = module.function("probe_affine")?;
    let vk_affine = ProbeAffine::new(ctx)?;
    let dev = &ctx.device;
    let seed: Vec<u32> = (0..n).map(|i| i.wrapping_mul(2654435761)).collect();
    let x = u32_tensor(ctx, exec, &seed)?;
    let y = u32_tensor(ctx, exec, &vec![0u32; n as usize])?;
    let (xa, ya) = (addr(ctx, &x), addr(ctx, &y));
    let groups = n.div_ceil(256);
    let (m1, a1, m2, a2) = (3u32, 7u32, 5u32, 11u32);
    rec.run(|cb| unsafe {
        vk_affine.cmd_dispatch(cb, xa, ya, n, m1, a1, groups); // y = x*m1+a1
        compute_barrier(dev, cb);
        cu.cmd_launch(
            cb,
            [groups, 1, 1],
            [256, 1, 1],
            0,
            &[
                KernelArg::U64(ya),
                KernelArg::U64(xa),
                KernelArg::U32(n),
                KernelArg::U32(m2),
                KernelArg::U32(a2),
            ],
        ); // x = y*m2+a2
        compute_barrier(dev, cb);
        vk_affine.cmd_dispatch_reuse_state(cb, groups); // y = x*m1+a1 (state reused)
    })?;
    let got = download_u32(exec, &y, n as usize)?;
    let bad = got
        .iter()
        .zip(&seed)
        .filter(|(g, s)| {
            **g != s
                .wrapping_mul(m1)
                .wrapping_add(a1)
                .wrapping_mul(m2)
                .wrapping_add(a2)
                .wrapping_mul(m1)
                .wrapping_add(a1)
        })
        .count();
    println!(
        "bound-state reuse across a CUDA launch (pipeline + push constants not re-recorded): mismatches={bad}/{n}"
    );
    Ok(())
}

// ---------------------------------------------------------------------
// overhead

fn overhead(ctx: &Arc<VulkanContext>, exec: &Executor) -> Result<()> {
    let rec = Rec::new(ctx, 4)?;
    let module = CudaModule::from_ptx(ctx, kernels::PROBE_PTX)?;
    let bump = module.function("probe_bump")?;
    let stamp = module.function("probe_stamp")?;
    let vk_affine = ProbeAffine::new(ctx)?;
    let dev = &ctx.device;
    let launches = env_u32("ML_PTX_LAUNCHES", 256);
    let reps = env_u32("ML_ROUNDS", 9);
    let counter = u32_tensor(ctx, exec, &[0u32; 64])?;
    let ca = addr(ctx, &counter);
    burn(ctx, exec, &rec)?;

    println!("== ptx overhead: {launches} launches per command buffer, {reps} reps (median)");
    println!(
        "{:<34} {:>10} {:>12} {:>12} {:>9}",
        "variant", "gpu_ms", "gpu_us/launch", "cpu_us/rec", "counter"
    );
    for (label, cuda, with_barrier) in [
        ("empty cb (timestamps only)", None, false),
        ("CUDA probe_bump, no barrier", Some(true), false),
        ("CUDA probe_bump, compute barrier", Some(true), true),
        ("Vulkan 1-WG dispatch, no barrier", Some(false), false),
        ("Vulkan 1-WG dispatch, barrier", Some(false), true),
    ] {
        let mut gpu = Vec::new();
        let mut cpu = Vec::new();
        let mut last_counter = 0;
        for _ in 0..reps {
            exec.upload(bytemuck_f32(&[0u32; 64]), &counter)?;
            let mut rec_us = 0.0;
            let ms = rec.timed(|cb| {
                let t0 = Instant::now();
                if let Some(is_cuda) = cuda {
                    for i in 0..launches {
                        if with_barrier && i > 0 {
                            compute_barrier(dev, cb);
                        }
                        unsafe {
                            if is_cuda {
                                bump.cmd_launch(
                                    cb,
                                    [1, 1, 1],
                                    [32, 1, 1],
                                    0,
                                    &[KernelArg::U64(ca)],
                                );
                            } else {
                                vk_affine.cmd_dispatch(cb, ca, ca + 128, 32, 1, 1, 1);
                            }
                        }
                    }
                }
                rec_us = t0.elapsed().as_secs_f64() * 1e6;
            })?;
            gpu.push(ms);
            cpu.push(rec_us);
            last_counter = download_u32(exec, &counter, 1)?[0];
        }
        let g = median(&mut gpu);
        let c = median(&mut cpu);
        let per = if cuda.is_some() { launches as f64 } else { 1.0 };
        println!(
            "{:<34} {:>10.4} {:>12.3} {:>12.3} {:>9}",
            label,
            g,
            g * 1e3 / per,
            c / per,
            if cuda == Some(true) {
                last_counter.to_string()
            } else {
                "-".into()
            }
        );
    }

    // Overlap: two ~spinning launches back to back, globaltimer stamps.
    let stamps = u32_tensor(ctx, exec, &[0u32; 32])?;
    let sa = addr(ctx, &stamps);
    let spin = env_u32("ML_PTX_SPIN", 20000);
    for with_barrier in [false, true] {
        let ms = rec.timed(|cb| unsafe {
            stamp.cmd_launch(
                cb,
                [1, 1, 1],
                [32, 1, 1],
                0,
                &[KernelArg::U64(sa), KernelArg::U32(spin)],
            );
            if with_barrier {
                compute_barrier(dev, cb);
            }
            stamp.cmd_launch(
                cb,
                [1, 1, 1],
                [32, 1, 1],
                0,
                &[KernelArg::U64(sa + 32), KernelArg::U32(spin)],
            );
        })?;
        let v = download_u32(exec, &stamps, 16)?;
        let t = |i: usize| (v[2 * i] as u64) | ((v[2 * i + 1] as u64) << 32);
        let (a0, a1, b0, b1) = (t(0), t(1), t(4), t(5));
        println!(
            "stamp pair ({}): A [{:.1}..{:.1}] us, B [{:.1}..{:.1}] us rel. to A start; overlap={}; vk-timestamp span {:.1} us",
            if with_barrier {
                "barrier"
            } else {
                "no barrier"
            },
            0.0,
            (a1 - a0) as f64 / 1e3,
            b0.wrapping_sub(a0) as i64 as f64 / 1e3,
            b1.wrapping_sub(a0) as i64 as f64 / 1e3,
            b0 < a1,
            ms * 1e3
        );
    }
    Ok(())
}

/// ~300 ms of real GPU work before timing (bench clock-state hazard).
fn burn(ctx: &Arc<VulkanContext>, exec: &Executor, rec: &Rec) -> Result<()> {
    let n = 1u32 << 24;
    let x = u32_tensor(ctx, exec, &vec![1u32; n as usize])?;
    let y = u32_tensor(ctx, exec, &vec![0u32; n as usize])?;
    let vk_affine = ProbeAffine::new(ctx)?;
    let (xa, ya) = (addr(ctx, &x), addr(ctx, &y));
    let deadline = Instant::now() + Duration::from_millis(env_u32("ML_BURN_MS", 300) as u64);
    while Instant::now() < deadline {
        rec.run(|cb| unsafe {
            for _ in 0..8 {
                vk_affine.cmd_dispatch(cb, xa, ya, n, 3, 1, 65535);
                compute_barrier(&ctx.device, cb);
            }
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// GEMM helpers

fn parse_shapes(args: &[String], default: &[(u32, u32, u32)]) -> Result<Vec<(u32, u32, u32)>> {
    if args.is_empty() {
        return Ok(default.to_vec());
    }
    args.iter()
        .map(|s| {
            let v: Vec<u32> = s
                .split('x')
                .map(|p| p.parse())
                .collect::<Result<_, _>>()
                .with_context(|| format!("bad shape '{s}' (want MxNxK)"))?;
            ensure!(v.len() == 3, "bad shape '{s}' (want MxNxK)");
            Ok((v[0], v[1], v[2]))
        })
        .collect()
}

/// Deterministic approx-normal fill (sum of 4 uniforms, var-normalized).
fn fill_normal(out: &mut [f32], seed: u64, std: f32) {
    let mut u = vec![0f32; out.len() * 4];
    fill_det(&mut u, seed);
    for (i, o) in out.iter_mut().enumerate() {
        let s: f32 = u[4 * i..4 * i + 4].iter().sum();
        // each U(-1,1) has var 1/3 -> sum var 4/3
        *o = s / (4.0f32 / 3.0).sqrt() * std;
    }
}

/// f64 reference for a spread of (row, col) samples of C = A*B.
fn sample_ref(a: &[f32], b: &[f32], m: u32, n: u32, k: u32, count: usize) -> Vec<(usize, f64)> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut out = Vec::with_capacity(count);
    let mut s = 0x1234_5678_9abc_def0u64;
    for i in 0..count {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let (r, c) = match i {
            0 => (0, 0),
            1 => (m - 1, n - 1),
            2 => (m - 1, 0),
            3 => (0, n - 1),
            _ => (((s >> 33) as usize) % m, ((s >> 11) as usize) % n),
        };
        let mut acc = 0f64;
        for kk in 0..k {
            acc += a[r * k + kk] as f64 * b[kk * n + c] as f64;
        }
        out.push((r * n + c, acc));
    }
    out
}

struct ErrStats {
    max_abs: f64,
    rms_err: f64,
    rms_ref: f64,
}

fn err_stats(got: &[f32], refs: &[(usize, f64)]) -> ErrStats {
    let mut max_abs = 0f64;
    let (mut se, mut sr) = (0f64, 0f64);
    for &(idx, want) in refs {
        let e = (got[idx] as f64 - want).abs();
        let e = if e.is_finite() { e } else { f64::INFINITY };
        max_abs = max_abs.max(e);
        se += e * e;
        sr += want * want;
    }
    let cnt = refs.len().max(1) as f64;
    ErrStats {
        max_abs,
        rms_err: (se / cnt).sqrt(),
        rms_ref: (sr / cnt).sqrt(),
    }
}

fn tensor_ash_run(exec: &Executor, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<f64> {
    let stats = exec.run_matmuls(&[MatmulCall {
        a,
        b,
        c,
        alpha: 1.0,
        accumulate: false,
    }])?;
    Ok(stats.gpu_time_ns.context("no GPU timestamps")? as f64 / 1e6)
}

struct Timing {
    med: f64,
    min: f64,
    p90: f64,
}

fn summarize(mut v: Vec<f64>) -> Timing {
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let p90 = pct(&mut v, 0.9);
    Timing {
        med: median(&mut v),
        min,
        p90,
    }
}

/// Interleaved A/B timing: per round, `iters` of each side, alternating
/// which side goes first.
fn interleave(
    rounds: u32,
    iters: u32,
    mut side_a: impl FnMut() -> Result<f64>,
    mut side_b: impl FnMut() -> Result<f64>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for r in 0..rounds {
        if r % 2 == 0 {
            for _ in 0..iters {
                ta.push(side_a()?);
            }
            for _ in 0..iters {
                tb.push(side_b()?);
            }
        } else {
            for _ in 0..iters {
                tb.push(side_b()?);
            }
            for _ in 0..iters {
                ta.push(side_a()?);
            }
        }
    }
    Ok((ta, tb))
}

const SGEMM_SHAPES: &[(u32, u32, u32)] = &[
    (4096, 4096, 4096),
    (2048, 2048, 2048),
    (1024, 1024, 1024),
    (1024, 3072, 512),
    (768, 768, 768),
    (1023, 1025, 1027),
];

fn sgemm(ctx: &Arc<VulkanContext>, exec: &Executor, args: &[String]) -> Result<()> {
    let shapes = parse_shapes(args, SGEMM_SHAPES)?;
    let rec = Rec::new(ctx, 4)?;
    let gemm = PtxGemm::new(ctx)?;
    println!(
        "== ptx sgemm on {} (JIT {:.1} ms)",
        ctx.device_name(),
        gemm.jit_ms.0
    );
    let rounds = env_u32("ML_ROUNDS", 7);
    let iters = env_u32("ML_ITERS", 5);
    let only = env_string("ML_PTX_KERNELS", "");
    burn(ctx, exec, &rec)?;
    println!(
        "{:<16} {:<28} {:>9} {:>7} | {:<26} {:>9} {:>7} {:>7} | {:>9} {:>9}",
        "shape",
        "tensor-ash route",
        "ta_ms",
        "TF/s",
        "ptx kernel",
        "ptx_ms",
        "TF/s",
        "ptx/ta",
        "ta_err",
        "ptx_err"
    );
    for (m, n, k) in shapes {
        let a = Tensor::uninit_device(ctx, &[m, k])?;
        let b = Tensor::uninit_device(ctx, &[k, n])?;
        let c_ta = Tensor::uninit_device(ctx, &[m, n])?;
        let c_px = Tensor::uninit_device(ctx, &[m, n])?;
        let mut ha = vec![0f32; (m * k) as usize];
        let mut hb = vec![0f32; (k * n) as usize];
        fill_det(&mut ha, 7);
        fill_det(&mut hb, 11);
        exec.upload(&ha, &a)?;
        exec.upload(&hb, &b)?;
        let refs = sample_ref(&ha, &hb, m, n, k, 512);
        let route = exec.dispatch_info_for(1, m, n, k, false);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let (aa, ba, ca) = (addr(ctx, &a), addr(ctx, &b), addr(ctx, &c_px));

        tensor_ash_run(exec, &a, &b, &c_ta)?;
        let mut h_ta = vec![0f32; (m * n) as usize];
        exec.download(&c_ta, &mut h_ta)?;
        let e_ta = err_stats(&h_ta, &refs);

        for (idx, (kern, _)) in gemm.sgemm.iter().enumerate() {
            if !kern.supports(m, n, k)
                || (!only.is_empty() && !only.split(',').any(|s| s == kern.name))
            {
                continue;
            }
            // Validate first: poison C, run once, compare.
            exec.upload(&vec![f32::NAN; (m * n) as usize], &c_px)?;
            let run_px = || {
                rec.timed(|cb| unsafe {
                    gemm.cmd_sgemm(cb, idx, aa, ba, ca, (m, n, k, 1), 1.0, 0.0)
                })
            };
            run_px()?;
            let mut h_px = vec![0f32; (m * n) as usize];
            exec.download(&c_px, &mut h_px)?;
            let e_px = err_stats(&h_px, &refs);
            let tol = 1e-5 * (k as f64).sqrt() * 4.0 + 1e-4;
            if !(e_px.max_abs <= tol.max(4.0 * e_ta.max_abs)) {
                println!(
                    "{m}x{n}x{k} {}: VALIDATION FAILED max_abs {:.3e} (ta {:.3e}) — not timed",
                    kern.name, e_px.max_abs, e_ta.max_abs
                );
                continue;
            }
            let (t_ta, t_px) = interleave(
                rounds,
                iters,
                || tensor_ash_run(exec, &a, &b, &c_ta),
                run_px,
            )?;
            let (s_ta, s_px) = (summarize(t_ta), summarize(t_px));
            println!(
                "{:<16} {:<28} {:>9.4} {:>7.2} | {:<26} {:>9.4} {:>7.2} {:>7.3} | {:>9.2e} {:>9.2e}",
                format!("{m}x{n}x{k}"),
                route.kernel,
                s_ta.med,
                flops / s_ta.med * 1e-9,
                kern.name,
                s_px.med,
                flops / s_px.med * 1e-9,
                s_ta.med / s_px.med,
                e_ta.max_abs,
                e_px.max_abs
            );
            log::info!(
                "{m}x{n}x{k} spread: ta min {:.4} p90 {:.4} | ptx min {:.4} p90 {:.4} (ms)",
                s_ta.min,
                s_ta.p90,
                s_px.min,
                s_px.p90
            );
        }
    }
    Ok(())
}

const HGEMM_SHAPES: &[(u32, u32, u32)] = &[
    (512, 2048, 2048),
    (512, 5632, 2048),
    (512, 2048, 5632),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
];

fn to_f16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_f16_bits(x)).collect()
}

fn hgemm(ctx: &Arc<VulkanContext>, exec: &Executor, args: &[String]) -> Result<()> {
    ensure!(
        ctx.coopmat_enabled,
        "tensor-ash a16 coopmat route needs coopmat"
    );
    let shapes = parse_shapes(args, HGEMM_SHAPES)?;
    let rec = Rec::new(ctx, 4)?;
    let gemm = PtxGemm::new(ctx)?;
    println!(
        "== ptx hgemm on {} (JIT {:.1} ms)",
        ctx.device_name(),
        gemm.jit_ms.1
    );
    let rounds = env_u32("ML_ROUNDS", 7);
    let iters = env_u32("ML_ITERS", 5);
    let only = env_string("ML_PTX_KERNELS", "");
    let a_std: f32 = env_string("ML_PTX_A_STD", "1.0").parse().unwrap_or(1.0);
    let b_std: f32 = env_string("ML_PTX_B_STD", "0.02").parse().unwrap_or(0.02);
    burn(ctx, exec, &rec)?;
    println!(
        "{:<16} {:>9} {:>7} | {:<26} {:>9} {:>7} {:>7} | {:>9} {:>9} {:>9}",
        "shape",
        "ta_ms",
        "TF/s",
        "ptx kernel",
        "ptx_ms",
        "TF/s",
        "ptx/ta",
        "ta_relrms",
        "ptx_relrms",
        "ptx_maxabs"
    );
    for (m, n, k) in shapes {
        let a = Tensor::uninit_device_f16(ctx, &[m, k])?;
        let b = Tensor::uninit_device_f16(ctx, &[k, n])?;
        let c_ta = Tensor::uninit_device_f16(ctx, &[m, n])?;
        let c_px = Tensor::uninit_device_f16(ctx, &[m, n])?;
        let c_px32 = Tensor::uninit_device(ctx, &[m, n])?;
        let mut ha = vec![0f32; (m * k) as usize];
        let mut hb = vec![0f32; (k * n) as usize];
        fill_normal(&mut ha, 5, a_std);
        fill_normal(&mut hb, 6, b_std);
        let (ha16, hb16) = (to_f16_bits(&ha), to_f16_bits(&hb));
        // Reference on exactly the f16 values the GPU multiplies.
        let ha_r: Vec<f32> = ha16.iter().map(|&h| f16_bits_to_f32(h)).collect();
        let hb_r: Vec<f32> = hb16.iter().map(|&h| f16_bits_to_f32(h)).collect();
        exec.upload_f16(&ha16, &a)?;
        exec.upload_f16(&hb16, &b)?;
        let refs = sample_ref(&ha_r, &hb_r, m, n, k, 512);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let (aa, ba) = (addr(ctx, &a), addr(ctx, &b));

        tensor_ash_run(exec, &a, &b, &c_ta)?;
        let mut h_ta = vec![0f32; (m * n) as usize];
        exec.download(&c_ta, &mut h_ta)?;
        let e_ta = err_stats(&h_ta, &refs);

        for (idx, (kern, _)) in gemm.hgemm.iter().enumerate() {
            if !kern.supports(m, n, k)
                || (!only.is_empty() && !only.split(',').any(|s| s == kern.name))
            {
                continue;
            }
            let c = if kern.out_f16 { &c_px } else { &c_px32 };
            let ca = addr(ctx, c);
            if kern.out_f16 {
                exec.upload_f16(&vec![0x7e00u16; (m * n) as usize], c)?;
            } else {
                exec.upload(&vec![f32::NAN; (m * n) as usize], c)?;
            }
            let run_px = || {
                rec.timed(|cb| unsafe { gemm.cmd_hgemm(cb, idx, aa, ba, ca, (m, n, k, 1), 1.0) })
            };
            if let Err(e) = run_px() {
                println!("{m}x{n}x{k} {}: LAUNCH FAILED ({e:#})", kern.name);
                return Err(e);
            }
            let mut h_px = vec![0f32; (m * n) as usize];
            exec.download(c, &mut h_px)?;
            let e_px = err_stats(&h_px, &refs);
            let rel = |e: &ErrStats| e.rms_err / e.rms_ref.max(1e-30);
            // f32-acc must match coopmat's error class; f16-acc gets a
            // looser gate (its error IS the finding, reported below).
            let gate = if kern.f16_acc && !kern.promoted() {
                0.05
            } else {
                (4.0 * rel(&e_ta)).max(2e-3)
            };
            if !(rel(&e_px) <= gate) {
                println!(
                    "{m}x{n}x{k} {}: VALIDATION FAILED rel_rms {:.3e} (ta {:.3e}, gate {gate:.1e}) — not timed",
                    kern.name,
                    rel(&e_px),
                    rel(&e_ta)
                );
                continue;
            }
            let (t_ta, t_px) = interleave(
                rounds,
                iters,
                || tensor_ash_run(exec, &a, &b, &c_ta),
                run_px,
            )?;
            let (s_ta, s_px) = (summarize(t_ta), summarize(t_px));
            println!(
                "{:<16} {:>9.4} {:>7.2} | {:<26} {:>9.4} {:>7.2} {:>7.3} | {:>9.2e} {:>9.2e} {:>9.2e}",
                format!("{m}x{n}x{k}"),
                s_ta.med,
                flops / s_ta.med * 1e-9,
                kern.name,
                s_px.med,
                flops / s_px.med * 1e-9,
                s_ta.med / s_px.med,
                rel(&e_ta),
                rel(&e_px),
                e_px.max_abs
            );
            log::info!(
                "{m}x{n}x{k} {} spread: ta min {:.4} p90 {:.4} | ptx min {:.4} p90 {:.4} (ms); ref rms {:.3e}",
                kern.name,
                s_ta.min,
                s_ta.p90,
                s_px.min,
                s_px.p90,
                e_px.rms_ref
            );
        }

        // Split-K variants (ML_PTX_SPLITS, default "2,3,4"; "0" skips).
        let splits_list: Vec<u32> = env_string("ML_PTX_SPLITS", "2,3,4")
            .split(',')
            .filter_map(|v| v.parse().ok())
            .filter(|&v| v >= 2)
            .collect();
        if (m * n) % 4 != 0 || splits_list.is_empty() {
            continue;
        }
        let max_splits = *splits_list.iter().max().unwrap();
        let w = Tensor::uninit_device(ctx, &[max_splits, m, n])?;
        let wa = addr(ctx, &w);
        let ca = addr(ctx, &c_px);
        for (idx, (kern, _)) in gemm.hgemm_splitk.iter().enumerate() {
            if !kern.supports(m, n, k)
                || (!only.is_empty() && !only.split(',').any(|s| s == kern.name))
            {
                continue;
            }
            for &splits in &splits_list {
                let label = format!("{}/{splits}", kern.name.trim_end_matches("_splitk"));
                exec.upload_f16(&vec![0x7e00u16; (m * n) as usize], &c_px)?;
                let run_px = || {
                    rec.timed(|cb| unsafe {
                        gemm.cmd_hgemm_splitk(cb, idx, aa, ba, wa, ca, (m, n, k), splits, 1.0)
                    })
                };
                run_px()?;
                let mut h_px = vec![0f32; (m * n) as usize];
                exec.download(&c_px, &mut h_px)?;
                let e_px = err_stats(&h_px, &refs);
                let rel = |e: &ErrStats| e.rms_err / e.rms_ref.max(1e-30);
                let gate = if kern.f16_acc && !kern.promoted() {
                    0.05
                } else {
                    (4.0 * rel(&e_ta)).max(2e-3)
                };
                if !(rel(&e_px) <= gate) {
                    println!(
                        "{m}x{n}x{k} {label}: VALIDATION FAILED rel_rms {:.3e} — not timed",
                        rel(&e_px)
                    );
                    continue;
                }
                let (t_ta, t_px) = interleave(
                    rounds,
                    iters,
                    || tensor_ash_run(exec, &a, &b, &c_ta),
                    run_px,
                )?;
                let (s_ta, s_px) = (summarize(t_ta), summarize(t_px));
                println!(
                    "{:<16} {:>9.4} {:>7.2} | {:<26} {:>9.4} {:>7.2} {:>7.3} | {:>9.2e} {:>9.2e} {:>9.2e}",
                    format!("{m}x{n}x{k}"),
                    s_ta.med,
                    flops / s_ta.med * 1e-9,
                    label,
                    s_px.med,
                    flops / s_px.med * 1e-9,
                    s_ta.med / s_px.med,
                    rel(&e_ta),
                    rel(&e_px),
                    e_px.max_abs
                );
            }
        }
    }
    Ok(())
}
