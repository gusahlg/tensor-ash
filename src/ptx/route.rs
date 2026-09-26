//! Opt-in PTX route for the a16 matmul family (f16 A, f16 B, f16 C —
//! the llama prefill contract).
//!
//! Enabled by `ML_PTX_GEMM=f32|f16|f16p` on a context created with
//! `ML_PTX=1` (`ML_NO_PTX=1` vetoes both).  The executor's matmul
//! recorder asks [`PtxRoute::select`] before binding a Vulkan
//! `*_a16_*` coopmat pipeline; when a PTX kernel is chosen it records
//! `vkCmdCudaLaunchKernelNV` in place of bind + push + dispatch.
//! Barriers, hazard tracking and prepared replay are unchanged: the
//! probes (`ml_bench ptx sync|state`) showed compute pipeline barriers
//! order CUDA launches in both directions, launch parameters are
//! captured at record time, and a CUDA launch leaves the bound Vulkan
//! compute pipeline/push constants intact.
//!
//! Accumulation policy:
//! * `f32`  — mma f16 x f16 -> f32 (same numerics class as coopmat1).
//! * `f16`  — mma f16 x f16 -> f16, 2x tensor rate on GeForce parts;
//!   what llama.cpp's CUDA backend gets from cuBLAS
//!   (`CUBLAS_COMPUTE_16F`) for f16 weights.
//! * `f16p` — f16-rate mma within one BK=32 k-tile, promoted into f32
//!   registers after every k-tile (tensor-ash-only; near-f32 error).
//!
//! `ML_PTX_GEMM_KERNEL=<entry name>` forces one kernel for every
//! eligible shape (experiments).

use std::sync::Arc;

use anyhow::{Result, bail};
use ash::vk;

use super::gemm::PtxGemm;
use crate::context::VulkanContext;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PtxAccum {
    F32,
    F16,
    F16Promoted,
}

pub struct PtxRoute {
    gemm: PtxGemm,
    accum: PtxAccum,
    forced: Option<usize>,
    sm_count: u32,
}

impl PtxRoute {
    /// Build the route from `ML_PTX_GEMM` / `ML_PTX_GEMM_KERNEL`.
    /// `Ok(None)` when not requested or when the context has no
    /// `VK_NV_cuda_kernel_launch`.
    pub fn from_env(ctx: &Arc<VulkanContext>) -> Result<Option<Self>> {
        let accum = match std::env::var("ML_PTX_GEMM").unwrap_or_default().as_str() {
            "" | "0" | "off" => return Ok(None),
            "f32" => PtxAccum::F32,
            "f16" => PtxAccum::F16,
            "f16p" => PtxAccum::F16Promoted,
            other => bail!("ML_PTX_GEMM={other}: expected f32, f16, f16p or off"),
        };
        if ctx.cuda_launch.is_none() {
            log::warn!(
                "tensor-ash: ML_PTX_GEMM set but VK_NV_cuda_kernel_launch is off (ML_PTX=1?)"
            );
            return Ok(None);
        }
        let t = std::time::Instant::now();
        let gemm = PtxGemm::new(ctx)?;
        let forced = match std::env::var("ML_PTX_GEMM_KERNEL") {
            Ok(name) if !name.is_empty() => match gemm.hgemm_index(&name) {
                Some(i) => Some(i),
                None => bail!("ML_PTX_GEMM_KERNEL={name}: no such hgemm entry"),
            },
            _ => None,
        };
        let sm_count = shader_sm_count(ctx).unwrap_or(0);
        log::info!(
            "tensor-ash: PTX a16 route on: accum={accum:?} forced={:?} sm_count={sm_count} \
             (modules ready in {:.1} ms)",
            forced.map(|i| gemm.hgemm[i].0.name),
            t.elapsed().as_secs_f64() * 1e3
        );
        Ok(Some(Self {
            gemm,
            accum,
            forced,
            sm_count,
        }))
    }

    pub fn accum(&self) -> PtxAccum {
        self.accum
    }

    pub fn kernel_name(&self, index: usize) -> &'static str {
        self.gemm.hgemm[index].0.name
    }

    /// Pick an hgemm entry for a batch-1 `M x N x K` a16 matmul, or
    /// `None` to keep the Vulkan route.  Heuristic from the 4060 / 3070
    /// sweeps in `benchmarks/campaign/ptx.md`; a measured per-device
    /// pick (the executor tuner) is the intended replacement.
    pub fn select(&self, m: u32, n: u32, k: u32) -> Option<usize> {
        if n % 8 != 0 || k % 8 != 0 {
            return None;
        }
        if let Some(i) = self.forced {
            return Some(i);
        }
        let tiles128 = u64::from(m.div_ceil(128)) * u64::from(n.div_ceil(128));
        let sms = u64::from(self.sm_count.max(1));
        let name = match self.accum {
            PtxAccum::F32 => {
                // Few, big tiles win once there are enough of them to
                // fill the device twice; below that 64x64 keeps more
                // CTAs in flight.
                if tiles128 >= 4 * sms {
                    "ta_hgemm_f32_128x128_s3"
                } else {
                    "ta_hgemm_f32_64x64_s4"
                }
            }
            PtxAccum::F16 => {
                if tiles128 >= 2 * sms {
                    "ta_hgemm_f16_128x128_s3"
                } else {
                    "ta_hgemm_f16_128x64_s3"
                }
            }
            PtxAccum::F16Promoted => "ta_hgemm_f16p_128x64_s3",
        };
        self.gemm.hgemm_index(name)
    }

    /// # Safety
    /// `cb` is recording; `a`/`b`/`c` are device addresses of f16
    /// buffers holding `M*K`, `K*N`, `M*N` elements.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn record(
        &self,
        cb: vk::CommandBuffer,
        index: usize,
        a: u64,
        b: u64,
        c: u64,
        (m, n, k): (u32, u32, u32),
        alpha: f32,
    ) {
        unsafe { self.gemm.cmd_hgemm(cb, index, a, b, c, (m, n, k, 1), alpha) };
    }
}

/// `VkPhysicalDeviceShaderSMBuiltinsPropertiesNV::shaderSMCount` when
/// the driver exposes `VK_NV_shader_sm_builtins` (a property query; the
/// extension need not be enabled).
fn shader_sm_count(ctx: &VulkanContext) -> Option<u32> {
    let exts = unsafe {
        ctx.instance
            .enumerate_device_extension_properties(ctx.physical_device)
            .ok()?
    };
    let present = exts.iter().any(|e| {
        e.extension_name_as_c_str()
            .is_ok_and(|n| n == ash::nv::shader_sm_builtins::NAME)
    });
    if !present {
        return None;
    }
    let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
    unsafe {
        ctx.instance
            .get_physical_device_properties2(ctx.physical_device, &mut props)
    };
    Some(sm.shader_sm_count)
}
