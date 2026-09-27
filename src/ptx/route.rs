//! PTX route for the a16 matmul family (f16 A, f16 B, f16 C — the
//! llama prefill contract).
//!
//! Two independent knobs decide what runs:
//!
//! * **Precision** is the executor's accumulator policy,
//!   [`CoopmatAcc`] (`ExecutorConfig::coopmat_acc`, or `ML_COOPMAT_ACC`
//!   for env-configured executors; default `F32`).  The route never
//!   picks a precision: it records a kernel of exactly the class the
//!   executor asks for ([`route_kernel_name`] maps every class to a
//!   kernel of that class, checked by the unit tests), or it declines
//!   and the op stays on its f32-accumulate coopmat1 kernel.  A
//!   reduced-precision policy is therefore a *permission*, never a
//!   silent downgrade: ops the route declines keep f32 accumulation.
//! * **Backend**: the route exists only on a context created with
//!   `ML_PTX=1` where `VK_NV_cuda_kernel_launch` is usable (see
//!   `ptx_gate` in `context/init.rs`; `ML_NO_PTX=1` vetoes), and
//!   `ML_PTX_GEMM` says which precision classes it serves
//!   ([`PtxGemmMode`]: `on`/`auto` (default), `all`, `off`).  The
//!   experiment branch's `ML_PTX_GEMM=f32|f16|f16p` still works for one
//!   release: backend on, and — for env-configured executors when
//!   `ML_COOPMAT_ACC` is unset — that policy, with a deprecation
//!   warning ([`legacy_env_acc`]).
//!
//! The executor's matmul recorder asks [`PtxRoute::select`] before
//! binding a Vulkan `*_a16_*` coopmat pipeline; when a PTX kernel is
//! chosen it records `vkCmdCudaLaunchKernelNV` in place of bind + push +
//! dispatch.  Barriers, hazard tracking and prepared replay are
//! unchanged: the probes (`ml_bench ptx sync|state`) showed compute
//! pipeline barriers order CUDA launches in both directions, launch
//! parameters are captured at record time, and a CUDA launch leaves the
//! bound Vulkan compute pipeline/push constants intact.
//!
//! Kernels per class (`benchmarks/campaign/ptx.md`):
//! * `F32` — mma f16 x f16 -> f32, the numerics class of coopmat1;
//!   served only with `ML_PTX_GEMM=all`, since it does not pay end to
//!   end (TinyLlama pp512 0.98x on the 4060).
//! * `F16` — mma f16 x f16 -> f16, 2x tensor rate on GeForce parts; the
//!   numerics class of cuBLAS `CUBLAS_COMPUTE_16F`, which is what
//!   llama.cpp's CUDA backend uses for f16-weight prefill GEMMs.
//! * `F16Promote` — f16-rate mma within one BK=64 k-tile, added into
//!   f32 registers after every k-tile (2.2-2.4x the f32-accumulate
//!   error, 5-8x below `F16`).
//!
//! `ML_PTX_GEMM_KERNEL=<entry name>` forces one kernel for every
//! eligible shape whose policy matches that kernel's class
//! (experiments).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Result, bail};
use ash::vk;

use super::gemm::HgemmKernel;
use super::{CudaFunction, CudaModule, kernels};
use crate::context::VulkanContext;
use crate::pipeline::CoopmatAcc;

/// Which accumulator classes the PTX route serves (`ML_PTX_GEMM`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PtxGemmMode {
    /// Never record PTX; the modules are not even loaded.
    Off,
    /// `F16` and `F16Promote`, where PTX is the measured-fastest
    /// implementation; `F32` stays on coopmat1.  Default.
    #[default]
    Auto,
    /// Every class, `F32` included (for tests and experiments).
    All,
}

impl PtxGemmMode {
    /// `auto` (= `on`, `1`), `all`, `off` (= `0`).  The deprecated
    /// precision spellings of the experiment branch map to the mode
    /// that serves that class (`f16`/`f16p` -> `auto`, `f32` -> `all`);
    /// the class itself is set by [`legacy_env_acc`].
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" | "1" | "on" | "f16" | "f16p" => Ok(Self::Auto),
            "0" | "off" => Ok(Self::Off),
            "all" | "f32" => Ok(Self::All),
            other => bail!("invalid ML_PTX_GEMM '{other}', expected on (auto), all or off"),
        }
    }

    /// `ML_PTX_GEMM` (unset = `Auto`).
    pub fn from_env() -> Result<Self> {
        match std::env::var("ML_PTX_GEMM") {
            Ok(value) => Self::parse(&value),
            Err(_) => Ok(Self::Auto),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::All => "all",
        }
    }

    pub fn serves(self, acc: CoopmatAcc) -> bool {
        match self {
            Self::Off => false,
            Self::Auto => acc != CoopmatAcc::F32,
            Self::All => true,
        }
    }

    fn to_u8(self) -> u8 {
        self as u8
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Off,
            1 => Self::Auto,
            _ => Self::All,
        }
    }
}

/// The accumulator class of a deprecated `ML_PTX_GEMM=f32|f16|f16p`
/// (the experiment branch's one knob for backend *and* precision).
fn legacy_class(value: &str) -> Option<CoopmatAcc> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Some(CoopmatAcc::F32),
        "f16" => Some(CoopmatAcc::F16),
        "f16p" => Some(CoopmatAcc::F16Promote),
        _ => None,
    }
}

/// Compatibility shim for env-configured executors (`Executor::new`),
/// for one release: when `ML_COOPMAT_ACC` is unset, a deprecated
/// precision value of `ML_PTX_GEMM` sets the policy, with a warning.
/// `ML_COOPMAT_ACC` always wins; a disagreement is reported too.
pub fn legacy_env_acc(policy: CoopmatAcc) -> CoopmatAcc {
    let Some(legacy) = std::env::var("ML_PTX_GEMM")
        .ok()
        .and_then(|v| legacy_class(&v))
    else {
        return policy;
    };
    if std::env::var_os("ML_COOPMAT_ACC").is_some() {
        if legacy != policy {
            log::warn!(
                "tensor-ash: ML_PTX_GEMM={} (deprecated precision spelling) ignored: \
                 ML_COOPMAT_ACC={} sets the accumulator policy",
                legacy.as_str(),
                policy.as_str()
            );
        }
        return policy;
    }
    log::warn!(
        "tensor-ash: ML_PTX_GEMM={0} is deprecated: it now means ML_PTX_GEMM={1} + \
         ML_COOPMAT_ACC={0}; set those instead",
        legacy.as_str(),
        if legacy == CoopmatAcc::F32 {
            "all"
        } else {
            "on"
        }
    );
    legacy
}

/// SM count assumed when `VK_NV_shader_sm_builtins` is absent (never
/// on the proprietary driver that exposes the launch extension).
const DEFAULT_SM_COUNT: u32 = 32;

/// The kernel the route records for a batch-1 `M x N` a16 GEMM at
/// accumulator class `acc` on a device with `sm_count` SMs.  Always a
/// kernel of class `acc`.  Heuristic from the 4060 / 3070 sweeps in
/// `benchmarks/campaign/ptx.md`; a measured per-device pick (the
/// executor tuner) is the intended replacement.
pub fn route_kernel_name(acc: CoopmatAcc, m: u32, n: u32, sm_count: u32) -> &'static str {
    let tiles128 = u64::from(m.div_ceil(128)) * u64::from(n.div_ceil(128));
    let sms = u64::from(sm_count.max(1));
    // Grouped CTA rasterization (_g8) once the grid is large enough
    // that a B panel row no longer stays L2-resident (4060 4096^3:
    // f16 42.7 -> 52.8 TF/s; 2048^3 it costs ~3%).
    let huge = tiles128 >= 16 * sms;
    match acc {
        // Few, big tiles win once there are enough of them to fill the
        // device; below that 64x64 keeps more CTAs in flight.
        CoopmatAcc::F32 if huge => "ta_hgemm_f32_128x128_s3_g8",
        CoopmatAcc::F32 if tiles128 >= 4 * sms => "ta_hgemm_f32_128x128_s3",
        CoopmatAcc::F32 => "ta_hgemm_f32_64x64_s4",
        CoopmatAcc::F16 if huge => "ta_hgemm_f16_128x128_s3_g8",
        CoopmatAcc::F16 if tiles128 >= 2 * sms => "ta_hgemm_f16_128x128_s3",
        CoopmatAcc::F16 => "ta_hgemm_f16_128x64_s3",
        CoopmatAcc::F16Promote if huge => "ta_hgemm_f16p_128x64_s3_g8",
        CoopmatAcc::F16Promote => "ta_hgemm_f16p_128x64_k64_s2",
    }
}

/// Every kernel [`route_kernel_name`] can return (resolved at load).
const ROUTE_KERNELS: &[&str] = &[
    "ta_hgemm_f32_128x128_s3_g8",
    "ta_hgemm_f32_128x128_s3",
    "ta_hgemm_f32_64x64_s4",
    "ta_hgemm_f16_128x128_s3_g8",
    "ta_hgemm_f16_128x128_s3",
    "ta_hgemm_f16_128x64_s3",
    "ta_hgemm_f16p_128x64_s3_g8",
    "ta_hgemm_f16p_128x64_k64_s2",
];

pub struct PtxRoute {
    /// Route kernels (plus a forced one), resolved from the hgemm module.
    kernels: Vec<(HgemmKernel, CudaFunction)>,
    forced: Option<usize>,
    sm_count: u32,
    mode: AtomicU8,
}

impl PtxRoute {
    /// Build the route from `ML_PTX_GEMM` / `ML_PTX_GEMM_KERNEL`.
    ///
    /// `Err` only for invalid settings (a configuration error, like an
    /// invalid `ML_KERNEL`).  `Ok(None)` when the mode is `off`, when
    /// the context has no usable `VK_NV_cuda_kernel_launch`, or when
    /// loading the kernels fails — the last is logged and leaves every
    /// matmul on its Vulkan kernel.
    pub fn from_env(ctx: &Arc<VulkanContext>) -> Result<Option<Self>> {
        let mode = PtxGemmMode::from_env()?;
        let forced = match std::env::var("ML_PTX_GEMM_KERNEL") {
            Ok(name) if !name.is_empty() => match HgemmKernel::by_name(&name) {
                Some(kernel) if kernel.out_f16 => Some(kernel.name),
                _ => bail!("ML_PTX_GEMM_KERNEL={name}: no such f16-C hgemm entry"),
            },
            _ => None,
        };
        if ctx.cuda_launch.is_none() {
            if std::env::var_os("ML_PTX_GEMM").is_some() {
                log::warn!(
                    "tensor-ash: ML_PTX_GEMM set but the PTX backend is off (needs ML_PTX=1 \
                     and a usable VK_NV_cuda_kernel_launch)"
                );
            }
            return Ok(None);
        }
        if mode == PtxGemmMode::Off {
            return Ok(None);
        }
        if !ctx.buffer_device_address_enabled {
            log::warn!("tensor-ash: PTX a16 route off: needs bufferDeviceAddress");
            return Ok(None);
        }
        match Self::load(ctx, mode, forced) {
            Ok(route) => Ok(Some(route)),
            Err(err) => {
                log::warn!("tensor-ash: PTX a16 route disabled (Vulkan kernels stay): {err:#}");
                Ok(None)
            }
        }
    }

    fn load(
        ctx: &Arc<VulkanContext>,
        mode: PtxGemmMode,
        forced: Option<&'static str>,
    ) -> Result<Self> {
        let t = std::time::Instant::now();
        let (module, cached) = CudaModule::from_ptx_cached(ctx, "hgemm", kernels::HGEMM_PTX)?;
        let mut names: Vec<&'static str> = ROUTE_KERNELS.to_vec();
        if let Some(name) = forced
            && !names.contains(&name)
        {
            names.push(name);
        }
        let kernels = names
            .iter()
            .map(|name| {
                let kernel = *HgemmKernel::by_name(name).expect("route kernel is in HGEMM_KERNELS");
                Ok((kernel, module.function(name)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let forced = forced.map(|name| names.iter().position(|n| *n == name).expect("pushed"));
        let sm_count = shader_sm_count(ctx).unwrap_or(DEFAULT_SM_COUNT);
        log::info!(
            "tensor-ash: PTX a16 route on: ML_PTX_GEMM={} forced={:?} sm_count={sm_count} \
             (hgemm module {} in {:.1} ms)",
            mode.as_str(),
            forced.map(|i| kernels[i].0.name),
            if cached { "cached" } else { "JIT" },
            t.elapsed().as_secs_f64() * 1e3
        );
        if let Some(i) = forced {
            log::info!(
                "tensor-ash: ML_PTX_GEMM_KERNEL applies only to ops whose accumulator \
                 policy is {}",
                kernels[i].0.acc().as_str()
            );
        }
        Ok(Self {
            kernels,
            forced,
            sm_count,
            mode: AtomicU8::new(mode.to_u8()),
        })
    }

    pub fn mode(&self) -> PtxGemmMode {
        PtxGemmMode::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Change the served classes for matmuls recorded from now on
    /// (already-prepared command buffers keep what they recorded).
    pub fn set_mode(&self, mode: PtxGemmMode) {
        let old = PtxGemmMode::from_u8(self.mode.swap(mode.to_u8(), Ordering::Relaxed));
        if old != mode {
            log::info!(
                "tensor-ash: PTX a16 route ML_PTX_GEMM {} -> {}",
                old.as_str(),
                mode.as_str()
            );
        }
    }

    pub fn serves(&self, acc: CoopmatAcc) -> bool {
        self.mode().serves(acc)
    }

    pub fn kernel(&self, index: usize) -> &HgemmKernel {
        &self.kernels[index].0
    }

    /// Pick the PTX kernel for a batch-1 `M x N x K` a16 matmul at
    /// accumulator class `acc`, or `None` to keep the Vulkan route
    /// (class not served, or a shape the kernels do not take).  The
    /// returned kernel's class is always `acc`.
    pub fn select(&self, acc: CoopmatAcc, m: u32, n: u32, k: u32) -> Option<usize> {
        if !self.serves(acc) || !n.is_multiple_of(8) || !k.is_multiple_of(8) {
            return None;
        }
        if let Some(i) = self.forced
            && self.kernels[i].0.acc() == acc
        {
            return Some(i);
        }
        let name = route_kernel_name(acc, m, n, self.sm_count);
        let index = self.kernels.iter().position(|(k, _)| k.name == name)?;
        debug_assert_eq!(self.kernels[index].0.acc(), acc);
        Some(index)
    }

    /// # Safety
    /// `cb` is recording; `a`/`b`/`c` are 16-byte aligned device
    /// addresses of f16 buffers holding `M*K`, `K*N`, `M*N` elements.
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
        let (kernel, func) = &self.kernels[index];
        unsafe { kernel.cmd_launch(func, cb, a, b, c, (m, n, k, 1), alpha) };
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
    Some(sm.shader_sm_count).filter(|&n| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLASSES: [CoopmatAcc; 3] = [CoopmatAcc::F32, CoopmatAcc::F16, CoopmatAcc::F16Promote];

    #[test]
    fn every_pick_has_the_requested_class() {
        for acc in CLASSES {
            for sms in [1, 24, 46, 128] {
                for m in [64, 128, 512, 2048, 4096, 8192] {
                    for n in [64, 256, 2048, 5632, 32000] {
                        let name = route_kernel_name(acc, m, n, sms);
                        assert!(ROUTE_KERNELS.contains(&name), "{name} not resolved at load");
                        let kernel = HgemmKernel::by_name(name).expect("route kernel exists");
                        assert_eq!(kernel.acc(), acc, "{name} for {acc:?}");
                        assert!(kernel.out_f16, "{name} must store f16 C");
                    }
                }
            }
        }
        for name in ROUTE_KERNELS {
            assert!(HgemmKernel::by_name(name).is_some(), "{name}");
        }
    }

    #[test]
    fn measured_picks_are_kept() {
        // TinyLlama prefill shapes on the 4060 (24 SMs) and 3070 (46).
        assert_eq!(
            route_kernel_name(CoopmatAcc::F16, 512, 5632, 24),
            "ta_hgemm_f16_128x128_s3"
        );
        assert_eq!(
            route_kernel_name(CoopmatAcc::F16, 512, 2048, 46),
            "ta_hgemm_f16_128x64_s3"
        );
        assert_eq!(
            route_kernel_name(CoopmatAcc::F16, 4096, 4096, 24),
            "ta_hgemm_f16_128x128_s3_g8"
        );
        assert_eq!(
            route_kernel_name(CoopmatAcc::F16Promote, 512, 2048, 24),
            "ta_hgemm_f16p_128x64_k64_s2"
        );
        assert_eq!(
            route_kernel_name(CoopmatAcc::F32, 512, 2048, 24),
            "ta_hgemm_f32_64x64_s4"
        );
    }

    #[test]
    fn mode_parsing_and_served_classes() {
        assert_eq!(PtxGemmMode::parse("").unwrap(), PtxGemmMode::Auto);
        assert_eq!(PtxGemmMode::parse(" AUTO ").unwrap(), PtxGemmMode::Auto);
        assert_eq!(PtxGemmMode::parse("off").unwrap(), PtxGemmMode::Off);
        assert_eq!(PtxGemmMode::parse("0").unwrap(), PtxGemmMode::Off);
        assert_eq!(PtxGemmMode::parse("all").unwrap(), PtxGemmMode::All);
        assert_eq!(PtxGemmMode::parse("on").unwrap(), PtxGemmMode::Auto);
        // Deprecated precision spellings: the mode that serves the
        // class (the class itself comes from `legacy_env_acc`).
        assert_eq!(PtxGemmMode::parse("f16").unwrap(), PtxGemmMode::Auto);
        assert_eq!(PtxGemmMode::parse("f16p").unwrap(), PtxGemmMode::Auto);
        assert_eq!(PtxGemmMode::parse("f32").unwrap(), PtxGemmMode::All);
        assert_eq!(legacy_class("f16p"), Some(CoopmatAcc::F16Promote));
        assert_eq!(legacy_class(" F16 "), Some(CoopmatAcc::F16));
        assert_eq!(legacy_class("f32"), Some(CoopmatAcc::F32));
        assert_eq!(legacy_class("on"), None);
        for class in CLASSES {
            // Every legacy class is served by the mode it maps to.
            assert!(PtxGemmMode::parse(class.as_str()).unwrap().serves(class));
        }
        assert!(PtxGemmMode::parse("fast").is_err());
        assert!(!PtxGemmMode::Auto.serves(CoopmatAcc::F32));
        assert!(PtxGemmMode::Auto.serves(CoopmatAcc::F16));
        assert!(PtxGemmMode::Auto.serves(CoopmatAcc::F16Promote));
        assert!(CLASSES.iter().all(|&acc| PtxGemmMode::All.serves(acc)));
        assert!(CLASSES.iter().all(|&acc| !PtxGemmMode::Off.serves(acc)));
        for mode in [PtxGemmMode::Off, PtxGemmMode::Auto, PtxGemmMode::All] {
            assert_eq!(PtxGemmMode::from_u8(mode.to_u8()), mode);
        }
    }
}
