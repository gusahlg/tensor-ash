//! Experimental: PTX kernels recorded into tensor-ash's own Vulkan
//! command buffers through `VK_NV_cuda_kernel_launch`.
//!
//! The NVIDIA proprietary driver exposes this (provisional) extension
//! on Ampere/Ada: a PTX module is JIT-compiled by the driver
//! (`vkCreateCudaModuleNV`), kernels are looked up by name
//! (`vkCreateCudaFunctionNV`) and launched with
//! `vkCmdCudaLaunchKernelNV` *inside an ordinary command buffer*, next
//! to compute dispatches, barriers and timestamp queries.  Kernel
//! parameters follow `cuLaunchKernel`'s convention (an array of pointers
//! to each parameter value); device pointers are Vulkan buffer device
//! addresses.  That gives the library the primitives GLSL/SPIR-V cannot
//! express: `cp.async` multi-stage pipelines, `ldmatrix`, `mma.sync`
//! (incl. f16 accumulate and int8/int4), `.maxnreg`.
//!
//! Everything here is opt-in: the extension is only enabled when
//! `ML_PTX=1` (and not `ML_NO_PTX=1`) at context creation — see
//! [`VulkanContext::cuda_launch`].  The PTX sources are checked in
//! (`ptx/*.cu` -> `ptx/*.ptx` via `ptx/build.sh`), so building
//! tensor-ash never needs a CUDA toolkit.
//!
//! `VK_NVX_binary_import` (the cubin loader DXVK-NVAPI uses) is wrapped
//! as well ([`NvxModule`]) so prebuilt SASS can be tested alongside.

pub mod gemm;
pub mod kernels;
mod probe;
mod route;

pub use probe::ProbeAffine;
pub use route::{PtxGemmMode, PtxRoute, route_kernel_name};

use std::ffi::{CString, c_void};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

use crate::context::VulkanContext;

/// SPIR-V of `shaders/ptx_probe_affine.comp`, the Vulkan twin of
/// `probe_affine` used by the ordering probes (`ml_bench ptx sync`).
pub const PROBE_AFFINE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/ptx_probe_affine.spv"));

fn launch_fns(ctx: &VulkanContext) -> Result<&ash::nv::cuda_kernel_launch::Device> {
    ctx.cuda_launch.as_ref().ok_or_else(|| {
        anyhow!("VK_NV_cuda_kernel_launch is not enabled (set ML_PTX=1 on an NVIDIA driver)")
    })
}

fn module_cache_path(ctx: &VulkanContext, label: &str, ptx: &str) -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in ptx.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let s = &ctx.device_summary;
    Some(base.join("tensor-ash").join(format!(
        "ptx_{label}_v{:04x}_{:04x}_d{}_{h:016x}.cubin",
        s.vendor_id, s.device_id, s.driver_version
    )))
}

/// One kernel parameter, passed by value like a `cuLaunchKernel`
/// argument.  Device pointers are `U64` buffer device addresses.
#[derive(Copy, Clone, Debug)]
pub enum KernelArg {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
}

impl KernelArg {
    fn bits(self) -> u64 {
        match self {
            KernelArg::U32(v) => v as u64,
            KernelArg::I32(v) => v as u32 as u64,
            KernelArg::F32(v) => v.to_bits() as u64,
            KernelArg::U64(v) => v,
        }
    }
}

/// A driver-compiled CUDA module (`VkCudaModuleNV`).
pub struct CudaModule {
    ctx: Arc<VulkanContext>,
    raw: vk::CudaModuleNV,
}

impl CudaModule {
    /// JIT-compile PTX text.  The driver wants a NUL-terminated image
    /// (cuModuleLoadData semantics); one is appended here.
    pub fn from_ptx(ctx: &Arc<VulkanContext>, ptx: &str) -> Result<Arc<Self>> {
        let image = CString::new(ptx).context("PTX text contains NUL")?;
        Self::from_data(ctx, image.as_bytes_with_nul())
    }

    /// Create a module from raw data exactly as given: PTX (NUL
    /// terminated), or whatever else the driver accepts (cubin, fatbin,
    /// or a blob from [`Self::cache_data`]) — the probe tests which.
    pub fn from_data(ctx: &Arc<VulkanContext>, data: &[u8]) -> Result<Arc<Self>> {
        let fns = launch_fns(ctx)?;
        let ci = vk::CudaModuleCreateInfoNV::default().data(data);
        let raw = unsafe { fns.create_cuda_module(&ci, None) }
            .map_err(|e| anyhow!("vkCreateCudaModuleNV: {e}"))?;
        Ok(Arc::new(Self {
            ctx: Arc::clone(ctx),
            raw,
        }))
    }

    /// [`Self::from_ptx`] through a persistent per-device cache of the
    /// driver-compiled cubin (`vkGetCudaModuleCacheNV`), keyed by
    /// device, driver version and a hash of the PTX text.  A cold JIT
    /// of the GEMM modules costs 0.4-0.7 s; reloading the cubin costs
    /// ~0.1 ms.  Any cache miss/corruption falls back to the JIT.
    /// Returns the module and whether it came from the cache.
    pub fn from_ptx_cached(
        ctx: &Arc<VulkanContext>,
        label: &str,
        ptx: &str,
    ) -> Result<(Arc<Self>, bool)> {
        let path = module_cache_path(ctx, label, ptx);
        if let Some(path) = path.as_ref()
            && let Ok(blob) = std::fs::read(path)
        {
            match Self::from_data(ctx, &blob) {
                Ok(module) => return Ok((module, true)),
                Err(err) => log::warn!(
                    "tensor-ash: ignoring PTX module cache {} ({err})",
                    path.display()
                ),
            }
        }
        let module = Self::from_ptx(ctx, ptx)?;
        if let Some(path) = path
            && let Ok(blob) = module.cache_data()
        {
            let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
            let written = path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&tmp, &blob))
                .and_then(|()| std::fs::rename(&tmp, &path));
            if let Err(err) = written {
                log::warn!("tensor-ash: could not persist PTX module cache ({err})");
                let _ = std::fs::remove_file(&tmp);
            }
        }
        Ok((module, false))
    }

    /// `vkGetCudaModuleCacheNV`: the driver's compiled form of the
    /// module, which can be fed back to [`Self::from_data`].
    pub fn cache_data(&self) -> Result<Vec<u8>> {
        let fns = launch_fns(&self.ctx)?;
        unsafe { fns.get_cuda_module_cache(self.raw) }
            .map_err(|e| anyhow!("vkGetCudaModuleCacheNV: {e}"))
    }

    pub fn raw(&self) -> vk::CudaModuleNV {
        self.raw
    }

    /// Look up a `.entry` by name.
    pub fn function(self: &Arc<Self>, name: &str) -> Result<CudaFunction> {
        let fns = launch_fns(&self.ctx)?;
        let cname = CString::new(name)?;
        let ci = vk::CudaFunctionCreateInfoNV::default()
            .module(self.raw)
            .name(&cname);
        let raw = unsafe { fns.create_cuda_function(&ci, None) }
            .map_err(|e| anyhow!("vkCreateCudaFunctionNV({name}): {e}"))?;
        Ok(CudaFunction {
            module: Arc::clone(self),
            raw,
            name: name.to_owned(),
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if let Some(fns) = self.ctx.cuda_launch.as_ref() {
            unsafe { fns.destroy_cuda_module(self.raw, None) };
        }
    }
}

/// A kernel entry point (`VkCudaFunctionNV`); keeps its module alive.
pub struct CudaFunction {
    module: Arc<CudaModule>,
    raw: vk::CudaFunctionNV,
    name: String,
}

impl CudaFunction {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Record a launch of this kernel into `cb`.
    ///
    /// # Safety
    /// `cb` must be in the recording state on the context's device, the
    /// argument list must match the kernel's `.param` list exactly, and
    /// every pointer argument must be a live buffer device address with
    /// the extent the kernel touches.  The argument values are captured
    /// at record time (the probe verifies this), so `args` need not
    /// outlive the call.
    pub unsafe fn cmd_launch(
        &self,
        cb: vk::CommandBuffer,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        args: &[KernelArg],
    ) {
        let fns = self
            .module
            .ctx
            .cuda_launch
            .as_ref()
            .expect("cuda launch enabled");
        let slots: Vec<u64> = args.iter().map(|a| a.bits()).collect();
        let ptrs: Vec<*const c_void> = slots.iter().map(|s| (s as *const u64).cast()).collect();
        let info = vk::CudaLaunchInfoNV::default()
            .function(self.raw)
            .grid_dim_x(grid[0])
            .grid_dim_y(grid[1])
            .grid_dim_z(grid[2])
            .block_dim_x(block[0])
            .block_dim_y(block[1])
            .block_dim_z(block[2])
            .shared_mem_bytes(shared_mem_bytes)
            .params(&ptrs);
        unsafe { fns.cmd_cuda_launch_kernel(cb, &info) };
    }
}

impl Drop for CudaFunction {
    fn drop(&mut self) {
        if let Some(fns) = self.module.ctx.cuda_launch.as_ref() {
            unsafe { fns.destroy_cuda_function(self.raw, None) };
        }
    }
}

/// A `VK_NVX_binary_import` module (`VkCuModuleNVX`): the cubin path
/// DXVK-NVAPI uses.  Probe-only.
pub struct NvxModule {
    ctx: Arc<VulkanContext>,
    raw: vk::CuModuleNVX,
}

impl NvxModule {
    pub fn from_data(ctx: &Arc<VulkanContext>, data: &[u8]) -> Result<Arc<Self>> {
        let Some(fns) = ctx.nvx_binary_import.as_ref() else {
            bail!("VK_NVX_binary_import is not enabled");
        };
        let ci = vk::CuModuleCreateInfoNVX::default().data(data);
        let mut raw = vk::CuModuleNVX::null();
        let r = unsafe {
            (fns.fp().create_cu_module_nvx)(fns.device(), &ci, std::ptr::null(), &mut raw)
        };
        if r != vk::Result::SUCCESS {
            bail!("vkCreateCuModuleNVX: {r}");
        }
        Ok(Arc::new(Self {
            ctx: Arc::clone(ctx),
            raw,
        }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<NvxFunction> {
        let fns = self.ctx.nvx_binary_import.as_ref().expect("nvx enabled");
        let cname = CString::new(name)?;
        let ci = vk::CuFunctionCreateInfoNVX::default()
            .module(self.raw)
            .name(&cname);
        let mut raw = vk::CuFunctionNVX::null();
        let r = unsafe {
            (fns.fp().create_cu_function_nvx)(fns.device(), &ci, std::ptr::null(), &mut raw)
        };
        if r != vk::Result::SUCCESS {
            bail!("vkCreateCuFunctionNVX({name}): {r}");
        }
        Ok(NvxFunction {
            module: Arc::clone(self),
            raw,
        })
    }
}

impl Drop for NvxModule {
    fn drop(&mut self) {
        if let Some(fns) = self.ctx.nvx_binary_import.as_ref() {
            unsafe { (fns.fp().destroy_cu_module_nvx)(fns.device(), self.raw, std::ptr::null()) };
        }
    }
}

pub struct NvxFunction {
    module: Arc<NvxModule>,
    raw: vk::CuFunctionNVX,
}

impl NvxFunction {
    /// # Safety
    /// Same contract as [`CudaFunction::cmd_launch`].
    pub unsafe fn cmd_launch(
        &self,
        cb: vk::CommandBuffer,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        args: &[KernelArg],
    ) {
        let fns = self
            .module
            .ctx
            .nvx_binary_import
            .as_ref()
            .expect("nvx enabled");
        let slots: Vec<u64> = args.iter().map(|a| a.bits()).collect();
        let ptrs: Vec<*const c_void> = slots.iter().map(|s| (s as *const u64).cast()).collect();
        let info = vk::CuLaunchInfoNVX::default()
            .function(self.raw)
            .grid_dim_x(grid[0])
            .grid_dim_y(grid[1])
            .grid_dim_z(grid[2])
            .block_dim_x(block[0])
            .block_dim_y(block[1])
            .block_dim_z(block[2])
            .shared_mem_bytes(shared_mem_bytes)
            .params(&ptrs);
        unsafe { (fns.fp().cmd_cu_launch_kernel_nvx)(cb, &info) };
    }
}

impl Drop for NvxFunction {
    fn drop(&mut self) {
        if let Some(fns) = self.module.ctx.nvx_binary_import.as_ref() {
            unsafe { (fns.fp().destroy_cu_function_nvx)(fns.device(), self.raw, std::ptr::null()) };
        }
    }
}
