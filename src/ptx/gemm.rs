//! PTX GEMM kernel descriptors and launch helpers.

use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use super::{CudaFunction, CudaModule, KernelArg, kernels};
use crate::context::VulkanContext;

/// One f32 SGEMM entry in `ptx/sgemm.cu`.
#[derive(Copy, Clone, Debug)]
pub struct SgemmKernel {
    pub name: &'static str,
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub stages: u32,
    pub threads: u32,
    /// 16-byte cp.async + vector stores: needs K % 4 == 0, N % 4 == 0.
    pub vec4: bool,
}

const fn sk(name: &'static str, bk: u32, stages: u32, vec4: bool) -> SgemmKernel {
    SgemmKernel {
        name,
        bm: 128,
        bn: 128,
        bk,
        stages,
        threads: 256,
        vec4,
    }
}

pub const SGEMM_KERNELS: &[SgemmKernel] = &[
    sk("ta_sgemm_128x128x8_s4_v4", 8, 4, true),
    sk("ta_sgemm_128x128x8_s3_v4", 8, 3, true),
    sk("ta_sgemm_128x128x16_s3_v4", 16, 3, true),
    sk("ta_sgemm_128x128x16_s2_v4", 16, 2, true),
    sk("ta_sgemm_128x128x8_s4_s1", 8, 4, false),
];

impl SgemmKernel {
    pub fn supports(&self, _m: u32, n: u32, k: u32) -> bool {
        !self.vec4 || (k % 4 == 0 && n % 4 == 0)
    }

    /// Total shared memory: `stages * (BM*(BK+4) + BK*BN)` floats.
    pub fn smem_bytes(&self) -> u32 {
        self.stages * (self.bm * (self.bk + 4) + self.bk * self.bn) * 4
    }

    /// Dynamic shared memory to request at launch: the kernel uses a
    /// static array up to 48 KiB and dynamic memory above.
    pub fn launch_smem_bytes(&self) -> u32 {
        let total = self.smem_bytes();
        if total <= 48 * 1024 { 0 } else { total }
    }
}

/// One f16 tensor-core GEMM entry in `ptx/hgemm.cu`.
#[derive(Copy, Clone, Debug)]
pub struct HgemmKernel {
    pub name: &'static str,
    pub bm: u32,
    pub bn: u32,
    pub threads: u32,
    pub stages: u32,
    /// f16 accumulate (`mma ...f16.f16.f16.f16`) instead of f32.
    pub f16_acc: bool,
    /// C stored as f16 (the a16 route contract) instead of f32.
    pub out_f16: bool,
}

const fn hk(
    name: &'static str,
    bm: u32,
    bn: u32,
    threads: u32,
    stages: u32,
    f16_acc: bool,
    out_f16: bool,
) -> HgemmKernel {
    HgemmKernel {
        name,
        bm,
        bn,
        threads,
        stages,
        f16_acc,
        out_f16,
    }
}

pub const HGEMM_KERNELS: &[HgemmKernel] = &[
    hk("ta_hgemm_f32_128x128_s3", 128, 128, 128, 3, false, true),
    hk("ta_hgemm_f32_128x128_s4", 128, 128, 128, 4, false, true),
    hk("ta_hgemm_f32_128x256_s3", 128, 256, 256, 3, false, true),
    hk("ta_hgemm_f32_128x64_s3", 128, 64, 128, 3, false, true),
    hk("ta_hgemm_f32_128x64_s4", 128, 64, 128, 4, false, true),
    hk("ta_hgemm_f32_64x64_s4", 64, 64, 128, 4, false, true),
    hk(
        "ta_hgemm_f32_128x128_s3_o32",
        128,
        128,
        128,
        3,
        false,
        false,
    ),
    hk("ta_hgemm_f16_128x128_s3", 128, 128, 128, 3, true, true),
    hk("ta_hgemm_f16_128x128_s4", 128, 128, 128, 4, true, true),
    hk("ta_hgemm_f16_128x256_s3", 128, 256, 256, 3, true, true),
    hk("ta_hgemm_f16_128x64_s3", 128, 64, 128, 3, true, true),
    hk("ta_hgemm_f16_128x64_s4", 128, 64, 128, 4, true, true),
    hk("ta_hgemm_f16_64x64_s4", 64, 64, 128, 4, true, true),
];

impl HgemmKernel {
    /// Dynamic shared memory: `stages * (BM*32 + 32*BN)` halves.
    pub fn smem_bytes(&self) -> u32 {
        self.stages * (self.bm * 32 + 32 * self.bn) * 2
    }

    pub fn supports(&self, _m: u32, n: u32, k: u32) -> bool {
        k % 8 == 0 && n % 8 == 0
    }
}

/// Both GEMM modules, JIT-compiled once, with every entry resolved.
pub struct PtxGemm {
    pub sgemm: Vec<(SgemmKernel, CudaFunction)>,
    pub hgemm: Vec<(HgemmKernel, CudaFunction)>,
    /// Driver JIT time for (sgemm, hgemm) modules.
    pub jit_ms: (f64, f64),
}

impl PtxGemm {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let t = std::time::Instant::now();
        let sm = CudaModule::from_ptx(ctx, kernels::SGEMM_PTX)?;
        let s_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let hm = CudaModule::from_ptx(ctx, kernels::HGEMM_PTX)?;
        let h_ms = t.elapsed().as_secs_f64() * 1e3;
        let sgemm = SGEMM_KERNELS
            .iter()
            .map(|k| Ok((*k, sm.function(k.name)?)))
            .collect::<Result<Vec<_>>>()?;
        let hgemm = HGEMM_KERNELS
            .iter()
            .map(|k| Ok((*k, hm.function(k.name)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sgemm,
            hgemm,
            jit_ms: (s_ms, h_ms),
        })
    }

    pub fn sgemm_index(&self, name: &str) -> Option<usize> {
        self.sgemm.iter().position(|(k, _)| k.name == name)
    }

    pub fn hgemm_index(&self, name: &str) -> Option<usize> {
        self.hgemm.iter().position(|(k, _)| k.name == name)
    }

    /// Record `C = alpha*A*B (+ beta*C)` (f32, row-major, dense batch).
    ///
    /// # Safety
    /// `cb` is recording; `a`/`b`/`c` are device addresses of buffers
    /// holding at least `batch*M*K`, `batch*K*N`, `batch*M*N` f32.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cmd_sgemm(
        &self,
        cb: vk::CommandBuffer,
        index: usize,
        a: u64,
        b: u64,
        c: u64,
        (m, n, k, batch): (u32, u32, u32, u32),
        alpha: f32,
        beta: f32,
    ) {
        let (kern, func) = &self.sgemm[index];
        let grid = [n.div_ceil(kern.bn), m.div_ceil(kern.bm), batch];
        let args = [
            KernelArg::U64(a),
            KernelArg::U64(b),
            KernelArg::U64(c),
            KernelArg::I32(m as i32),
            KernelArg::I32(n as i32),
            KernelArg::I32(k as i32),
            KernelArg::F32(alpha),
            KernelArg::F32(beta),
        ];
        unsafe {
            func.cmd_launch(
                cb,
                grid,
                [kern.threads, 1, 1],
                kern.launch_smem_bytes(),
                &args,
            )
        };
    }

    /// Record `C = alpha*A*B` with f16 A/B (row-major, dense batch).
    ///
    /// # Safety
    /// As [`Self::cmd_sgemm`], with f16 A/B and f16 or f32 C per kernel.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cmd_hgemm(
        &self,
        cb: vk::CommandBuffer,
        index: usize,
        a: u64,
        b: u64,
        c: u64,
        (m, n, k, batch): (u32, u32, u32, u32),
        alpha: f32,
    ) {
        let (kern, func) = &self.hgemm[index];
        let grid = [n.div_ceil(kern.bn), m.div_ceil(kern.bm), batch];
        let args = [
            KernelArg::U64(a),
            KernelArg::U64(b),
            KernelArg::U64(c),
            KernelArg::I32(m as i32),
            KernelArg::I32(n as i32),
            KernelArg::I32(k as i32),
            KernelArg::F32(alpha),
        ];
        unsafe { func.cmd_launch(cb, grid, [kern.threads, 1, 1], kern.smem_bytes(), &args) };
    }
}
