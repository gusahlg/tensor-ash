//! Checked-in PTX images (generated from `ptx/*.cu` by `ptx/build.sh`,
//! target `sm_80`, JIT-compiled by the driver for the actual GPU).

/// Feasibility-probe kernels: `probe_fill`, `probe_affine`,
/// `probe_bump`, `probe_smem`, `probe_stamp`.
pub const PROBE_PTX: &str = include_str!("../../ptx/probe.ptx");

/// f32 SIMT SGEMM with a 3/4-stage cp.async pipeline.
pub const SGEMM_PTX: &str = include_str!("../../ptx/sgemm.ptx");

/// f16 tensor-core GEMM (mma.sync m16n8k16 + ldmatrix + cp.async),
/// f32 or f16 accumulate.
pub const HGEMM_PTX: &str = include_str!("../../ptx/hgemm.ptx");
