//! The compute pipeline + descriptor-set-layout for the matmul kernel.
//!
//! Descriptor *pools* live in the executor (one per command-buffer slot),
//! so this module owns only kernel-shaped resources.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::context::VulkanContext;

/// Push constants for the matmul shader.  Bit-for-bit identical to the
/// GLSL `PC` block.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatmulPushConstants {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_c: u32,
    pub flags: u32, // bit 0: accumulate
    pub alpha: f32,
}

/// Default large-kernel output-tile dimensions.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 128;
pub const SMALL_TILE_M: u32 = 64;
pub const SMALL_TILE_N: u32 = 64;

const SPV_MATMUL_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32.spv"));
const SPV_MATMUL_F32_SMALL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_small.spv"));

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelSelection {
    Auto,
    Large,
    Small,
}

impl KernelSelection {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "large" | "large_128" | "128" => Ok(Self::Large),
            "small" | "small_64" | "64" => Ok(Self::Small),
            other => bail!("invalid ML_KERNEL '{other}', expected auto, large, or small"),
        }
    }

    fn from_env() -> Result<Self> {
        let value = std::env::var("ML_KERNEL").unwrap_or_else(|_| "auto".into());
        Self::parse(&value)
    }
}

/// Per-call pipeline specialization.  Selects one of the precompiled
/// variants of a kernel so the shader sees these as compile-time
/// constants and can fold out the corresponding branches.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct KernelVariant {
    /// `C += alpha*A@B` when true, `C = alpha*A@B` when false.
    pub accumulate: bool,
    /// Host knows alpha==1.0 — shader skips the multiply.
    pub alpha_is_one: bool,
    /// Host knows M and N are multiples of the tile size — shader
    /// drops all m_full/n_full bounds checks.
    pub interior_only: bool,
}

impl KernelVariant {
    /// Number of distinct variants compiled per kernel (one per
    /// combination of the three booleans).
    pub const COUNT: usize = 8;

    #[inline]
    pub const fn index(self) -> usize {
        (self.accumulate as usize)
            | ((self.alpha_is_one as usize) << 1)
            | ((self.interior_only as usize) << 2)
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Self {
        Self {
            accumulate: (idx & 0b001) != 0,
            alpha_is_one: (idx & 0b010) != 0,
            interior_only: (idx & 0b100) != 0,
        }
    }
}

pub struct MatmulKernel {
    pub name: &'static str,
    pub tile_m: u32,
    pub tile_n: u32,
    pub shader_module: vk::ShaderModule,
    /// One pipeline per `KernelVariant`; indexed by `KernelVariant::index()`.
    pub variants: [vk::Pipeline; KernelVariant::COUNT],
}

impl MatmulKernel {
    #[inline]
    pub fn pipeline_for(&self, variant: KernelVariant) -> vk::Pipeline {
        self.variants[variant.index()]
    }
}

pub struct MatmulPipeline {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub large: MatmulKernel,
    pub small: MatmulKernel,
    selection: KernelSelection,
}

impl MatmulPipeline {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        Self::new_with_kernel_selection(ctx, KernelSelection::from_env()?)
    }

    pub fn new_with_kernel_selection(
        ctx: &Arc<VulkanContext>,
        selection: KernelSelection,
    ) -> Result<Self> {
        unsafe {
            // ---- Descriptor set layout: A, B, C storage buffers ----
            let bindings = (0u32..3)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let set_layout = ctx
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .context("create_descriptor_set_layout")?;

            // ---- Pipeline layout: 1 set + push constants ----
            let pc_size = std::mem::size_of::<MatmulPushConstants>() as u32;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size)];
            let set_layouts = [set_layout];
            let pipeline_layout = match ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&pc_ranges),
                None,
            ) {
                Ok(pipeline_layout) => pipeline_layout,
                Err(err) => {
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create_pipeline_layout");
                }
            };

            // ---- Compute pipeline ----
            let large = match create_kernel(
                ctx,
                pipeline_layout,
                "large_128",
                TILE_M,
                TILE_N,
                SPV_MATMUL_F32,
            ) {
                Ok(large) => large,
                Err(err) => {
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create large matmul kernel");
                }
            };
            let small = match create_kernel(
                ctx,
                pipeline_layout,
                "small_64",
                SMALL_TILE_M,
                SMALL_TILE_N,
                SPV_MATMUL_F32_SMALL,
            ) {
                Ok(small) => small,
                Err(err) => {
                    for p in large.variants.iter().copied() {
                        if p != vk::Pipeline::null() {
                            ctx.device.destroy_pipeline(p, None);
                        }
                    }
                    ctx.device.destroy_shader_module(large.shader_module, None);
                    ctx.device.destroy_pipeline_layout(pipeline_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(err).context("create small matmul kernel");
                }
            };

            Ok(Self {
                ctx: Arc::clone(ctx),
                set_layout,
                pipeline_layout,
                large,
                small,
                selection,
            })
        }
    }

    pub fn select_kernel(&self, m: u32, n: u32, k: u32) -> &MatmulKernel {
        match self.selection {
            KernelSelection::Large => &self.large,
            KernelSelection::Small => &self.small,
            KernelSelection::Auto if auto_selects_small_kernel(m, n, k) => &self.small,
            KernelSelection::Auto => &self.large,
        }
    }
}

fn auto_selects_small_kernel(m: u32, n: u32, k: u32) -> bool {
    // Alignment is a hard constraint: the large 128x128 kernel has no
    // tile-edge fast path that would let it accept off-tile M or N.
    if !m.is_multiple_of(TILE_M) || !n.is_multiple_of(TILE_N) {
        return true;
    }
    // For tiny K the fixed per-workgroup load + barrier cost dominates,
    // so the smaller-tile kernel (4x more workgroups, 4x more chances to
    // overlap latency) wins.
    if k < 128 {
        return true;
    }
    // The large 128-tile kernel needs roughly 2 workgroups per SM to hide
    // memory latency.  Mid-range GPUs have 30-80 SMs, so ~256 large-tile
    // workgroups is the saturation point.  Below that the small kernel
    // (4x more workgroups for the same output) wins on occupancy despite
    // its lower arithmetic intensity.
    //
    // This was measured on RTX 3070 (46 SMs): at 1024^3 the small kernel
    // is faster (7.8 vs 7.0 TFLOPS), but at 2048^3 large wins decisively
    // (9.8 vs 8.6 TFLOPS).
    let large_tiles = (m / TILE_M) as u64 * (n / TILE_N) as u64;
    if large_tiles < 256 {
        return true;
    }
    false
}

impl Drop for MatmulPipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            for kernel in [&self.large, &self.small] {
                for p in kernel.variants.iter().copied() {
                    if p != vk::Pipeline::null() {
                        self.ctx.device.destroy_pipeline(p, None);
                    }
                }
                self.ctx
                    .device
                    .destroy_shader_module(kernel.shader_module, None);
            }
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn create_kernel(
    ctx: &Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    name: &'static str,
    tile_m: u32,
    tile_n: u32,
    spv: &[u8],
) -> Result<MatmulKernel> {
    unsafe {
        assert!(spv.len().is_multiple_of(4), "SPIR-V size not 4-aligned");
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_module = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .context("create_shader_module")?;
        let entry = std::ffi::CString::new("main").unwrap();

        // ---- One pipeline per (accumulate, alpha_is_one, interior_only) tuple,
        // ---- built in a single batched vkCreateComputePipelines call so the
        // ---- driver can amortize and de-duplicate ISA compilation.

        // SPIR-V `OpConstantTrue/False` for a bool spec constant takes a
        // 32-bit value (driver reads the LSB).  Layout: 3 x u32 per variant.
        let spec_entries = [
            vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(1)
                .offset(4)
                .size(4),
            vk::SpecializationMapEntry::default()
                .constant_id(2)
                .offset(8)
                .size(4),
        ];

        let spec_data: Vec<[u32; 3]> = (0..KernelVariant::COUNT)
            .map(|i| {
                let v = KernelVariant::from_index(i);
                [
                    v.accumulate as u32,
                    v.alpha_is_one as u32,
                    v.interior_only as u32,
                ]
            })
            .collect();

        let spec_infos: Vec<vk::SpecializationInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::SpecializationInfo::default()
                    .map_entries(&spec_entries)
                    .data(bytemuck::cast_slice(&spec_data[i]))
            })
            .collect();

        let stages: Vec<vk::PipelineShaderStageCreateInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(&entry)
                    .specialization_info(&spec_infos[i])
            })
            .collect();

        let create_infos: Vec<vk::ComputePipelineCreateInfo> = (0..KernelVariant::COUNT)
            .map(|i| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(stages[i])
                    .layout(pipeline_layout)
            })
            .collect();

        let pipelines = ctx
            .device
            .create_compute_pipelines(ctx.pipeline_cache, &create_infos, None)
            .map_err(|(_, err)| {
                ctx.device.destroy_shader_module(shader_module, None);
                anyhow::anyhow!("create_compute_pipelines {name}: {err}")
            })?;

        let mut variants = [vk::Pipeline::null(); KernelVariant::COUNT];
        for (i, p) in pipelines.iter().enumerate() {
            variants[i] = *p;
        }

        Ok(MatmulKernel {
            name,
            tile_m,
            tile_n,
            shader_module,
            variants,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_selection() {
        assert_eq!(KernelSelection::parse("").unwrap(), KernelSelection::Auto);
        assert_eq!(
            KernelSelection::parse("auto").unwrap(),
            KernelSelection::Auto
        );
        assert_eq!(
            KernelSelection::parse("large_128").unwrap(),
            KernelSelection::Large
        );
        assert_eq!(
            KernelSelection::parse("64").unwrap(),
            KernelSelection::Small
        );
        assert!(KernelSelection::parse("wide").is_err());
    }

    #[test]
    fn auto_selector_prefers_small_for_edge_or_small_shapes() {
        // Tile-misaligned M or N must always go small (large kernel can't
        // produce partial tiles).
        assert!(auto_selects_small_kernel(1023, 2048, 1024));
        assert!(auto_selects_small_kernel(2048, 1025, 1024));
        // Tiny K can't amortize the large kernel's fixed per-WG cost.
        assert!(auto_selects_small_kernel(2048, 2048, 64));
        // Few large-tile workgroups: small kernel saturates the GPU
        // better.  64 large WGs at 1024^2, 16 at 512^2.
        assert!(auto_selects_small_kernel(512, 512, 512));
        assert!(auto_selects_small_kernel(1024, 1024, 1024));
        // Enough large-tile workgroups to saturate (>=256).
        assert!(!auto_selects_small_kernel(2048, 2048, 128));
        assert!(!auto_selects_small_kernel(2048, 2048, 2048));
        assert!(!auto_selects_small_kernel(4096, 1024, 1024));
        assert!(!auto_selects_small_kernel(1024, 4096, 1024));
    }
}
