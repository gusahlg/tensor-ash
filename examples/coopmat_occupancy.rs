//! Driver-reported register counts and theoretical CTAs/SM for the
//! coopmat GEMM tiles (the occupancy input of the SM-aware tile model,
//! `benchmarks/campaign/coopmat-sm.md`).
//!
//! Builds each selected kernel on a *second* logical device that
//! enables `VK_KHR_pipeline_executable_properties` (the library's own
//! device never does, so production pipelines stay untouched), with
//! `CAPTURE_STATISTICS`, and prints every statistic the driver reports
//! plus the occupancy those numbers imply on an NVIDIA SM.
//!
//!     ML_STATS_FILTER=coopmat cargo run --release --example coopmat_occupancy
//!
//! `ML_STATS_FILTER` is a comma-separated list of kernel-name
//! substrings (default `coopmat`); `ML_MAX_CTAS_PER_SM` overrides the
//! per-SM CTA cap (default 16 for sm_86, 24 when the device name says
//! RTX 40).

use std::ffi::{CStr, c_char};

use anyhow::{Context, Result, bail};
use ash::vk;
use tensor_ash::{KERNEL_SPECS, MatmulPushConstants, VulkanContext};

/// NVIDIA sm_86 / sm_89 per-SM limits (CUDA occupancy calculator).
const REGS_PER_SM: u32 = 65_536;
const SMEM_PER_SM: u32 = 102_400;
const SMEM_RESERVED_PER_CTA: u32 = 1_024;
const SMEM_GRANULE: u32 = 128;
const REG_ALLOC_UNIT_PER_WARP: u32 = 256;

fn c_str(chars: &[c_char]) -> String {
    unsafe { CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// `OpExecutionMode %main LocalSize x y z` from the SPIR-V words.
fn local_size(spv: &[u8]) -> Option<[u32; 3]> {
    const OP_EXECUTION_MODE: u32 = 16;
    const MODE_LOCAL_SIZE: u32 = 17;
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut pc = 5;
    while pc < words.len() {
        let count = (words[pc] >> 16) as usize;
        let opcode = words[pc] & 0xFFFF;
        if count == 0 {
            return None;
        }
        if opcode == OP_EXECUTION_MODE && count >= 6 && words[pc + 2] == MODE_LOCAL_SIZE {
            return Some([words[pc + 3], words[pc + 4], words[pc + 5]]);
        }
        pc += count;
    }
    None
}

fn statistic_value(stat: &vk::PipelineExecutableStatisticKHR<'_>) -> String {
    unsafe {
        match stat.format {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (stat.value.b32 != 0).to_string(),
            vk::PipelineExecutableStatisticFormatKHR::INT64 => stat.value.i64.to_string(),
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => stat.value.u64.to_string(),
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => stat.value.f64.to_string(),
            _ => "?".into(),
        }
    }
}

fn statistic_u64(stat: &vk::PipelineExecutableStatisticKHR<'_>) -> Option<u64> {
    unsafe {
        match stat.format {
            vk::PipelineExecutableStatisticFormatKHR::INT64 => u64::try_from(stat.value.i64).ok(),
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => Some(stat.value.u64),
            _ => None,
        }
    }
}

struct Occupancy {
    by_regs: u32,
    by_smem: u32,
    by_warps: u32,
    ctas: u32,
}

fn occupancy(threads: u32, regs: u32, smem: u32, max_warps: u32, max_ctas: u32) -> Occupancy {
    let warps = threads.div_ceil(32);
    let regs_per_warp = (regs * 32).div_ceil(REG_ALLOC_UNIT_PER_WARP) * REG_ALLOC_UNIT_PER_WARP;
    let by_regs = REGS_PER_SM / (regs_per_warp * warps).max(1);
    let smem_alloc = (smem + SMEM_RESERVED_PER_CTA).div_ceil(SMEM_GRANULE) * SMEM_GRANULE;
    let by_smem = SMEM_PER_SM / smem_alloc.max(1);
    let by_warps = max_warps / warps.max(1);
    Occupancy {
        by_regs,
        by_smem,
        by_warps,
        ctas: by_regs.min(by_smem).min(by_warps).min(max_ctas),
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let ctx = VulkanContext::new(false)?;
    let filters: Vec<String> = std::env::var("ML_STATS_FILTER")
        .unwrap_or_else(|_| "coopmat".into())
        .split(',')
        .map(|f| f.trim().to_ascii_lowercase())
        .filter(|f| !f.is_empty())
        .collect();
    let max_ctas = std::env::var("ML_MAX_CTAS_PER_SM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if ctx.device_name().contains("RTX 40") {
            24
        } else {
            16
        });
    let max_warps = ctx
        .shader_cores
        .and_then(|cores| cores.warps_per_core)
        .unwrap_or(48);
    println!("{}", ctx.diagnostics());
    println!(
        "occupancy limits: {max_warps} warps/SM, {max_ctas} CTAs/SM, {REGS_PER_SM} regs/SM, {SMEM_PER_SM} B shared/SM (+{SMEM_RESERVED_PER_CTA} B/CTA reserved)"
    );

    unsafe {
        let exts = ctx
            .instance
            .enumerate_device_extension_properties(ctx.physical_device)?;
        let has = |name: &CStr| {
            exts.iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
        };
        if !has(ash::khr::pipeline_executable_properties::NAME) {
            bail!("device lacks VK_KHR_pipeline_executable_properties");
        }
        let coopmat = has(ash::khr::cooperative_matrix::NAME) && ctx.coopmat_enabled;
        let mut ext_names = vec![ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        if coopmat {
            ext_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        let priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(ctx.compute_family)
            .queue_priorities(&priorities)];
        let mut exec_props = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(ctx.f16_storage_enabled);
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(ctx.buffer_device_address_enabled)
            .shader_float16(ctx.f16_storage_enabled)
            .vulkan_memory_model(coopmat);
        let mut cm =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(coopmat);
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_extension_names(&ext_names)
            .push_next(&mut exec_props)
            .push_next(&mut v11)
            .push_next(&mut v12);
        if coopmat {
            device_ci = device_ci.push_next(&mut cm);
        }
        let device = ctx
            .instance
            .create_device(ctx.physical_device, &device_ci, None)
            .context("create stats device")?;
        let exec_fn = ash::khr::pipeline_executable_properties::Device::new(&ctx.instance, &device);

        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<MatmulPushConstants>() as u32)];
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_range),
            None,
        )?;

        println!(
            "\n{:<30} {:>7} {:>7} {:>6} {:>5} {:>5} {:>6} {:>6} {:>6}  driver statistics",
            "kernel", "tile", "threads", "smem", "regs", "ctas", "byReg", "bySmem", "byWarp"
        );
        for spec in KERNEL_SPECS {
            let lower = spec.name.to_ascii_lowercase();
            if !filters.iter().any(|f| lower.contains(f)) {
                continue;
            }
            if spec.uses_descriptors
                || (spec.name.contains("coopmat") && !coopmat)
                || spec.name.contains("cm2")
            {
                continue;
            }
            let words: Vec<u32> = spec
                .spv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR)
                .stage(stage)
                .layout(layout);
            let pipeline = match device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&ci),
                None,
            ) {
                Ok(p) => p[0],
                Err((_, err)) => {
                    println!("{:<30} create failed: {err}", spec.name);
                    device.destroy_shader_module(module, None);
                    continue;
                }
            };
            let executables = exec_fn.get_pipeline_executable_properties(
                &vk::PipelineInfoKHR::default().pipeline(pipeline),
            )?;
            let threads = local_size(spec.spv).map_or(0, |[x, y, z]| x * y * z);
            let smem = spec.shared_memory_bytes();
            for index in 0..executables.len() as u32 {
                let stats = exec_fn.get_pipeline_executable_statistics(
                    &vk::PipelineExecutableInfoKHR::default()
                        .pipeline(pipeline)
                        .executable_index(index),
                )?;
                let regs = stats
                    .iter()
                    .find(|s| c_str(&s.name).to_ascii_lowercase().contains("register"))
                    .and_then(statistic_u64)
                    .map(|r| r as u32);
                let rendered = stats
                    .iter()
                    .map(|s| format!("{}={}", c_str(&s.name), statistic_value(s)))
                    .collect::<Vec<_>>()
                    .join("; ");
                let (regs_s, occ) = match regs {
                    Some(r) => (
                        r.to_string(),
                        Some(occupancy(threads, r, smem, max_warps, max_ctas)),
                    ),
                    None => ("?".into(), None),
                };
                println!(
                    "{:<30} {:>7} {:>7} {:>6} {:>5} {:>5} {:>6} {:>6} {:>6}  {}",
                    spec.name,
                    format!("{}x{}", spec.tile_m, spec.tile_n),
                    threads,
                    smem,
                    regs_s,
                    occ.as_ref().map_or("?".into(), |o| o.ctas.to_string()),
                    occ.as_ref().map_or("?".into(), |o| o.by_regs.to_string()),
                    occ.as_ref().map_or("?".into(), |o| o.by_smem.to_string()),
                    occ.as_ref().map_or("?".into(), |o| o.by_warps.to_string()),
                    rendered
                );
            }
            device.destroy_pipeline(pipeline, None);
            device.destroy_shader_module(module, None);
        }
        device.destroy_pipeline_layout(layout, None);
        device.destroy_device(None);
    }
    Ok(())
}
