//! Persistent-threads experimental kernel.
//!
//! This is an opt-in side-channel pipeline that lives outside of
//! `MatmulPipeline` / `KERNEL_SPECS` so it doesn't have to round-trip
//! through the regular auto-selector or fit the standard
//! push-constant layout.
//!
//! Idea: instead of one workgroup per output tile, the host dispatches
//! a fixed number of workgroups and each WG loops, atomically pulling
//! the next tile index off a global counter until the counter exceeds
//! the total tile count.  This amortises kernel launch / drain
//! overhead on shapes that only generate a few tiles.
//!
//! Same compute hot path as the m128n64k64 BDA v4 kernel: the only
//! delta is the prologue (decode `tile_idx` -> `(batch, block_row,
//! block_col)` from a shared-broadcast `atomicAdd` instead of
//! `gl_WorkGroupID`).

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::matmul::{MatmulCall, ResolvedMatmul, RunStats};

/// Output-tile dimensions of the persistent kernel.  Must match the
/// `BM`/`BN`/`BK` defines in `matmul_f32_persistent_v4.comp`.
pub const TILE_M: u32 = 128;
pub const TILE_N: u32 = 64;
pub const TILE_K: u32 = 64;

/// Workgroup launch multiplier: `dispatch_x = min(num_tiles, MULTIPLIER * sm_count)`.
/// 2x SM count is the classic persistent-threads number — high enough to
/// hide latency in the body of each WG, low enough that the atomicAdd
/// contention stays bounded.
const WG_PER_SM: u32 = 2;
/// Fallback SM count when we can't query the device.  46 = RTX 3070, the
/// development target; benign default on other GPUs because it only
/// affects the WG count, which is bounded above by `num_tiles`.
const FALLBACK_SM_COUNT: u32 = 46;

/// Push constants for the persistent kernel.  Bit-for-bit identical to
/// the GLSL `PC` block in `matmul_persistent_kernel.glsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PersistentPushConstants {
    m: u32,
    n: u32,
    k: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_c: u32,
    flags: u32,
    alpha: f32,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    counter_ptr: u64,
    grid_x: u32,
    grid_y: u32,
    num_tiles: u32,
    _pad: u32,
}

/// Specialization constants matching `matmul_persistent_kernel.glsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpecData {
    accumulate: u32,
    alpha_is_one: u32,
    interior_only: u32,
    k_multiple: u32,
}

/// The 16-pipeline cross product of (ACCUMULATE, ALPHA_IS_ONE,
/// INTERIOR_ONLY, K_MULTIPLE) for the persistent shader.
const VARIANT_COUNT: usize = 16;

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_persistent_v4.spv"));

pub struct PersistentMatmul {
    ctx: Arc<VulkanContext>,
    pipeline_layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
    pipelines: [vk::Pipeline; VARIANT_COUNT],
    sm_count: u32,
    // Per-submit state.  Single command pool + one command buffer +
    // fence + 4-byte atomic counter buffer.  No descriptor set: every
    // I/O happens through buffer-device-address push constants.
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    counter_buffer: Buffer,
    counter_addr: u64,
    used: bool,
}

impl PersistentMatmul {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        if !ctx.buffer_device_address_enabled {
            bail!(
                "persistent kernel requires Vulkan 1.2 bufferDeviceAddress \
                 which is not enabled on this device"
            );
        }

        unsafe {
            // No descriptor set: the persistent kernel uses BDA for
            // every storage buffer (A, B, C, atomic counter), so the
            // pipeline layout has zero set layouts.  This matches the
            // GLSL: no `layout(set=, binding=)` declarations anywhere.
            let pc_size = std::mem::size_of::<PersistentPushConstants>() as u32;
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size)];
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&pc_ranges),
                    None,
                )
                .context("create_pipeline_layout (persistent)")?;
            let pipeline_layout_guard = scopeguard::guard(pipeline_layout, |l| {
                ctx.device.destroy_pipeline_layout(l, None)
            });

            assert!(SPV.len().is_multiple_of(4), "SPIR-V not 4-aligned");
            let words: Vec<u32> = SPV
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shader = ctx
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .context("create_shader_module (persistent)")?;
            let shader_guard =
                scopeguard::guard(shader, |m| ctx.device.destroy_shader_module(m, None));

            let entry = std::ffi::CString::new("main").unwrap();
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
                vk::SpecializationMapEntry::default()
                    .constant_id(3)
                    .offset(12)
                    .size(4),
            ];
            let spec_data: Vec<SpecData> = (0..VARIANT_COUNT)
                .map(|i| SpecData {
                    accumulate: ((i & 0b0001) != 0) as u32,
                    alpha_is_one: ((i & 0b0010) != 0) as u32,
                    interior_only: ((i & 0b0100) != 0) as u32,
                    k_multiple: ((i & 0b1000) != 0) as u32,
                })
                .collect();
            let spec_infos: Vec<vk::SpecializationInfo> = (0..VARIANT_COUNT)
                .map(|i| {
                    vk::SpecializationInfo::default()
                        .map_entries(&spec_entries)
                        .data(bytemuck::bytes_of(&spec_data[i]))
                })
                .collect();
            let stages: Vec<vk::PipelineShaderStageCreateInfo> = (0..VARIANT_COUNT)
                .map(|i| {
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(&entry)
                        .specialization_info(&spec_infos[i])
                })
                .collect();
            let create_infos: Vec<vk::ComputePipelineCreateInfo> = (0..VARIANT_COUNT)
                .map(|i| {
                    vk::ComputePipelineCreateInfo::default()
                        .stage(stages[i])
                        .layout(pipeline_layout)
                })
                .collect();
            let pipelines =
                match ctx
                    .device
                    .create_compute_pipelines(ctx.pipeline_cache, &create_infos, None)
                {
                    Ok(pipelines) => pipelines,
                    Err((partial, err)) => {
                        for p in partial {
                            if p != vk::Pipeline::null() {
                                ctx.device.destroy_pipeline(p, None);
                            }
                        }
                        bail!("create_compute_pipelines (persistent): {err}");
                    }
                };
            let mut variants = [vk::Pipeline::null(); VARIANT_COUNT];
            for (i, p) in pipelines.iter().enumerate() {
                variants[i] = *p;
            }
            let pipelines_guard = scopeguard::guard(variants, |variants| {
                for p in variants {
                    if p != vk::Pipeline::null() {
                        ctx.device.destroy_pipeline(p, None);
                    }
                }
            });

            // Single command pool + fence for the experiment.  The
            // mainline executor uses a pool-of-slots; here we run one
            // dispatch at a time so the minimal version is fine.
            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.compute_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .context("create_command_pool (persistent)")?;
            let cmd_pool_guard =
                scopeguard::guard(cmd_pool, |p| ctx.device.destroy_command_pool(p, None));
            let cmd = ctx
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .context("allocate_command_buffers (persistent)")?[0];
            let fence = ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .context("create_fence (persistent)")?;
            let fence_guard = scopeguard::guard(fence, |f| ctx.device.destroy_fence(f, None));

            let query_pool = if ctx.timestamps_supported {
                ctx.device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .context("create_query_pool (persistent)")?
            } else {
                vk::QueryPool::null()
            };
            let query_pool_guard = scopeguard::guard(query_pool, |q| {
                if q != vk::QueryPool::null() {
                    ctx.device.destroy_query_pool(q, None);
                }
            });

            // 4 bytes is enough but allocate 16 for alignment / safety
            // (Buffer::new asserts size > 0; everything else doesn't care).
            let counter_buffer = Buffer::new(
                ctx,
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                BufferLocation::Device,
            )
            .context("counter buffer allocation")?;
            let counter_addr = ctx.buffer_device_address(counter_buffer.raw);

            // SM count: best-effort.  shaderSMCount lives in the NV
            // shader-SM-builtins extension which we don't enable here;
            // fall back to a sane RTX-3070-ish constant.
            let sm_count = FALLBACK_SM_COUNT;

            Ok(Self {
                ctx: Arc::clone(ctx),
                pipeline_layout: ScopeGuard::into_inner(pipeline_layout_guard),
                shader: ScopeGuard::into_inner(shader_guard),
                pipelines: ScopeGuard::into_inner(pipelines_guard),
                sm_count,
                cmd_pool: ScopeGuard::into_inner(cmd_pool_guard),
                cmd,
                fence: ScopeGuard::into_inner(fence_guard),
                query_pool: ScopeGuard::into_inner(query_pool_guard),
                counter_buffer,
                counter_addr,
                used: false,
            })
        }
    }

    /// Run a single matmul through the persistent kernel.  Blocks until
    /// GPU completion.
    pub fn run(&mut self, call: &MatmulCall<'_>) -> Result<RunStats> {
        let dims = ResolvedMatmul::from_call(call)?;
        let total_flops = dims.total_flops;

        let a_addr = self.ctx.buffer_device_address(call.a.raw_buffer());
        let b_addr = self.ctx.buffer_device_address(call.b.raw_buffer());
        let c_addr = self.ctx.buffer_device_address(call.c.raw_buffer());

        let grid_x = dims.n.div_ceil(TILE_N);
        let grid_y = dims.m.div_ceil(TILE_M);
        let tiles_per_batch = grid_x as u64 * grid_y as u64;
        let num_tiles = tiles_per_batch
            .checked_mul(dims.batch as u64)
            .context("persistent kernel: tile count overflows u64")?;
        if num_tiles > u32::MAX as u64 {
            bail!("persistent kernel: tile count {num_tiles} > u32::MAX");
        }
        let num_tiles = num_tiles as u32;
        if num_tiles == 0 {
            return Ok(RunStats {
                gpu_time_ns: None,
                n_calls: 0,
                total_flops: 0,
            });
        }

        let pc = PersistentPushConstants {
            m: dims.m,
            n: dims.n,
            k: dims.k,
            batch_stride_a: dims.batch_stride_a,
            batch_stride_b: dims.batch_stride_b,
            batch_stride_c: dims.batch_stride_c,
            flags: if call.accumulate { 1 } else { 0 },
            alpha: call.alpha,
            a_ptr: a_addr,
            b_ptr: b_addr,
            c_ptr: c_addr,
            counter_ptr: self.counter_addr,
            grid_x,
            grid_y,
            num_tiles,
            _pad: 0,
        };

        // Variant index: identical encoding to KernelVariant::index().
        let interior_only = dims.m.is_multiple_of(TILE_M) && dims.n.is_multiple_of(TILE_N);
        let k_multiple = dims.k.is_multiple_of(TILE_K);
        let alpha_is_one = call.alpha == 1.0;
        let variant_idx = (call.accumulate as usize)
            | ((alpha_is_one as usize) << 1)
            | ((interior_only as usize) << 2)
            | ((k_multiple as usize) << 3);
        let pipeline = self.pipelines[variant_idx];

        // Dispatch width: min(num_tiles, WG_PER_SM * sm_count).  Each
        // workgroup loops until the global counter exceeds num_tiles,
        // so an over-allocation just costs the extra WGs one stall on
        // the atomicAdd that sees `>= num_tiles` and exits.
        let max_wg = WG_PER_SM.saturating_mul(self.sm_count);
        let dispatch_x = num_tiles.min(max_wg);

        let want_timestamps = self.ctx.timestamps_supported;
        let dev = &self.ctx.device;

        unsafe {
            if self.used {
                dev.reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())
                    .context("reset_command_pool (persistent)")?;
            }
            self.used = true;

            dev.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("begin_command_buffer (persistent)")?;

            if want_timestamps {
                dev.cmd_reset_query_pool(self.cmd, self.query_pool, 0, 2);
            }

            // Zero the atomic counter every submit.  fill_buffer is
            // mandatory in core Vulkan and races nothing because the
            // last submission has already fenced out.
            dev.cmd_fill_buffer(self.cmd, self.counter_buffer.raw, 0, 4, 0);
            // Make the zero visible to compute shaders that will read
            // it via buffer-device-address atomicAdd.
            let buf_barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.counter_buffer.raw)
                .offset(0)
                .size(4)];
            dev.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &buf_barrier,
                &[],
            );

            if want_timestamps {
                dev.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.query_pool,
                    0,
                );
            }

            dev.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            dev.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            dev.cmd_dispatch(self.cmd, dispatch_x, 1, 1);

            if want_timestamps {
                dev.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    1,
                );
            }

            dev.end_command_buffer(self.cmd)
                .context("end_command_buffer (persistent)")?;

            let cbs = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            {
                let queue = self.ctx.queue.lock();
                dev.queue_submit(*queue, &[submit], self.fence)
                    .context("queue_submit (persistent)")?;
            }
            dev.wait_for_fences(&[self.fence], true, u64::MAX)
                .context("wait_for_fences (persistent)")?;
            dev.reset_fences(&[self.fence])
                .context("reset_fences (persistent)")?;

            let gpu_time_ns = if want_timestamps {
                let mut data = [0u64; 2];
                dev.get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .context("get_query_pool_results (persistent)")?;
                let ticks = data[1].wrapping_sub(data[0]);
                Some((ticks as f64 * self.ctx.timestamp_period_ns) as u64)
            } else {
                None
            };

            Ok(RunStats {
                gpu_time_ns,
                n_calls: 1,
                total_flops,
            })
        }
    }

    /// Best-of-`iters` benchmark with `warmup` pre-iterations.
    /// Returns `(gpu_ms, wall_ms, tflops)` based on the best-of-iters
    /// GPU timestamp.  `flops` is supplied by the caller so the
    /// callee doesn't need to re-derive batch/m/n/k from tensor shapes.
    pub fn bench(
        &mut self,
        call: &MatmulCall<'_>,
        iters: u32,
        warmup: u32,
        flops: f64,
    ) -> Result<(f64, f64, f64)> {
        for _ in 0..warmup {
            self.run(call)?;
        }
        let mut best_gpu_ns = u64::MAX;
        let mut best_wall_ns = u128::MAX;
        for _ in 0..iters.max(1) {
            let t0 = Instant::now();
            let stats = self.run(call)?;
            let wall_ns = t0.elapsed().as_nanos();
            best_wall_ns = best_wall_ns.min(wall_ns);
            if let Some(gpu_ns) = stats.gpu_time_ns {
                best_gpu_ns = best_gpu_ns.min(gpu_ns);
            }
        }
        let gpu_ms = if best_gpu_ns != u64::MAX {
            best_gpu_ns as f64 / 1e6
        } else {
            f64::NAN
        };
        let wall_ms = best_wall_ns as f64 / 1e6;
        let tflops = if best_gpu_ns != u64::MAX {
            flops / best_gpu_ns as f64 * 1e-3
        } else {
            f64::NAN
        };
        Ok((gpu_ms, wall_ms, tflops))
    }
}

impl Drop for PersistentMatmul {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            if self.query_pool != vk::QueryPool::null() {
                self.ctx.device.destroy_query_pool(self.query_pool, None);
            }
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx.device.destroy_command_pool(self.cmd_pool, None);
            for p in self.pipelines {
                if p != vk::Pipeline::null() {
                    self.ctx.device.destroy_pipeline(p, None);
                }
            }
            self.ctx.device.destroy_shader_module(self.shader, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
