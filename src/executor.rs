//! Thread-safe GEMM dispatcher.
//!
//! Architecture
//! ============
//! An `Executor` owns a small pool of *slots*.  A slot bundles every
//! resource that is per-submission (and therefore can't be shared
//! across in-flight submissions): a command pool, a command buffer, a
//! fence, a descriptor pool, and a timestamp query pool.
//!
//! `run_matmuls` checks out one slot under a mutex, releases the mutex
//! while it does the actual recording + submit + wait, then returns
//! the slot.  Multiple host threads can therefore record concurrently
//! up to `n_slots`, after which they block at checkout.
//!
//! The queue submission is serialized by a separate mutex (Vulkan
//! requires external synchronization on a single VkQueue).
//!
//! Why not multi-queue?  On consumer GPUs, multiple compute queues from
//! the same family time-multiplex on the hardware, so for the GEMM
//! workload here they provide ~0 throughput benefit but lots of
//! synchronization complexity.  The pool-of-slots design instead
//! captures the real benefit (CPU-side recording parallelism, CPU/GPU
//! overlap of consecutive submissions).

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::{Condvar, Mutex};

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::matmul::{ResolvedMatmul, ResolvedMatmulBatch, total_flops};
use crate::pipeline::MatmulPipeline;
use crate::tensor::Tensor;

pub use crate::matmul::{MatmulCall, RunStats};

struct Slot {
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    query_pool: vk::QueryPool, // 2 timestamps: start, end
    upload_staging: Option<Buffer>,
    download_staging: Option<Buffer>,
    /// Whether the slot was used at least once (fence/query/pool need a reset).
    used: bool,
}

pub struct Executor {
    ctx: Arc<VulkanContext>,
    pipeline: Arc<MatmulPipeline>,
    /// Available-slot queue with a condvar for blocking checkout.
    slots: Mutex<VecDeque<Slot>>,
    slot_avail: Condvar,
    /// Maximum descriptor sets we'll allocate per submission (= max
    /// matmul calls in one `run_matmuls`).
    max_calls_per_submit: u32,
}

impl Executor {
    /// `n_slots` = how many submissions can be in flight at once. 2 is
    /// the sweet spot: one being recorded by the host while the other
    /// runs on the GPU.  Higher values benefit hosts that submit
    /// concurrently from many threads.
    pub fn new(
        ctx: Arc<VulkanContext>,
        pipeline: Arc<MatmulPipeline>,
        n_slots: usize,
        max_calls_per_submit: u32,
    ) -> Result<Self> {
        let n_slots = n_slots.max(1);
        let max_calls_per_submit = max_calls_per_submit.max(1);
        let mut slots = VecDeque::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push_back(Self::create_slot(&ctx, max_calls_per_submit)?);
        }
        Ok(Self {
            ctx,
            pipeline,
            slots: Mutex::new(slots),
            slot_avail: Condvar::new(),
            max_calls_per_submit,
        })
    }

    fn create_slot(ctx: &Arc<VulkanContext>, max_sets: u32) -> Result<Slot> {
        unsafe {
            let cmd_pool = ctx
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(ctx.compute_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .context("create_command_pool")?;
            let cmd = match ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            ) {
                Ok(command_buffers) => command_buffers[0],
                Err(err) => {
                    ctx.device.destroy_command_pool(cmd_pool, None);
                    return Err(err).context("allocate_command_buffers");
                }
            };
            let fence = match ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
            {
                Ok(fence) => fence,
                Err(err) => {
                    ctx.device.destroy_command_pool(cmd_pool, None);
                    return Err(err).context("create_fence");
                }
            };

            // Descriptor pool sized for `max_sets` matmul calls (3 STORAGE_BUFFER each).
            let pool_size = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_sets * 3)];
            let descriptor_pool = match ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&pool_size),
                None,
            ) {
                Ok(descriptor_pool) => descriptor_pool,
                Err(err) => {
                    ctx.device.destroy_fence(fence, None);
                    ctx.device.destroy_command_pool(cmd_pool, None);
                    return Err(err).context("create_descriptor_pool");
                }
            };

            let query_pool = if ctx.timestamps_supported {
                match ctx.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                ) {
                    Ok(query_pool) => query_pool,
                    Err(err) => {
                        ctx.device.destroy_descriptor_pool(descriptor_pool, None);
                        ctx.device.destroy_fence(fence, None);
                        ctx.device.destroy_command_pool(cmd_pool, None);
                        return Err(err).context("create_query_pool");
                    }
                }
            } else {
                vk::QueryPool::null()
            };

            Ok(Slot {
                cmd_pool,
                cmd,
                fence,
                descriptor_pool,
                query_pool,
                upload_staging: None,
                download_staging: None,
                used: false,
            })
        }
    }

    /// Synchronous host→device upload via the slot-local staging buffer.
    pub fn upload(&self, src: &[f32], dst: &Tensor) -> Result<()> {
        let size = size_of_slice(src)?;
        if size > dst.size_bytes() {
            bail!(
                "upload: {} bytes > tensor capacity {}",
                size,
                dst.size_bytes()
            );
        }
        if size == 0 {
            return Ok(());
        }
        let mut slot = self.checkout_slot();
        let res = self.upload_with_slot(&mut slot, src, dst, size);
        self.checkin_slot(slot);
        res
    }

    /// Synchronous device→host download via the slot-local staging buffer.
    pub fn download(&self, src: &Tensor, dst: &mut [f32]) -> Result<()> {
        let size = size_of_slice(dst)?;
        if size > src.size_bytes() {
            bail!(
                "download: {} bytes > tensor capacity {}",
                size,
                src.size_bytes()
            );
        }
        if size == 0 {
            return Ok(());
        }
        let mut slot = self.checkout_slot();
        let res = self.download_with_slot(&mut slot, src, dst, size);
        self.checkin_slot(slot);
        res
    }

    /// Run a batch of matmul calls.  Blocks until GPU completion.
    pub fn run_matmuls(&self, calls: &[MatmulCall<'_>]) -> Result<RunStats> {
        if calls.is_empty() {
            return Ok(RunStats {
                gpu_time_ns: None,
                n_calls: 0,
                total_flops: 0,
            });
        }
        if calls.len() > self.max_calls_per_submit as usize {
            bail!(
                "run_matmuls: {} calls > max_calls_per_submit {}",
                calls.len(),
                self.max_calls_per_submit
            );
        }

        let resolved = ResolvedMatmulBatch::from_calls(calls)?;
        let resolved = resolved.as_slice();
        let total_flops = total_flops(resolved)?;

        let mut slot = self.checkout_slot();
        let result = self.record_and_run(&mut slot, calls, resolved);
        let gpu_time_ns = match &result {
            Ok(t) => *t,
            Err(_) => None,
        };
        self.checkin_slot(slot);
        result.map(|_| RunStats {
            gpu_time_ns,
            n_calls: calls.len(),
            total_flops,
        })
    }

    // ---- Internals --------------------------------------------------------

    fn checkout_slot(&self) -> Slot {
        let mut slots = self.slots.lock();
        loop {
            if let Some(s) = slots.pop_front() {
                return s;
            }
            self.slot_avail.wait(&mut slots);
        }
    }

    fn checkin_slot(&self, slot: Slot) {
        self.slots.lock().push_back(slot);
        self.slot_avail.notify_one();
    }

    fn upload_with_slot(
        &self,
        slot: &mut Slot,
        src: &[f32],
        dst: &Tensor,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let staging_raw = {
            let staging = self.ensure_upload_staging(slot, size)?;
            staging.write_pod_slice(src)?;
            staging.raw
        };
        self.run_copy_on_slot(slot, staging_raw, dst.raw_buffer(), size)
    }

    fn download_with_slot(
        &self,
        slot: &mut Slot,
        src: &Tensor,
        dst: &mut [f32],
        size: vk::DeviceSize,
    ) -> Result<()> {
        let staging_raw = self.ensure_download_staging(slot, size)?.raw;
        self.run_copy_on_slot(slot, src.raw_buffer(), staging_raw, size)?;
        slot.download_staging
            .as_ref()
            .expect("download staging exists after ensure_download_staging")
            .read_pod_slice(dst)
    }

    fn ensure_upload_staging<'a>(
        &self,
        slot: &'a mut Slot,
        size: vk::DeviceSize,
    ) -> Result<&'a Buffer> {
        let needs_new = slot
            .upload_staging
            .as_ref()
            .is_none_or(|buffer| buffer.size < size);
        if needs_new {
            slot.upload_staging = Some(Buffer::new(
                &self.ctx,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                BufferLocation::Host,
            )?);
        }
        Ok(slot
            .upload_staging
            .as_ref()
            .expect("upload staging buffer is initialized"))
    }

    fn ensure_download_staging<'a>(
        &self,
        slot: &'a mut Slot,
        size: vk::DeviceSize,
    ) -> Result<&'a Buffer> {
        let needs_new = slot
            .download_staging
            .as_ref()
            .is_none_or(|buffer| buffer.size < size);
        if needs_new {
            slot.download_staging = Some(Buffer::new(
                &self.ctx,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                BufferLocation::HostCached,
            )?);
        }
        Ok(slot
            .download_staging
            .as_ref()
            .expect("download staging buffer is initialized"))
    }

    fn record_and_run(
        &self,
        slot: &mut Slot,
        calls: &[MatmulCall<'_>],
        resolved: &[ResolvedMatmul],
    ) -> Result<Option<u64>> {
        debug_assert_eq!(calls.len(), resolved.len());
        let dev = &self.ctx.device;
        let want_timestamps = self.ctx.timestamps_supported;

        unsafe {
            // Reset slot state from any prior use.
            if slot.used {
                dev.reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())
                    .context("reset_command_pool")?;
                dev.reset_descriptor_pool(
                    slot.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .context("reset_descriptor_pool")?;
            }
            slot.used = true;

            let descriptor_sets = allocate_matmul_descriptor_sets(
                &self.ctx,
                &self.pipeline,
                slot.descriptor_pool,
                calls,
            )?;

            // ---- Record -----------------------------------------------------
            dev.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("begin_command_buffer")?;

            if want_timestamps {
                dev.cmd_reset_query_pool(slot.cmd, slot.query_pool, 0, 2);
                dev.cmd_write_timestamp(
                    slot.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    slot.query_pool,
                    0,
                );
            }

            record_matmul_commands(
                &self.ctx,
                &self.pipeline,
                slot.cmd,
                &descriptor_sets,
                calls,
                resolved,
            );

            if want_timestamps {
                dev.cmd_write_timestamp(
                    slot.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    slot.query_pool,
                    1,
                );
            }

            dev.end_command_buffer(slot.cmd)
                .context("end_command_buffer")?;

            // ---- Submit (serialized on the single queue) -------------------
            let cbs = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            {
                let queue = self.ctx.queue.lock();
                dev.queue_submit(*queue, &[submit], slot.fence)
                    .context("queue_submit")?;
            }

            // ---- Wait + read timestamps ------------------------------------
            dev.wait_for_fences(&[slot.fence], true, u64::MAX)
                .context("wait_for_fences")?;
            dev.reset_fences(&[slot.fence]).context("reset_fences")?;

            let gpu_time_ns = if want_timestamps {
                let mut data = [0u64; 2];
                dev.get_query_pool_results(
                    slot.query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .context("get_query_pool_results")?;
                let ticks = data[1].wrapping_sub(data[0]);
                Some((ticks as f64 * self.ctx.timestamp_period_ns) as u64)
            } else {
                None
            };

            Ok(gpu_time_ns)
        }
    }

    fn run_copy_on_slot(
        &self,
        slot: &mut Slot,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        unsafe {
            let dev = &self.ctx.device;
            if slot.used {
                dev.reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())?;
            }
            slot.used = true;
            dev.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let region = [vk::BufferCopy::default().size(size)];
            dev.cmd_copy_buffer(slot.cmd, src, dst, &region);
            dev.end_command_buffer(slot.cmd)?;

            let cbs = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            {
                let q = self.ctx.queue.lock();
                dev.queue_submit(*q, &[submit], slot.fence)?;
            }
            dev.wait_for_fences(&[slot.fence], true, u64::MAX)?;
            dev.reset_fences(&[slot.fence])?;
            Ok(())
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        let slots = std::mem::take(&mut *self.slots.lock());
        for s in slots {
            unsafe {
                if s.query_pool != vk::QueryPool::null() {
                    self.ctx.device.destroy_query_pool(s.query_pool, None);
                }
                self.ctx
                    .device
                    .destroy_descriptor_pool(s.descriptor_pool, None);
                self.ctx.device.destroy_fence(s.fence, None);
                self.ctx.device.destroy_command_pool(s.cmd_pool, None);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Recording helpers
// ----------------------------------------------------------------------------

fn allocate_matmul_descriptor_sets(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    descriptor_pool: vk::DescriptorPool,
    calls: &[MatmulCall<'_>],
) -> Result<Vec<vk::DescriptorSet>> {
    macro_rules! storage_buffer_write {
        ($set:expr, $binding:expr, $info:expr) => {
            vk::WriteDescriptorSet::default()
                .dst_set($set)
                .dst_binding($binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref($info))
        };
    }

    if let [call] = calls {
        let layouts = [pipeline.set_layout];
        let sets = unsafe {
            ctx.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )
                .context("allocate_descriptor_sets")?
        };
        let set = sets[0];
        let buffer_infos = [
            tensor_descriptor(call.a),
            tensor_descriptor(call.b),
            tensor_descriptor(call.c),
        ];
        let writes = [
            storage_buffer_write!(set, 0, &buffer_infos[0]),
            storage_buffer_write!(set, 1, &buffer_infos[1]),
            storage_buffer_write!(set, 2, &buffer_infos[2]),
        ];

        unsafe {
            ctx.device.update_descriptor_sets(&writes, &[]);
        }

        return Ok(sets);
    }

    let layouts = vec![pipeline.set_layout; calls.len()];
    let sets = unsafe {
        ctx.device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .context("allocate_descriptor_sets")?
    };

    let mut buffer_infos = Vec::with_capacity(calls.len() * 3);
    for call in calls {
        buffer_infos.push(tensor_descriptor(call.a));
        buffer_infos.push(tensor_descriptor(call.b));
        buffer_infos.push(tensor_descriptor(call.c));
    }

    let mut writes = Vec::with_capacity(calls.len() * 3);
    for (i, set) in sets.iter().copied().enumerate() {
        let base = i * 3;
        writes.push(storage_buffer_write!(set, 0, &buffer_infos[base]));
        writes.push(storage_buffer_write!(set, 1, &buffer_infos[base + 1]));
        writes.push(storage_buffer_write!(set, 2, &buffer_infos[base + 2]));
    }

    unsafe {
        ctx.device.update_descriptor_sets(&writes, &[]);
    }

    Ok(sets)
}

fn record_matmul_commands(
    ctx: &VulkanContext,
    pipeline: &MatmulPipeline,
    cb: vk::CommandBuffer,
    descriptor_sets: &[vk::DescriptorSet],
    calls: &[MatmulCall<'_>],
    resolved: &[ResolvedMatmul],
) {
    let mut bound_pipeline = vk::Pipeline::null();
    for ((set, call), dims) in descriptor_sets
        .iter()
        .copied()
        .zip(calls.iter())
        .zip(resolved.iter())
    {
        let pc = dims.push_constants(call.alpha, call.accumulate);
        let kernel = pipeline.select_kernel(dims.m, dims.n, dims.k);

        unsafe {
            if bound_pipeline != kernel.pipeline {
                ctx.device
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                bound_pipeline = kernel.pipeline;
            }
            ctx.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            ctx.device.cmd_push_constants(
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            ctx.device.cmd_dispatch(
                cb,
                dims.n.div_ceil(kernel.tile_n),
                dims.m.div_ceil(kernel.tile_m),
                dims.batch,
            );
        }
    }
}

fn tensor_descriptor(tensor: &Tensor) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(tensor.raw_buffer())
        .offset(0)
        .range(tensor.size_bytes())
}

fn size_of_slice<T>(slice: &[T]) -> Result<vk::DeviceSize> {
    let bytes = std::mem::size_of_val(slice);
    vk::DeviceSize::try_from(bytes).map_err(|_| anyhow!("slice size does not fit VkDeviceSize"))
}
