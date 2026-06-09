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

mod recording;
mod slot;

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

use recording::{record_matmul_commands, update_matmul_descriptor_sets};
use slot::Slot;

pub use crate::matmul::{MatmulCall, RunStats};

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
            slots.push_back(Slot::create(
                &ctx,
                max_calls_per_submit,
                pipeline.set_layout,
            )?);
        }
        Ok(Self {
            ctx,
            pipeline,
            slots: Mutex::new(slots),
            slot_avail: Condvar::new(),
            max_calls_per_submit,
        })
    }

    /// Synchronous host->device upload via the slot-local staging buffer.
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

    /// Synchronous device->host download via the slot-local staging buffer.
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
    ///
    /// The calls in `calls` are dispatched back-to-back in a single
    /// command buffer with **no pipeline barriers between them**.  That
    /// is intentional — the common usage is many *independent* matmuls
    /// (different C tensors per call), and barriers would serialize
    /// them on the GPU for no benefit.
    ///
    /// If you need to chain calls — i.e. the second call reads a `C`
    /// that the first call writes — split them into separate
    /// `run_matmuls` invocations.  Each `run_matmuls` waits for the
    /// fence, which provides a full GPU sync.
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
        self.checkin_slot(slot);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: calls.len(),
            total_flops,
        })
    }

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
        let want_timestamps = self.ctx.timestamps_supported;
        let query_pool = slot.query_pool;
        let descriptor_set_count = calls.len();

        unsafe {
            // Point the slot's pre-allocated descriptor sets at this
            // submission's tensors.  By now the previous submission on
            // this slot has fenced out (see submit_recorded), so the
            // sets are safe to update.
            update_matmul_descriptor_sets(&self.ctx, &slot.descriptor_sets[..calls.len()], calls);

            self.submit_recorded(slot, |dev, cb, slot| {
                if want_timestamps {
                    dev.cmd_reset_query_pool(cb, query_pool, 0, 2);
                    dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0);
                }

                let descriptor_sets = &slot.descriptor_sets[..descriptor_set_count];
                record_matmul_commands(
                    &self.ctx,
                    &self.pipeline,
                    cb,
                    descriptor_sets,
                    calls,
                    resolved,
                )?;

                if want_timestamps {
                    dev.cmd_write_timestamp(
                        cb,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query_pool,
                        1,
                    );
                }
                Ok(())
            })?;

            let gpu_time_ns = if want_timestamps {
                let mut data = [0u64; 2];
                self.ctx
                    .device
                    .get_query_pool_results(
                        query_pool,
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
            self.submit_recorded(slot, |dev, cb, _slot| {
                let region = [vk::BufferCopy::default().size(size)];
                dev.cmd_copy_buffer(cb, src, dst, &region);
                Ok(())
            })
        }
    }

    /// Reset the slot's command pool (if dirty), begin/record/end the
    /// command buffer via `record`, submit on the single compute queue,
    /// then block until the fence signals.  Centralizes the boilerplate
    /// shared by `record_and_run` and `run_copy_on_slot`.
    unsafe fn submit_recorded<F>(&self, slot: &mut Slot, record: F) -> Result<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer, &Slot) -> Result<()>,
    {
        let dev = &self.ctx.device;
        unsafe {
            if slot.used {
                dev.reset_command_pool(slot.cmd_pool, vk::CommandPoolResetFlags::empty())
                    .context("reset_command_pool")?;
            }
            slot.used = true;

            dev.begin_command_buffer(
                slot.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("begin_command_buffer")?;

            record(dev, slot.cmd, slot)?;

            dev.end_command_buffer(slot.cmd)
                .context("end_command_buffer")?;

            let cbs = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            {
                let queue = self.ctx.queue.lock();
                dev.queue_submit(*queue, &[submit], slot.fence)
                    .context("queue_submit")?;
            }

            dev.wait_for_fences(&[slot.fence], true, u64::MAX)
                .context("wait_for_fences")?;
            dev.reset_fences(&[slot.fence]).context("reset_fences")?;
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

fn size_of_slice<T>(slice: &[T]) -> Result<vk::DeviceSize> {
    let bytes = std::mem::size_of_val(slice);
    vk::DeviceSize::try_from(bytes).map_err(|_| anyhow!("slice size does not fit VkDeviceSize"))
}
