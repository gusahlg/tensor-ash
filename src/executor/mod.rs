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
mod splitk;
mod streamk;

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::{Condvar, Mutex};

use crate::buffer::{Buffer, BufferLocation};
use crate::context::VulkanContext;
use crate::matmul::{ResolvedMatmul, ResolvedMatmulBatch, total_flops};
use crate::pipeline::MatmulPipeline;
use crate::tensor::Tensor;

use recording::{
    record_matmul_commands, record_matmul_split_k_commands, update_matmul_descriptor_sets,
};
use slot::Slot;
pub use splitk::default_num_k_splits;
use splitk::SplitKPipeline;
pub use streamk::{StreamKPushConstants, StreamKSchedule, default_stream_k_grid};
use streamk::StreamKPipeline;

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
    /// Experimental split-K pipeline, lazily built on first use of
    /// `run_matmuls_split_k`.  See `executor/splitk.rs` for the design
    /// rationale.
    split_k: OnceLock<SplitKPipeline>,
    /// Experimental Stream-K pipeline, lazily built on first use of
    /// `run_matmuls_stream_k`.  See `executor/streamk.rs`.
    stream_k: OnceLock<StreamKPipeline>,
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
            split_k: OnceLock::new(),
            stream_k: OnceLock::new(),
        })
    }

    /// Experimental: dispatch a single matmul using the split-K
    /// kernel.  Splits the K reduction into `num_k_splits` chunks
    /// (`0` = pick a default based on the shape), zero-fills C, then
    /// dispatches one workgroup per (block_row, block_col, batch,
    /// k_split) and atomically reduces partial sums into C.
    ///
    /// Returns the same `RunStats` shape as `run_matmuls`, including
    /// GPU time over the zero-fill **plus** the dispatch — both
    /// happen inside the timestamp window.
    ///
    /// Caveats:
    ///   * `accumulate=true` is rejected: split-K's correctness
    ///     story relies on starting from a zeroed C.
    ///   * Atomic-add ordering is non-deterministic, so results
    ///     differ across runs in the low-order bit.  Tests should
    ///     budget `tolerance(K) * num_k_splits` for the extra
    ///     rounding.
    ///
    /// This entry point is intentionally a *separate* method from
    /// `run_matmuls` so callers opt in explicitly.
    pub fn run_matmuls_split_k(
        &self,
        call: MatmulCall<'_>,
        num_k_splits: u32,
    ) -> Result<RunStats> {
        if call.accumulate {
            bail!("run_matmuls_split_k: accumulate=true is not supported");
        }

        let resolved = crate::matmul::ResolvedMatmul::from_call(&call)?;

        // Stable equivalent of `get_or_try_init`: try-init outside, then
        // race the get_or_init.  Worst case we build the pipeline twice
        // on first concurrent access but only the first init wins.
        let split_k = if let Some(s) = self.split_k.get() {
            s
        } else {
            let built = SplitKPipeline::new(&self.ctx, self.pipeline.pipeline_layout)?;
            self.split_k.get_or_init(|| built)
        };
        let kernel = split_k.pick(resolved.m, resolved.n);
        let num_k_splits = if num_k_splits == 0 {
            default_num_k_splits(
                resolved.batch,
                resolved.m,
                resolved.n,
                resolved.k,
                kernel.tile_m,
                kernel.tile_n,
            )
        } else {
            num_k_splits
        };
        if num_k_splits == 0 || num_k_splits > 0xFFFF {
            bail!(
                "run_matmuls_split_k: num_k_splits={num_k_splits} out of range [1, 65535]"
            );
        }

        let mut slot = self.checkout_slot();
        let result = self.record_and_run_split_k(&mut slot, &call, &resolved, kernel, num_k_splits);
        self.checkin_slot(slot);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: resolved.total_flops,
        })
    }

    fn record_and_run_split_k(
        &self,
        slot: &mut Slot,
        call: &MatmulCall<'_>,
        resolved: &crate::matmul::ResolvedMatmul,
        kernel: &splitk::SplitKKernel,
        num_k_splits: u32,
    ) -> Result<Option<u64>> {
        let want_timestamps = self.ctx.timestamps_supported;
        let query_pool = slot.query_pool;
        let set_count = 1usize;
        let calls_slice = std::slice::from_ref(call);
        let max_groups = self.ctx.device_properties.limits.max_compute_work_group_count;
        let gx = resolved.n.div_ceil(kernel.tile_n);
        let gy = resolved.m.div_ceil(kernel.tile_m);
        let gz = resolved
            .batch
            .checked_mul(num_k_splits)
            .ok_or_else(|| anyhow!("split-K dispatch grid Z overflows u32"))?;
        if gx > max_groups[0] || gy > max_groups[1] || gz > max_groups[2] {
            bail!(
                "split-K dispatch ({gx}, {gy}, {gz}) for M={}, N={}, K={}, batch={}, splits={} \
                 exceeds device maxComputeWorkGroupCount ({}, {}, {})",
                resolved.m,
                resolved.n,
                resolved.k,
                resolved.batch,
                num_k_splits,
                max_groups[0],
                max_groups[1],
                max_groups[2],
            );
        }

        unsafe {
            update_matmul_descriptor_sets(
                &self.ctx,
                &slot.descriptor_sets[..set_count],
                calls_slice,
            );

            self.submit_recorded(slot, |dev, cb, slot| {
                if want_timestamps {
                    dev.cmd_reset_query_pool(cb, query_pool, 0, 2);
                    dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0);
                }

                let descriptor_sets = &slot.descriptor_sets[..set_count];
                record_matmul_split_k_commands(
                    &self.ctx,
                    &self.pipeline,
                    kernel,
                    cb,
                    descriptor_sets,
                    call,
                    resolved,
                    num_k_splits,
                    gx,
                    gy,
                    gz,
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

    /// Experimental: dispatch a single matmul using the Stream-K
    /// kernel.  Restrictions (v1):
    ///   * M%128 == N%128 == K%32 == 0
    ///   * batch == 1
    ///   * accumulate == false
    /// When restrictions are violated the call is rejected; callers
    /// fall back to `run_matmuls` for those shapes.
    pub fn run_matmuls_stream_k(&self, call: MatmulCall<'_>) -> Result<RunStats> {
        if call.accumulate {
            bail!("run_matmuls_stream_k: accumulate=true is not supported");
        }
        let resolved = crate::matmul::ResolvedMatmul::from_call(&call)?;
        if resolved.batch != 1 {
            bail!(
                "run_matmuls_stream_k: only batch==1 is supported (got batch={})",
                resolved.batch
            );
        }
        const BM: u32 = 128;
        const BN: u32 = 128;
        const BK: u32 = 32;
        if !resolved.m.is_multiple_of(BM)
            || !resolved.n.is_multiple_of(BN)
            || !resolved.k.is_multiple_of(BK)
        {
            bail!(
                "run_matmuls_stream_k: shape {}x{}x{} not aligned to ({},{},{})",
                resolved.m, resolved.n, resolved.k, BM, BN, BK
            );
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("run_matmuls_stream_k: bufferDeviceAddress not available on this device");
        }
        if !self.ctx.shader_buffer_float32_atomic_add_enabled {
            bail!(
                "run_matmuls_stream_k: VK_EXT_shader_atomic_float \
                 (shaderBufferFloat32AtomicAdd) not available on this device"
            );
        }

        let stream_k = if let Some(s) = self.stream_k.get() {
            s
        } else {
            let built = StreamKPipeline::new(&self.ctx)?;
            self.stream_k.get_or_init(|| built)
        };
        let kernel = stream_k.pick(resolved.m, resolved.n);

        let schedule = StreamKSchedule::for_shape(
            resolved.m,
            resolved.n,
            resolved.k,
            kernel.tile_m,
            kernel.tile_n,
            kernel.tile_k,
            // Hard-coded to 46 for the RTX 3070 we benchmark on; this
            // can be looked up via VK_NV_compute_shader_derivatives or
            // a runtime probe in a follow-up.  Today the value is only
            // used to size the preferred persistent grid (G_pref =
            // sm_count * 2) within the two-tile invariant clamp on the
            // tail, so being slightly off is harmless.
            46,
        );

        let mut slot = self.checkout_slot();
        let result = self.record_and_run_stream_k(&mut slot, &call, &resolved, stream_k, &schedule);
        self.checkin_slot(slot);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: resolved.total_flops,
        })
    }

    fn record_and_run_stream_k(
        &self,
        slot: &mut Slot,
        call: &MatmulCall<'_>,
        resolved: &crate::matmul::ResolvedMatmul,
        stream_k: &StreamKPipeline,
        schedule: &StreamKSchedule,
    ) -> Result<Option<u64>> {
        let want_timestamps = self.ctx.timestamps_supported;
        let query_pool = slot.query_pool;
        let kernel = stream_k.pick(resolved.m, resolved.n);
        let layout = stream_k.pipeline_layout;
        let max_groups = self.ctx.device_properties.limits.max_compute_work_group_count;
        if schedule.grid_total > max_groups[0] {
            bail!(
                "stream-K grid_total={} exceeds device maxComputeWorkGroupCount[0]={}",
                schedule.grid_total,
                max_groups[0]
            );
        }
        if schedule.grid_total == 0 {
            bail!(
                "stream-K schedule produced empty grid for {}x{}x{} \
                 (m_tiles*n_tiles=0)",
                resolved.m,
                resolved.n,
                resolved.k,
            );
        }
        let a_ptr = self.ctx.buffer_device_address(call.a.raw_buffer());
        let b_ptr = self.ctx.buffer_device_address(call.b.raw_buffer());
        let c_ptr = self.ctx.buffer_device_address(call.c.raw_buffer());

        let pc = StreamKPushConstants {
            m: resolved.m,
            n: resolved.n,
            k: resolved.k,
            batch_stride_a: resolved.batch_stride_a,
            batch_stride_b: resolved.batch_stride_b,
            batch_stride_c: resolved.batch_stride_c,
            flags: 0,
            alpha: call.alpha,
            a_ptr,
            b_ptr,
            c_ptr,
            iters_per_tile: schedule.iters_per_tile,
            iters_per_wg_sk: schedule.iters_per_wg_sk,
            rem_sk: schedule.rem_sk,
            n_tiles: schedule.n_tiles,
            dp_tiles_total: schedule.dp_tiles_total,
            g_sk: schedule.g_sk,
            total_iters_sk: schedule.total_iters_sk,
            _pad: 0,
        };

        unsafe {
            self.submit_recorded(slot, |dev, cb, _slot| {
                if want_timestamps {
                    dev.cmd_reset_query_pool(cb, query_pool, 0, 2);
                    dev.cmd_write_timestamp(
                        cb,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        query_pool,
                        0,
                    );
                }

                // Zero-fill C so the atomic-add path is correct.
                dev.cmd_fill_buffer(cb, call.c.raw_buffer(), 0, vk::WHOLE_SIZE, 0);

                let buf_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(call.c.raw_buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                dev.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    std::slice::from_ref(&buf_barrier),
                    &[],
                );

                dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                dev.cmd_push_constants(
                    cb,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&pc),
                );
                dev.cmd_dispatch(cb, schedule.grid_total, 1, 1);

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
                    .context("get_query_pool_results (stream-k)")?;
                let ticks = data[1].wrapping_sub(data[0]);
                Some((ticks as f64 * self.ctx.timestamp_period_ns) as u64)
            } else {
                None
            };
            Ok(gpu_time_ns)
        }
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
