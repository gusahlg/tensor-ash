//! Dispatch preparation, submission, timing, and benchmarking.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::context::timestamp_delta;
use crate::matmul::{MatmulCall, ResolvedMatmul, RunStats};

use super::{PersistentMatmul, PersistentPushConstants, TILE_K, TILE_M, TILE_N, WG_PER_SM};

struct PreparedDispatch {
    pc: PersistentPushConstants,
    pipeline: vk::Pipeline,
    group_count_x: u32,
    total_flops: u64,
}

impl PersistentMatmul {
    /// Run a single matmul through the persistent kernel. Blocks until completion.
    pub fn run(&mut self, call: &MatmulCall<'_>) -> Result<RunStats> {
        let dispatch = self.prepare_dispatch(call)?;
        let gpu_time_ns = self.submit(&dispatch)?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: dispatch.total_flops,
        })
    }

    fn prepare_dispatch(&self, call: &MatmulCall<'_>) -> Result<PreparedDispatch> {
        for (label, tensor) in [("A", call.a), ("B", call.b), ("C", call.c)] {
            if !tensor.belongs_to(&self.ctx) {
                bail!("persistent kernel: tensor {label} belongs to a different Vulkan context");
            }
        }
        if call.c.raw_buffer() == call.a.raw_buffer() || call.c.raw_buffer() == call.b.raw_buffer()
        {
            bail!("persistent kernel: output C must not alias input A or B");
        }
        let dims = ResolvedMatmul::from_call(call)?;
        let grid_x = dims.n.div_ceil(TILE_N);
        let grid_y = dims.m.div_ceil(TILE_M);
        let num_tiles = u64::from(grid_x)
            .checked_mul(u64::from(grid_y))
            .and_then(|tiles| tiles.checked_mul(u64::from(dims.batch)))
            .context("persistent kernel: tile count overflows u64")?;
        let num_tiles =
            u32::try_from(num_tiles).context("persistent kernel: tile count exceeds u32::MAX")?;

        let variant = usize::from(call.accumulate)
            | (usize::from(call.alpha == 1.0) << 1)
            | (usize::from(dims.m.is_multiple_of(TILE_M) && dims.n.is_multiple_of(TILE_N)) << 2)
            | (usize::from(dims.k.is_multiple_of(TILE_K)) << 3);
        let persistent_limit = WG_PER_SM.saturating_mul(self.sm_count);
        let device_limit = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count[0];
        let group_count_x = num_tiles.min(persistent_limit).min(device_limit);
        if group_count_x == 0 {
            bail!("persistent kernel: device reports a zero dispatch-X limit");
        }

        Ok(PreparedDispatch {
            pc: PersistentPushConstants {
                m: dims.m,
                n: dims.n,
                k: dims.k,
                batch_stride_a: dims.batch_stride_a,
                batch_stride_b: dims.batch_stride_b,
                batch_stride_c: dims.batch_stride_c,
                flags: u32::from(call.accumulate),
                alpha: call.alpha,
                a_ptr: self.ctx.buffer_device_address(call.a.raw_buffer()),
                b_ptr: self.ctx.buffer_device_address(call.b.raw_buffer()),
                c_ptr: self.ctx.buffer_device_address(call.c.raw_buffer()),
                counter_ptr: self.counter_addr,
                grid_x,
                grid_y,
                num_tiles,
                _pad: 0,
            },
            pipeline: self.pipelines[variant],
            group_count_x,
            total_flops: dims.total_flops,
        })
    }

    fn submit(&mut self, dispatch: &PreparedDispatch) -> Result<Option<u64>> {
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

            if self.ctx.timestamps_supported {
                dev.cmd_reset_query_pool(self.cmd, self.query_pool, 0, 2);
            }
            dev.cmd_fill_buffer(self.cmd, self.counter_buffer.raw, 0, 4, 0);
            let counter_barrier = [vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.counter_buffer.raw)
                .size(4)];
            dev.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &counter_barrier,
                &[],
            );
            if self.ctx.timestamps_supported {
                dev.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.query_pool,
                    0,
                );
            }
            dev.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, dispatch.pipeline);
            dev.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&dispatch.pc),
            );
            dev.cmd_dispatch(self.cmd, dispatch.group_count_x, 1, 1);
            if self.ctx.timestamps_supported {
                dev.cmd_write_timestamp(
                    self.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    1,
                );
            }
            dev.end_command_buffer(self.cmd)
                .context("end_command_buffer (persistent)")?;

            let command_buffers = [self.cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            {
                let queue = self.ctx.queue.lock();
                dev.queue_submit(*queue, &submits, self.fence)
                    .context("queue_submit (persistent)")?;
            }
            dev.wait_for_fences(&[self.fence], true, u64::MAX)
                .context("wait_for_fences (persistent)")?;
            dev.reset_fences(&[self.fence])
                .context("reset_fences (persistent)")?;
            self.read_gpu_time()
        }
    }

    unsafe fn read_gpu_time(&self) -> Result<Option<u64>> {
        if !self.ctx.timestamps_supported {
            return Ok(None);
        }
        let mut data = [0u64; 2];
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .context("get_query_pool_results (persistent)")?;
        let ticks = timestamp_delta(data[0], data[1], self.timestamp_valid_bits);
        Ok(Some((ticks as f64 * self.ctx.timestamp_period_ns) as u64))
    }

    /// Best-of-`iters` benchmark after `warmup` unmeasured iterations.
    pub fn bench(
        &mut self,
        call: &MatmulCall<'_>,
        iters: u32,
        warmup: u32,
    ) -> Result<(f64, f64, f64)> {
        for _ in 0..warmup {
            self.run(call)?;
        }
        let mut best_gpu_ns = u64::MAX;
        let mut best_wall_ns = u128::MAX;
        let mut total_flops = 0;
        for _ in 0..iters.max(1) {
            let started = Instant::now();
            let stats = self.run(call)?;
            total_flops = stats.total_flops;
            best_wall_ns = best_wall_ns.min(started.elapsed().as_nanos());
            if let Some(gpu_ns) = stats.gpu_time_ns {
                best_gpu_ns = best_gpu_ns.min(gpu_ns);
            }
        }
        let measured = best_gpu_ns != u64::MAX;
        let gpu_ms = if measured {
            best_gpu_ns as f64 / 1e6
        } else {
            f64::NAN
        };
        let tflops = if measured && best_gpu_ns > 0 {
            total_flops as f64 / best_gpu_ns as f64 * 1e-3
        } else {
            f64::NAN
        };
        Ok((gpu_ms, best_wall_ns as f64 / 1e6, tflops))
    }
}

#[cfg(test)]
mod tests {
    use crate::context::timestamp_delta;

    #[test]
    fn timestamp_delta_respects_device_counter_width() {
        assert_eq!(timestamp_delta(250, 5, 8), 11);
        assert_eq!(timestamp_delta(u64::MAX - 2, 3, 64), 6);
    }
}
