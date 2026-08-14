//! Stream-K validation, scheduling, and execution.

use anyhow::{Result, bail};
use ash::vk;

use super::streamk::{StreamKPipeline, StreamKPushConstants};
use super::streamk_schedule::{StreamKSchedule, stream_k_should_fire};
use super::{Executor, MatmulCall, RunStats, STREAMK_FALLBACK_SM_COUNT, Slot};

impl Executor {
    /// Auto-routing wrapper: consult the Stream-K gate and dispatch
    /// to either Stream-K or the regular DP path.  Single-call only.
    ///
    /// Falls through to `run_matmuls(&[call])` whenever the gate
    /// refuses (unaligned shape, no wave-quantization tail, tail
    /// fraction too large, batch>1, accumulate=true, extension
    /// unavailable, etc.).  Callers that want guaranteed Stream-K
    /// dispatch should use [`run_matmuls_stream_k`] directly.
    ///
    /// `tail_fraction_max` matches [`stream_k_should_fire`]; the
    /// default 0.05 reflects the current hybrid kernel's ~7%
    /// overhead floor versus the regular DP kernel.  Raise once the
    /// SK overhead gap is closed.
    pub fn run_matmuls_auto_stream_k(
        &self,
        call: MatmulCall<'_>,
        tail_fraction_max: f64,
    ) -> Result<RunStats> {
        let want_sk = !call.accumulate
            && self.ctx.buffer_device_address_enabled
            && self.ctx.shader_buffer_float32_atomic_add_enabled
            && {
                let resolved = crate::matmul::ResolvedMatmul::from_call(&call).ok();
                match resolved {
                    Some(r) if r.batch == 1 => stream_k_should_fire(
                        r.m,
                        r.n,
                        r.k,
                        STREAMK_FALLBACK_SM_COUNT,
                        tail_fraction_max,
                    ),
                    _ => false,
                }
            };
        if want_sk {
            self.run_matmuls_stream_k(call)
        } else {
            self.run_matmuls(std::slice::from_ref(&call))
        }
    }

    /// Experimental: dispatch a single matmul using the Stream-K
    /// kernel.  Restrictions (v1):
    ///   * M%128 == N%128 == K%32 == 0
    ///   * batch == 1
    ///   * accumulate == false
    ///
    /// When restrictions are violated the call is rejected; callers
    /// fall back to `run_matmuls` for those shapes.
    pub fn run_matmuls_stream_k(&self, call: MatmulCall<'_>) -> Result<RunStats> {
        self.validate_call_context(&call)?;
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
                resolved.m,
                resolved.n,
                resolved.k,
                BM,
                BN,
                BK
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
        let kernel = stream_k.pick_dp(resolved.m, resolved.n);

        let schedule = StreamKSchedule::try_for_shape(
            resolved.m,
            resolved.n,
            resolved.k,
            kernel.tile_m,
            kernel.tile_n,
            kernel.tile_k,
            STREAMK_FALLBACK_SM_COUNT,
        )?;

        let mut slot = self.checkout_slot();
        let result = self.record_and_run_stream_k(&mut slot, &call, &resolved, stream_k, &schedule);

        let gpu_time_ns = result?;
        Ok(RunStats {
            gpu_time_ns,
            n_calls: 1,
            total_flops: resolved.total_flops,
        })
    }

    /// Debug probe: dispatch the Stream-K DP-flat kernel covering *every*
    /// tile (no SK-tail, no pre-fill).  This isolates the DP-flat
    /// kernel's per-tile throughput from the persistent-grid /
    /// atomic-add / fill overhead of the full hybrid dispatch.  Use only
    /// for benchmarking; for production traffic use
    /// [`run_matmuls_stream_k`] or [`run_matmuls_auto_stream_k`].
    pub fn run_matmuls_stream_k_dp_only(&self, call: MatmulCall<'_>) -> Result<RunStats> {
        self.validate_call_context(&call)?;
        if call.accumulate {
            bail!("run_matmuls_stream_k_dp_only: accumulate=true is not supported");
        }
        let resolved = crate::matmul::ResolvedMatmul::from_call(&call)?;
        if resolved.batch != 1 {
            bail!(
                "run_matmuls_stream_k_dp_only: only batch==1 is supported (got batch={})",
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
                "run_matmuls_stream_k_dp_only: shape {}x{}x{} not aligned",
                resolved.m,
                resolved.n,
                resolved.k,
            );
        }
        if !self.ctx.buffer_device_address_enabled {
            bail!("run_matmuls_stream_k_dp_only: bufferDeviceAddress not available");
        }
        if !self.ctx.shader_buffer_float32_atomic_add_enabled {
            bail!("run_matmuls_stream_k_dp_only: VK_EXT_shader_atomic_float not available");
        }
        let stream_k = if let Some(s) = self.stream_k.get() {
            s
        } else {
            let built = StreamKPipeline::new(&self.ctx)?;
            self.stream_k.get_or_init(|| built)
        };
        let kernel = stream_k.pick_dp(resolved.m, resolved.n);
        // Construct a schedule where every tile is owned by DP and
        // there is no SK-tail dispatch.
        let m_tiles = resolved.m / kernel.tile_m;
        let n_tiles = resolved.n / kernel.tile_n;
        let total_tiles = m_tiles * n_tiles;
        let iters_per_tile = (resolved.k / kernel.tile_k).max(1);
        let schedule = StreamKSchedule {
            iters_per_tile,
            iters_per_wg_sk: 0,
            rem_sk: 0,
            n_tiles,
            dp_tiles_total: total_tiles,
            g_sk: 0,
            total_iters_sk: 0,
            grid_total: total_tiles,
        };
        let mut slot = self.checkout_slot();
        let result = self.record_and_run_stream_k(&mut slot, &call, &resolved, stream_k, &schedule);
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
        // `StreamKSchedule::for_shape` guarantees `dp_tiles_total` is
        // a clean multiple of `n_tiles`, so the DP dispatch grid is
        // always a clean 2D rectangle (no idle workgroups) and we
        // can use the branchless large_bda_v4 SPIR-V directly.
        let dp_kernel = stream_k.pick_dp(resolved.m, resolved.n);
        let tail_kernel = stream_k.pick_tail(resolved.m, resolved.n);
        let layout = stream_k.pipeline_layout;
        let max_groups = self
            .ctx
            .device_properties
            .limits
            .max_compute_work_group_count;
        let dp_rows = if schedule.dp_tiles_total == 0 {
            0
        } else {
            schedule.dp_tiles_total / schedule.n_tiles
        };
        if schedule.n_tiles > max_groups[0]
            || dp_rows > max_groups[1]
            || schedule.g_sk > max_groups[0]
        {
            bail!(
                "stream-K dispatch DP=({}, {}, 1) SK=({}, 1, 1) exceeds device \
                 maxComputeWorkGroupCount ({}, {}, {})",
                schedule.n_tiles,
                dp_rows,
                schedule.g_sk,
                max_groups[0],
                max_groups[1],
                max_groups[2],
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
        let a_ptr = call.a.device_address();
        let b_ptr = call.b.device_address();
        let c_ptr = call.c.device_address();

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
            self.submit_timed(
                slot,
                "get_query_pool_results (Stream-K)",
                |dev, cb, _slot| {
                    // Zero-fill C so the SK-tail atomic-add path is correct.
                    // DP-flat tiles plain-store, so when the schedule is
                    // pure-DP (g_sk == 0) the fill+barrier are pure overhead
                    // and we skip them entirely.  Hybrid mode still fills
                    // the whole buffer; the SK-tail tiles may be scattered
                    // across rows and a sub-range fill costs more in CPU
                    // bookkeeping than it saves in transfer bytes for the
                    // typical (small tail) hybrid case.
                    if schedule.g_sk > 0 {
                        dev.cmd_fill_buffer(cb, call.c.raw_buffer(), 0, vk::WHOLE_SIZE, 0);

                        let buf_barrier = vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )
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
                    }

                    dev.cmd_push_constants(
                        cb,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&pc),
                    );

                    // DP-flat dispatch covers tiles [0, dp_tiles_total).
                    // Skipped when the schedule is pure-SK (small shape).
                    // `dp_tiles_total` is guaranteed to be a multiple of
                    // `n_tiles` by `StreamKSchedule::for_shape`, so the 2D
                    // dispatch (n_tiles, dp_rows) is exact and contains
                    // no idle workgroups; the NVIDIA work distributor
                    // applies its L2-friendly swizzle just like the
                    // standalone BDA_V4 dispatch.
                    if schedule.dp_tiles_total > 0 {
                        dev.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::COMPUTE,
                            dp_kernel.pipeline,
                        );
                        dev.cmd_dispatch(cb, schedule.n_tiles, dp_rows, 1);
                    }

                    // SK-tail dispatch covers tiles
                    // [dp_tiles_total, dp_tiles_total + tail_tiles) via
                    // `g_sk` persistent workgroups.  Skipped when the
                    // schedule is pure-DP.  Both dispatches write to
                    // disjoint C tiles, so no compute->compute barrier is
                    // strictly required by data dependency — but we add
                    // one because Vulkan requires explicit sync between
                    // dispatches that touch the same buffer.
                    if schedule.g_sk > 0 {
                        if schedule.dp_tiles_total > 0 {
                            let sk_barrier = vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                )
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .buffer(call.c.raw_buffer())
                                .offset(0)
                                .size(vk::WHOLE_SIZE);
                            dev.cmd_pipeline_barrier(
                                cb,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &[],
                                std::slice::from_ref(&sk_barrier),
                                &[],
                            );
                        }
                        dev.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::COMPUTE,
                            tail_kernel.pipeline,
                        );
                        dev.cmd_dispatch(cb, schedule.g_sk, 1, 1);
                    }

                    Ok(())
                },
            )
        }
    }
}
