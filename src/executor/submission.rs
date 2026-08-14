//! Slot checkout and the one synchronous Vulkan submission path.

use anyhow::{Context, Result};
use ash::vk;

use crate::context::timestamp_delta;

use super::{Executor, Slot};

pub(super) struct SlotLease<'a> {
    executor: &'a Executor,
    slot: Option<Slot>,
}

impl std::ops::Deref for SlotLease<'_> {
    type Target = Slot;

    fn deref(&self) -> &Self::Target {
        self.slot.as_ref().expect("slot lease is always populated")
    }
}

impl std::ops::DerefMut for SlotLease<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slot.as_mut().expect("slot lease is always populated")
    }
}

impl Drop for SlotLease<'_> {
    fn drop(&mut self) {
        self.executor
            .return_slot(self.slot.take().expect("slot lease is always populated"));
    }
}

impl Executor {
    pub(super) fn checkout_slot(&self) -> SlotLease<'_> {
        let mut slots = self.slots.lock();
        loop {
            if let Some(slot) = slots.pop_front() {
                return SlotLease {
                    executor: self,
                    slot: Some(slot),
                };
            }
            self.slot_avail.wait(&mut slots);
        }
    }

    fn return_slot(&self, slot: Slot) {
        self.slots.lock().push_back(slot);
        self.slot_avail.notify_one();
    }

    /// Record and synchronously submit one command buffer, optionally
    /// wrapping the recorded commands in the slot's timestamp query pair.
    pub(super) unsafe fn submit_timed<F>(
        &self,
        slot: &mut Slot,
        result_context: &'static str,
        record: F,
    ) -> Result<Option<u64>>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer, &Slot) -> Result<()>,
    {
        let want_timestamps = self.ctx.timestamps_supported;
        let query_pool = slot.query_pool;
        unsafe {
            self.submit_recorded(slot, |dev, cb, slot| {
                if want_timestamps {
                    dev.cmd_reset_query_pool(cb, query_pool, 0, 2);
                    dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, query_pool, 0);
                }
                record(dev, cb, slot)?;
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

            if !want_timestamps {
                return Ok(None);
            }
            let mut data = [0u64; 2];
            self.ctx
                .device
                .get_query_pool_results(
                    query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .context(result_context)?;
            let ticks = timestamp_delta(data[0], data[1], self.ctx.timestamp_valid_bits);
            Ok(Some((ticks as f64 * self.ctx.timestamp_period_ns) as u64))
        }
    }

    /// Reset, record, submit on the externally synchronized queue, and wait.
    pub(super) unsafe fn submit_recorded<F>(&self, slot: &mut Slot, record: F) -> Result<()>
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

            let command_buffers = [slot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            {
                let queue = self.ctx.queue.lock();
                dev.queue_submit(*queue, &[submit], slot.fence)
                    .context("queue_submit")?;
            }

            wait_fence_spin(dev, slot.fence)?;
            dev.reset_fences(&[slot.fence]).context("reset_fences")?;
            Ok(())
        }
    }
}

/// Wait for a fence with a short bounded spin before blocking.  Small
/// dispatches signal within microseconds, and the blocking wait's
/// scheduler wakeup costs more than the GPU work itself; the spin
/// budget caps the wasted CPU on long dispatches at a few microseconds.
pub(super) fn wait_fence_spin(dev: &ash::Device, fence: vk::Fence) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_micros(50);
    loop {
        match unsafe { dev.get_fence_status(fence) } {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(err) => return Err(err).context("get_fence_status"),
        }
        if std::time::Instant::now() >= deadline {
            return unsafe {
                dev.wait_for_fences(&[fence], true, u64::MAX)
                    .context("wait_for_fences")
            };
        }
        std::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::timestamp_delta;

    #[test]
    fn timestamp_delta_masks_queue_counter_width() {
        assert_eq!(timestamp_delta(4, 7, 0), 3);
        assert_eq!(timestamp_delta(1, 2, 1), 1);
        assert_eq!(timestamp_delta(250, 5, 8), 11);
        assert_eq!(timestamp_delta((1u64 << 63) - 2, 3, 63), 5);
        assert_eq!(timestamp_delta(u64::MAX - 2, 3, 64), 6);
    }
}
