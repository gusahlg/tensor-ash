//! Slot checkout and the one synchronous Vulkan submission path.

use anyhow::{Context, Result};
use ash::vk;

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
            let ticks = data[1].wrapping_sub(data[0]);
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

            dev.wait_for_fences(&[slot.fence], true, u64::MAX)
                .context("wait_for_fences")?;
            dev.reset_fences(&[slot.fence]).context("reset_fences")?;
            Ok(())
        }
    }
}
