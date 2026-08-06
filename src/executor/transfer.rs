//! Synchronous tensor transfers and reusable staging buffers.

use anyhow::{Result, anyhow, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::tensor::Tensor;

use super::{Executor, Slot};

impl Executor {
    /// Synchronous host->device upload via the slot-local staging buffer.
    pub fn upload(&self, src: &[f32], dst: &Tensor) -> Result<()> {
        self.validate_tensor_context(dst, "upload destination")?;
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
        self.upload_with_slot(&mut slot, src, dst, size)
    }

    /// Synchronous device->host download via the slot-local staging buffer.
    pub fn download(&self, src: &Tensor, dst: &mut [f32]) -> Result<()> {
        self.validate_tensor_context(src, "download source")?;
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
        self.download_with_slot(&mut slot, src, dst, size)
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
}

fn size_of_slice<T>(slice: &[T]) -> Result<vk::DeviceSize> {
    let bytes = std::mem::size_of_val(slice);
    vk::DeviceSize::try_from(bytes).map_err(|_| anyhow!("slice size does not fit VkDeviceSize"))
}
