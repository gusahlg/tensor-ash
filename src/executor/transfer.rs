//! Synchronous tensor transfers and reusable staging buffers.

use anyhow::{Result, anyhow, bail};
use ash::vk;

use crate::buffer::{Buffer, BufferLocation};
use crate::dtype::{DType, f16_bits_to_f32, f32_to_f16_bits};
use crate::tensor::Tensor;

use super::{Executor, Slot};

impl Executor {
    /// Synchronous host->device upload via the slot-local staging buffer.
    /// The host side is always `&[f32]`; values headed for an f16 tensor
    /// round to the nearest half during staging.
    pub fn upload(&self, src: &[f32], dst: &Tensor) -> Result<()> {
        self.validate_tensor_context(dst, "upload destination")?;
        if src.is_empty() {
            return Ok(());
        }
        match dst.dtype() {
            DType::F32 => {
                let size = checked_transfer_size("upload", size_of_slice(src)?, dst)?;
                let mut slot = self.checkout_slot();
                self.upload_with_slot(&mut slot, src, dst, size)
            }
            DType::F16 => {
                let halves: Vec<u16> = src.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let size = checked_transfer_size("upload", size_of_slice(&halves)?, dst)?;
                let mut slot = self.checkout_slot();
                self.upload_with_slot(&mut slot, &halves, dst, size)
            }
        }
    }

    /// Synchronous device->host download via the slot-local staging buffer.
    /// f16 tensors are widened to f32 during staging.
    pub fn download(&self, src: &Tensor, dst: &mut [f32]) -> Result<()> {
        self.validate_tensor_context(src, "download source")?;
        if dst.is_empty() {
            return Ok(());
        }
        match src.dtype() {
            DType::F32 => {
                let size = checked_transfer_size("download", size_of_slice(dst)?, src)?;
                let mut slot = self.checkout_slot();
                self.download_with_slot(&mut slot, src, dst, size)
            }
            DType::F16 => {
                let mut halves = vec![0u16; dst.len()];
                let size = checked_transfer_size("download", size_of_slice(&halves)?, src)?;
                let mut slot = self.checkout_slot();
                self.download_with_slot(&mut slot, src, &mut halves, size)?;
                for (out, &half) in dst.iter_mut().zip(&halves) {
                    *out = f16_bits_to_f32(half);
                }
                Ok(())
            }
        }
    }

    fn upload_with_slot<T: bytemuck::Pod>(
        &self,
        slot: &mut Slot,
        src: &[T],
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

    fn download_with_slot<T: bytemuck::Pod>(
        &self,
        slot: &mut Slot,
        src: &Tensor,
        dst: &mut [T],
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

fn checked_transfer_size(
    operation: &str,
    size: vk::DeviceSize,
    tensor: &Tensor,
) -> Result<vk::DeviceSize> {
    if size > tensor.size_bytes() {
        bail!(
            "{operation}: {} bytes > tensor capacity {}",
            size,
            tensor.size_bytes()
        );
    }
    Ok(size)
}
