//! GPU buffer + its backing device memory.
//!
//! Two flavors:
//!   * `Device` — device-local memory.  Fastest for the GPU.  Host I/O
//!                must go through a staging buffer (see `Executor::upload`).
//!   * `Host`   — host-visible + coherent.  Persistently mapped for the
//!                whole lifetime of the buffer; cheap to read/write from
//!                the CPU, slower for the GPU on discrete cards.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;

use crate::context::VulkanContext;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BufferLocation { Device, Host }

pub struct Buffer {
    ctx:           Arc<VulkanContext>,
    pub raw:       vk::Buffer,
    pub memory:    vk::DeviceMemory,
    pub size:      vk::DeviceSize,
    pub location:  BufferLocation,
    /// Persistent mapping pointer for host-visible buffers; null otherwise.
    mapped:        *mut u8,
}

// The pointer is owned: no aliasing across threads other than synchronized
// reads/writes via this object.  The Vulkan handles themselves are thread-safe.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(
        ctx:      &Arc<VulkanContext>,
        size:     vk::DeviceSize,
        usage:    vk::BufferUsageFlags,
        location: BufferLocation,
    ) -> Result<Self> {
        assert!(size > 0, "Buffer::new with size=0");
        unsafe {
            let raw = ctx.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            ).context("create_buffer")?;
            let mem_req = ctx.device.get_buffer_memory_requirements(raw);

            let props = match location {
                BufferLocation::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
                BufferLocation::Host   => vk::MemoryPropertyFlags::HOST_VISIBLE
                                        | vk::MemoryPropertyFlags::HOST_COHERENT,
            };
            let mem_type = ctx.find_memory_type(mem_req, props)?;
            let memory = ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type),
                None,
            ).map_err(|e| {
                ctx.device.destroy_buffer(raw, None);
                anyhow::anyhow!("allocate_memory: {e}")
            })?;
            ctx.device.bind_buffer_memory(raw, memory, 0).context("bind_buffer_memory")?;

            let mapped = if location == BufferLocation::Host {
                ctx.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .context("map_memory")? as *mut u8
            } else {
                std::ptr::null_mut()
            };

            Ok(Self { ctx: Arc::clone(ctx), raw, memory, size, location, mapped })
        }
    }

    /// Copy CPU bytes into a host-visible buffer.
    pub fn write_from_slice(&self, bytes: &[u8]) -> Result<()> {
        if self.location != BufferLocation::Host {
            bail!("write_from_slice: buffer is not host-visible");
        }
        if bytes.len() as vk::DeviceSize > self.size {
            bail!("write_from_slice: {} bytes > buffer size {}", bytes.len(), self.size);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, bytes.len());
        }
        Ok(())
    }

    /// Copy a POD slice into a host-visible buffer.
    pub fn write_pod_slice<T: bytemuck::Pod>(&self, values: &[T]) -> Result<()> {
        self.write_from_slice(bytemuck::cast_slice(values))
    }

    /// Copy host-visible buffer bytes into an existing POD slice.
    pub fn read_pod_slice<T: bytemuck::Pod>(&self, values: &mut [T]) -> Result<()> {
        if self.location != BufferLocation::Host {
            bail!("read_pod_slice: buffer is not host-visible");
        }
        let bytes = bytemuck::cast_slice_mut(values);
        if bytes.len() as vk::DeviceSize > self.size {
            bail!("read_pod_slice: {} bytes > buffer size {}", bytes.len(), self.size);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, bytes.as_mut_ptr(), bytes.len());
        }
        Ok(())
    }

    /// Copy host-visible buffer bytes into a new Vec.
    pub fn read_into_vec(&self) -> Result<Vec<u8>> {
        if self.location != BufferLocation::Host {
            bail!("read_into_vec: buffer is not host-visible");
        }
        let len = self.size as usize;
        let mut out = vec![0u8; len];
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr(), len); }
        Ok(out)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.location == BufferLocation::Host && !self.mapped.is_null() {
                self.ctx.device.unmap_memory(self.memory);
            }
            self.ctx.device.destroy_buffer(self.raw, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}
