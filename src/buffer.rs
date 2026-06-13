//! GPU buffer + its backing device memory.
//!
//! Two flavors:
//!   * `Device` — device-local memory.  Fastest for the GPU.  Host I/O
//!     must go through a staging buffer (see `Executor::upload`).
//!   * `Host` — host-visible memory, preferring coherent writes.
//!   * `HostCached` — host-visible memory, preferring cached CPU reads.
//!
//! Host buffers are persistently mapped for their whole lifetime.  Non-coherent
//! memory is flushed after CPU writes and invalidated before CPU reads.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ash::vk;
use scopeguard::ScopeGuard;

use crate::context::VulkanContext;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BufferLocation {
    Device,
    Host,
    HostCached,
}

pub struct Buffer {
    ctx: Arc<VulkanContext>,
    pub raw: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub location: BufferLocation,
    memory_flags: vk::MemoryPropertyFlags,
    /// Persistent mapping pointer for host-visible buffers; null otherwise.
    mapped: *mut u8,
}

// The pointer is owned: no aliasing across threads other than synchronized
// reads/writes via this object.  The Vulkan handles themselves are thread-safe.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(
        ctx: &Arc<VulkanContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: BufferLocation,
    ) -> Result<Self> {
        assert!(size > 0, "Buffer::new with size=0");
        unsafe {
            // Each Vulkan handle we allocate is wrapped in a scopeguard that
            // destroys it on early return.  On the happy path we disarm the
            // guards at the end with ScopeGuard::into_inner.
            //
            // When the context has Vulkan 1.2 `bufferDeviceAddress`
            // enabled, every buffer gets `SHADER_DEVICE_ADDRESS` so the
            // kernel-side `buffer_reference` path can dereference it.
            // We also have to flag the memory allocation with
            // `MEMORY_ALLOCATE_DEVICE_ADDRESS`.
            let usage = if ctx.buffer_device_address_enabled {
                usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            } else {
                usage
            };
            let raw = ctx
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .context("create_buffer")?;
            let raw_guard = scopeguard::guard(raw, |raw| ctx.device.destroy_buffer(raw, None));

            let mem_req = ctx.device.get_buffer_memory_requirements(raw);

            let (required, preferred) = match location {
                BufferLocation::Device => (
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                ),
                BufferLocation::Host => (
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    vk::MemoryPropertyFlags::HOST_COHERENT,
                ),
                BufferLocation::HostCached => (
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                ),
            };
            let mem_type = ctx.find_memory_type_preferred(mem_req, required, preferred)?;
            let memory_flags = ctx.memory_properties.memory_types[mem_type as usize].property_flags;

            let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default();
            if ctx.buffer_device_address_enabled {
                alloc_flags = alloc_flags.flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            }
            let mut alloc_ci = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type);
            if ctx.buffer_device_address_enabled {
                alloc_ci = alloc_ci.push_next(&mut alloc_flags);
            }
            let memory = ctx
                .device
                .allocate_memory(&alloc_ci, None)
                .context("allocate_memory")?;
            let memory_guard = scopeguard::guard(memory, |m| ctx.device.free_memory(m, None));

            ctx.device
                .bind_buffer_memory(raw, memory, 0)
                .context("bind_buffer_memory")?;

            let mapped = if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                ctx.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .context("map_memory")? as *mut u8
            } else {
                std::ptr::null_mut()
            };

            Ok(Self {
                ctx: Arc::clone(ctx),
                raw: ScopeGuard::into_inner(raw_guard),
                memory: ScopeGuard::into_inner(memory_guard),
                size,
                location,
                memory_flags,
                mapped,
            })
        }
    }

    fn is_host_visible(&self) -> bool {
        self.memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    fn is_host_coherent(&self) -> bool {
        self.memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    }

    fn flush_host_writes(&self) -> Result<()> {
        if self.is_host_coherent() {
            return Ok(());
        }
        let range = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        unsafe {
            self.ctx
                .device
                .flush_mapped_memory_ranges(&range)
                .context("flush_mapped_memory_ranges")?;
        }
        Ok(())
    }

    fn invalidate_host_reads(&self) -> Result<()> {
        if self.is_host_coherent() {
            return Ok(());
        }
        let range = [vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        unsafe {
            self.ctx
                .device
                .invalidate_mapped_memory_ranges(&range)
                .context("invalidate_mapped_memory_ranges")?;
        }
        Ok(())
    }

    /// Copy CPU bytes into a host-visible buffer.
    pub fn write_from_slice(&self, bytes: &[u8]) -> Result<()> {
        if !self.is_host_visible() {
            bail!("write_from_slice: buffer is not host-visible");
        }
        if bytes.len() as vk::DeviceSize > self.size {
            bail!(
                "write_from_slice: {} bytes > buffer size {}",
                bytes.len(),
                self.size
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, bytes.len());
        }
        self.flush_host_writes()?;
        Ok(())
    }

    /// Copy a POD slice into a host-visible buffer.
    pub fn write_pod_slice<T: bytemuck::Pod>(&self, values: &[T]) -> Result<()> {
        self.write_from_slice(bytemuck::cast_slice(values))
    }

    /// Copy host-visible buffer bytes into an existing POD slice.
    pub fn read_pod_slice<T: bytemuck::Pod>(&self, values: &mut [T]) -> Result<()> {
        if !self.is_host_visible() {
            bail!("read_pod_slice: buffer is not host-visible");
        }
        let bytes = bytemuck::cast_slice_mut(values);
        if bytes.len() as vk::DeviceSize > self.size {
            bail!(
                "read_pod_slice: {} bytes > buffer size {}",
                bytes.len(),
                self.size
            );
        }
        self.invalidate_host_reads()?;
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, bytes.as_mut_ptr(), bytes.len());
        }
        Ok(())
    }

    /// Copy host-visible buffer bytes into a new Vec.
    pub fn read_into_vec(&self) -> Result<Vec<u8>> {
        if !self.is_host_visible() {
            bail!("read_into_vec: buffer is not host-visible");
        }
        let len = self.size as usize;
        let mut out = vec![0u8; len];
        self.invalidate_host_reads()?;
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, out.as_mut_ptr(), len);
        }
        Ok(out)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if !self.mapped.is_null() {
                self.ctx.device.unmap_memory(self.memory);
            }
            self.ctx.device.destroy_buffer(self.raw, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}
