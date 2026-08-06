//! Vulkan instance, physical device, logical device, and the one
//! compute queue we'll submit to.
//!
//! Deliberately minimal: most matmul-heavy workloads don't benefit
//! from multiple compute queues on the same family. Async transfer is a
//! separate concern that can be layered on top if needed.

mod cache;
mod debug;
mod device;
mod init;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};
use ash::vk;
use parking_lot::Mutex;

pub use device::{DeviceKind, DevicePreference, DeviceSummary};

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device_summary: DeviceSummary,
    pub device: ash::Device,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub compute_family: u32,
    /// Compute queue.  Vulkan requires external sync on a queue, hence the mutex.
    pub queue: Mutex<vk::Queue>,
    /// Nanoseconds-per-tick reported by the driver for GPU timestamps.
    pub timestamp_period_ns: f64,
    pub timestamps_supported: bool,
    /// Whether `VK_KHR_buffer_device_address` (= Vulkan 1.2
    /// `bufferDeviceAddress`) was successfully enabled.  Required for
    /// the buffer-reference GLSL path used by the LDG.128 kernels.
    pub buffer_device_address_enabled: bool,
    /// Whether `VK_EXT_shader_atomic_float` was enabled with
    /// `shaderBufferFloat32AtomicAdd`.  Required for the hardware
    /// `atomicAdd(float, float)` path in the Stream-K kernel; absent
    /// this, Stream-K pipeline creation is rejected and callers fall
    /// back to the regular DP path.
    pub shader_buffer_float32_atomic_add_enabled: bool,
    /// Pipeline cache, seeded from disk on init and flushed back on drop.
    /// Persisting it avoids the SPIR-V -> ISA recompile (50-200 ms on
    /// NVIDIA) on every cold start.
    pub pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<PathBuf>,
    debug_loader: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanContext {
    pub fn new(enable_validation: bool) -> Result<Arc<Self>> {
        Self::new_with_device_preference(enable_validation, DevicePreference::Auto)
    }

    pub fn new_with_device_preference(
        enable_validation: bool,
        preference: DevicePreference,
    ) -> Result<Arc<Self>> {
        init::create(enable_validation, preference).map(Arc::new)
    }

    /// Find a memory type satisfying `requirements` containing every flag in `props`.
    pub fn find_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        props: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            let bit = 1u32 << i;
            if (requirements.memory_type_bits & bit) == 0 {
                continue;
            }
            if self.memory_properties.memory_types[i as usize]
                .property_flags
                .contains(props)
            {
                return Ok(i);
            }
        }
        bail!("no memory type with properties {:?}", props);
    }

    /// Find a memory type satisfying `required`, preferring all flags in
    /// `preferred` when the device exposes a compatible type.
    pub fn find_memory_type_preferred(
        &self,
        requirements: vk::MemoryRequirements,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        if !preferred.is_empty() {
            let preferred_props = required | preferred;
            if let Ok(index) = self.find_memory_type(requirements, preferred_props) {
                return Ok(index);
            }
        }
        self.find_memory_type(requirements, required)
    }

    /// Query the GPU virtual address of `buffer`.  Panics if buffer
    /// device address was not enabled at context creation (call
    /// `buffer_device_address_enabled` first).
    ///
    /// The address is a 64-bit GPU pointer that can be passed as a
    /// push constant and dereferenced inside the kernel via
    /// `GL_EXT_buffer_reference`.
    pub fn buffer_device_address(&self, buffer: vk::Buffer) -> u64 {
        assert!(
            self.buffer_device_address_enabled,
            "buffer_device_address requested but feature not enabled"
        );
        unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_summary.name
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.device_summary.kind
    }

    pub fn diagnostics(&self) -> String {
        format!(
            "device #{}: {} ({}, Vulkan {}, vendor=0x{:04x}, device=0x{:04x}, driver={}, compute_family={}, timestamps={})",
            self.device_summary.index,
            self.device_summary.name,
            self.device_summary.kind.as_str(),
            self.device_summary.api_version_string(),
            self.device_summary.vendor_id,
            self.device_summary.device_id,
            self.device_summary.driver_version,
            self.compute_family,
            self.timestamps_supported,
        )
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if self.pipeline_cache != vk::PipelineCache::null() {
                cache::persist_pipeline_cache(
                    &self.device,
                    self.pipeline_cache,
                    self.pipeline_cache_path.as_deref(),
                );
                self.device
                    .destroy_pipeline_cache(self.pipeline_cache, None);
            }
            self.device.destroy_device(None);
            if let (Some(loader), Some(msg)) = (self.debug_loader.as_ref(), self.debug_messenger) {
                loader.destroy_debug_utils_messenger(msg, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
