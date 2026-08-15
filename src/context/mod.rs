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

pub(crate) fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> u64 {
    let delta = end.wrapping_sub(start);
    match valid_bits {
        1..64 => delta & ((1u64 << valid_bits) - 1),
        _ => delta,
    }
}

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device_summary: DeviceSummary,
    pub device: ash::Device,
    pub device_properties: vk::PhysicalDeviceProperties,
    /// `VkPhysicalDeviceDriverProperties::driverID` (core in Vulkan
    /// 1.2); `None` on pre-1.2 devices.  Distinguishes drivers that
    /// share a vendor id (NVIDIA proprietary vs Mesa NVK) for
    /// driver-scoped quirk handling — see
    /// [`Self::workgroup_shared_budget`].
    pub driver_id: Option<vk::DriverId>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub compute_family: u32,
    /// Compute queue.  Vulkan requires external sync on a queue, hence the mutex.
    pub queue: Mutex<vk::Queue>,
    /// Nanoseconds-per-tick reported by the driver for GPU timestamps.
    pub timestamp_period_ns: f64,
    /// Number of valid low bits in timestamps from the selected queue.
    pub timestamp_valid_bits: u32,
    pub timestamps_supported: bool,
    /// Whether `VK_KHR_buffer_device_address` (= Vulkan 1.2
    /// `bufferDeviceAddress`) was successfully enabled.  Required for
    /// the buffer-reference GLSL path used by the LDG.128 kernels.
    pub buffer_device_address_enabled: bool,
    /// Whether `shaderFloat16` + `storageBuffer16BitAccess` were both
    /// enabled.  Required for the f16-storage kernel variants; without
    /// it f16 tensors are rejected at matmul time and the f16 kernels
    /// are not built.
    pub f16_storage_enabled: bool,
    /// Whether `VK_KHR_cooperative_matrix` was enabled (with the
    /// Vulkan memory model).  Required for the tensor-core `coopmat`
    /// kernels; `ML_NO_COOPMAT=1` forces it off.
    pub coopmat_enabled: bool,
    /// Whether `VK_NV_cooperative_matrix2` was enabled with every
    /// feature bit the cm2 flash kernels need (workgroup scope,
    /// flexible dimensions, reductions, conversions, per-element ops,
    /// tensor addressing, block loads) on a Vulkan 1.3+ device.
    /// Requires [`Self::coopmat_enabled`]; `ML_NO_COOPMAT2=1` forces
    /// it off.
    pub coopmat2_enabled: bool,
    /// Pipeline cache, seeded from disk on init and flushed back on drop.
    /// Persisting it avoids the SPIR-V -> ISA recompile (50-200 ms on
    /// NVIDIA) on every cold start.
    pub pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<PathBuf>,
    debug_loader: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanContext {
    /// Largest workgroup-memory declaration currently shipped in the
    /// kernel catalog (the 128x64/BK=64 tiles: `As[128][65]` f32 +
    /// `Bs[64][16]` uvec4).  Caps the NVIDIA-proprietary allowance in
    /// [`Self::workgroup_shared_budget`] so a future, even larger tile
    /// is still gated everywhere; `catalog.rs` tests pin the registry
    /// maximum to this value.
    pub const MAX_SHIPPED_WORKGROUP_BYTES: u32 = 49_664;

    /// Create a context honoring `ML_DEVICE` (`auto`, `discrete`,
    /// `integrated`, `virtual`, `cpu`, `index:N`, `name:TEXT`, or a
    /// bare name substring; unset = auto).  Reading the variable here,
    /// at the library entry point, lets the test suite and every tool
    /// select a device on multi-GPU hosts without loader tricks.
    pub fn new(enable_validation: bool) -> Result<Arc<Self>> {
        let preference = DevicePreference::parse(&std::env::var("ML_DEVICE").unwrap_or_default())?;
        Self::new_with_device_preference(enable_validation, preference)
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

    /// Per-workgroup shared-memory budget the kernel registry gates
    /// against.  Spec-wise this is `maxComputeSharedMemorySize`: a
    /// shader declaring more is invalid
    /// (VUID-RuntimeSpirv-Workgroup-06530), and NVK on Turing enforces
    /// it the hard way — pipeline runs lose the device at dispatch
    /// (Xid 13 `SKEDCHECK18_L1_CONFIG_TOO_SMALL`).
    ///
    /// Exception: the NVIDIA proprietary driver reports the universal
    /// 49,152 B limit but demonstrably does not enforce it (the
    /// hardware shared carveout is larger), and the measured tuning
    /// corpus was built on it with the 49,664 B BK=64 tiles winning
    /// their shape classes.  That driver therefore keeps a bounded
    /// allowance up to the largest shipped declaration, preserving
    /// measured routing.  Trimming those tiles under 49,152 B is the
    /// real fix and would retire this exception.
    pub fn workgroup_shared_budget(&self) -> u32 {
        let limit = self.device_properties.limits.max_compute_shared_memory_size;
        if self.driver_id == Some(vk::DriverId::NVIDIA_PROPRIETARY) {
            limit.max(Self::MAX_SHIPPED_WORKGROUP_BYTES)
        } else {
            limit
        }
    }

    /// True when this driver is known to compile identical shader
    /// arithmetic into identical instruction sequences across the
    /// spec-constant variants of a pipeline, so a fused store epilogue
    /// reproduces its composed reference bit-for-bit (the row-GEMV
    /// RoPE/scatter store contract).  Verified to hold on the NVIDIA
    /// proprietary driver and Mesa RADV.  Mesa NVK's backend currently
    /// applies value-changing transforms non-uniformly between
    /// variants (~1 ULP into the rotation inputs, persisting even
    /// under `precise`-qualified arithmetic), so bitwise
    /// fused-vs-composed comparisons must relax to tolerance there.
    /// Unknown drivers report true: the shaders pin the reduce and
    /// rotation orders with explicit fma, so any backend that refrains
    /// from value-changing transforms reproduces them exactly.
    pub fn fused_store_bit_reproducible(&self) -> bool {
        self.driver_id != Some(vk::DriverId::MESA_NVK)
    }

    pub fn device_name(&self) -> &str {
        &self.device_summary.name
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.device_summary.kind
    }

    pub fn diagnostics(&self) -> String {
        format!(
            "device #{}: {} ({}, Vulkan {}, vendor=0x{:04x}, device=0x{:04x}, driver={}, compute_family={}, timestamps={} ({} bits))",
            self.device_summary.index,
            self.device_summary.name,
            self.device_summary.kind.as_str(),
            self.device_summary.api_version_string(),
            self.device_summary.vendor_id,
            self.device_summary.device_id,
            self.device_summary.driver_version,
            self.compute_family,
            self.timestamps_supported,
            self.timestamp_valid_bits,
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
