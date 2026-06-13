//! Vulkan instance, physical device, logical device, and the one
//! compute queue we'll submit to.
//!
//! Deliberately minimal: most matmul-heavy workloads don't benefit
//! from multiple compute queues on the same family. Async transfer is a
//! separate concern that can be layered on top if needed.

mod cache;
mod debug;
mod device;

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::Mutex;

use cache::pipeline_cache_path_for;
use debug::debug_callback;
use device::{device_summaries, select_physical_device};

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
        unsafe {
            let entry =
                ash::Entry::load().map_err(|e| anyhow!("failed to load Vulkan loader: {e}"))?;

            let app_name = CString::new("tensor-ash").unwrap();
            let engine_name = CString::new("tensor-ash").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .engine_name(&engine_name)
                .api_version(vk::API_VERSION_1_2);

            let validation_name: &CStr = c"VK_LAYER_KHRONOS_validation";
            let have_validation = enable_validation
                && entry
                    .enumerate_instance_layer_properties()?
                    .iter()
                    .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);

            let debug_utils_name = ash::ext::debug_utils::NAME;
            let have_debug_utils = have_validation
                && entry
                    .enumerate_instance_extension_properties(None)?
                    .iter()
                    .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == debug_utils_name);

            let mut layers: Vec<*const i8> = Vec::new();
            if have_validation {
                layers.push(validation_name.as_ptr());
            }
            let mut exts: Vec<*const i8> = Vec::new();
            if have_debug_utils {
                exts.push(debug_utils_name.as_ptr());
            }

            let instance_ci = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&exts);
            let instance = entry
                .create_instance(&instance_ci, None)
                .context("create_instance")?;

            let (debug_loader, debug_messenger) = if have_debug_utils {
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback));
                let msg = loader
                    .create_debug_utils_messenger(&ci, None)
                    .context("create_debug_utils_messenger")?;
                (Some(loader), Some(msg))
            } else {
                (None, None)
            };

            let phys_devs = instance
                .enumerate_physical_devices()
                .context("enumerate_physical_devices")?;
            if phys_devs.is_empty() {
                bail!("no Vulkan-capable physical devices found");
            }
            let device_summaries = device_summaries(&instance, &phys_devs);
            let selected_index = select_physical_device(&device_summaries, &preference)?;
            let physical_device = phys_devs[selected_index];
            let device_properties = instance.get_physical_device_properties(physical_device);
            let device_summary = device_summaries[selected_index].clone();
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            log::info!(
                "tensor-ash: using device #{}: {} ({}, Vulkan {})",
                device_summary.index,
                device_summary.name,
                device_summary.kind.as_str(),
                device_summary.api_version_string(),
            );

            let qf_props = instance.get_physical_device_queue_family_properties(physical_device);
            let compute_family = qf_props
                .iter()
                .enumerate()
                .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .min_by_key(|(_, p)| {
                    if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        1
                    } else {
                        0
                    }
                })
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("no compute-capable queue family"))?;

            let timestamps_supported = qf_props[compute_family as usize].timestamp_valid_bits > 0
                && device_properties.limits.timestamp_period > 0.0;

            let priorities = [1.0f32];
            let queue_ci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&priorities)];

            // Probe whether the physical device advertises Vulkan 1.2's
            // `bufferDeviceAddress` and the Stream-K
            // `shaderBufferFloat32AtomicAdd` from
            // VK_EXT_shader_atomic_float.  Both are core for the
            // post-2020 NVIDIA / AMD drivers we target but we still
            // gate enablement on the actual `get` queries so the lib
            // remains runnable on llvmpipe / older mobile GPUs.
            let mut bda_query = vk::PhysicalDeviceVulkan12Features::default();
            let mut atomic_float_query =
                vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut bda_query)
                .push_next(&mut atomic_float_query);
            instance.get_physical_device_features2(physical_device, &mut features2);
            let buffer_device_address_supported =
                bda_query.buffer_device_address == vk::TRUE;
            let shader_buffer_float32_atomic_add_supported =
                atomic_float_query.shader_buffer_float32_atomic_add == vk::TRUE;

            // The feature-bit query above is necessary but not
            // sufficient: the extension must also appear in the
            // device's enabled-extension list at create-device time.
            // Enumerate and check before flipping the enable flag.
            let device_exts = instance
                .enumerate_device_extension_properties(physical_device)
                .context("enumerate_device_extension_properties")?;
            let atomic_float_ext_name = ash::ext::shader_atomic_float::NAME;
            let atomic_float_ext_present = device_exts.iter().any(|ext| {
                CStr::from_ptr(ext.extension_name.as_ptr()) == atomic_float_ext_name
            });
            let enable_shader_atomic_float =
                shader_buffer_float32_atomic_add_supported && atomic_float_ext_present;

            let mut enabled_device_exts: Vec<*const i8> = Vec::new();
            if enable_shader_atomic_float {
                enabled_device_exts.push(atomic_float_ext_name.as_ptr());
            }

            let features = vk::PhysicalDeviceFeatures::default();
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
            if buffer_device_address_supported {
                vulkan12 = vulkan12.buffer_device_address(true);
            }
            let mut atomic_float_features =
                vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            if enable_shader_atomic_float {
                atomic_float_features =
                    atomic_float_features.shader_buffer_float32_atomic_add(true);
            }
            let mut device_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_ci)
                .enabled_features(&features)
                .enabled_extension_names(&enabled_device_exts);
            if buffer_device_address_supported {
                device_ci = device_ci.push_next(&mut vulkan12);
            }
            if enable_shader_atomic_float {
                device_ci = device_ci.push_next(&mut atomic_float_features);
            }
            let device = instance
                .create_device(physical_device, &device_ci, None)
                .context("create_device")?;
            let queue = device.get_device_queue(compute_family, 0);

            let pipeline_cache_path = pipeline_cache_path_for(&device_summary);
            let cache_data: Vec<u8> = pipeline_cache_path
                .as_ref()
                .and_then(|p| std::fs::read(p).ok())
                .unwrap_or_default();
            let pipeline_cache = device
                .create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo::default().initial_data(&cache_data),
                    None,
                )
                .unwrap_or_else(|err| {
                    log::warn!(
                        "tensor-ash: create_pipeline_cache failed ({err}); proceeding without persistence",
                    );
                    vk::PipelineCache::null()
                });

            Ok(Arc::new(Self {
                entry,
                instance,
                physical_device,
                device,
                device_summary,
                device_properties,
                memory_properties,
                compute_family,
                queue: Mutex::new(queue),
                timestamp_period_ns: device_properties.limits.timestamp_period as f64,
                timestamps_supported,
                buffer_device_address_enabled: buffer_device_address_supported,
                shader_buffer_float32_atomic_add_enabled: enable_shader_atomic_float,
                pipeline_cache,
                pipeline_cache_path,
                debug_loader,
                debug_messenger,
            }))
        }
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
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer),
            )
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
                if let Some(path) = self.pipeline_cache_path.as_ref()
                    && let Ok(data) = self.device.get_pipeline_cache_data(self.pipeline_cache)
                {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if let Err(err) = std::fs::write(path, &data) {
                        log::warn!(
                            "tensor-ash: failed to write pipeline cache to {}: {err}",
                            path.display(),
                        );
                    }
                }
                self.device
                    .destroy_pipeline_cache(self.pipeline_cache, None);
            }
            if let (Some(loader), Some(msg)) = (self.debug_loader.as_ref(), self.debug_messenger) {
                loader.destroy_debug_utils_messenger(msg, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
