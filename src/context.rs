//! Vulkan instance, physical device, logical device, and the one
//! compute queue we'll submit to.
//!
//! Deliberately minimal — most matmul-heavy workloads don't benefit
//! from multiple compute queues on the same family (they time-multiplex
//! the same SMs).  Async transfer is a separate concern that can be
//! layered on top if needed.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::Mutex;

pub struct VulkanContext {
    pub entry:             ash::Entry,
    pub instance:          ash::Instance,
    pub physical_device:   vk::PhysicalDevice,
    pub device:            ash::Device,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub compute_family:    u32,
    /// Compute queue.  Vulkan requires external sync on a queue, hence the mutex.
    pub queue:             Mutex<vk::Queue>,
    /// Nanoseconds-per-tick reported by the driver for GPU timestamps.
    pub timestamp_period_ns: f64,
    pub timestamps_supported: bool,
    debug_loader:    Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanContext {
    pub fn new(enable_validation: bool) -> Result<Arc<Self>> { unsafe {
        let entry = ash::Entry::load()
            .map_err(|e| anyhow!("failed to load Vulkan loader: {e}"))?;

        // ---- Instance ------------------------------------------------------
        let app_name    = CString::new("ml_project").unwrap();
        let engine_name = CString::new("ml_project").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::API_VERSION_1_2);

        let validation_name: &CStr =
            CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
        let have_validation = enable_validation
            && entry.enumerate_instance_layer_properties()?
                .iter()
                .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);

        let debug_utils_name = ash::ext::debug_utils::NAME;
        let have_debug_utils = have_validation
            && entry.enumerate_instance_extension_properties(None)?
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == debug_utils_name);

        let mut layers: Vec<*const i8> = Vec::new();
        if have_validation { layers.push(validation_name.as_ptr()); }
        let mut exts: Vec<*const i8> = Vec::new();
        if have_debug_utils { exts.push(debug_utils_name.as_ptr()); }

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&exts);
        let instance = entry.create_instance(&instance_ci, None)
            .context("create_instance")?;

        // ---- Debug messenger ----------------------------------------------
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
            let msg = loader.create_debug_utils_messenger(&ci, None)
                .context("create_debug_utils_messenger")?;
            (Some(loader), Some(msg))
        } else { (None, None) };

        // ---- Pick physical device -----------------------------------------
        let phys_devs = instance.enumerate_physical_devices()
            .context("enumerate_physical_devices")?;
        if phys_devs.is_empty() {
            bail!("no Vulkan-capable physical devices found");
        }
        let physical_device = pick_physical_device(&instance, &phys_devs);
        let device_properties = instance.get_physical_device_properties(physical_device);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let device_name = CStr::from_ptr(device_properties.device_name.as_ptr())
            .to_string_lossy().into_owned();
        log::info!(
            "ml_project: using {device_name} (Vulkan {}.{}.{})",
            vk::api_version_major(device_properties.api_version),
            vk::api_version_minor(device_properties.api_version),
            vk::api_version_patch(device_properties.api_version),
        );

        // ---- Pick compute queue family ------------------------------------
        let qf_props = instance.get_physical_device_queue_family_properties(physical_device);
        let compute_family = qf_props.iter().enumerate()
            .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            // Prefer a dedicated compute family (no GRAPHICS bit) when available.
            .min_by_key(|(_, p)|
                if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) { 1 } else { 0 })
            .map(|(i, _)| i as u32)
            .ok_or_else(|| anyhow!("no compute-capable queue family"))?;

        let timestamps_supported =
            qf_props[compute_family as usize].timestamp_valid_bits > 0
            && device_properties.limits.timestamp_period > 0.0;

        let priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&priorities)];

        let features = vk::PhysicalDeviceFeatures::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .host_query_reset(true);
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_features(&features)
            .push_next(&mut features12);
        let device = instance.create_device(physical_device, &device_ci, None)
            .context("create_device")?;
        let queue = device.get_device_queue(compute_family, 0);

        Ok(Arc::new(Self {
            entry, instance, physical_device, device,
            device_properties, memory_properties,
            compute_family,
            queue: Mutex::new(queue),
            timestamp_period_ns: device_properties.limits.timestamp_period as f64,
            timestamps_supported,
            debug_loader, debug_messenger,
        }))
    }}

    /// Find a memory type satisfying `requirements` containing every flag in `props`.
    pub fn find_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        props:        vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            let bit = 1u32 << i;
            if (requirements.memory_type_bits & bit) == 0 { continue; }
            if self.memory_properties.memory_types[i as usize]
                .property_flags.contains(props)
            {
                return Ok(i);
            }
        }
        bail!("no memory type with properties {:?}", props);
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let (Some(loader), Some(msg)) =
                (self.debug_loader.as_ref(), self.debug_messenger)
            {
                loader.destroy_debug_utils_messenger(msg, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    devices:  &[vk::PhysicalDevice],
) -> vk::PhysicalDevice {
    // Score: type-class (discrete > integrated > virtual > cpu) + compute capacity.
    devices.iter().copied().max_by_key(|&pd| {
        let p = unsafe { instance.get_physical_device_properties(pd) };
        let type_score: i64 = match p.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU   => 4_000_000_000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2_000_000_000,
            vk::PhysicalDeviceType::VIRTUAL_GPU    => 1_000_000_000,
            vk::PhysicalDeviceType::CPU            => 100_000,
            _                                      => 0,
        };
        type_score + p.limits.max_compute_work_group_invocations as i64
    }).expect("non-empty device list")
}

unsafe extern "system" fn debug_callback(
    _sev:  vk::DebugUtilsMessageSeverityFlagsEXT,
    _typ:  vk::DebugUtilsMessageTypeFlagsEXT,
    data:  *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() { return vk::FALSE; }
    let msg_ptr = unsafe { (*data).p_message };
    if !msg_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(msg_ptr) }.to_string_lossy();
        eprintln!("[vulkan] {msg}");
    }
    vk::FALSE
}
