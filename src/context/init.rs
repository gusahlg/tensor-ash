//! Fallible Vulkan setup.
//!
//! Vulkan handles do not implement `Drop`, so every handle created here is
//! guarded until ownership is transferred to [`VulkanContext`]. This keeps an
//! error during device discovery or creation from leaking the instance (and,
//! when enabled, its debug messenger).

use std::ffi::CStr;

use anyhow::{Context, Result, anyhow};
use ash::vk;
use parking_lot::Mutex;
use scopeguard::ScopeGuard;

use super::VulkanContext;
use super::cache::{create_pipeline_cache, pipeline_cache_path_for};
use super::debug::debug_callback;
use super::device::{DevicePreference, device_summaries, select_physical_device};

pub(super) fn create(
    enable_validation: bool,
    preference: DevicePreference,
) -> Result<VulkanContext> {
    unsafe {
        let entry =
            ash::Entry::load().map_err(|err| anyhow!("failed to load Vulkan loader: {err}"))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"tensor-ash")
            .engine_name(c"tensor-ash")
            .api_version(vk::API_VERSION_1_2);
        let validation_name = c"VK_LAYER_KHRONOS_validation";
        let have_validation = enable_validation
            && entry
                .enumerate_instance_layer_properties()?
                .iter()
                .any(|layer| CStr::from_ptr(layer.layer_name.as_ptr()) == validation_name);
        let have_debug_utils = have_validation
            && entry
                .enumerate_instance_extension_properties(None)?
                .iter()
                .any(|ext| {
                    CStr::from_ptr(ext.extension_name.as_ptr()) == ash::ext::debug_utils::NAME
                });

        let layers = have_validation
            .then_some(validation_name.as_ptr())
            .into_iter()
            .collect::<Vec<_>>();
        let extensions = have_debug_utils
            .then_some(ash::ext::debug_utils::NAME.as_ptr())
            .into_iter()
            .collect::<Vec<_>>();
        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = entry
            .create_instance(&instance_ci, None)
            .context("create_instance")?;
        let instance = scopeguard::guard(instance, |instance| instance.destroy_instance(None));

        let debug = if have_debug_utils {
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
            let messenger = loader
                .create_debug_utils_messenger(&ci, None)
                .context("create_debug_utils_messenger")?;
            Some(scopeguard::guard(
                (loader, messenger),
                |(loader, messenger)| loader.destroy_debug_utils_messenger(messenger, None),
            ))
        } else {
            None
        };

        let physical_devices = instance
            .enumerate_physical_devices()
            .context("enumerate_physical_devices")?;
        let summaries = device_summaries(&instance, &physical_devices);
        let selected_index = select_physical_device(&summaries, &preference)?;
        let physical_device = physical_devices[selected_index];
        let device_summary = summaries[selected_index].clone();
        let device_properties = instance.get_physical_device_properties(physical_device);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        log::info!(
            "tensor-ash: using device #{}: {} ({}, Vulkan {})",
            device_summary.index,
            device_summary.name,
            device_summary.kind.as_str(),
            device_summary.api_version_string(),
        );

        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let compute_family = queue_families
            .iter()
            .enumerate()
            .filter(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .min_by_key(|(_, properties)| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| anyhow!("no compute-capable queue family"))?;
        let timestamps_supported = queue_families[compute_family as usize].timestamp_valid_bits > 0
            && device_properties.limits.timestamp_period > 0.0;
        let timestamp_valid_bits = queue_families[compute_family as usize].timestamp_valid_bits;

        let mut vulkan11_query = vk::PhysicalDeviceVulkan11Features::default();
        let mut vulkan12_query = vk::PhysicalDeviceVulkan12Features::default();
        let mut atomic_float_query = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
        let mut features_query = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan11_query)
            .push_next(&mut vulkan12_query)
            .push_next(&mut atomic_float_query);
        instance.get_physical_device_features2(physical_device, &mut features_query);
        let buffer_device_address_supported = vulkan12_query.buffer_device_address == vk::TRUE;
        let atomic_float_supported =
            atomic_float_query.shader_buffer_float32_atomic_add == vk::TRUE;
        // f16 storage kernels need both halves: `shaderFloat16` for the
        // arithmetic types and `storageBuffer16BitAccess` for 16-bit
        // access through physical-storage-buffer pointers (the SPIR-V
        // capability covers BDA loads too, not just descriptor SSBOs).
        let f16_storage_supported = vulkan12_query.shader_float16 == vk::TRUE
            && vulkan11_query.storage_buffer16_bit_access == vk::TRUE;

        // A supported feature bit is not enough for an extension feature: the
        // corresponding extension must also be enabled when creating the device.
        let device_extensions = instance
            .enumerate_device_extension_properties(physical_device)
            .context("enumerate_device_extension_properties")?;
        let atomic_float_name = ash::ext::shader_atomic_float::NAME;
        let atomic_float_present = device_extensions
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == atomic_float_name);
        let enable_atomic_float = atomic_float_supported && atomic_float_present;
        let enabled_device_extensions = enable_atomic_float
            .then_some(atomic_float_name.as_ptr())
            .into_iter()
            .collect::<Vec<_>>();

        let priorities = [1.0];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default();
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f16_storage_supported);
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(buffer_device_address_supported)
            .shader_float16(f16_storage_supported);
        let mut atomic_float = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomic_add(enable_atomic_float);
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_features(&features)
            .enabled_extension_names(&enabled_device_extensions);
        if buffer_device_address_supported || f16_storage_supported {
            device_ci = device_ci.push_next(&mut vulkan12);
        }
        if f16_storage_supported {
            device_ci = device_ci.push_next(&mut vulkan11);
        }
        if enable_atomic_float {
            device_ci = device_ci.push_next(&mut atomic_float);
        }
        let device = instance
            .create_device(physical_device, &device_ci, None)
            .context("create_device")?;
        let device = scopeguard::guard(device, |device| device.destroy_device(None));
        let queue = device.get_device_queue(compute_family, 0);

        let pipeline_cache_path = pipeline_cache_path_for(&device_summary);
        let pipeline_cache = create_pipeline_cache(&device, pipeline_cache_path.as_deref());
        let pipeline_cache = scopeguard::guard(pipeline_cache, |pipeline_cache| {
            if pipeline_cache != vk::PipelineCache::null() {
                device.destroy_pipeline_cache(pipeline_cache, None);
            }
        });

        // From here on there are no fallible operations. Disarm the guards in
        // child-before-parent order and transfer the handles to VulkanContext.
        let pipeline_cache = ScopeGuard::into_inner(pipeline_cache);
        let device = ScopeGuard::into_inner(device);
        let (debug_loader, debug_messenger) = match debug {
            Some(debug) => {
                let (loader, messenger) = ScopeGuard::into_inner(debug);
                (Some(loader), Some(messenger))
            }
            None => (None, None),
        };
        let instance = ScopeGuard::into_inner(instance);

        Ok(VulkanContext {
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
            timestamp_valid_bits,
            timestamps_supported,
            buffer_device_address_enabled: buffer_device_address_supported,
            shader_buffer_float32_atomic_add_enabled: enable_atomic_float,
            f16_storage_enabled: f16_storage_supported,
            pipeline_cache,
            pipeline_cache_path,
            debug_loader,
            debug_messenger,
        })
    }
}
