//! Fallible Vulkan setup.
//!
//! Vulkan handles do not implement `Drop`, so every handle created here is
//! guarded until ownership is transferred to [`VulkanContext`]. This keeps an
//! error during device discovery or creation from leaking the instance (and,
//! when enabled, its debug messenger).

use std::ffi::{CStr, c_void};

use anyhow::{Context, Result, anyhow};
use ash::vk;
use parking_lot::Mutex;
use scopeguard::ScopeGuard;

use super::VulkanContext;
use super::cache::{create_pipeline_cache, pipeline_cache_path_for};
use super::debug::debug_callback;
use super::device::{DevicePreference, device_summaries, select_physical_device};

/// Keep one bare `VkInstance` (and the loader library itself) alive for
/// the remaining lifetime of the process.
///
/// The Vulkan loader dlcloses every ICD shared library when the last
/// `VkInstance` is destroyed and re-dlopens them all on the next
/// `vkCreateInstance`.  Some ICDs carry static-TLS dependencies (the
/// NVIDIA proprietary driver's `libnvidia-tls.so` uses the initial-exec
/// TLS model), and glibc's fixed static-TLS surplus is consumed a slice
/// at a time by repeated dlopen cycles when fresh threads are involved.
/// After roughly 15-20 context create/destroy cycles on different
/// threads (e.g. one per test in a test harness), the surplus runs out,
/// the ICD fails to load with "cannot allocate memory in static TLS
/// block", and the GPU silently vanishes from enumeration — device
/// selection then falls back to a CPU driver, or fails outright.
///
/// Pinning a single instance keeps every ICD resident so its TLS block
/// is allocated exactly once.  The pin instance and the cloned entry
/// are leaked deliberately; the OS reclaims them at process exit.
/// `ML_NO_LOADER_PIN=1` disables the pin.
fn pin_loader(entry: &ash::Entry) {
    use std::sync::OnceLock;
    static LOADER_PIN: OnceLock<()> = OnceLock::new();
    if std::env::var("ML_NO_LOADER_PIN").is_ok_and(|v| v == "1") {
        return;
    }
    LOADER_PIN.get_or_init(|| {
        let instance_ci = vk::InstanceCreateInfo::default();
        match unsafe { entry.create_instance(&instance_ci, None) } {
            Ok(_pin) => {
                // `ash::Instance` does not destroy on drop; dropping the
                // wrapper leaks the VkInstance handle, which is exactly
                // what we want.  Keep libvulkan itself loaded too, so
                // the pinned instance can never dangle.
                std::mem::forget(entry.clone());
            }
            Err(err) => {
                log::warn!("tensor-ash: loader pin instance creation failed: {err}");
            }
        }
    });
}

pub(super) fn create(
    enable_validation: bool,
    preference: DevicePreference,
) -> Result<VulkanContext> {
    unsafe {
        let entry =
            ash::Entry::load().map_err(|err| anyhow!("failed to load Vulkan loader: {err}"))?;
        pin_loader(&entry);

        // 1.3 rather than 1.2 so the NV_cooperative_matrix2 shaders
        // (SPIR-V 1.6) are loadable; the 1.1/1.2 feature-struct chains
        // below are version-agnostic and unaffected.  Requesting a
        // higher apiVersion than the loader supports is not an error
        // on Vulkan >= 1.1 loaders.
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"tensor-ash")
            .engine_name(c"tensor-ash")
            .api_version(vk::API_VERSION_1_3);
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
        // `VkPhysicalDeviceDriverProperties` is core in 1.2; on older
        // devices the driver identity stays unknown and driver-scoped
        // quirk handling (workgroup_shared_budget) falls back to the
        // strict spec behavior.
        let driver_id = (device_properties.api_version >= vk::API_VERSION_1_2).then(|| {
            let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver_props);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            driver_props.driver_id
        });
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
        let mut coopmat_query = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features_query = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan11_query)
            .push_next(&mut vulkan12_query)
            .push_next(&mut coopmat_query);
        instance.get_physical_device_features2(physical_device, &mut features_query);
        let buffer_device_address_supported = vulkan12_query.buffer_device_address == vk::TRUE;
        // f16 storage kernels need both halves: `shaderFloat16` for the
        // arithmetic types and `storageBuffer16BitAccess` for 16-bit
        // access through physical-storage-buffer pointers (the SPIR-V
        // capability covers BDA loads too, not just descriptor SSBOs).
        let f16_storage_supported = vulkan12_query.shader_float16 == vk::TRUE
            && vulkan11_query.storage_buffer16_bit_access == vk::TRUE;
        // Cooperative-matrix (tensor core) kernels additionally need
        // the extension + feature and the Vulkan memory model (glslang
        // emits `OpMemoryModel ... Vulkan` for coopmat shaders), and
        // build on the f16 storage path.  `ML_NO_COOPMAT=1` is the
        // kill-switch if a driver misbehaves.
        let coopmat_disabled = std::env::var("ML_NO_COOPMAT")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        // A supported feature bit is not enough for an extension feature: the
        // corresponding extension must also be enabled when creating the device.
        let device_extensions = instance
            .enumerate_device_extension_properties(physical_device)
            .context("enumerate_device_extension_properties")?;
        let coopmat_name = ash::khr::cooperative_matrix::NAME;
        let coopmat_present = device_extensions
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == coopmat_name);
        let enable_coopmat = !coopmat_disabled
            && f16_storage_supported
            && coopmat_present
            && coopmat_query.cooperative_matrix == vk::TRUE
            && vulkan12_query.vulkan_memory_model == vk::TRUE;
        // VK_NV_cooperative_matrix2 (workgroup-scope matrices, tensor
        // addressing, reductions, per-element ops) for the cm2 flash
        // kernels.  Unknown to ash 0.38, so the feature/property
        // structs are hand-rolled below and chained by raw pointer.
        // Gate: coopmat1 gate + Vulkan 1.3 device (the shaders are
        // SPIR-V 1.6) + all seven feature bits (llama.cpp's proven
        // envelope; the f32-KV decode callback needs BlockLoads) + a
        // flexible-dimensions config compatible with the 128-thread
        // Br=Bc=64 f16xf16->f32 kernels.  `ML_NO_COOPMAT2=1` is the
        // kill-switch.
        let coopmat2_disabled = std::env::var("ML_NO_COOPMAT2")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let coopmat2_present = device_extensions
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == COOPMAT2_EXTENSION_NAME);
        let mut enable_coopmat2 = false;
        if !coopmat2_disabled
            && enable_coopmat
            && coopmat2_present
            && device_properties.api_version >= vk::API_VERSION_1_3
        {
            let mut coopmat2_query = PhysicalDeviceCooperativeMatrix2FeaturesNV::default();
            let mut features2_query = vk::PhysicalDeviceFeatures2 {
                p_next: (&raw mut coopmat2_query).cast::<c_void>(),
                ..Default::default()
            };
            instance.get_physical_device_features2(physical_device, &mut features2_query);
            let features_ok = [
                coopmat2_query.cooperative_matrix_workgroup_scope,
                coopmat2_query.cooperative_matrix_flexible_dimensions,
                coopmat2_query.cooperative_matrix_reductions,
                coopmat2_query.cooperative_matrix_conversions,
                coopmat2_query.cooperative_matrix_per_element_operations,
                coopmat2_query.cooperative_matrix_tensor_addressing,
                coopmat2_query.cooperative_matrix_block_loads,
            ]
            .iter()
            .all(|&bit| bit == vk::TRUE);

            let mut coopmat2_props = PhysicalDeviceCooperativeMatrix2PropertiesNV::default();
            let mut props2_query = vk::PhysicalDeviceProperties2 {
                p_next: (&raw mut coopmat2_props).cast::<c_void>(),
                ..Default::default()
            };
            instance.get_physical_device_properties2(physical_device, &mut props2_query);
            // The flash kernels declare dimensions up to 128 (dh128).
            let dims_ok =
                coopmat2_props.cooperative_matrix_flexible_dimensions_max_dimension >= 128;

            let config_ok = coopmat2_flash_config_supported(&entry, &instance, physical_device);
            enable_coopmat2 = features_ok && dims_ok && config_ok;
            log::info!(
                "tensor-ash: VK_NV_cooperative_matrix2: features_ok={features_ok} \
                 (ws={} fd={} red={} conv={} pe={} ta={} bl={}), max_dim={}, \
                 reserved_shmem={}, flash_config_ok={config_ok} -> enabled={enable_coopmat2}",
                coopmat2_query.cooperative_matrix_workgroup_scope,
                coopmat2_query.cooperative_matrix_flexible_dimensions,
                coopmat2_query.cooperative_matrix_reductions,
                coopmat2_query.cooperative_matrix_conversions,
                coopmat2_query.cooperative_matrix_per_element_operations,
                coopmat2_query.cooperative_matrix_tensor_addressing,
                coopmat2_query.cooperative_matrix_block_loads,
                coopmat2_props.cooperative_matrix_flexible_dimensions_max_dimension,
                coopmat2_props.cooperative_matrix_workgroup_scope_reserved_shared_memory,
            );
        }

        let mut enabled_device_extensions = Vec::new();
        if enable_coopmat {
            enabled_device_extensions.push(coopmat_name.as_ptr());
        }
        if enable_coopmat2 {
            enabled_device_extensions.push(COOPMAT2_EXTENSION_NAME.as_ptr());
        }

        let priorities = [1.0];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default();
        let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f16_storage_supported);
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(buffer_device_address_supported)
            .shader_float16(f16_storage_supported)
            .vulkan_memory_model(enable_coopmat);
        let mut coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(enable_coopmat);
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
        if enable_coopmat {
            device_ci = device_ci.push_next(&mut coopmat);
        }
        // ash's typed `push_next` cannot chain a struct it does not
        // know, so the coopmat2 features link in by raw pointer (same
        // p_next contract, hand-managed).
        let mut coopmat2_features = PhysicalDeviceCooperativeMatrix2FeaturesNV {
            cooperative_matrix_workgroup_scope: vk::TRUE,
            cooperative_matrix_flexible_dimensions: vk::TRUE,
            cooperative_matrix_reductions: vk::TRUE,
            cooperative_matrix_conversions: vk::TRUE,
            cooperative_matrix_per_element_operations: vk::TRUE,
            cooperative_matrix_tensor_addressing: vk::TRUE,
            cooperative_matrix_block_loads: vk::TRUE,
            ..Default::default()
        };
        if enable_coopmat2 {
            coopmat2_features.p_next = device_ci.p_next.cast_mut();
            device_ci.p_next = (&raw const coopmat2_features).cast::<c_void>();
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
            driver_id,
            memory_properties,
            compute_family,
            queue: Mutex::new(queue),
            timestamp_period_ns: device_properties.limits.timestamp_period as f64,
            timestamp_valid_bits,
            timestamps_supported,
            buffer_device_address_enabled: buffer_device_address_supported,
            f16_storage_enabled: f16_storage_supported,
            coopmat_enabled: enable_coopmat,
            coopmat2_enabled: enable_coopmat2,
            pipeline_cache,
            pipeline_cache_path,
            debug_loader,
            debug_messenger,
        })
    }
}

// ---------------------------------------------------------------------------
// VK_NV_cooperative_matrix2 (extension #594) definitions.
//
// ash 0.38 is generated from Vulkan-Headers 1.3.281; this extension
// entered the headers at 1.3.300, so nothing below exists in `ash::vk`.
// Layouts hand-rolled to match `vulkan_core.h` (sTypes
// 1000593000/1000593001/1000593002 from the extension's block).

const COOPMAT2_EXTENSION_NAME: &CStr = c"VK_NV_cooperative_matrix2";

/// `VkPhysicalDeviceCooperativeMatrix2FeaturesNV`. Exactly seven
/// feature bits — this is the full struct.
#[repr(C)]
struct PhysicalDeviceCooperativeMatrix2FeaturesNV {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    cooperative_matrix_workgroup_scope: vk::Bool32,
    cooperative_matrix_flexible_dimensions: vk::Bool32,
    cooperative_matrix_reductions: vk::Bool32,
    cooperative_matrix_conversions: vk::Bool32,
    cooperative_matrix_per_element_operations: vk::Bool32,
    cooperative_matrix_tensor_addressing: vk::Bool32,
    cooperative_matrix_block_loads: vk::Bool32,
}

impl Default for PhysicalDeviceCooperativeMatrix2FeaturesNV {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1000593000),
            p_next: std::ptr::null_mut(),
            cooperative_matrix_workgroup_scope: vk::FALSE,
            cooperative_matrix_flexible_dimensions: vk::FALSE,
            cooperative_matrix_reductions: vk::FALSE,
            cooperative_matrix_conversions: vk::FALSE,
            cooperative_matrix_per_element_operations: vk::FALSE,
            cooperative_matrix_tensor_addressing: vk::FALSE,
            cooperative_matrix_block_loads: vk::FALSE,
        }
    }
}

/// `VkPhysicalDeviceCooperativeMatrix2PropertiesNV`.
#[repr(C)]
struct PhysicalDeviceCooperativeMatrix2PropertiesNV {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    cooperative_matrix_workgroup_scope_max_workgroup_size: u32,
    cooperative_matrix_flexible_dimensions_max_dimension: u32,
    /// Subtract from the per-workgroup shared-memory budget: the
    /// driver reserves this much for compiler-managed matrix staging.
    cooperative_matrix_workgroup_scope_reserved_shared_memory: u32,
}

impl Default for PhysicalDeviceCooperativeMatrix2PropertiesNV {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1000593002),
            p_next: std::ptr::null_mut(),
            cooperative_matrix_workgroup_scope_max_workgroup_size: 0,
            cooperative_matrix_flexible_dimensions_max_dimension: 0,
            cooperative_matrix_workgroup_scope_reserved_shared_memory: 0,
        }
    }
}

/// `VkCooperativeMatrixFlexibleDimensionsPropertiesNV`.
#[derive(Clone)]
#[repr(C)]
struct CooperativeMatrixFlexibleDimensionsPropertiesNV {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    m_granularity: u32,
    n_granularity: u32,
    k_granularity: u32,
    a_type: vk::ComponentTypeKHR,
    b_type: vk::ComponentTypeKHR,
    c_type: vk::ComponentTypeKHR,
    result_type: vk::ComponentTypeKHR,
    saturating_accumulation: vk::Bool32,
    scope: vk::ScopeKHR,
    workgroup_invocations: u32,
}

impl Default for CooperativeMatrixFlexibleDimensionsPropertiesNV {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1000593001),
            p_next: std::ptr::null_mut(),
            m_granularity: 0,
            n_granularity: 0,
            k_granularity: 0,
            a_type: vk::ComponentTypeKHR::FLOAT16,
            b_type: vk::ComponentTypeKHR::FLOAT16,
            c_type: vk::ComponentTypeKHR::FLOAT16,
            result_type: vk::ComponentTypeKHR::FLOAT16,
            saturating_accumulation: vk::FALSE,
            scope: vk::ScopeKHR::WORKGROUP,
            workgroup_invocations: 0,
        }
    }
}

type PfnGetFlexibleDimensionsProperties = unsafe extern "system" fn(
    vk::PhysicalDevice,
    *mut u32,
    *mut CooperativeMatrixFlexibleDimensionsPropertiesNV,
) -> vk::Result;

/// Whether a workgroup-scope flexible-dimensions config exists that
/// the cm2 flash kernels can use: f16 A/B with f32 accumulate at 128
/// invocations, granularities dividing the Br=Bc=64 tiles (dh 64/128
/// are multiples of 64's divisors too).  NVIDIA reports 32x16x16@128
/// on RTX, which passes.
unsafe fn coopmat2_flash_config_supported(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    unsafe {
        let Some(raw) = (entry.static_fn().get_instance_proc_addr)(
            instance.handle(),
            c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr(),
        ) else {
            log::warn!(
                "tensor-ash: vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV \
                 missing despite VK_NV_cooperative_matrix2"
            );
            return false;
        };
        let get_props: PfnGetFlexibleDimensionsProperties = std::mem::transmute(raw);

        let mut count = 0u32;
        if get_props(physical_device, &mut count, std::ptr::null_mut()) != vk::Result::SUCCESS {
            return false;
        }
        // Output structs must carry their sType in (two-call pattern).
        let mut configs =
            vec![CooperativeMatrixFlexibleDimensionsPropertiesNV::default(); count as usize];
        let result = get_props(physical_device, &mut count, configs.as_mut_ptr());
        if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
            return false;
        }
        configs.truncate(count as usize);
        configs.iter().any(|config| {
            config.scope == vk::ScopeKHR::WORKGROUP
                && config.workgroup_invocations == 128
                && config.a_type == vk::ComponentTypeKHR::FLOAT16
                && config.b_type == vk::ComponentTypeKHR::FLOAT16
                && config.c_type == vk::ComponentTypeKHR::FLOAT32
                && config.result_type == vk::ComponentTypeKHR::FLOAT32
                && config.saturating_accumulation == vk::FALSE
                && [
                    config.m_granularity,
                    config.n_granularity,
                    config.k_granularity,
                ]
                .iter()
                .all(|&granularity| granularity != 0 && 64 % granularity == 0)
        })
    }
}
