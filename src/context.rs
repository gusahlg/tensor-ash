//! Vulkan instance, physical device, logical device, and the one
//! compute queue we'll submit to.
//!
//! Deliberately minimal — most matmul-heavy workloads don't benefit
//! from multiple compute queues on the same family (they time-multiplex
//! the same SMs).  Async transfer is a separate concern that can be
//! layered on top if needed.

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;
use parking_lot::Mutex;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DevicePreference {
    Auto,
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Index(usize),
    NameContains(String),
}

impl DevicePreference {
    pub fn parse(raw: &str) -> Result<Self> {
        let raw = raw.trim();
        let raw_lc = raw.to_ascii_lowercase();
        if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if raw.eq_ignore_ascii_case("discrete") {
            return Ok(Self::Discrete);
        }
        if raw.eq_ignore_ascii_case("integrated") {
            return Ok(Self::Integrated);
        }
        if raw.eq_ignore_ascii_case("virtual") {
            return Ok(Self::Virtual);
        }
        if raw.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(index) = raw_lc.strip_prefix("index:") {
            let index = index
                .parse::<usize>()
                .with_context(|| format!("invalid ML_DEVICE index: {index}"))?;
            return Ok(Self::Index(index));
        }
        if raw_lc.starts_with("name:") {
            let name = &raw[5..];
            let name = name.trim();
            if name.is_empty() {
                bail!("ML_DEVICE name filter must not be empty");
            }
            return Ok(Self::NameContains(name.to_ascii_lowercase()));
        }
        if let Ok(index) = raw.parse::<usize>() {
            return Ok(Self::Index(index));
        }
        Ok(Self::NameContains(raw.to_ascii_lowercase()))
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Discrete => "discrete".into(),
            Self::Integrated => "integrated".into(),
            Self::Virtual => "virtual".into(),
            Self::Cpu => "cpu".into(),
            Self::Index(index) => format!("index:{index}"),
            Self::NameContains(name) => format!("name contains '{name}'"),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DeviceKind {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DiscreteGpu => "discrete",
            Self::IntegratedGpu => "integrated",
            Self::VirtualGpu => "virtual",
            Self::Cpu => "cpu",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceSummary {
    pub index: usize,
    pub name: String,
    pub kind: DeviceKind,
    pub api_version: (u32, u32, u32),
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
}

impl DeviceSummary {
    fn from_properties(index: usize, props: vk::PhysicalDeviceProperties) -> Self {
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Self {
            index,
            name,
            kind: device_kind(props.device_type),
            api_version: (
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version),
            ),
            driver_version: props.driver_version,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
        }
    }

    pub fn api_version_string(&self) -> String {
        format!(
            "{}.{}.{}",
            self.api_version.0, self.api_version.1, self.api_version.2
        )
    }
}

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

            // ---- Instance ------------------------------------------------------
            let app_name = CString::new("ml_project").unwrap();
            let engine_name = CString::new("ml_project").unwrap();
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
                let msg = loader
                    .create_debug_utils_messenger(&ci, None)
                    .context("create_debug_utils_messenger")?;
                (Some(loader), Some(msg))
            } else {
                (None, None)
            };

            // ---- Pick physical device -----------------------------------------
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
                "ml_project: using device #{}: {} ({}, Vulkan {})",
                device_summary.index,
                device_summary.name,
                device_summary.kind.as_str(),
                device_summary.api_version_string(),
            );

            // ---- Pick compute queue family ------------------------------------
            let qf_props = instance.get_physical_device_queue_family_properties(physical_device);
            let compute_family = qf_props
                .iter()
                .enumerate()
                .filter(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                // Prefer a dedicated compute family (no GRAPHICS bit) when available.
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

            let features = vk::PhysicalDeviceFeatures::default();
            let device_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_ci)
                .enabled_features(&features);
            let device = instance
                .create_device(physical_device, &device_ci, None)
                .context("create_device")?;
            let queue = device.get_device_queue(compute_family, 0);

            // ---- Persistent pipeline cache -----------------------------------
            // Seeded from disk; written back on Drop.  If the on-disk cache
            // was produced by a different driver/device the loader silently
            // ignores it and we get an empty cache (no functional impact).
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
                        "ml_project: create_pipeline_cache failed ({err}); proceeding without persistence",
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
                            "ml_project: failed to write pipeline cache to {}: {err}",
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

/// Per-device location for the persistent pipeline cache.  Uses
/// `$XDG_CACHE_HOME/ml_project/` (or `$HOME/.cache/ml_project/`) and a
/// vendor/device-id-qualified filename so caches from different GPUs on
/// the same host don't stomp on each other.
fn pipeline_cache_path_for(summary: &DeviceSummary) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let filename = format!(
        "pipeline_cache_v{:04x}_{:04x}.bin",
        summary.vendor_id, summary.device_id
    );
    Some(base.join("ml_project").join(filename))
}

fn device_summaries(
    instance: &ash::Instance,
    devices: &[vk::PhysicalDevice],
) -> Vec<DeviceSummary> {
    devices
        .iter()
        .enumerate()
        .map(|(index, &pd)| {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            DeviceSummary::from_properties(index, props)
        })
        .collect()
}

fn select_physical_device(
    devices: &[DeviceSummary],
    preference: &DevicePreference,
) -> Result<usize> {
    if devices.is_empty() {
        bail!("no Vulkan-capable physical devices found");
    }
    let selected = match preference {
        DevicePreference::Auto => auto_select_device(devices),
        DevicePreference::Discrete => devices.iter().find(|d| d.kind == DeviceKind::DiscreteGpu),
        DevicePreference::Integrated => {
            devices.iter().find(|d| d.kind == DeviceKind::IntegratedGpu)
        }
        DevicePreference::Virtual => devices.iter().find(|d| d.kind == DeviceKind::VirtualGpu),
        DevicePreference::Cpu => devices.iter().find(|d| d.kind == DeviceKind::Cpu),
        DevicePreference::Index(index) => devices.iter().find(|d| d.index == *index),
        DevicePreference::NameContains(needle) => devices
            .iter()
            .find(|d| d.name.to_ascii_lowercase().contains(needle)),
    };
    selected.map(|d| d.index).ok_or_else(|| {
        anyhow!(
            "no Vulkan device matches ML_DEVICE={} (available: {})",
            preference.describe(),
            describe_available_devices(devices),
        )
    })
}

fn describe_available_devices(devices: &[DeviceSummary]) -> String {
    devices
        .iter()
        .map(|d| format!("#{} {} ({})", d.index, d.name, d.kind.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn device_kind(kind: vk::PhysicalDeviceType) -> DeviceKind {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::DiscreteGpu,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::IntegratedGpu,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::VirtualGpu,
        vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
        _ => DeviceKind::Other,
    }
}

fn auto_select_device(devices: &[DeviceSummary]) -> Option<&DeviceSummary> {
    for kind in [
        DeviceKind::DiscreteGpu,
        DeviceKind::IntegratedGpu,
        DeviceKind::VirtualGpu,
        DeviceKind::Other,
        DeviceKind::Cpu,
    ] {
        if let Some(device) = devices.iter().find(|d| d.kind == kind) {
            return Some(device);
        }
    }
    None
}

unsafe extern "system" fn debug_callback(
    _sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _typ: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    let msg_ptr = unsafe { (*data).p_message };
    if !msg_ptr.is_null() {
        let msg = unsafe { CStr::from_ptr(msg_ptr) }.to_string_lossy();
        eprintln!("[vulkan] {msg}");
    }
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_preferences() {
        assert_eq!(DevicePreference::parse("").unwrap(), DevicePreference::Auto);
        assert_eq!(
            DevicePreference::parse("auto").unwrap(),
            DevicePreference::Auto
        );
        assert_eq!(
            DevicePreference::parse("discrete").unwrap(),
            DevicePreference::Discrete
        );
        assert_eq!(
            DevicePreference::parse("integrated").unwrap(),
            DevicePreference::Integrated
        );
        assert_eq!(
            DevicePreference::parse("virtual").unwrap(),
            DevicePreference::Virtual
        );
        assert_eq!(
            DevicePreference::parse("cpu").unwrap(),
            DevicePreference::Cpu
        );
        assert_eq!(
            DevicePreference::parse("index:2").unwrap(),
            DevicePreference::Index(2)
        );
        assert_eq!(
            DevicePreference::parse("INDEX:4").unwrap(),
            DevicePreference::Index(4)
        );
        assert_eq!(
            DevicePreference::parse("3").unwrap(),
            DevicePreference::Index(3)
        );
        assert_eq!(
            DevicePreference::parse("name:RTX 3070").unwrap(),
            DevicePreference::NameContains("rtx 3070".into())
        );
        assert_eq!(
            DevicePreference::parse("NAME:RTX 4090").unwrap(),
            DevicePreference::NameContains("rtx 4090".into())
        );
        assert_eq!(
            DevicePreference::parse("llvmpipe").unwrap(),
            DevicePreference::NameContains("llvmpipe".into())
        );
    }

    #[test]
    fn selects_requested_device_kind() {
        let devices = vec![
            DeviceSummary {
                index: 0,
                name: "llvmpipe".into(),
                kind: DeviceKind::Cpu,
                api_version: (1, 3, 0),
                driver_version: 1,
                vendor_id: 1,
                device_id: 1,
            },
            DeviceSummary {
                index: 1,
                name: "NVIDIA GeForce RTX 3070".into(),
                kind: DeviceKind::DiscreteGpu,
                api_version: (1, 4, 0),
                driver_version: 2,
                vendor_id: 0x10de,
                device_id: 0x2488,
            },
        ];

        assert_eq!(
            select_physical_device(&devices, &DevicePreference::Auto).unwrap(),
            1
        );
        assert_eq!(
            select_physical_device(&devices, &DevicePreference::Cpu).unwrap(),
            0
        );
        assert_eq!(
            select_physical_device(&devices, &DevicePreference::Index(1)).unwrap(),
            1
        );
        assert_eq!(
            select_physical_device(&devices, &DevicePreference::NameContains("rtx".into()))
                .unwrap(),
            1
        );
    }

    #[test]
    fn reports_available_devices_when_selection_fails() {
        let devices = vec![DeviceSummary {
            index: 0,
            name: "llvmpipe".into(),
            kind: DeviceKind::Cpu,
            api_version: (1, 3, 0),
            driver_version: 1,
            vendor_id: 1,
            device_id: 1,
        }];

        let err = select_physical_device(&devices, &DevicePreference::Discrete)
            .unwrap_err()
            .to_string();

        assert!(err.contains("ML_DEVICE=discrete"));
        assert!(err.contains("llvmpipe"));
    }
}
