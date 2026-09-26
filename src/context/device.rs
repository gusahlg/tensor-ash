use std::ffi::CStr;

use anyhow::{Context, Result, anyhow, bail};
use ash::vk;

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

/// Where [`ShaderCores::count`] came from.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ShaderCoreSource {
    /// `VkPhysicalDeviceShaderSMBuiltinsPropertiesNV::shaderSMCount`
    /// (`VK_NV_shader_sm_builtins`): streaming multiprocessors.
    NvSmBuiltins,
    /// `VkPhysicalDeviceShaderCoreProperties2AMD::activeComputeUnitCount`
    /// (`VK_AMD_shader_core_properties2`): active compute units.
    AmdActiveComputeUnits,
    /// `shaderEngineCount * shaderArraysPerEngineCount *
    /// computeUnitsPerShaderArray` (`VK_AMD_shader_core_properties`):
    /// the physical CU count, an upper bound on the active count.
    AmdPhysicalComputeUnits,
    /// `VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM::shaderCoreCount`
    /// (`VK_ARM_shader_core_builtins`).
    ArmShaderCores,
    /// `ML_SM_COUNT=N` override (experiments and cross-GPU what-ifs).
    EnvOverride,
}

impl ShaderCoreSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NvSmBuiltins => "VK_NV_shader_sm_builtins",
            Self::AmdActiveComputeUnits => "VK_AMD_shader_core_properties2",
            Self::AmdPhysicalComputeUnits => "VK_AMD_shader_core_properties",
            Self::ArmShaderCores => "VK_ARM_shader_core_builtins",
            Self::EnvOverride => "ML_SM_COUNT",
        }
    }
}

/// The device's count of independently scheduled shader cores (NVIDIA
/// SMs, AMD CUs, ARM shader cores) — the unit a workgroup is resident
/// on, and therefore the denominator of wave quantization.  Routing
/// uses it to size tile grids (see `pipeline::routing::coopmat_tile`);
/// devices exposing none of the vendor queries report `None` and
/// routing keeps its SM-agnostic rules.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ShaderCores {
    pub count: u32,
    /// Maximum resident warps (waves) per core when the driver reports
    /// it (`shaderWarpsPerSM` / `shaderWarpsPerCore`).
    pub warps_per_core: Option<u32>,
    pub source: ShaderCoreSource,
}

/// Parse an `ML_SM_COUNT` value: `Ok(Some(Some(n)))` pins `n` cores,
/// `Ok(Some(None))` (`0` / `none`) forces the unknown-count fallback,
/// `Ok(None)` (unset / empty / `auto`) keeps the driver's answer.
pub(super) fn parse_sm_count_override(raw: Option<&str>) -> Result<Option<Option<u32>>> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(None);
    };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    if raw.eq_ignore_ascii_case("none") {
        return Ok(Some(None));
    }
    let count = raw
        .parse::<u32>()
        .with_context(|| format!("invalid ML_SM_COUNT '{raw}' (expected N, 0/none, or auto)"))?;
    Ok(Some((count > 0).then_some(count)))
}

/// Query the shader-core count through whichever vendor extension the
/// device exposes.  A supported extension's properties struct may be
/// chained into `vkGetPhysicalDeviceProperties2` without enabling the
/// extension on the logical device, so this costs nothing at device
/// creation.  Pre-1.1 devices (no properties2) report `None`.
pub(super) fn query_shader_cores(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    api_version: u32,
    device_extensions: &[vk::ExtensionProperties],
) -> Option<ShaderCores> {
    if api_version < vk::API_VERSION_1_1 {
        return None;
    }
    let has = |name: &CStr| {
        device_extensions
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) } == name)
    };
    // SAFETY: each chained struct outlives its properties2 call and is
    // the matching sType for an extension the device advertises.
    unsafe {
        if has(ash::nv::shader_sm_builtins::NAME) {
            let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            if sm.shader_sm_count > 0 {
                return Some(ShaderCores {
                    count: sm.shader_sm_count,
                    warps_per_core: (sm.shader_warps_per_sm > 0).then_some(sm.shader_warps_per_sm),
                    source: ShaderCoreSource::NvSmBuiltins,
                });
            }
        }
        if has(ash::amd::shader_core_properties2::NAME) {
            let mut core2 = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut core2);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            if core2.active_compute_unit_count > 0 {
                return Some(ShaderCores {
                    count: core2.active_compute_unit_count,
                    warps_per_core: None,
                    source: ShaderCoreSource::AmdActiveComputeUnits,
                });
            }
        }
        if has(ash::amd::shader_core_properties::NAME) {
            let mut core = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut core);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            let count = core
                .shader_engine_count
                .saturating_mul(core.shader_arrays_per_engine_count)
                .saturating_mul(core.compute_units_per_shader_array);
            if count > 0 {
                return Some(ShaderCores {
                    count,
                    warps_per_core: None,
                    source: ShaderCoreSource::AmdPhysicalComputeUnits,
                });
            }
        }
        if has(ash::arm::shader_core_builtins::NAME) {
            let mut arm = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut arm);
            instance.get_physical_device_properties2(physical_device, &mut props2);
            if arm.shader_core_count > 0 {
                return Some(ShaderCores {
                    count: arm.shader_core_count,
                    warps_per_core: (arm.shader_warps_per_core > 0)
                        .then_some(arm.shader_warps_per_core),
                    source: ShaderCoreSource::ArmShaderCores,
                });
            }
        }
    }
    None
}

pub(super) fn device_summaries(
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

pub(super) fn select_physical_device(
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
        assert!(DevicePreference::parse("name:   ").is_err());
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
    fn parses_sm_count_override() {
        assert_eq!(parse_sm_count_override(None).unwrap(), None);
        assert_eq!(parse_sm_count_override(Some("")).unwrap(), None);
        assert_eq!(parse_sm_count_override(Some(" AUTO ")).unwrap(), None);
        assert_eq!(parse_sm_count_override(Some("24")).unwrap(), Some(Some(24)));
        assert_eq!(
            parse_sm_count_override(Some(" 46 ")).unwrap(),
            Some(Some(46))
        );
        assert_eq!(parse_sm_count_override(Some("0")).unwrap(), Some(None));
        assert_eq!(parse_sm_count_override(Some("none")).unwrap(), Some(None));
        assert!(parse_sm_count_override(Some("-3")).is_err());
        assert!(parse_sm_count_override(Some("many")).is_err());
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
