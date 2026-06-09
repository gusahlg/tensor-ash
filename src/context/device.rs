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
