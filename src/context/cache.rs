use std::path::{Path, PathBuf};

use ash::vk;

use super::DeviceSummary;

/// Per-device location for the persistent pipeline cache.  Uses
/// `$XDG_CACHE_HOME/tensor-ash/` (or `$HOME/.cache/tensor-ash/`) and a
/// vendor/device-id-qualified filename so caches from different GPUs on
/// the same host don't stomp on each other.
pub(super) fn pipeline_cache_path_for(summary: &DeviceSummary) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let filename = format!(
        "pipeline_cache_v{:04x}_{:04x}.bin",
        summary.vendor_id, summary.device_id
    );
    Some(base.join("tensor-ash").join(filename))
}

pub(super) fn create_pipeline_cache(
    device: &ash::Device,
    path: Option<&Path>,
) -> vk::PipelineCache {
    let cached = path
        .and_then(|path| std::fs::read(path).ok())
        .unwrap_or_default();
    let create = |data: &[u8]| unsafe {
        device.create_pipeline_cache(
            &vk::PipelineCacheCreateInfo::default().initial_data(data),
            None,
        )
    };

    match create(&cached) {
        Ok(cache) => cache,
        Err(err) if !cached.is_empty() => {
            log::warn!("tensor-ash: ignoring invalid pipeline cache ({err})");
            create(&[]).unwrap_or_else(|retry_err| {
                log::warn!(
                    "tensor-ash: create_pipeline_cache failed ({retry_err}); proceeding without persistence"
                );
                vk::PipelineCache::null()
            })
        }
        Err(err) => {
            log::warn!(
                "tensor-ash: create_pipeline_cache failed ({err}); proceeding without persistence"
            );
            vk::PipelineCache::null()
        }
    }
}

pub(super) fn persist_pipeline_cache(
    device: &ash::Device,
    cache: vk::PipelineCache,
    path: Option<&Path>,
) {
    let Some(path) = path else { return };
    let Ok(data) = (unsafe { device.get_pipeline_cache_data(cache) }) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(err) = std::fs::write(path, data) {
        log::warn!(
            "tensor-ash: failed to write pipeline cache to {}: {err}",
            path.display()
        );
    }
}
