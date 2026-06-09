use std::path::PathBuf;

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
