//! Measured kernel selection with a persistent per-device store.
//!
//! The static heuristic in `selection.rs` is the prior; when tuning is
//! enabled (`ML_TUNE=1`) the executor measures every eligible kernel on
//! the real shape and records the winner here.  Winners persist across
//! processes in `$XDG_CACHE_HOME/tensor-ash/`, keyed by GPU
//! vendor/device id in the filename and validated against the driver
//! version + a hash of every embedded SPIR-V binary in the header, so a
//! driver update or kernel rebuild invalidates stale entries instead of
//! silently serving them.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::context::VulkanContext;

use super::catalog::KERNEL_SPECS;

const SPLITK2_SPIRV: [&[u8]; 3] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_m128n128.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_m64n64.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_f32_splitk2_reduce.spv")),
];
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// One GEMM problem shape.  `accumulate` / `alpha` / epilogues change
/// only the store path, which never flips the kernel ranking, so they
/// are deliberately not part of the key.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct TuneKey {
    pub batch: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// f16-storage B routes tune within the `f16w_*` candidate set;
    /// the same (B,M,N,K) shape may hold a different winner per
    /// storage type.
    pub b_f16: bool,
}

/// Measured winner for one shape.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TuneEntry {
    /// Best data-parallel kernel (index into `KERNEL_SPECS`).  Used by
    /// every dispatch of this shape, including batched submissions,
    /// accumulate, and epilogue ops.
    pub kernel: usize,
    /// When set, single plain (non-accumulate, no-epilogue) calls
    /// route through the two-stage split-K path with this split count
    /// instead — it measured faster than every DP kernel.
    pub splitk2_splits: Option<u32>,
}

/// FNV-1a over every embedded kernel binary.  Any shader edit or
/// registry reorder changes this, invalidating persisted winners.
pub(super) fn shader_registry_hash() -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for spec in KERNEL_SPECS {
        for &b in spec.spv {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    for spv in SPLITK2_SPIRV {
        for &b in spv {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

pub(super) fn tune_store_path(ctx: &VulkanContext) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let filename = format!(
        "tuned_kernels_v{:04x}_{:04x}.txt",
        ctx.device_summary.vendor_id, ctx.device_summary.device_id
    );
    Some(base.join("tensor-ash").join(filename))
}

fn header_line(ctx: &VulkanContext, shader_hash: u64) -> String {
    format!(
        "tensor-ash-tune-v3 driver={} shaders={shader_hash:016x}",
        ctx.device_summary.driver_version
    )
}

/// Load persisted winners.  Returns an empty map when the file is
/// missing, malformed, or was written for a different driver / shader
/// build.
pub(super) fn load_tuned(ctx: &VulkanContext, shader_hash: u64) -> HashMap<TuneKey, TuneEntry> {
    let mut map = HashMap::new();
    let Some(path) = tune_store_path(ctx) else {
        return map;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return map;
    };
    let mut lines = text.lines();
    if lines.next() != Some(header_line(ctx, shader_hash).as_str()) {
        log::info!(
            "tensor-ash: discarding stale tuning store {} (driver or shader build changed)",
            path.display()
        );
        return map;
    }
    for line in lines {
        let mut it = line.split_whitespace();
        let (Some(b), Some(m), Some(n), Some(k), Some(name)) =
            (it.next(), it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (Ok(batch), Ok(m), Ok(n), Ok(k)) = (b.parse(), m.parse(), n.parse(), k.parse()) else {
            continue;
        };
        let Some(idx) = KERNEL_SPECS.iter().position(|s| s.name == name) else {
            continue;
        };
        let mut splitk2_splits = None;
        let mut b_f16 = false;
        for tok in it {
            if let Some(splits) = tok.strip_prefix("splitk2=") {
                splitk2_splits = splits
                    .parse::<u32>()
                    .ok()
                    .filter(|s| (2..=0xFFFF).contains(s));
            } else if tok == "bf16" {
                b_f16 = true;
            }
        }
        map.insert(
            TuneKey {
                batch,
                m,
                n,
                k,
                b_f16,
            },
            TuneEntry {
                kernel: idx,
                splitk2_splits,
            },
        );
    }
    map
}

/// Rewrite the whole store (it is small: one line per tuned shape).
pub(super) fn save_tuned(ctx: &VulkanContext, shader_hash: u64, map: &HashMap<TuneKey, TuneEntry>) {
    let Some(path) = tune_store_path(ctx) else {
        return;
    };
    let write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = String::new();
        out.push_str(&header_line(ctx, shader_hash));
        out.push('\n');
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(key, _)| (key.batch, key.m, key.n, key.k, key.b_f16));
        for (key, entry) in entries {
            out.push_str(&format!(
                "{} {} {} {} {}",
                key.batch, key.m, key.n, key.k, KERNEL_SPECS[entry.kernel].name
            ));
            if let Some(splits) = entry.splitk2_splits {
                out.push_str(&format!(" splitk2={splits}"));
            }
            if key.b_f16 {
                out.push_str(" bf16");
            }
            out.push('\n');
        }
        let temporary = path.with_extension(format!(
            "tmp.{}.{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut f = std::fs::File::create(&temporary)?;
        f.write_all(out.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(temporary, &path)
    };
    if let Err(err) = write() {
        log::warn!(
            "tensor-ash: failed to persist tuning store {}: {err}",
            path.display()
        );
    }
}
