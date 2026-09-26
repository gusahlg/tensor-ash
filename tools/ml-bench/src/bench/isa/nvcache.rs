//! NVIDIA `VkPipelineCache` blob -> per-pipeline `NVuc` ISA containers.
//!
//! Layout reverse-engineered on driver 595.x (GA104 and AD107 agree);
//! every field below was checked against `nvdisasm`'s own reading of
//! the container (`nvdisasm -b SM86 <container>` prints the same
//! section table, register count and code size).
//!
//! ```text
//! cache blob
//!   0x00  VkPipelineCacheHeaderVersionOne (32 B: size, version,
//!         vendorID, deviceID, pipelineCacheUUID[16])
//!   0x20  u32 entry_count
//!   0x24  entry[entry_count]:
//!           key[16] | u32 len | "CPKV" | u32 0 | key[16] | u64 hash
//!           | u32 len | u32 flags(0) | u32 raw_size | zstd frame (len-4 B)
//!                                                  -> 60 + len bytes
//! zstd frame -> payload (raw_size bytes) holding one or more
//!   "NVDANVVM" + NVuc container (+ pipeline-level records we do not
//!   decode: workgroup size, knob hash dump, the SPIR-V itself)
//! NVuc container (offsets below are relative to the "NVuc" magic)
//!   0x00  "NVuc"   0x04 u32 version (1)
//!   0x08  u16 section_count, u16 x5 + u32 (arch / flags, undecoded)
//!   0x18  u64 container_size (bytes, from the magic)
//!   0x20  section header[section_count], 32 B each:
//!           u32 type | u32 size | u64 offset | u32 v0 | v1 | v2 | v3
//! section types (names as nvdisasm prints them):
//!   0x01 UCODE (raw SASS), 0x03 REG (u32 count, u32 requested),
//!   0x18 BARRIER, 0x21 STRING_TABLE, 0x37 DEBUG_FRAME_TABLE,
//!   0x39 SYMBOL_TABLE, 0x3a DEBUG_FUNCTION_MAP, 0x45 PROGRAM_INFO,
//!   0x1204 CTAID_INFO; others are printed as unknown.
//! ```

use std::io::Read;

use anyhow::{Context, Result, bail, ensure};

const VK_HEADER_BYTES: usize = 32;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];
const NVDA_MAGIC: &[u8; 8] = b"NVDANVVM";
const NVUC_MAGIC: &[u8; 4] = b"NVuc";

/// One cache entry after decompression.  Some entries' zstd frames
/// reference a dictionary the driver keeps internally (seen in fresh
/// single-pipeline caches on AD107: a second, small entry next to the
/// ISA entry); those cannot be decoded here and carry `Err(reason)`.
pub(crate) struct CacheEntry {
    pub(crate) payload: Result<Vec<u8>, String>,
    /// Stored (possibly compressed) bytes and the declared raw size.
    pub(crate) stored_bytes: usize,
    pub(crate) raw_bytes: usize,
}

impl CacheEntry {
    /// NVuc containers in a decodable payload (none otherwise).
    pub(crate) fn containers(&self) -> Result<Vec<NvucContainer>> {
        match &self.payload {
            Ok(p) => find_containers(p),
            Err(_) => Ok(Vec::new()),
        }
    }
}

/// PCI device id from the Vulkan cache header.
pub(crate) fn cache_device_id(blob: &[u8]) -> Option<u32> {
    u32_at(blob, 12).ok()
}

/// Parse a NVIDIA pipeline-cache blob into its decompressed entries.
pub(crate) fn parse_cache_blob(blob: &[u8]) -> Result<Vec<CacheEntry>> {
    ensure!(
        blob.len() >= VK_HEADER_BYTES + 4,
        "pipeline cache blob too short ({} B)",
        blob.len()
    );
    let vendor = u32_at(blob, 8)?;
    ensure!(
        vendor == 0x10de,
        "pipeline cache vendor 0x{vendor:04x} is not NVIDIA (0x10de); \
         only the NVIDIA proprietary cache layout is decoded"
    );
    let count = u32_at(blob, VK_HEADER_BYTES)? as usize;
    let mut pos = VK_HEADER_BYTES + 4;
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let len = u32_at(blob, pos + 16)? as usize;
        let body = pos + 20;
        ensure!(
            blob.get(body..body + 4) == Some(b"CPKV"),
            "cache entry {index} at 0x{pos:x}: missing CPKV magic"
        );
        ensure!(
            u32_at(blob, body + 32)? as usize == len,
            "cache entry {index}: inner length disagrees with outer"
        );
        // [u32 flags] then `len` bytes: [u32 raw_size][data (len - 4)]
        let raw_size = u32_at(blob, body + 40)? as usize;
        let data_start = body + 44;
        let data = blob
            .get(data_start..data_start + len - 4)
            .with_context(|| format!("cache entry {index}: truncated data"))?;
        let payload = if data.starts_with(&ZSTD_MAGIC) {
            let mut out = Vec::with_capacity(raw_size);
            let mut cursor = data;
            match ruzstd::decoding::StreamingDecoder::new(&mut cursor) {
                Ok(mut decoder) => {
                    decoder
                        .read_to_end(&mut out)
                        .with_context(|| format!("cache entry {index}: zstd decode"))?;
                    ensure!(
                        out.len() == raw_size,
                        "cache entry {index}: decompressed {} B, header says {raw_size}",
                        out.len()
                    );
                    Ok(out)
                }
                Err(err) => Err(format!("zstd frame header: {err}")),
            }
        } else {
            Ok(data.to_vec())
        };
        entries.push(CacheEntry {
            payload,
            stored_bytes: data.len(),
            raw_bytes: raw_size,
        });
        // key+len (20) + CPKV..len (36) + flags (4) + len.
        pos += 60 + len;
    }
    Ok(entries)
}

/// One NVuc section header.
#[derive(Clone, Debug)]
pub(crate) struct Section {
    pub(crate) kind: u32,
    pub(crate) size: u32,
    pub(crate) offset: u64,
    pub(crate) vals: [u32; 4],
}

impl Section {
    pub(crate) fn name(&self) -> &'static str {
        match self.kind {
            0x01 => "UCODE",
            0x03 => "REG",
            0x18 => "BARRIER",
            0x21 => "STRING_TABLE",
            0x37 => "DEBUG_FRAME_TABLE",
            0x39 => "SYMBOL_TABLE",
            0x3a => "DEBUG_FUNCTION_MAP",
            0x45 => "PROGRAM_INFO",
            0x1204 => "CTAID_INFO",
            _ => "unknown",
        }
    }
}

/// A decoded NVuc ISA container.
pub(crate) struct NvucContainer {
    /// The container bytes, starting at the "NVuc" magic.
    pub(crate) bytes: Vec<u8>,
    pub(crate) sections: Vec<Section>,
    /// Header words 0x08..0x18 (u16 x6 + u32), kept for the report.
    pub(crate) header: [u16; 6],
}

impl NvucContainer {
    pub(crate) fn section(&self, kind: u32) -> Option<&Section> {
        self.sections.iter().find(|s| s.kind == kind)
    }

    pub(crate) fn section_bytes(&self, section: &Section) -> &[u8] {
        let start = section.offset as usize;
        &self.bytes[start..start + section.size as usize]
    }

    /// Raw SASS (the UCODE section), exact boundaries.
    pub(crate) fn ucode(&self) -> Option<&[u8]> {
        self.section(0x01).map(|s| self.section_bytes(s))
    }

    /// (allocated registers, requested cap) from the REG section.
    pub(crate) fn registers(&self) -> Option<(u32, u32)> {
        let s = self.section(0x03)?;
        let b = self.section_bytes(s);
        (b.len() >= 8).then(|| {
            (
                u32::from_le_bytes(b[0..4].try_into().unwrap()),
                u32::from_le_bytes(b[4..8].try_into().unwrap()),
            )
        })
    }

    /// Named-barrier count from the BARRIER section header.
    pub(crate) fn barrier_count(&self) -> Option<u32> {
        self.section(0x18).map(|s| s.vals[1])
    }

    /// First NUL-terminated string of the STRING_TABLE (the symbol).
    pub(crate) fn symbol(&self) -> Option<String> {
        let s = self.section(0x21)?;
        let b = self.section_bytes(s);
        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        Some(String::from_utf8_lossy(&b[..end]).into_owned())
    }

    pub(crate) fn describe(&self) -> String {
        let mut out = format!(
            "NVuc container: {} B, header u16s {:?}, {} sections\n",
            self.bytes.len(),
            self.header,
            self.sections.len()
        );
        for s in &self.sections {
            out.push_str(&format!(
                "  type 0x{:04x} {:<18} size {:>6} @0x{:06x} vals {:?}\n",
                s.kind,
                s.name(),
                s.size,
                s.offset,
                s.vals
            ));
        }
        out
    }
}

/// Every NVuc container embedded in a decompressed cache payload.
pub(crate) fn find_containers(payload: &[u8]) -> Result<Vec<NvucContainer>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = find(&payload[from..], NVDA_MAGIC) {
        let at = from + rel + NVDA_MAGIC.len();
        from = at;
        if payload.get(at..at + 4) != Some(NVUC_MAGIC) {
            continue;
        }
        out.push(parse_container(&payload[at..])?);
    }
    Ok(out)
}

/// Parse one NVuc container starting at its magic; trailing bytes
/// beyond the header's container size are ignored.
pub(crate) fn parse_container(bytes: &[u8]) -> Result<NvucContainer> {
    ensure!(bytes.starts_with(NVUC_MAGIC), "not an NVuc container");
    let version = u32_at(bytes, 4)?;
    ensure!(version == 1, "unsupported NVuc version {version}");
    let nsec = u16_at(bytes, 8)? as usize;
    let mut header = [0u16; 6];
    for (i, h) in header.iter_mut().enumerate() {
        *h = u16_at(bytes, 8 + 2 * i)?;
    }
    let size = u64_at(bytes, 0x18)? as usize;
    ensure!(
        size <= bytes.len() && size >= 0x20 + 0x20 * nsec,
        "NVuc container size {size} out of range ({} B available, {nsec} sections)",
        bytes.len()
    );
    let mut sections = Vec::with_capacity(nsec);
    for i in 0..nsec {
        let h = 0x20 + 0x20 * i;
        let section = Section {
            kind: u32_at(bytes, h)?,
            size: u32_at(bytes, h + 4)?,
            offset: u64_at(bytes, h + 8)?,
            vals: [
                u32_at(bytes, h + 16)?,
                u32_at(bytes, h + 20)?,
                u32_at(bytes, h + 24)?,
                u32_at(bytes, h + 28)?,
            ],
        };
        if section.size > 0 {
            ensure!(
                section.offset as usize + section.size as usize <= size,
                "NVuc section {i} (type 0x{:x}) overruns the container",
                section.kind
            );
        }
        sections.push(section);
    }
    let container = NvucContainer {
        bytes: bytes[..size].to_vec(),
        sections,
        header,
    };
    if let Some(code) = container.ucode() {
        ensure!(
            code.len().is_multiple_of(16),
            "UCODE size {} is not a multiple of the 16 B instruction",
            code.len()
        );
    }
    Ok(container)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    match b.get(at..at + 2) {
        Some(s) => Ok(u16::from_le_bytes(s.try_into().unwrap())),
        None => bail!("read past end at 0x{at:x}"),
    }
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    match b.get(at..at + 4) {
        Some(s) => Ok(u32::from_le_bytes(s.try_into().unwrap())),
        None => bail!("read past end at 0x{at:x}"),
    }
}

fn u64_at(b: &[u8], at: usize) -> Result<u64> {
    match b.get(at..at + 8) {
        Some(s) => Ok(u64::from_le_bytes(s.try_into().unwrap())),
        None => bail!("read past end at 0x{at:x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal synthetic container: REG + UCODE sections.
    fn synthetic() -> Vec<u8> {
        let nsec = 2u16;
        let data_start = 0x20 + 0x20 * nsec as usize;
        let reg_off = data_start;
        let code_off = reg_off + 8;
        let size = code_off + 32;
        let mut b = vec![0u8; size];
        b[0..4].copy_from_slice(NVUC_MAGIC);
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8..10].copy_from_slice(&nsec.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&(size as u64).to_le_bytes());
        // REG
        b[0x20..0x24].copy_from_slice(&3u32.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&8u32.to_le_bytes());
        b[0x28..0x30].copy_from_slice(&(reg_off as u64).to_le_bytes());
        // UCODE
        b[0x40..0x44].copy_from_slice(&1u32.to_le_bytes());
        b[0x44..0x48].copy_from_slice(&32u32.to_le_bytes());
        b[0x48..0x50].copy_from_slice(&(code_off as u64).to_le_bytes());
        b[reg_off..reg_off + 4].copy_from_slice(&96u32.to_le_bytes());
        b[reg_off + 4..reg_off + 8].copy_from_slice(&255u32.to_le_bytes());
        b[code_off] = 0xAA;
        b
    }

    #[test]
    fn parses_sections_registers_and_exact_ucode() {
        let mut payload = b"junkNVDANVVM".to_vec();
        payload.extend(synthetic());
        payload.extend(b"trailing records");
        let containers = find_containers(&payload).unwrap();
        assert_eq!(containers.len(), 1);
        let entry = CacheEntry {
            payload: Ok(payload.clone()),
            stored_bytes: payload.len(),
            raw_bytes: payload.len(),
        };
        assert_eq!(entry.containers().unwrap().len(), 1);
        let c = &containers[0];
        assert_eq!(c.registers(), Some((96, 255)));
        let code = c.ucode().unwrap();
        assert_eq!(code.len(), 32);
        assert_eq!(code[0], 0xAA);
    }

    #[test]
    fn rejects_overrunning_sections() {
        let mut b = synthetic();
        b[0x44..0x48].copy_from_slice(&4096u32.to_le_bytes());
        assert!(parse_container(&b).is_err());
    }
}
