//! Checked-in per-GPU thesis expectations.
//!
//! `benchmarks/thesis-expectations.toml` records, per device name, the
//! numeric predictions the harness holds each thesis to (bandwidth
//! fractions, barrier cost, submission cost, TFLOPS floors).  The file
//! is a strict TOML subset parsed here without external dependencies:
//! `["quoted section"]` headers and `key = <float>` pairs, `#`
//! comments.  Devices without a section still get every measurement,
//! but all verdicts degrade to INFO — record a section to turn the
//! harness into a regression gate for that GPU.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

/// One device's recorded expectations: bare TOML keys to float values.
pub(crate) type Section = BTreeMap<String, f64>;

pub(crate) struct Expectations {
    sections: BTreeMap<String, Section>,
}

impl Expectations {
    pub(crate) fn device(&self, name: &str) -> Option<&Section> {
        self.sections.get(name)
    }
}

/// Locate the expectations file: `ML_THESIS_EXPECT`, then the
/// workspace-relative checked-in path, then the compile-time manifest
/// fallback (for invocations from outside the workspace root).
pub(crate) fn default_path() -> PathBuf {
    if let Ok(path) = std::env::var("ML_THESIS_EXPECT") {
        return PathBuf::from(path);
    }
    let relative = PathBuf::from("benchmarks/thesis-expectations.toml");
    if relative.exists() {
        return relative;
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/thesis-expectations.toml"
    ))
}

/// Load and parse the expectations file; a missing file is not an
/// error (the harness then reports without verdicts).
pub(crate) fn load(path: &PathBuf) -> Result<Option<Expectations>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("reading {}", path.display()));
        }
    };
    parse(&text)
        .with_context(|| format!("parsing {}", path.display()))
        .map(Some)
}

fn parse(text: &str) -> Result<Expectations> {
    let mut sections: BTreeMap<String, Section> = BTreeMap::new();
    let mut current: Option<String> = None;
    for (lineno, raw) in text.lines().enumerate() {
        // Values are numeric and section names are device strings, so
        // stripping from the first '#' can never truncate a value.
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('[') {
            let header = header
                .strip_suffix(']')
                .with_context(|| format!("line {}: unterminated section header", lineno + 1))?
                .trim();
            let name = header
                .strip_prefix('"')
                .and_then(|h| h.strip_suffix('"'))
                .unwrap_or(header);
            if name.is_empty() {
                bail!("line {}: empty section name", lineno + 1);
            }
            sections.entry(name.to_string()).or_default();
            current = Some(name.to_string());
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .with_context(|| format!("line {}: expected key = value", lineno + 1))?;
        let section = current
            .as_ref()
            .with_context(|| format!("line {}: key before any [section]", lineno + 1))?;
        let value: f64 = value
            .trim()
            .parse()
            .with_context(|| format!("line {}: value is not a number", lineno + 1))?;
        sections
            .get_mut(section)
            .expect("section inserted on header")
            .insert(key.trim().to_string(), value);
    }
    Ok(Expectations { sections })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sections_keys_and_comments() {
        let text = r#"
# top comment
["NVIDIA GeForce RTX 3070"]
mem_bw_gbps = 448.0   # spec sheet
t3_barrier_us = 7.7

[bare-name]
x = 1
"#;
        let parsed = parse(text).unwrap();
        let ga104 = parsed.device("NVIDIA GeForce RTX 3070").unwrap();
        assert_eq!(ga104.get("mem_bw_gbps"), Some(&448.0));
        assert_eq!(ga104.get("t3_barrier_us"), Some(&7.7));
        assert_eq!(parsed.device("bare-name").unwrap().get("x"), Some(&1.0));
        assert!(parsed.device("unknown").is_none());
    }

    #[test]
    fn rejects_malformed_lines() {
        assert!(parse("key = 1.0").is_err(), "key before section");
        assert!(parse("[sec]\nkey 1.0").is_err(), "missing equals");
        assert!(parse("[sec]\nkey = fast").is_err(), "non-numeric value");
        assert!(parse("[unterminated\n").is_err(), "unterminated header");
    }
}
