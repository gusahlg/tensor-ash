//! Minimal GGUF v2/v3 reader: metadata scalars/strings, tensor infos,
//! and raw tensor data access.  Supports F32 and F16 tensor types only.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;

/// Metadata value.  Arrays are walked and discarded (the tokenizer
/// vocab is thousands of strings we never need on this path).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Array or other payload we parsed past without keeping.
    Skipped,
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    #[allow(dead_code)]
    pub name: String,
    /// Dimensions as stored: `ne[0]` is the fastest-varying axis.
    pub ne: Vec<u64>,
    pub ggml_type: u32,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.ne.iter().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Absolute byte offset of the aligned data section.
    pub data_start: u64,
    reader: BufReader<File>,
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).context("unexpected end of GGUF file")
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > 1 << 24 {
        bail!("GGUF string length {len} is implausible");
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(r, &mut buf)?;
    String::from_utf8(buf).context("GGUF string is not UTF-8")
}

/// Fixed byte width of a scalar metadata type, if it has one.
fn scalar_width(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,            // 8 = string, 9 = array
    }
}

fn read_value<R: Read + Seek>(r: &mut R, ty: u32, keep: bool) -> Result<Value> {
    macro_rules! scalar {
        ($n:expr, $conv:expr) => {{
            let mut b = [0u8; $n];
            read_exact(r, &mut b)?;
            $conv(b)
        }};
    }
    Ok(match ty {
        0 => Value::U64(scalar!(1, |b: [u8; 1]| b[0] as u64)),
        1 => Value::I64(scalar!(1, |b: [u8; 1]| b[0] as i8 as i64)),
        2 => Value::U64(scalar!(2, |b| u16::from_le_bytes(b) as u64)),
        3 => Value::I64(scalar!(2, |b| i16::from_le_bytes(b) as i64)),
        4 => Value::U64(scalar!(4, |b| u32::from_le_bytes(b) as u64)),
        5 => Value::I64(scalar!(4, |b| i32::from_le_bytes(b) as i64)),
        6 => Value::F64(scalar!(4, |b| f32::from_le_bytes(b) as f64)),
        7 => Value::Bool(scalar!(1, |b: [u8; 1]| b[0] != 0)),
        8 => {
            let s = read_string(r)?;
            if keep { Value::Str(s) } else { Value::Skipped }
        }
        9 => {
            // Array: element type, count, elements.  Skip the payload;
            // fixed-width element arrays seek past in one step.
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            if let Some(width) = scalar_width(elem_ty) {
                let bytes = width
                    .checked_mul(count)
                    .context("GGUF array size overflow")?;
                r.seek(SeekFrom::Current(bytes as i64))?;
            } else if elem_ty == 8 {
                for _ in 0..count {
                    let len = read_u64(r)?;
                    r.seek(SeekFrom::Current(len as i64))?;
                }
            } else {
                bail!("GGUF nested/unknown array element type {elem_ty}");
            }
            Value::Skipped
        }
        10 => Value::U64(scalar!(8, u64::from_le_bytes)),
        11 => Value::I64(scalar!(8, i64::from_le_bytes)),
        12 => Value::F64(scalar!(8, f64::from_le_bytes)),
        _ => bail!("unknown GGUF metadata type {ty}"),
    })
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut r = BufReader::with_capacity(1 << 20, file);

        let mut magic = [0u8; 4];
        read_exact(&mut r, &mut magic)?;
        if &magic != b"GGUF" {
            bail!("not a GGUF file (bad magic {magic:?})");
        }
        let version = read_u32(&mut r)?;
        if version != 2 && version != 3 {
            bail!("unsupported GGUF version {version} (want 2 or 3)");
        }
        let tensor_count = read_u64(&mut r)?;
        let kv_count = read_u64(&mut r)?;
        if tensor_count > 1 << 20 || kv_count > 1 << 20 {
            bail!("implausible GGUF header (tensors {tensor_count}, kvs {kv_count})");
        }

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(&mut r)?;
            let ty = read_u32(&mut r)?;
            let value = read_value(&mut r, ty, true)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            if n_dims == 0 || n_dims > 4 {
                bail!("tensor {name}: bad n_dims {n_dims}");
            }
            let mut ne = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                ne.push(read_u64(&mut r)?);
            }
            let ggml_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    ne,
                    ggml_type,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("bad general.alignment {alignment}");
        }
        let here = r.stream_position()?;
        let data_start = here.div_ceil(alignment) * alignment;

        Ok(Self {
            metadata,
            tensors,
            data_start,
            reader: r,
        })
    }

    pub fn require_u64(&self, key: &str) -> Result<u64> {
        self.metadata
            .get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("GGUF metadata missing integer key {key}"))
    }

    pub fn f64_or(&self, key: &str, default: f64) -> f64 {
        self.metadata
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or(default)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("GGUF tensor {name} not found"))
    }

    /// Reads a tensor's data as f32, widening f16.  Bails on any other
    /// (quantized) type: this runner needs an f16 GGUF.
    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?.clone();
        let count = info.element_count() as usize;
        self.reader
            .seek(SeekFrom::Start(self.data_start + info.offset))?;
        match info.ggml_type {
            GGML_TYPE_F32 => {
                let mut bytes = vec![0u8; count * 4];
                read_exact(&mut self.reader, &mut bytes)?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            GGML_TYPE_F16 => {
                let mut bytes = vec![0u8; count * 2];
                read_exact(&mut self.reader, &mut bytes)?;
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| tensor_ash::dtype::f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect())
            }
            other => bail!(
                "tensor {name} has ggml type {other}; only F32/F16 are supported — \
                 use an f16 GGUF (e.g. *.f16.gguf), not a quantized one"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn put_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// Builds a synthetic v3 GGUF: scalar + string + array metadata,
    /// one f32 tensor and one f16 tensor, 32-byte alignment.
    fn synthetic_gguf() -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&5u64.to_le_bytes()); // kv_count

        // general.architecture = "llama" (string)
        put_string(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        put_string(&mut out, "llama");
        // llama.block_count = 22 (u32)
        put_string(&mut out, "llama.block_count");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&22u32.to_le_bytes());
        // llama.attention.layer_norm_rms_epsilon = 1e-5 (f32)
        put_string(&mut out, "llama.attention.layer_norm_rms_epsilon");
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&1e-5f32.to_le_bytes());
        // tokenizer.ggml.tokens = ["a", "bc"] (string array, must be skipped)
        put_string(&mut out, "tokenizer.ggml.tokens");
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes()); // elem type string
        out.extend_from_slice(&2u64.to_le_bytes());
        put_string(&mut out, "a");
        put_string(&mut out, "bc");
        // tokenizer.ggml.scores = [1.0, 2.0] (f32 array, skipped)
        put_string(&mut out, "tokenizer.ggml.scores");
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&1.0f32.to_le_bytes());
        out.extend_from_slice(&2.0f32.to_le_bytes());

        // Tensor infos: "w32" f32 [3, 2] and "w16" f16 [4].
        let w32: Vec<f32> = vec![0.5, -1.0, 2.0, 3.5, -0.25, 8.0];
        let w16: Vec<f32> = vec![1.0, -2.0, 0.5, 4.0];
        put_string(&mut out, "w32");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        let w16_offset = ((w32.len() * 4) as u64).div_ceil(32) * 32;
        put_string(&mut out, "w16");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&GGML_TYPE_F16.to_le_bytes());
        out.extend_from_slice(&w16_offset.to_le_bytes());

        // Align to 32, then the data blob.
        while out.len() % 32 != 0 {
            out.push(0);
        }
        let data_start = out.len();
        for v in &w32 {
            out.extend_from_slice(&v.to_le_bytes());
        }
        while !((out.len() - data_start) as u64).is_multiple_of(32) {
            out.push(0);
        }
        assert_eq!((out.len() - data_start) as u64, w16_offset);
        for v in &w16 {
            out.extend_from_slice(&tensor_ash::dtype::f32_to_f16_bits(*v).to_le_bytes());
        }
        (out, w32, w16)
    }

    #[test]
    fn parses_synthetic_gguf() {
        let (bytes, w32, w16) = synthetic_gguf();
        let dir = std::env::temp_dir().join("llama_ash_gguf_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mini.gguf");
        File::create(&path).unwrap().write_all(&bytes).unwrap();

        let mut gguf = GgufFile::open(&path).unwrap();
        assert_eq!(
            gguf.metadata.get("general.architecture"),
            Some(&Value::Str("llama".into()))
        );
        assert_eq!(gguf.require_u64("llama.block_count").unwrap(), 22);
        let eps = gguf.f64_or("llama.attention.layer_norm_rms_epsilon", 0.0);
        assert!((eps - 1e-5).abs() < 1e-12);
        assert_eq!(gguf.f64_or("llama.rope.freq_base", 10000.0), 10000.0);
        assert_eq!(
            gguf.metadata.get("tokenizer.ggml.tokens"),
            Some(&Value::Skipped)
        );

        let info = gguf.info("w32").unwrap();
        assert_eq!(info.ne, vec![3, 2]);
        assert_eq!(gguf.read_f32("w32").unwrap(), w32);
        let info = gguf.info("w16").unwrap();
        assert_eq!(info.ne, vec![4]);
        assert_eq!(info.ggml_type, GGML_TYPE_F16);
        assert_eq!(gguf.read_f32("w16").unwrap(), w16);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let dir = std::env::temp_dir().join("llama_ash_gguf_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        File::create(&path).unwrap().write_all(b"NOPE").unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("magic")
        );

        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        assert!(
            GgufFile::open(&path)
                .unwrap_err()
                .to_string()
                .contains("version")
        );
    }
}
