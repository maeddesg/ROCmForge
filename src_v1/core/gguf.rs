//! GGUF binary-format parser for v1.0.
//!
//! Parses header → KV metadata (typed) → tensor descriptors → memory
//! maps the file. Zero-copy: tensor data stays in the mmap until the
//! explicit upload step in [`super::model_loader`] copies it into the
//! VRAM arena.
//!
//! The parser is structurally adapted from the v0.x loader but stores
//! metadata as typed [`GGUFValue`]s rather than stringified values, so
//! downstream code can match on e.g. `U32(_)` or `F32(_)` without
//! re-parsing strings.
//!
//! Supported versions: GGUF v2 and v3 (as produced by
//! `llama.cpp/convert_hf_to_gguf.py`).

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::tensor_info::{GgmlType, TensorInfo};

// --- Constants ---------------------------------------------------------------

const MAGIC: [u8; 4] = *b"GGUF";
const VERSION_MIN: u32 = 2;
const VERSION_MAX: u32 = 3;
/// GGUF spec: tensor-data section starts on a 32-byte boundary after the
/// last tensor descriptor.
pub const TENSOR_ALIGNMENT: u64 = 32;

// --- Error -------------------------------------------------------------------

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    UnknownTensorType(u32),
    UnknownValueType(u32),
    InvalidUtf8,
    StringTooLong(usize),
    OutOfBounds {
        offset: u64,
        size: usize,
        file_size: usize,
    },
    MissingKey(String),
    TypeMismatch {
        key: String,
        expected: &'static str,
    },
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::InvalidMagic(m) => write!(f, "invalid GGUF magic: {m:?}"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::UnknownTensorType(t) => write!(f, "unknown ggml_type: {t}"),
            GgufError::UnknownValueType(t) => write!(f, "unknown metadata value type: {t}"),
            GgufError::InvalidUtf8 => write!(f, "invalid UTF-8 in metadata string"),
            GgufError::StringTooLong(n) => write!(f, "string too long: {n} bytes"),
            GgufError::OutOfBounds { offset, size, file_size } => write!(
                f,
                "tensor out of bounds: offset={offset} size={size} file_size={file_size}"
            ),
            GgufError::MissingKey(k) => write!(f, "missing metadata key: {k}"),
            GgufError::TypeMismatch { key, expected } => {
                write!(f, "metadata key {key}: expected {expected}")
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

pub type GgufResult<T> = Result<T, GgufError>;

// --- Typed metadata values ---------------------------------------------------

/// GGUF metadata value-type discriminants — see GGUF spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> GgufResult<Self> {
        Ok(match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

/// Typed GGUF metadata value.
///
/// Arrays of scalar primitives are preserved; arrays of strings keep
/// their elements. Nested arrays (rare in practice — only token arrays
/// ever use this shape) are stored as `Array(GgufValueType::Array, ...)`
/// with a single-level payload.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayString(Vec<String>),
    /// Large / unsupported-element arrays: we remember the element type
    /// and count so callers can report "arrayf N elements" without the
    /// parser having to materialise multi-million-entry vectors.
    ArraySummary {
        element_type: GgufValueType,
        count: u64,
    },
}

impl GgufValue {
    pub fn as_string(&self) -> GgufResult<&str> {
        match self {
            GgufValue::String(s) => Ok(s),
            _ => Err(GgufError::TypeMismatch {
                key: String::new(),
                expected: "string",
            }),
        }
    }

    /// Best-effort unsigned-int coercion: accepts any integer-type value
    /// that fits into `u64`.
    pub fn as_u64(&self) -> GgufResult<u64> {
        Ok(match self {
            GgufValue::U8(v) => *v as u64,
            GgufValue::U16(v) => *v as u64,
            GgufValue::U32(v) => *v as u64,
            GgufValue::U64(v) => *v,
            GgufValue::I8(v) if *v >= 0 => *v as u64,
            GgufValue::I16(v) if *v >= 0 => *v as u64,
            GgufValue::I32(v) if *v >= 0 => *v as u64,
            GgufValue::I64(v) if *v >= 0 => *v as u64,
            GgufValue::Bool(b) => *b as u64,
            _ => {
                return Err(GgufError::TypeMismatch {
                    key: String::new(),
                    expected: "unsigned integer",
                })
            }
        })
    }

    pub fn as_u32(&self) -> GgufResult<u32> {
        let v = self.as_u64()?;
        if v > u32::MAX as u64 {
            return Err(GgufError::TypeMismatch {
                key: String::new(),
                expected: "u32",
            });
        }
        Ok(v as u32)
    }

    pub fn as_f32(&self) -> GgufResult<f32> {
        Ok(match self {
            GgufValue::F32(v) => *v,
            GgufValue::F64(v) => *v as f32,
            GgufValue::U32(v) => *v as f32,
            GgufValue::U64(v) => *v as f32,
            GgufValue::I32(v) => *v as f32,
            GgufValue::I64(v) => *v as f32,
            _ => {
                return Err(GgufError::TypeMismatch {
                    key: String::new(),
                    expected: "float",
                })
            }
        })
    }

    pub fn as_bool(&self) -> GgufResult<bool> {
        match self {
            GgufValue::Bool(b) => Ok(*b),
            GgufValue::U8(v) => Ok(*v != 0),
            GgufValue::U32(v) => Ok(*v != 0),
            _ => Err(GgufError::TypeMismatch {
                key: String::new(),
                expected: "bool",
            }),
        }
    }
}

// --- Header ------------------------------------------------------------------

#[derive(Debug)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

// --- Primitive readers -------------------------------------------------------

fn read_u32<R: Read>(r: &mut R) -> GgufResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> GgufResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> GgufResult<String> {
    let len = read_u64(r)? as usize;
    if len > 100_000_000 {
        return Err(GgufError::StringTooLong(len));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8)
}

fn read_header<R: Read>(r: &mut R) -> GgufResult<Header> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }
    let version = read_u32(r)?;
    if !(VERSION_MIN..=VERSION_MAX).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;
    Ok(Header {
        version,
        tensor_count,
        kv_count,
    })
}

// --- Value readers -----------------------------------------------------------

fn read_value<R: Read>(r: &mut R, ty: GgufValueType) -> GgufResult<GgufValue> {
    Ok(match ty {
        GgufValueType::U8 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::U8(b[0])
        }
        GgufValueType::I8 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::I8(b[0] as i8)
        }
        GgufValueType::U16 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            GgufValue::U16(u16::from_le_bytes(b))
        }
        GgufValueType::I16 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            GgufValue::I16(i16::from_le_bytes(b))
        }
        GgufValueType::U32 => GgufValue::U32(read_u32(r)?),
        GgufValueType::I32 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            GgufValue::I32(i32::from_le_bytes(b))
        }
        GgufValueType::F32 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            GgufValue::F32(f32::from_le_bytes(b))
        }
        GgufValueType::Bool => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            GgufValue::Bool(b[0] != 0)
        }
        GgufValueType::String => GgufValue::String(read_string(r)?),
        GgufValueType::U64 => GgufValue::U64(read_u64(r)?),
        GgufValueType::I64 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            GgufValue::I64(i64::from_le_bytes(b))
        }
        GgufValueType::F64 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            GgufValue::F64(f64::from_le_bytes(b))
        }
        GgufValueType::Array => read_array(r)?,
    })
}

fn read_array<R: Read>(r: &mut R) -> GgufResult<GgufValue> {
    let elem_ty = GgufValueType::from_u32(read_u32(r)?)?;
    let count = read_u64(r)?;

    // Large-count guardrail: materialise only into compact primitive
    // vecs, summarise everything else.
    const MAX_MATERIALISE: u64 = 256;
    // String arrays are always materialised — the tokenizer vocabulary
    // has up to ~200 000 entries and must be available in full.
    let materialise = match elem_ty {
        GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => true,
        GgufValueType::String => true,
        _ => count <= MAX_MATERIALISE,
    };

    if !materialise {
        // Skip elements without allocating.
        for _ in 0..count {
            skip_value(r, elem_ty)?;
        }
        return Ok(GgufValue::ArraySummary {
            element_type: elem_ty,
            count,
        });
    }

    match elem_ty {
        GgufValueType::U32 => {
            let mut v = Vec::with_capacity(count as usize);
            for _ in 0..count {
                v.push(read_u32(r)?);
            }
            Ok(GgufValue::ArrayU32(v))
        }
        GgufValueType::I32 => {
            let mut v = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                v.push(i32::from_le_bytes(b));
            }
            Ok(GgufValue::ArrayI32(v))
        }
        GgufValueType::F32 => {
            let mut v = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let mut b = [0u8; 4];
                r.read_exact(&mut b)?;
                v.push(f32::from_le_bytes(b));
            }
            Ok(GgufValue::ArrayF32(v))
        }
        GgufValueType::String => {
            let mut v = Vec::with_capacity(count as usize);
            for _ in 0..count {
                v.push(read_string(r)?);
            }
            Ok(GgufValue::ArrayString(v))
        }
        other => {
            // Small array of non-materialised element type — skip but
            // summarise.
            for _ in 0..count {
                skip_value(r, other)?;
            }
            Ok(GgufValue::ArraySummary {
                element_type: other,
                count,
            })
        }
    }
}

fn skip_value<R: Read>(r: &mut R, ty: GgufValueType) -> GgufResult<()> {
    match ty {
        GgufValueType::U8 | GgufValueType::I8 | GgufValueType::Bool => {
            r.read_exact(&mut [0u8; 1])?;
        }
        GgufValueType::U16 | GgufValueType::I16 => {
            r.read_exact(&mut [0u8; 2])?;
        }
        GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => {
            r.read_exact(&mut [0u8; 4])?;
        }
        GgufValueType::U64 | GgufValueType::I64 | GgufValueType::F64 => {
            r.read_exact(&mut [0u8; 8])?;
        }
        GgufValueType::String => {
            read_string(r)?;
        }
        GgufValueType::Array => {
            let inner = GgufValueType::from_u32(read_u32(r)?)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, inner)?;
            }
        }
    }
    Ok(())
}

// --- Main parser entry points ------------------------------------------------

/// Parsed GGUF file with typed metadata and the raw mmap.
///
/// `TensorInfo.file_offset` is **relative to `data_start`** — the byte
/// range `data_start + file_offset .. + byte_size` is the tensor's data
/// inside the mmap.
pub struct GGUFFile {
    _file: File,
    mmap: Mmap,
    header: Header,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_start: u64,
}

impl GGUFFile {
    pub fn open(path: impl AsRef<Path>) -> GgufResult<Self> {
        let path = resolve_tilde(path.as_ref());
        let file = File::open(&path)?;
        let mut reader = BufReader::new(&file);

        let header = read_header(&mut reader)?;

        let mut metadata: HashMap<String, GgufValue> =
            HashMap::with_capacity(header.kv_count as usize);
        for _ in 0..header.kv_count {
            let key = read_string(&mut reader)?;
            let ty = GgufValueType::from_u32(read_u32(&mut reader)?)?;
            let value = read_value(&mut reader, ty)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let name = read_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(&mut reader)?);
            }
            let type_code = read_u32(&mut reader)?;
            let ggml_type = GgmlType::from_u32(type_code)?;
            let offset = read_u64(&mut reader)?;

            let n_elements: u64 = shape.iter().product();
            let byte_size = ggml_type.bytes_for_elements(n_elements as usize) as u64;

            tensors.push(TensorInfo {
                name,
                shape,
                ggml_type,
                file_offset: offset,
                byte_size,
                n_elements,
            });
        }

        // Tensor data section starts at the next TENSOR_ALIGNMENT
        // boundary after the descriptor section.
        let pos = reader.seek(SeekFrom::Current(0))?;
        let data_start = pos.div_ceil(TENSOR_ALIGNMENT) * TENSOR_ALIGNMENT;

        drop(reader);

        // SAFETY: read-only mapping, file is kept alive via `_file`.
        let mmap = unsafe { Mmap::map(&file) }?;

        Ok(Self {
            _file: file,
            mmap,
            header,
            metadata,
            tensors,
            data_start,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    pub fn metadata_count(&self) -> usize {
        self.metadata.len()
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn mmap_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Raw tensor data slice inside the mmap. Returns up to `max_bytes`
    /// starting at the tensor's data offset — used by the `--list-tensors`
    /// CLI and the readback spot-check test.
    pub fn tensor_data(&self, tensor: &TensorInfo, max_bytes: usize) -> GgufResult<&[u8]> {
        let start = (self.data_start + tensor.file_offset) as usize;
        let len = std::cmp::min(tensor.byte_size as usize, max_bytes);
        let end = start + len;
        if end > self.mmap.len() {
            return Err(GgufError::OutOfBounds {
                offset: self.data_start + tensor.file_offset,
                size: len,
                file_size: self.mmap.len(),
            });
        }
        Ok(&self.mmap[start..end])
    }

    /// Full tensor data slice — zero-copy view into the mmap for the
    /// loader's H2D upload.
    pub fn tensor_data_full(&self, tensor: &TensorInfo) -> GgufResult<&[u8]> {
        let start = (self.data_start + tensor.file_offset) as usize;
        let end = start + tensor.byte_size as usize;
        if end > self.mmap.len() {
            return Err(GgufError::OutOfBounds {
                offset: self.data_start + tensor.file_offset,
                size: tensor.byte_size as usize,
                file_size: self.mmap.len(),
            });
        }
        Ok(&self.mmap[start..end])
    }
}

/// Expand `~` / `~/…` path prefixes into the current user's home dir.
pub fn resolve_tilde(path: &Path) -> std::path::PathBuf {
    let s = path.to_string_lossy();
    if let Some(stripped) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    } else if s == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    path.to_path_buf()
}
