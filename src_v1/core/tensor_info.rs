//! Tensor inventory: typed [`GgmlType`], per-tensor [`TensorInfo`],
//! architecture-agnostic [`TensorRole`] classification, and layer
//! grouping.
//!
//! Tensor-role detection is metadata-driven: role is derived from the
//! GGUF name alone (`blk.{N}.attn_q.weight` → `AttentionQ`, layer `N`).
//! No model-name matching.
//!
//! The role set covers the three supported Phase-1 models:
//!   - Qwen2.5: dense transformer + **attention biases** (Q/K/V bias
//!     tensors present in the GGUF)
//!   - Qwen3: dense transformer with per-head Q-/K-norm
//!   - Llama-3.1: dense transformer with explicit `rope_freqs.weight`
//!
//! Any tensor whose name does not match a known pattern becomes
//! [`TensorRole::Unknown`] — the loader logs a warning and downstream
//! passes decide whether the tensor is critical.

use std::collections::HashMap;

use super::gguf::GgufError;

// --- GgmlType ----------------------------------------------------------------

/// GGUF tensor data types — matches `ggml_type` from ggml.h.
///
/// Only the types ROCmForge v1.0 understands are represented; any
/// unknown code returned by the GGUF file produces
/// [`GgufError::UnknownTensorType`] at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            other => return Err(GgufError::UnknownTensorType(other)),
        })
    }

    /// Bytes required to store `n` elements of this type.
    ///
    /// Block sizes match llama.cpp `ggml-common.h`:
    /// Q4_0: 32/18, Q4_1: 32/20, Q5_0: 32/22, Q5_1: 32/24, Q8_0: 32/34,
    /// Q2_K: 256/82, Q3_K: 256/110, Q4_K: 256/144, Q5_K: 256/176,
    /// Q6_K: 256/210, Q8_K: 256/292.
    pub fn bytes_for_elements(&self, n: usize) -> usize {
        match self {
            Self::F32 => n * 4,
            Self::F16 => n * 2,
            Self::BF16 => n * 2,
            Self::Q4_0 => n.div_ceil(32) * 18,
            Self::Q4_1 => n.div_ceil(32) * 20,
            Self::Q5_0 => n.div_ceil(32) * 22,
            Self::Q5_1 => n.div_ceil(32) * 24,
            Self::Q8_0 => n.div_ceil(32) * 34,
            Self::Q2_K => n.div_ceil(256) * 82,
            Self::Q3_K => n.div_ceil(256) * 110,
            Self::Q4_K => n.div_ceil(256) * 144,
            Self::Q5_K => n.div_ceil(256) * 176,
            Self::Q6_K => n.div_ceil(256) * 210,
            Self::Q8_K => n.div_ceil(256) * 292,
        }
    }

    /// `true` iff Phase-1 dequant kernels support this type. Dequant-IR
    /// spec §3: Q4_0, Q4_1, Q4_K, Q6_K, Q8_0, F16, F32 are supported;
    /// BF16 is a planned pass-through. Everything else is a parse-OK /
    /// warn-at-validate case.
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q4_K
                | Self::Q6_K
                | Self::Q8_0
        )
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
        };
        f.write_str(s)
    }
}

// --- TensorInfo --------------------------------------------------------------

/// Metadata for a single tensor. **No weight data** — data stays in the
/// mmap and is copied into the VRAM arena by [`super::model_loader`].
///
/// `file_offset` is relative to the GGUF tensor-data section (the
/// `GGUFFile::data_start` boundary). After the upload step, consumer
/// code uses `ArenaSlice` offsets exclusively; `file_offset` is only
/// relevant while reading from disk.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    pub file_offset: u64,
    pub byte_size: u64,
    pub n_elements: u64,
}

// --- TensorRole --------------------------------------------------------------

/// Architecture-agnostic role label derived purely from the tensor
/// name. Roles with a `*Bias` suffix cover Qwen2.5-style attention
/// projections that include bias tensors (the architecture doc exposes
/// optional bias on `Gemv { weight, bias: Option<WeightRef> }`, so
/// biases are first-class roles rather than "unknown" junk).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorRole {
    // Global
    Embedding,
    OutputNorm,
    LMHead,

    // Per-layer weights
    AttentionNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOutput,
    AttentionQNorm,
    AttentionKNorm,
    FFNNorm,
    FFNGate,
    FFNUp,
    FFNDown,

    // Per-layer biases (Qwen2.5)
    AttentionQBias,
    AttentionKBias,
    AttentionVBias,
    AttentionOutputBias,
    FFNGateBias,
    FFNUpBias,
    FFNDownBias,

    // Optional globals
    RopeFreqs,

    // Unknown — loader warns, graph builder decides criticality.
    Unknown(String),
}

/// Parsed `(role, layer_idx)` from a tensor name.
///
/// `layer_idx` is `None` for global tensors and for `Unknown` entries.
pub fn parse_tensor_name(name: &str) -> (TensorRole, Option<usize>) {
    // Global tensors first — no layer prefix.
    match name {
        "token_embd.weight" => return (TensorRole::Embedding, None),
        "output_norm.weight" => return (TensorRole::OutputNorm, None),
        "output.weight" => return (TensorRole::LMHead, None),
        "rope_freqs.weight" => return (TensorRole::RopeFreqs, None),
        _ => {}
    }

    // Layer tensors: blk.{N}.{suffix}
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            let layer_str = &rest[..dot];
            let suffix = &rest[dot + 1..];
            if let Ok(layer) = layer_str.parse::<usize>() {
                let role = match suffix {
                    "attn_norm.weight" => TensorRole::AttentionNorm,
                    "attn_q.weight" => TensorRole::AttentionQ,
                    "attn_k.weight" => TensorRole::AttentionK,
                    "attn_v.weight" => TensorRole::AttentionV,
                    "attn_output.weight" => TensorRole::AttentionOutput,
                    "attn_q_norm.weight" => TensorRole::AttentionQNorm,
                    "attn_k_norm.weight" => TensorRole::AttentionKNorm,
                    "ffn_norm.weight" => TensorRole::FFNNorm,
                    "ffn_gate.weight" => TensorRole::FFNGate,
                    "ffn_up.weight" => TensorRole::FFNUp,
                    "ffn_down.weight" => TensorRole::FFNDown,
                    "attn_q.bias" => TensorRole::AttentionQBias,
                    "attn_k.bias" => TensorRole::AttentionKBias,
                    "attn_v.bias" => TensorRole::AttentionVBias,
                    "attn_output.bias" => TensorRole::AttentionOutputBias,
                    "ffn_gate.bias" => TensorRole::FFNGateBias,
                    "ffn_up.bias" => TensorRole::FFNUpBias,
                    "ffn_down.bias" => TensorRole::FFNDownBias,
                    other => {
                        return (
                            TensorRole::Unknown(format!("blk.{layer}.{other}")),
                            Some(layer),
                        );
                    }
                };
                return (role, Some(layer));
            }
        }
    }

    (TensorRole::Unknown(name.to_string()), None)
}

// --- Layer grouping ----------------------------------------------------------

/// All tensors belonging to a single transformer block.
#[derive(Debug, Default, Clone)]
pub struct LayerTensors {
    pub layer_idx: usize,
    pub tensors: HashMap<TensorRole, TensorInfo>,
}

/// Group tensors by their layer index. Global tensors and `Unknown`
/// entries without a layer index are skipped (they are still available
/// via `GGUFFile::tensors()` if needed).
pub fn group_tensors_by_layer(tensors: &[TensorInfo]) -> Vec<LayerTensors> {
    let mut per_layer: HashMap<usize, LayerTensors> = HashMap::new();
    for t in tensors {
        let (role, layer) = parse_tensor_name(&t.name);
        if let Some(idx) = layer {
            let entry = per_layer.entry(idx).or_insert_with(|| LayerTensors {
                layer_idx: idx,
                tensors: HashMap::new(),
            });
            entry.tensors.insert(role, t.clone());
        }
    }
    let mut out: Vec<LayerTensors> = per_layer.into_values().collect();
    out.sort_by_key(|l| l.layer_idx);
    out
}

// --- Validation --------------------------------------------------------------

/// Warn about tensors whose quant type Phase-1 cannot dequantise.
/// Returns one message per unsupported tensor; empty vec means clean.
pub fn validate_quant_types(tensors: &[TensorInfo]) -> Vec<String> {
    tensors
        .iter()
        .filter(|t| !t.ggml_type.is_supported())
        .map(|t| {
            format!(
                "Tensor '{}' has unsupported quant type {} — will be skipped",
                t.name, t.ggml_type
            )
        })
        .collect()
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_global_tensors() {
        assert_eq!(
            parse_tensor_name("token_embd.weight").0,
            TensorRole::Embedding
        );
        assert_eq!(parse_tensor_name("output.weight").0, TensorRole::LMHead);
        assert_eq!(
            parse_tensor_name("output_norm.weight").0,
            TensorRole::OutputNorm
        );
        assert_eq!(
            parse_tensor_name("rope_freqs.weight").0,
            TensorRole::RopeFreqs
        );
    }

    #[test]
    fn parse_layer_tensors() {
        let (role, layer) = parse_tensor_name("blk.17.attn_q.weight");
        assert_eq!(role, TensorRole::AttentionQ);
        assert_eq!(layer, Some(17));

        let (role, layer) = parse_tensor_name("blk.0.ffn_gate.weight");
        assert_eq!(role, TensorRole::FFNGate);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn parse_bias_tensors() {
        assert_eq!(
            parse_tensor_name("blk.3.attn_q.bias"),
            (TensorRole::AttentionQBias, Some(3))
        );
        assert_eq!(
            parse_tensor_name("blk.3.attn_k.bias"),
            (TensorRole::AttentionKBias, Some(3))
        );
        assert_eq!(
            parse_tensor_name("blk.3.attn_v.bias"),
            (TensorRole::AttentionVBias, Some(3))
        );
    }

    #[test]
    fn parse_qk_norm() {
        assert_eq!(
            parse_tensor_name("blk.5.attn_q_norm.weight"),
            (TensorRole::AttentionQNorm, Some(5))
        );
        assert_eq!(
            parse_tensor_name("blk.5.attn_k_norm.weight"),
            (TensorRole::AttentionKNorm, Some(5))
        );
    }

    #[test]
    fn parse_unknown_is_not_fatal() {
        let (role, layer) = parse_tensor_name("blk.1.weird_thing.weight");
        assert!(matches!(role, TensorRole::Unknown(_)));
        assert_eq!(layer, Some(1));

        let (role, layer) = parse_tensor_name("totally_unexpected");
        assert!(matches!(role, TensorRole::Unknown(_)));
        assert_eq!(layer, None);
    }

    #[test]
    fn ggml_type_support_flags() {
        assert!(GgmlType::Q4_K.is_supported());
        assert!(GgmlType::Q6_K.is_supported());
        assert!(GgmlType::Q4_0.is_supported());
        assert!(GgmlType::F16.is_supported());
        assert!(!GgmlType::Q3_K.is_supported());
        assert!(!GgmlType::Q5_K.is_supported());
    }
}
