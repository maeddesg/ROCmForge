//! Architecture-agnostic model configuration, **derived entirely from
//! GGUF metadata + tensor inventory** — no per-model hardcoding.
//!
//! Robust key resolution: the primary GGUF key is `{arch}.{name}`, but
//! some producers ship slightly different names (`hidden_size` instead
//! of `embedding_length`, `rope_theta` instead of `rope.freq_base`,
//! etc.). [`get_arch_key`] tries the canonical name first and falls
//! back to known aliases, so all three Phase-1 models load cleanly.
//!
//! Feature detection (`has_qk_norm`, `has_rope_freqs`, `is_gqa`) is
//! driven off the tensor inventory, not metadata — a flag in the
//! metadata can lie, the presence of an actual tensor cannot.

use std::collections::HashMap;

use super::gguf::{GgufError, GgufResult, GgufValue};
use super::tensor_info::{GgmlType, TensorInfo};

/// Resolve a GGUF metadata key of the form `{arch}.{name}`, trying a
/// small set of known aliases so that producers with slightly different
/// naming conventions still load.
pub fn get_arch_key<'a>(
    metadata: &'a HashMap<String, GgufValue>,
    arch: &str,
    name: &str,
) -> Option<&'a GgufValue> {
    let primary = format!("{arch}.{name}");
    if let Some(v) = metadata.get(&primary) {
        return Some(v);
    }

    let aliases: &[&str] = match name {
        "block_count" => &["layers", "num_hidden_layers", "n_layers"],
        "embedding_length" => &["hidden_size", "n_embd"],
        "feed_forward_length" => &["intermediate_size", "ffn_dim"],
        "attention.head_count" => &["num_attention_heads", "n_heads"],
        "attention.head_count_kv" => &["num_key_value_heads", "n_heads_kv"],
        "rope.freq_base" => &["rope_theta"],
        "vocab_size" => &[],
        "context_length" => &["max_position_embeddings", "n_ctx"],
        "attention.layer_norm_rms_epsilon" => &[
            "rms_norm_eps",
            "attention_norm_epsilon",
            "layer_norm_rms_epsilon",
        ],
        "attention.key_length" => &["head_dim"],
        _ => &[],
    };
    for alias in aliases {
        let key = format!("{arch}.{alias}");
        if let Some(v) = metadata.get(&key) {
            return Some(v);
        }
    }

    // Last resort: cross-architecture fallback via bare `llama.*` —
    // matches the v0.x `GgufMetadata::get` behaviour.
    if arch != "llama" {
        let llama_primary = format!("llama.{name}");
        if let Some(v) = metadata.get(&llama_primary) {
            return Some(v);
        }
        for alias in aliases {
            let key = format!("llama.{alias}");
            if let Some(v) = metadata.get(&key) {
                return Some(v);
            }
        }
    }

    None
}

/// Missing-key error with a stable message shape for callers.
fn missing(key: impl Into<String>) -> GgufError {
    GgufError::MissingKey(key.into())
}

/// Fully derived model configuration. Every field is either read from
/// GGUF metadata or computed from the tensor inventory — no per-model
/// match statements.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Architecture.
    pub architecture: String,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub context_length: usize,
    pub rms_norm_eps: f32,

    // RoPE.
    pub rope_freq_base: f32,
    pub rope_scaling_type: Option<String>,
    pub has_rope_freqs: bool,

    // Auto-detected features (from tensor inventory).
    pub has_qk_norm: bool,
    pub has_attention_bias: bool,
    pub is_gqa: bool,

    // Tokenizer.
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub chat_template: Option<String>,

    // Inventory.
    pub quant_formats_used: Vec<GgmlType>,
}

impl ModelConfig {
    pub fn from_metadata(
        metadata: &HashMap<String, GgufValue>,
        tensors: &[TensorInfo],
    ) -> GgufResult<Self> {
        let architecture = metadata
            .get("general.architecture")
            .ok_or_else(|| missing("general.architecture"))?
            .as_string()
            .map_err(|_| GgufError::TypeMismatch {
                key: "general.architecture".into(),
                expected: "string",
            })?
            .to_string();

        let n_layers = get_arch_key(metadata, &architecture, "block_count")
            .ok_or_else(|| missing(format!("{architecture}.block_count")))?
            .as_u64()? as usize;

        let n_heads = get_arch_key(metadata, &architecture, "attention.head_count")
            .ok_or_else(|| missing(format!("{architecture}.attention.head_count")))?
            .as_u64()? as usize;

        // GQA default: kv_heads = heads when metadata is silent.
        let n_kv_heads = match get_arch_key(metadata, &architecture, "attention.head_count_kv") {
            Some(v) => v.as_u64()? as usize,
            None => n_heads,
        };

        let hidden_dim = get_arch_key(metadata, &architecture, "embedding_length")
            .ok_or_else(|| missing(format!("{architecture}.embedding_length")))?
            .as_u64()? as usize;

        let ffn_dim = get_arch_key(metadata, &architecture, "feed_forward_length")
            .ok_or_else(|| missing(format!("{architecture}.feed_forward_length")))?
            .as_u64()? as usize;

        // Vocab: explicit metadata if present, otherwise derive from
        // the embedding or output tensor (outer dimension).
        let vocab_size = if let Some(v) = get_arch_key(metadata, &architecture, "vocab_size") {
            v.as_u64()? as usize
        } else {
            tensors
                .iter()
                .find(|t| t.name == "token_embd.weight" || t.name == "output.weight")
                .map(|t| {
                    // GGUF shape innermost-first: [hidden, vocab] for
                    // the embedding / lm_head projections.
                    *t.shape.last().unwrap_or(&0) as usize
                })
                .unwrap_or(0)
        };

        // head_dim: explicit first, otherwise computed.
        let head_dim = if let Some(v) = get_arch_key(metadata, &architecture, "attention.key_length") {
            v.as_u64()? as usize
        } else if n_heads > 0 {
            hidden_dim / n_heads
        } else {
            0
        };

        let context_length = get_arch_key(metadata, &architecture, "context_length")
            .and_then(|v| v.as_u64().ok())
            .map(|n| n as usize)
            .unwrap_or(2048);

        let rms_norm_eps = get_arch_key(metadata, &architecture, "attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32().ok())
            .unwrap_or(1e-5);

        let rope_freq_base = get_arch_key(metadata, &architecture, "rope.freq_base")
            .and_then(|v| v.as_f32().ok())
            .unwrap_or(10_000.0);

        let rope_scaling_type = get_arch_key(metadata, &architecture, "rope.scaling.type")
            .and_then(|v| v.as_string().ok())
            .map(|s| s.to_string());

        // Feature detection from tensor names (name-substring match; the
        // name parser lives in `tensor_info`, but detecting features
        // here avoids a dependency cycle).
        let has_qk_norm = tensors
            .iter()
            .any(|t| t.name.contains("attn_q_norm") || t.name.contains("attn_k_norm"));
        let has_rope_freqs = tensors.iter().any(|t| t.name == "rope_freqs.weight");
        let has_attention_bias = tensors.iter().any(|t| {
            t.name.ends_with("attn_q.bias")
                || t.name.ends_with("attn_k.bias")
                || t.name.ends_with("attn_v.bias")
        });
        let is_gqa = n_heads != n_kv_heads;

        // Tokenizer.
        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32().ok());
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32().ok());
        let chat_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_string().ok())
            .map(|s| s.to_string());

        // Quant inventory (sorted for deterministic output).
        let mut quant_formats_used: Vec<GgmlType> = tensors
            .iter()
            .map(|t| t.ggml_type)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        quant_formats_used.sort_by_key(|t| *t as u32);

        Ok(Self {
            architecture,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            ffn_dim,
            vocab_size,
            head_dim,
            context_length,
            rms_norm_eps,
            rope_freq_base,
            rope_scaling_type,
            has_rope_freqs,
            has_qk_norm,
            has_attention_bias,
            is_gqa,
            bos_token_id,
            eos_token_id,
            chat_template,
            quant_formats_used,
        })
    }
}
