//! BPE tokenizer for v1.0. Thin adapter around the v0.x
//! `rocmforge::tokenizer::BpeTokenizer` (786 lines of byte-level BPE
//! logic with Qwen2 / Llama-BPE presets — battle-tested on the three
//! Phase-1 target models). The adapter translates v1's typed GGUF
//! metadata into v0.x's `TokenizerData`, calls `from_gguf`, and
//! exposes the subset of the API that the inference pipeline needs.
//!
//! Chat-template formatting is done in pure Rust via a small
//! architecture dispatch — the GGUF ships a Jinja2 string that a full
//! Jinja2 parser would be overkill for. All three Phase-1 models use
//! one of two layouts: ChatML (Qwen) or Llama-3 header blocks.

use std::collections::HashMap;

use crate::loader::TokenizerData as V0xTokenizerData;
use crate::tokenizer::BpeTokenizer as V0xBpe;

use super::gguf::GgufValue;

/// Phase-1 Tokenizer — wraps the v0.x BPE implementation.
pub struct Tokenizer {
    inner: V0xBpe,
    architecture: String,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Build from v1 GGUF metadata. Returns an error only when the
    /// metadata doesn't carry the minimum fields a BPE vocabulary needs.
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, GgufValue>,
        architecture: &str,
    ) -> Result<Self, String> {
        // Required: vocab array of strings.
        let tokens_val = metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| "missing tokenizer.ggml.tokens".to_string())?;
        let tokens_strs = match tokens_val {
            GgufValue::ArrayString(v) => v.clone(),
            other => {
                return Err(format!(
                    "tokenizer.ggml.tokens has unexpected type {other:?}"
                ))
            }
        };
        // Convert strings to Vec<Vec<u8>> — tokenizer stores raw bytes
        // so non-UTF-8 tokens (common for byte-fallback) survive the
        // roundtrip.
        let tokens: Vec<Vec<u8>> =
            tokens_strs.iter().map(|s| s.as_bytes().to_vec()).collect();

        // Merges — the GGUF stores "first second" space-joined lines.
        let merges = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| match v {
                GgufValue::ArrayString(arr) => Some(arr),
                _ => None,
            })
            .map(|arr| {
                arr.iter()
                    .filter_map(|line| {
                        let mut it = line.splitn(2, ' ');
                        let a = it.next()?;
                        let b = it.next()?;
                        Some((a.as_bytes().to_vec(), b.as_bytes().to_vec()))
                    })
                    .collect::<Vec<(Vec<u8>, Vec<u8>)>>()
            })
            .unwrap_or_default();

        let bos_token_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32().ok());
        let eos_token_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32().ok());
        let unk_token_id = metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32().ok());

        let model = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_string().ok())
            .map(|s| s.to_string());
        let pre = metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_string().ok())
            .map(|s| s.to_string());

        let add_bos = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool().ok());
        let add_eos = metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool().ok());

        let chat_template = metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_string().ok())
            .map(|s| s.to_string());

        let td = V0xTokenizerData {
            tokens,
            merges,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            model,
            pre,
            add_bos,
            add_eos,
        };
        let inner = V0xBpe::from_gguf(&td);

        Ok(Self {
            inner,
            architecture: architecture.to_string(),
            chat_template,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.inner.bos_id()
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.inner.eos_id()
    }

    /// Returns true if the token is any end-of-generation token
    /// (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, etc.).
    /// Qwen3 and Llama-3.1 both have multiple EOG candidates; v0.x
    /// tracks them in `eog_ids()`.
    pub fn is_eos(&self, id: u32) -> bool {
        self.inner.is_eog(id)
    }

    /// Encode text to token ids. `add_special=false` keeps control
    /// tokens (like `<|im_start|>`) out of the returned sequence unless
    /// the tokenizer configuration requests BOS.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.inner.encode(text, add_special)
    }

    /// Decode token ids back to text. `skip_special=true` drops control
    /// tokens; set to `false` when debugging chat-template output.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        self.inner.decode(tokens, skip_special)
    }

    /// Apply an architecture-appropriate chat template to a single
    /// user turn. Phase-1 hardcodes the two layouts ChatML (all Qwen
    /// variants) and Llama-3 header blocks. Returns the formatted
    /// prompt ready for `encode(..., add_special=true)`.
    pub fn apply_chat_template(&self, user_prompt: &str, system_prompt: Option<&str>) -> String {
        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        match self.architecture.as_str() {
            "llama" => format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            _ => format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{user_prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
        }
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn chat_template_raw(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
}
