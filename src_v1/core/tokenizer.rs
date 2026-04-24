//! BPE tokenizer for v1.0. Thin adapter around the v0.x
//! `rocmforge::tokenizer::BpeTokenizer` (786 lines of byte-level BPE
//! logic with Qwen2 / Llama-BPE presets — battle-tested on the three
//! Phase-1 target models). The adapter translates v1's typed GGUF
//! metadata into v0.x's `TokenizerData`, calls `from_gguf`, and
//! exposes the subset of the API that the inference pipeline needs.
//!
//! Chat-template formatting is done in pure Rust via a small
//! architecture dispatch — the GGUF ships a Jinja2 string that a full
//! Jinja2 parser would be overkill for. Phase-1 models covered ChatML
//! (Qwen) and Llama-3 header blocks; Phase-3 adds Mistral's `[INST]
//! ... [/INST]` flavour. `general.architecture = "llama"` is not
//! unique (Llama-2 / Llama-3 / Mistral / DeepSeek-Distill all use it)
//! so the template kind is resolved at construction time from
//! vocab_size + bos_id, see [`ChatTemplate::detect`].

use std::collections::HashMap;
use std::collections::HashSet;

use crate::loader::TokenizerData as V0xTokenizerData;
use crate::tokenizer::BpeTokenizer as V0xBpe;

use super::gguf::GgufValue;

/// Extra end-of-generation literals that the v0.x `is_eog()` list
/// misses. `<|endoftext|>` (id 151643 in Qwen3) has no underscore and
/// is distinct from `<|end_of_text|>`; a model that drops out with
/// this token would otherwise run to `max_tokens`.
const EXTRA_EOG_LITERALS: &[&str] = &[
    "<|endoftext|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end|>",
    "<end_of_turn>",
];

/// Concrete chat-template flavour. Resolved at construction time
/// from `general.architecture` + vocab size + BOS id, because the
/// `"llama"` arch tag in GGUF is shared by Llama-2, Llama-3, Mistral,
/// and DeepSeek-Distill — each of which needs a different wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Qwen2 / Qwen2.5 — ChatML, no `/no_think` directive.
    Qwen2,
    /// Qwen3 — ChatML + `/no_think` directive to suppress reasoning.
    Qwen3,
    /// Llama-3 / Llama-3.1 / DeepSeek-R1-Distill-Llama — header blocks.
    Llama3,
    /// Llama-2 chat — `<s>[INST] <<SYS>> … <</SYS>> … [/INST]`.
    /// Not currently exercised by a test model but kept for parity.
    Llama2,
    /// Mistral-7B-Instruct v0.1 / v0.2 / v0.3 — `[INST] … [/INST]`,
    /// no separate system block (any system prompt is folded into
    /// the first user turn).
    MistralV3,
    /// Gemma-2 / Gemma-3 — `<start_of_turn>role\n…<end_of_turn>`.
    Gemma,
    /// Last-resort ChatML (unknown architecture).
    Generic,
}

impl ChatTemplate {
    fn detect(architecture: &str, vocab_size: usize, bos_id: Option<u32>) -> Self {
        match architecture {
            "qwen3" => ChatTemplate::Qwen3,
            "qwen2" => ChatTemplate::Qwen2,
            // Gemma-1, -2, -3 all share the same prompt wrapping.
            "gemma" | "gemma2" | "gemma3" | "gemma4" => ChatTemplate::Gemma,
            // `llama` is ambiguous — disambiguate via vocab + BOS.
            "llama" => match (vocab_size, bos_id) {
                // Llama-3 / 3.1 / DeepSeek-R1-Distill-Llama share the
                // 128k vocab + BOS 128000.
                (128256, Some(128000)) => ChatTemplate::Llama3,
                // Mistral-7B-Instruct v0.3.
                (32768, Some(1)) => ChatTemplate::MistralV3,
                // Llama-2 — 32k vocab, BOS 1.
                (32000, Some(1)) => ChatTemplate::Llama2,
                _ => {
                    eprintln!(
                        "  ⚠ Unknown llama-variant (vocab={vocab_size}, bos={bos_id:?}), \
                         defaulting to Llama3 chat template",
                    );
                    ChatTemplate::Llama3
                }
            },
            _ => ChatTemplate::Generic,
        }
    }
}

/// Phase-1 Tokenizer — wraps the v0.x BPE implementation.
pub struct Tokenizer {
    inner: V0xBpe,
    architecture: String,
    chat_template_kind: ChatTemplate,
    chat_template: Option<String>,
    /// Supplementary EOG ids resolved from the vocab at load time.
    /// Takes the union of `inner.eog_ids()` and whatever extra literals
    /// we can find in the vocabulary.
    extra_eog: HashSet<u32>,
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
            tokens: tokens.clone(),
            merges,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            model,
            pre,
            add_bos,
            add_eos,
        };
        let mut inner = V0xBpe::from_gguf(&td);

        // Register control / user-defined tokens from
        // `tokenizer.ggml.token_type` as "special" so the BPE does not
        // byte-fragment literals like `[INST]`, `[/INST]`,
        // `[TOOL_CALLS]`. v0.x's `<…>` heuristic misses the bracket
        // family. GGUF token-type encoding: 3 = CONTROL, 4 = USER_DEFINED
        // (per llama.cpp's `LLAMA_TOKEN_TYPE_*`).
        if let Some(token_types) = metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| match v {
                GgufValue::ArrayI32(a) => Some(a),
                _ => None,
            })
        {
            let extra: Vec<u32> = token_types
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == 3 || **t == 4)
                .map(|(i, _)| i as u32)
                .collect();
            if !extra.is_empty() {
                inner.register_special_token_ids(&extra);
            }
        }

        // Walk the vocab once and collect ids of any literal that
        // should terminate generation. v0.x's `is_eog()` handles most
        // of these, but it misses `<|endoftext|>` (Qwen3 id 151643).
        let mut extra_eog: HashSet<u32> = HashSet::new();
        for (i, bytes) in tokens.iter().enumerate() {
            if let Ok(s) = std::str::from_utf8(bytes) {
                if EXTRA_EOG_LITERALS.contains(&s) {
                    extra_eog.insert(i as u32);
                }
            }
        }
        if let Some(id) = eos_token_id {
            extra_eog.insert(id);
        }

        let chat_template_kind =
            ChatTemplate::detect(architecture, tokens.len(), bos_token_id);
        eprintln!(
            "  chat_template: {:?} (arch={architecture}, vocab={}, bos={:?})",
            chat_template_kind,
            tokens.len(),
            bos_token_id,
        );

        Ok(Self {
            inner,
            architecture: architecture.to_string(),
            chat_template_kind,
            chat_template,
            extra_eog,
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
    /// tracks most in `eog_ids()`, and `extra_eog` covers the rest.
    pub fn is_eos(&self, id: u32) -> bool {
        self.inner.is_eog(id) || self.extra_eog.contains(&id)
    }

    /// List every token id that should terminate generation. Useful
    /// for GPU-side stop-token lookup tables.
    pub fn eog_ids(&self) -> Vec<u32> {
        let mut ids: HashSet<u32> = self.inner.eog_ids().into_iter().collect();
        for id in &self.extra_eog {
            ids.insert(*id);
        }
        let mut out: Vec<u32> = ids.into_iter().collect();
        out.sort();
        out
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
    /// user turn. Phase-1 hardcodes two layouts: ChatML (Qwen variants)
    /// and Llama-3 header blocks.
    ///
    /// Qwen3 is a reasoning model that emits a `<think>…</think>`
    /// preamble by default. For Phase-1 validation we want the answer
    /// directly, so the user turn is suffixed with `/no_think` — a
    /// directive Qwen3 recognises to suppress the reasoning block. If
    /// the model still leaks a think block, `strip_think_block()`
    /// cleans up the decoded output as a backup.
    pub fn apply_chat_template(&self, user_prompt: &str, system_prompt: Option<&str>) -> String {
        let system = system_prompt.unwrap_or("You are a helpful assistant.");
        match self.chat_template_kind {
            ChatTemplate::Llama3 => format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            ChatTemplate::Qwen3 => format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{user_prompt} /no_think<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            ChatTemplate::Qwen2 | ChatTemplate::Generic => format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{user_prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            // Mistral v0.1 / v0.2 / v0.3 — no separate system block.
            // BOS (`<s>`) is prepended by `encode(..., add_special=true)`.
            // The HuggingFace reference chat template folds any system
            // prompt into the first [INST] block; we prepend it with a
            // blank line separator so downstream tokenisation stays
            // close to the reference.
            ChatTemplate::MistralV3 => {
                if system_prompt.is_some() {
                    format!("[INST] {system}\n\n{user_prompt} [/INST]")
                } else {
                    format!("[INST] {user_prompt} [/INST]")
                }
            }
            ChatTemplate::Llama2 => format!(
                "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_prompt} [/INST]"
            ),
            ChatTemplate::Gemma => format!(
                "<start_of_turn>user\n{user_prompt}<end_of_turn>\n\
                 <start_of_turn>model\n"
            ),
        }
    }

    /// Phase 2.4 — continuation template for a follow-up turn in a
    /// multi-turn conversation. The system prompt and the first user
    /// turn are already in the KV cache; this returns only the text
    /// that needs to be prefilled to add a new user message: an
    /// `<|im_end|>` to close the previous (unfinished — no EOS was
    /// committed to the cache) assistant turn, followed by a fresh
    /// user-turn wrapper and the opener for the next assistant turn.
    ///
    /// For Llama-family models this is the `<|eot_id|>` + new
    /// user/assistant header block without a fresh system prompt.
    ///
    /// `apply_chat_template(prompt, None)` is still the right call
    /// for the **first** turn of a conversation; this method is only
    /// for turns 2+.
    pub fn apply_chat_template_continuation(&self, user_prompt: &str) -> String {
        match self.chat_template_kind {
            ChatTemplate::Llama3 => format!(
                "<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            ChatTemplate::Qwen3 => format!(
                "<|im_end|>\n\
                 <|im_start|>user\n{user_prompt} /no_think<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            ChatTemplate::Qwen2 | ChatTemplate::Generic => format!(
                "<|im_end|>\n\
                 <|im_start|>user\n{user_prompt}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            // Close the previous assistant turn with `</s>`, then open
            // a fresh [INST] block. Reference Mistral multi-turn string
            // is `<s>[INST] t1 [/INST] a1</s>[INST] t2 [/INST] …` —
            // the continuation pays the `</s>` and the `[INST]` wrapper.
            ChatTemplate::MistralV3 | ChatTemplate::Llama2 => {
                format!("</s>[INST] {user_prompt} [/INST]")
            }
            ChatTemplate::Gemma => format!(
                "<end_of_turn>\n\
                 <start_of_turn>user\n{user_prompt}<end_of_turn>\n\
                 <start_of_turn>model\n"
            ),
        }
    }

    /// Strip any `<think>…</think>` block from a decoded assistant
    /// reply. Qwen3's reasoning output lives inside those tags; the
    /// validation suite wants only the final answer. Also trims the
    /// leading whitespace left behind after the closing tag.
    pub fn strip_think_block(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(open) = rest.find("<think>") {
            out.push_str(&rest[..open]);
            match rest[open..].find("</think>") {
                Some(rel_close) => {
                    rest = &rest[open + rel_close + "</think>".len()..];
                }
                None => {
                    // Unterminated think block — drop everything from
                    // here. A truncated model output would otherwise
                    // dump the whole reasoning trace.
                    return out.trim_start().to_string();
                }
            }
        }
        out.push_str(rest);
        out.trim_start().to_string()
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn chat_template_raw(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
}
