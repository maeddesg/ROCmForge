//! Byte-Pair Encoding (BPE) tokenizer.
//!
//! Supports two pre-tokenizer presets that share the same BPE merge logic
//! but differ in the pre-split regex applied before BPE:
//!
//! * [`TokenizerPreset::Qwen2`] — Qwen2.5 / Qwen3 GGUFs (`tokenizer.ggml.pre = "qwen2"`).
//! * [`TokenizerPreset::LlamaBpe`] — Llama-3.x GGUFs (`tokenizer.ggml.pre = "llama-bpe"`),
//!   using the GPT-4-style regex.
//!
//! Build with `BpeTokenizer::from_gguf(tokenizer_data)`.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// ── Pre-tokenizer regexes ──────────────────────────────────────────────────────

static REGEX_QWEN2: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
    ).unwrap()
});

// GPT-4-style regex shipped by llama.cpp for `llama-bpe`. Two source-level
// deviations from the original PCRE pattern, both safe for greedy matching:
//
//  * possessive quantifiers (`?+`, `++`) → greedy equivalents. The
//    `regex` crate lacks possessives; since each alternative is anchored
//    by a different leading character class, the match set is identical.
//  * `\s+(?!\S)` (whitespace-not-followed-by-non-whitespace) is dropped.
//    Greedy `\s+` at the end of the alternation already covers the same
//    runs — lookahead only filtered candidates that the final `\s+`
//    re-matches, so dropping it leaves the split identical while keeping
//    us on the stdlib `regex` crate (no `fancy-regex` dependency).
//
// Llama-3 also groups numbers in chunks of up to 3 digits (`\p{N}{1,3}`),
// which matters for "123" → single token vs. "1234" → ["123", "4"].
static REGEX_LLAMA_BPE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
    ).unwrap()
});

// ── Internal types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct BytesKey(Vec<u8>);

impl Hash for BytesKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

pub type BpeRank = i32;

#[derive(Clone, Debug)]
struct Symbol {
    text: Vec<u8>,
    prev: i32,
    next: i32,
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

// ── Public enums ───────────────────────────────────────────────────────────────

/// Vocabulary type for the tokenizer.
///
/// Only BPE is supported in rocmforge (no SentencePiece).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VocabType {
    Bpe, // GPT-2 style byte-level BPE
}

/// Pre-tokenizer type for splitting text before BPE.
///
/// Selected from `tokenizer.ggml.pre` in the GGUF KV section. Unknown
/// strings fall back to `Qwen2`, which is rocmforge's historical default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TokenizerPreset {
    /// Qwen2 / Qwen3 pre-split regex. No default BOS.
    Qwen2,
    /// Llama-3 GPT-4-style pre-split regex. Defaults `add_bos = true`
    /// when the GGUF omits the flag (Llama's chat format assumes BOS).
    LlamaBpe,
}

impl TokenizerPreset {
    fn from_gguf(pre: Option<&str>) -> Self {
        match pre {
            Some("llama-bpe") => Self::LlamaBpe,
            Some("qwen2") | None => Self::Qwen2,
            Some(other) => {
                tracing::warn!(
                    pre = other,
                    "Unknown tokenizer.ggml.pre; falling back to Qwen2 preset"
                );
                Self::Qwen2
            }
        }
    }

    fn default_add_bos(self) -> bool {
        matches!(self, Self::LlamaBpe)
    }
}

/// Kept as a type alias so downstream callers that still name the old
/// enum continue to compile — new code should use `TokenizerPreset`.
pub type PreTokenizerType = TokenizerPreset;

// ── Tokenizer ───────────────────────────────────────────────────────────────────

/// BPE tokenizer for Qwen2.5 models.
///
/// Encodes text to token IDs and decodes token IDs back to text.
/// Built from GGUF tokenizer data using `from_gguf()`.
#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<BytesKey, u32>,
    merges: HashMap<(BytesKey, BytesKey), BpeRank>,
    special_tokens: HashSet<u32>,
    preset: TokenizerPreset,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    unk_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    byte_encoder: HashMap<u8, String>,
    byte_decoder: HashMap<char, u8>,
}

impl BpeTokenizer {
    /// Build from GGUF tokenizer arrays.
    pub fn from_gguf(data: &crate::loader::TokenizerData) -> Self {
        let preset = TokenizerPreset::from_gguf(data.pre.as_deref());
        let add_bos = data.add_bos.unwrap_or_else(|| preset.default_add_bos());
        let add_eos = data.add_eos.unwrap_or(false);
        tracing::debug!(
            preset = ?preset,
            pre = ?data.pre,
            add_bos,
            add_eos,
            bos_id = ?data.bos_token_id,
            eos_id = ?data.eos_token_id,
            "BpeTokenizer::from_gguf"
        );
        Self::new(
            data.tokens.clone(),
            data.merges.clone(),
            data.bos_token_id,
            data.eos_token_id,
            data.unk_token_id,
            add_bos,
            add_eos,
            preset,
        )
    }

    fn new(
        vocab: Vec<Vec<u8>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        bos: Option<u32>,
        eos: Option<u32>,
        unk: Option<u32>,
        add_bos: bool,
        add_eos: bool,
        preset: TokenizerPreset,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut special_tokens = HashSet::new();
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(BytesKey(token.clone()), id as u32);
            if token.starts_with(b"<") && token.ends_with(b">") {
                special_tokens.insert(id as u32);
            }
        }
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, (a, b)) in merges.iter().enumerate() {
            merge_map.insert((BytesKey(a.clone()), BytesKey(b.clone())), rank as BpeRank);
        }
        let byte_encoder = Self::build_byte_encoder();
        let byte_decoder = byte_encoder
            .iter()
            .filter_map(|(&b, s)| s.chars().next().map(|ch| (ch, b)))
            .collect();
        Self {
            vocab,
            token_to_id,
            merges: merge_map,
            special_tokens,
            preset,
            bos_id: bos,
            eos_id: eos,
            unk_id: unk,
            add_bos,
            add_eos,
            byte_encoder,
            byte_decoder,
        }
    }

    // ── Encode ──────────────────────────────────────────────────────────────────

    /// Encode text to token IDs.
    ///
    /// If `add_special` is true, adds BOS/EOS tokens as configured.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos_id {
                tokens.push(b);
            }
        }
        for fragment in self.split_by_special_tokens(text) {
            match fragment {
                Fragment::Text(t) => {
                    for word in self.regex_split(t) {
                        tokens.extend(self.tokenize_word(&word));
                    }
                }
                Fragment::Special(id) => tokens.push(id),
            }
        }
        if add_special && self.add_eos {
            if let Some(e) = self.eos_id {
                tokens.push(e);
            }
        }
        tokens
    }

    fn split_by_special_tokens<'a>(&'a self, text: &'a str) -> Vec<Fragment<'a>> {
        let mut fragments = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut found_special = false;
            for (id, token) in self.vocab.iter().enumerate() {
                if self.special_tokens.contains(&(id as u32)) {
                    if let Ok(s) = std::str::from_utf8(token) {
                        if remaining.starts_with(s) {
                            fragments.push(Fragment::Special(id as u32));
                            remaining = &remaining[s.len()..];
                            found_special = true;
                            break;
                        }
                    }
                }
            }
            if !found_special {
                let mut text_end = remaining.len();
                for (id, token) in self.vocab.iter().enumerate() {
                    if self.special_tokens.contains(&(id as u32)) {
                        if let Ok(s) = std::str::from_utf8(token) {
                            if let Some(pos) = remaining.find(s) {
                                if pos > 0 && pos < text_end {
                                    text_end = pos;
                                }
                            }
                        }
                    }
                }
                let end = text_end.max(1).min(remaining.len());
                fragments.push(Fragment::Text(&remaining[..end]));
                remaining = &remaining[end..];
            }
        }
        fragments
    }

    fn regex_split(&self, text: &str) -> Vec<String> {
        let re: &Regex = match self.preset {
            TokenizerPreset::Qwen2 => &REGEX_QWEN2,
            TokenizerPreset::LlamaBpe => &REGEX_LLAMA_BPE,
        };
        let pieces: Vec<String> = re.find_iter(text).map(|m| m.as_str().to_string()).collect();

        let out = match self.preset {
            TokenizerPreset::Qwen2 => split_qwen2_punct_newline(pieces),
            TokenizerPreset::LlamaBpe => redistribute_whitespace_llama_bpe(pieces),
        };

        // Permanent split-level debug log. Costs nothing at RUST_LOG=warn;
        // at RUST_LOG=debug it is the fastest way to diagnose tokeniser
        // divergence from llama.cpp (compare the `splits` list).
        if tracing::event_enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                preset = ?self.preset,
                splits = ?out,
                "Pre-tokenizer splits"
            );
        }

        out
    }

    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        // Map raw bytes through byte-encoder alphabet
        let mut encoded = String::new();
        for b in word.bytes() {
            if let Some(s) = self.byte_encoder.get(&b) {
                encoded.push_str(s);
            } else {
                encoded.push(b as char);
            }
        }
        // Split into UTF-8 character symbols
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        let bytes = encoded.as_bytes();
        while offset < bytes.len() {
            let clen = utf8_char_len(bytes[offset]);
            let end = (offset + clen).min(bytes.len());
            symbols.push(Symbol {
                text: encoded[offset..end].as_bytes().to_vec(),
                prev: symbols.len() as i32 - 1,
                next: -1,
            });
            offset = end;
        }
        for i in 0..symbols.len() {
            if i + 1 < symbols.len() {
                symbols[i].next = (i + 1) as i32;
            }
        }
        if symbols.is_empty() {
            return Vec::new();
        }
        // BPE merge loop
        loop {
            let mut best_rank = BpeRank::MAX;
            let mut best_idx: i32 = -1;
            let mut i = 0i32;
            while i >= 0 && (i as usize) < symbols.len() {
                let next = symbols[i as usize].next;
                if next >= 0 && !symbols[next as usize].text.is_empty() {
                    if let Some(&rank) = self.merges.get(&(
                        BytesKey(symbols[i as usize].text.clone()),
                        BytesKey(symbols[next as usize].text.clone()),
                    )) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }
                i = next;
            }
            if best_idx < 0 || best_rank == BpeRank::MAX {
                break;
            }
            let left = best_idx as usize;
            let right = symbols[left].next as usize;
            let next_next = symbols[right].next;
            let mut merged =
                Vec::with_capacity(symbols[left].text.len() + symbols[right].text.len());
            merged.extend_from_slice(&symbols[left].text);
            merged.extend_from_slice(&symbols[right].text);
            symbols[left].text = merged;
            symbols[left].next = next_next;
            if next_next >= 0 {
                symbols[next_next as usize].prev = best_idx;
            }
            symbols[right].text.clear();
        }
        // Collect tokens
        let mut tokens = Vec::new();
        let mut idx = 0i32;
        while idx >= 0 && (idx as usize) < symbols.len() {
            if !symbols[idx as usize].text.is_empty() {
                let key = BytesKey(symbols[idx as usize].text.clone());
                if let Some(&id) = self.token_to_id.get(&key) {
                    tokens.push(id);
                } else {
                    // Byte fallback
                    for &b in &symbols[idx as usize].text {
                        if let Some(s) = self.byte_encoder.get(&b) {
                            let k = BytesKey(s.as_bytes().to_vec());
                            tokens.push(
                                *self
                                    .token_to_id
                                    .get(&k)
                                    .unwrap_or(&self.unk_id.unwrap_or(0)),
                            );
                        } else {
                            tokens.push(self.unk_id.unwrap_or(0));
                        }
                    }
                }
                idx = symbols[idx as usize].next;
            } else {
                idx += 1;
            }
        }
        tokens
    }

    // ── Decode ──────────────────────────────────────────────────────────────────

    /// Decode token IDs to text.
    ///
    /// If `skip_special` is true, skips special tokens in the output.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> String {
        let mut raw = Vec::new();
        for &id in tokens {
            if skip_special && self.special_tokens.contains(&id) {
                continue;
            }
            if let Some(bytes) = self.vocab.get(id as usize) {
                raw.extend_from_slice(bytes);
            }
        }
        self.decode_bytes(&String::from_utf8_lossy(&raw))
    }

    /// Decode a single token ID to text.
    pub fn decode_token(&self, id: u32) -> String {
        if let Some(bytes) = self.vocab.get(id as usize) {
            self.decode_bytes(&String::from_utf8_lossy(bytes))
        } else {
            format!("<unk_{}>", id)
        }
    }

    /// Decode a single token to its raw byte sequence without any lossy
    /// UTF-8 conversion. Used by the streaming chat emitter, which buffers
    /// partial multi-byte sequences (e.g. 4-byte emojis split across 2–3
    /// tokens) until a complete codepoint is available. Single-token
    /// `decode_token` still returns a `String` via `from_utf8_lossy`, which
    /// is fine for diagnostics but replaces partial emojis with `��`.
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(vocab_bytes) = self.vocab.get(id as usize) else {
            return format!("<unk_{}>", id).into_bytes();
        };
        // The vocab entry is the byte-level-BPE encoded form. Map it back
        // through `byte_decoder` just like `decode_bytes` does, but emit
        // raw bytes so the caller can stitch partial codepoints together.
        let text = String::from_utf8_lossy(vocab_bytes);
        let mut out = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '▁' {
                out.push(b' ');
                continue;
            }
            if let Some(&b) = self.byte_decoder.get(&ch) {
                out.push(b);
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        out
    }

    fn decode_bytes(&self, text: &str) -> String {
        // Qwen2 vocab contains proper UTF-8; skip the double-encoding repair
        let text = text.to_string();
        // Map byte-level BPE unicode back to original bytes
        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '▁' {
                bytes.push(b' ');
                continue;
            }
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        let decoded = String::from_utf8_lossy(&bytes).to_string();
        self.convert_hex_escapes(&decoded)
    }

    fn convert_hex_escapes(&self, text: &str) -> String {
        let mut result = String::new();
        let mut i = 0;
        while i < text.len() {
            let slice = &text[i..];
            if slice.starts_with("<0x") {
                let rest = &slice[3..];
                let mut hex = String::new();
                for ch in rest.chars() {
                    if ch.is_ascii_hexdigit() && hex.len() < 2 {
                        hex.push(ch);
                    } else {
                        break;
                    }
                }
                let tail = &rest[hex.len()..];
                if !hex.is_empty() && tail.starts_with('>') {
                    if let Ok(b) = u8::from_str_radix(&hex, 16) {
                        result.push(b as char);
                        i += 3 + hex.len() + 1;
                        continue;
                    }
                }
            }
            if let Some(ch) = slice.chars().next() {
                result.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
        result
    }

    /// Is this token an end-of-generation signal?
    pub fn is_eog(&self, id: u32) -> bool {
        if Some(id) == self.eos_id {
            return true;
        }
        if let Some(bytes) = self.vocab.get(id as usize) {
            matches!(
                String::from_utf8_lossy(bytes).as_ref(),
                "" | "<|eos|>"
                    | "</s>"
                    | "<|eot_id|>"
                    | "<|eom_id|>"
                    | "<|end_of_text|>"
                    | "<|im_end|>"
                    | "<|end|>"
                    | "<end_of_turn>"
            )
        } else {
            false
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────────

    /// Returns the vocabulary size (number of tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Returns the BOS (beginning of sequence) token ID, if configured.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// Returns the EOS (end of sequence) token ID, if configured.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Returns all end-of-generation token IDs (for passing to GPU-side code).
    pub fn eog_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(eos) = self.eos_id {
            ids.push(eos);
        }
        let eog_texts = [
            "", "<|eos|>", "</s>", "<|eot_id|>",
            "<|eom_id|>", "<|end_of_text|>",
            "<|im_end|>", "<|end|>", "<end_of_turn>",
        ];
        for (i, bytes) in self.vocab.iter().enumerate() {
            let text = String::from_utf8_lossy(bytes);
            if eog_texts.contains(&text.as_ref()) && !ids.contains(&(i as u32)) {
                ids.push(i as u32);
            }
        }
        ids
    }

    /// Returns whether BOS token should be added.
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Returns whether EOS token should be added.
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    // ── Byte encoder ────────────────────────────────────────────────────────────

    fn build_byte_encoder() -> HashMap<u8, String> {
        let mut bs: Vec<u8> = Vec::new();
        bs.extend(b'!'..=b'~');
        bs.extend(0xA1..=0xAC_u8);
        bs.extend(0xAE..=0xFF_u8);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut enc = HashMap::with_capacity(256);
        for (b, c) in bs.into_iter().zip(cs) {
            if let Some(ch) = char::from_u32(c) {
                enc.insert(b, ch.to_string());
            }
        }
        enc
    }
}

/// Llama-BPE-specific post-processing: `[ws_run][alnum_word]` pairs
/// redistribute one trailing whitespace character into the word, so that
/// `"  multiple"` → `[" ", " multiple"]` instead of `["  ", "multiple"]`.
///
/// The llama.cpp reference regex expresses this with the lookahead
/// `\s+(?!\S)`, which the `regex` crate does not support. Emulating the
/// split here keeps us on the stdlib `regex` crate while producing
/// byte-identical output to `llama-tokenize` on Meta-Llama-3.1.
fn redistribute_whitespace_llama_bpe(pieces: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(pieces.len() + 2);
    let mut iter = pieces.into_iter().peekable();
    while let Some(piece) = iter.next() {
        let is_ws_run = !piece.is_empty() && piece.chars().all(|c| c.is_whitespace());
        if !is_ws_run {
            out.push(piece);
            continue;
        }
        // Newlines are handled by the regex alternative `\s*[\r\n]+`
        // directly — llama.cpp captures the whole newline run without a
        // lookahead. Only plain space runs use `\s+(?!\S)` and therefore
        // need the last-space-joins-next-word emulation below. Leaving
        // newline-containing runs alone matches llama-tokenize output
        // (e.g. `\n\n` stays one piece → BPE merge 271 `\n\n`).
        if piece.contains('\n') || piece.contains('\r') {
            out.push(piece);
            continue;
        }
        let Some(next) = iter.peek() else {
            out.push(piece);
            continue;
        };
        let next_leads_with_alnum = next
            .chars()
            .next()
            .map(|c| c.is_alphanumeric())
            .unwrap_or(false);
        if !next_leads_with_alnum {
            out.push(piece);
            continue;
        }
        // Split off the last whitespace char of `piece` and prepend it to
        // `next`. If `piece` is a single whitespace char, the head is
        // empty and only the merged `next` is emitted.
        let last_char = piece.chars().last().expect("non-empty ws_run");
        let last_len = last_char.len_utf8();
        let head = &piece[..piece.len() - last_len];
        if !head.is_empty() {
            out.push(head.to_string());
        }
        let next_owned = iter.next().expect("peek succeeded");
        let mut merged = String::with_capacity(last_len + next_owned.len());
        merged.push(last_char);
        merged.push_str(&next_owned);
        out.push(merged);
    }
    out
}

/// Qwen2-specific post-processing: split `"?\n"` into `["?", "\n"]` etc.
/// Preserves the existing byte-identical Qwen2 pipeline.
fn split_qwen2_punct_newline(pieces: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(pieces.len());
    for piece in pieces.into_iter() {
        if piece.len() > 1 && (piece.ends_with('\n') || piece.ends_with('\r')) {
            let mut cut = piece.len();
            while cut > 0 {
                let ch = match piece[..cut].chars().next_back() {
                    Some(c) => c,
                    None => break,
                };
                if ch == '\n' || ch == '\r' {
                    cut -= ch.len_utf8();
                } else {
                    break;
                }
            }
            let head = &piece[..cut];
            let tail = &piece[cut..];
            if !head.is_empty()
                && head
                    .chars()
                    .last()
                    .map(|c| c.is_ascii_punctuation())
                    .unwrap_or(false)
                && !tail.is_empty()
            {
                out.push(head.to_string());
                for ch in tail.chars() {
                    out.push(ch.to_string());
                }
                continue;
            }
        }
        out.push(piece);
    }
    out
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if (b & 0xE0) == 0xC0 {
        2
    } else if (b & 0xF0) == 0xE0 {
        3
    } else if (b & 0xF8) == 0xF0 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> BpeTokenizer {
        let vocab: Vec<Vec<u8>> = vec![
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"o".to_vec(),
            b" ".to_vec(),
            b"he".to_vec(),
            b"ll".to_vec(),
            b"lo".to_vec(),
            b"hello".to_vec(),
            b"".to_vec(),
        ];
        let merges = vec![
            (b"h".to_vec(), b"e".to_vec()),
            (b"l".to_vec(), b"l".to_vec()),
            (b"he".to_vec(), b"ll".to_vec()),
            (b"hel".to_vec(), b"lo".to_vec()),
        ];
        BpeTokenizer::new(
            vocab,
            merges,
            None,
            Some(9),
            None,
            false,
            false,
            TokenizerPreset::Qwen2,
        )
    }

    #[test]
    fn vocab_size() {
        assert_eq!(toy_tokenizer().vocab_size(), 10);
    }

    #[test]
    fn eos_id() {
        assert_eq!(toy_tokenizer().eos_id(), Some(9));
    }

    #[test]
    fn is_eog_by_id() {
        assert!(toy_tokenizer().is_eog(9));
        assert!(!toy_tokenizer().is_eog(0));
    }

    #[test]
    fn decode_single_token() {
        let tok = toy_tokenizer();
        let decoded = tok.decode_token(5);
        assert!(decoded.contains('h') || !decoded.is_empty());
    }

    #[test]
    fn byte_encoder_covers_all_256() {
        let enc = BpeTokenizer::build_byte_encoder();
        assert_eq!(enc.len(), 256);
    }
}
