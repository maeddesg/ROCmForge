//! BPE tokenizer module.
//!
//! Provides text tokenization for Qwen2/3 and Llama-3 models using
//! byte-pair encoding with per-architecture pre-tokenizer presets.

mod bpe;

pub use bpe::{BpeTokenizer, PreTokenizerType, TokenizerPreset, VocabType};
