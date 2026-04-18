//! Tokenizer parity check: ROCmForge's `llama-bpe` preset must produce
//! byte-identical token-ID sequences to llama.cpp on the Meta Llama-3.1
//! GGUF. Reference IDs were extracted with
//! `llama-tokenize --ids -p '<prompt>'` on the same model file.
//!
//! The test is gated on the GGUF being present at `~/models`; CI runners
//! without the weights just skip. Set `ROCMFORGE_SKIP_LLAMA_PARITY=1` to
//! force-skip even when the file is there (useful for offline work).

use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn llama_model_path() -> Option<PathBuf> {
    if std::env::var("ROCMFORGE_SKIP_LLAMA_PARITY").is_ok() {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

fn tokenizer_from(path: &PathBuf) -> BpeTokenizer {
    let file = GgufFile::open(path).expect("open llama-3.1 gguf");
    BpeTokenizer::from_gguf(file.tokenizer_data())
}

fn assert_parity(tok: &BpeTokenizer, text: &str, expected: &[u32]) {
    let got = tok.encode(text, true);
    assert_eq!(
        got, expected,
        "tokenizer mismatch for input {:?}\n  expected: {:?}\n  got:      {:?}",
        text, expected, got
    );
}

/// Golden token-ID sequences produced by llama.cpp's `llama-tokenize`
/// on Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf. BOS (128000) is the
/// first element in each sequence, added by `add_bos`.
const GOLDEN: &[(&str, &[u32])] = &[
    ("Hello, world!", &[128000, 9906, 11, 1917, 0]),
    (
        "The capital of France is",
        &[128000, 791, 6864, 315, 9822, 374],
    ),
    (
        "I'm trying to create",
        &[128000, 40, 2846, 4560, 311, 1893],
    ),
    (
        "123 + 456 = 579",
        &[128000, 4513, 489, 220, 10961, 284, 220, 24847],
    ),
    (
        "Héllo wörld",
        &[128000, 39, 19010, 385, 289, 9603, 509],
    ),
    (
        "  multiple   spaces  ",
        &[128000, 220, 5361, 256, 12908, 256],
    ),
    (
        "def foo():\n    return 42",
        &[128000, 755, 15586, 4019, 262, 471, 220, 2983],
    ),
];

#[test]
fn llama_bpe_matches_llama_cpp_token_ids() {
    let Some(path) = llama_model_path() else {
        eprintln!("Skipping: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf not found under $HOME/models");
        return;
    };
    let tok = tokenizer_from(&path);
    for (text, expected) in GOLDEN {
        assert_parity(&tok, text, expected);
    }
}

#[test]
fn llama_bpe_prepends_single_bos_with_add_special() {
    let Some(path) = llama_model_path() else {
        eprintln!("Skipping: Llama-3.1 GGUF not available");
        return;
    };
    let tok = tokenizer_from(&path);
    let ids = tok.encode("Hello", true);
    assert_eq!(ids.first(), Some(&128000_u32), "expected BOS first");
    // And exactly one BOS — not doubled.
    let bos_count = ids.iter().filter(|&&id| id == 128000).count();
    assert_eq!(bos_count, 1, "BOS appeared {} times in {:?}", bos_count, ids);
}

#[test]
fn llama_bpe_skips_bos_without_add_special() {
    let Some(path) = llama_model_path() else {
        eprintln!("Skipping: Llama-3.1 GGUF not available");
        return;
    };
    let tok = tokenizer_from(&path);
    let ids = tok.encode("Hello", false);
    assert_ne!(ids.first(), Some(&128000_u32), "BOS must not appear when add_special=false");
}
