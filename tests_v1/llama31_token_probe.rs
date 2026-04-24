//! C1 diagnostic — encode the Llama-3.1 special tokens and a full
//! Turn-1 chat template with our tokenizer, report the produced IDs
//! so we can compare against the expected single-token IDs.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::tokenizer::Tokenizer;

fn expected_id(md: &std::collections::HashMap<String, rocmforge::v1::core::gguf::GgufValue>,
               literal: &str) -> Option<u32> {
    // Walk tokenizer.ggml.tokens, return the index of the exact literal
    // if present. Used to cross-check that the vocab actually contains
    // these tokens before complaining that our encoder can't find them.
    if let Some(rocmforge::v1::core::gguf::GgufValue::ArrayString(arr))
        = md.get("tokenizer.ggml.tokens")
    {
        for (i, s) in arr.iter().enumerate() {
            if s == literal {
                return Some(i as u32);
            }
        }
    }
    None
}

#[test]
fn c1_llama31_special_token_encoding() {
    let path = dirs::home_dir()
        .expect("HOME")
        .join("models")
        .join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skipping — {} missing", path.display());
        return;
    }
    let gguf = GGUFFile::open(&path).expect("open gguf");
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("cfg");
    let tok = Tokenizer::from_gguf_metadata(gguf.metadata(), &cfg.architecture)
        .expect("tokenizer");

    println!("\n=== C1 · Llama-3.1 Special-Token-Probe ===");
    println!("arch = {}", cfg.architecture);

    // ── Vocab presence + expected IDs ─────────────────────────────────
    let specials = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|finetune_right_pad_id|>",
    ];
    println!("\n--- Vocab presence (ground truth = vocab index) ---");
    for s in &specials {
        match expected_id(gguf.metadata(), s) {
            Some(id) => println!("  {s:30} = vocab[{id}]"),
            None => println!("  {s:30} = NOT IN VOCAB"),
        }
    }

    // ── Encode each literal in isolation ──────────────────────────────
    println!("\n--- Encode each literal (add_bos=false) ---");
    let mut any_split = false;
    for s in &specials {
        let ids = tok.encode(s, false);
        let expected = expected_id(gguf.metadata(), s);
        let ok = ids.len() == 1 && Some(ids[0]) == expected;
        if !ok { any_split = true; }
        println!("  {s:30} → {ids:?}  expected=[{:?}]  {}",
                 expected,
                 if ok { "OK" } else { "SPLIT/MISSING" });
    }

    // ── Encode the full Turn-1 template (no add_bos — template has BOS) ──
    let turn1 = "<|begin_of_text|>\
<|start_header_id|>user<|end_header_id|>\n\n\
My name is Alice<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n";
    println!("\n--- Full Turn-1 template (len bytes = {}) ---", turn1.len());
    println!("{turn1}");
    let full = tok.encode(turn1, false);
    println!("\nour encoder → {} tokens:", full.len());
    println!("{:?}", full);
    println!("\nexpected (from prompt): [128000, 128006, 882, 128007, 271, 5765, 836, 374, 30505, 128009, 128006, 78191, 128007, 271]");
    println!("                        = 14 tokens, special-tokens are single IDs");

    // Decode back to see what the tokens actually are. skip_special=false
    // so we can eyeball the split (if any) in the role strings and body.
    println!("\ndecoded round-trip: {:?}", tok.decode(&full, false));

    // ── Compact verdict ───────────────────────────────────────────────
    println!("\n--- Verdict ---");
    if any_split {
        println!("  At least one special token WAS SPLIT — this is the root cause.");
    } else {
        println!("  Special tokens encode as single IDs — C1 root-cause HYPOTHESIS RULED OUT.");
    }
}
