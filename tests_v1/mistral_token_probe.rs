//! Mistral chat-template token probe — diagnose whether our BPE
//! encodes [INST] and [/INST] as single token IDs.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::tokenizer::Tokenizer;

#[test]
fn mistral_special_token_encoding() {
    let path = dirs::home_dir()
        .expect("HOME")
        .join("models")
        .join("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("skipping — {} missing", path.display());
        return;
    }
    let gguf = GGUFFile::open(&path).expect("open gguf");
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("cfg");
    let tok = Tokenizer::from_gguf_metadata(gguf.metadata(), &cfg.architecture)
        .expect("tokenizer");

    println!("\n=== Mistral · Special-Token-Probe ===");

    // Find the vocab indices of [INST] and [/INST].
    if let Some(rocmforge::v1::core::gguf::GgufValue::ArrayString(arr))
        = gguf.metadata().get("tokenizer.ggml.tokens")
    {
        for (i, s) in arr.iter().enumerate().take(20) {
            println!("  vocab[{i}] = {:?}", s);
        }
    }

    let literals = ["[INST]", " [INST]", "[/INST]", " [/INST]", "[INST] hello [/INST]"];
    for s in &literals {
        let ids = tok.encode(s, false);
        let dec = tok.decode(&ids, false);
        println!("  encode({s:?}) → {ids:?}  decode → {dec:?}");
    }

    // Full Turn-1 prompt exactly as our template now emits it.
    let turn1 = "[INST] Explain what a mutex is in one paragraph. [/INST]";
    let ids_no_bos = tok.encode(turn1, false);
    let ids_bos = tok.encode(turn1, true);
    println!("\n  full turn-1 (no BOS): {ids_no_bos:?}  ({} tok)", ids_no_bos.len());
    println!("  full turn-1 (w/ BOS): {ids_bos:?}  ({} tok)", ids_bos.len());
    println!("  decoded (no BOS): {:?}", tok.decode(&ids_no_bos, false));
}
