//! CPU-only root-cause probe for the Llama-3.1-Q4_K_M quality issue.
//! Answers the 4 diagnostic hypotheses by reading GGUF metadata.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;

fn probe(path: &std::path::Path, label: &str) {
    if !path.exists() {
        eprintln!("[{label}] missing — skip: {}", path.display());
        return;
    }
    println!("\n===================== {} =====================", label);
    let gguf = GGUFFile::open(path).expect("open gguf");

    // ── (1) Tied weights? ──────────────────────────────────────────────
    let names: std::collections::HashSet<&str> =
        gguf.tensors().iter().map(|t| t.name.as_str()).collect();
    let has_output = names.contains("output.weight");
    let has_embd = names.contains("token_embd.weight");
    let has_rope_freqs = names.contains("rope_freqs.weight");
    println!("(1) TIED WEIGHTS:");
    println!("    token_embd.weight: {has_embd}");
    println!("    output.weight:     {has_output}  {}",
             if has_output { "(separate lm_head)" } else { "(TIED to token_embd)" });
    println!("    rope_freqs.weight: {has_rope_freqs}");

    // ── Raw metadata dump ──────────────────────────────────────────────
    let md = gguf.metadata();
    let print_key = |k: &str| {
        match md.get(k) {
            Some(v) => println!("    {:48} = {:?}", k, v),
            None => println!("    {:48} = MISSING", k),
        }
    };
    println!("\n(2) RoPE metadata:");
    print_key("llama.rope.freq_base");
    print_key("qwen3.rope.freq_base");
    print_key("qwen2.rope.freq_base");
    print_key("general.architecture");
    print_key("llama.rope.scaling.factor");
    print_key("llama.rope.scaling.type");
    print_key("llama.rope.scaling.original_context_length");

    println!("\n(3) GQA / attention metadata:");
    print_key("llama.attention.head_count");
    print_key("llama.attention.head_count_kv");
    print_key("qwen3.attention.head_count");
    print_key("qwen3.attention.head_count_kv");
    print_key("llama.embedding_length");
    print_key("llama.context_length");

    println!("\n(4) BOS / tokenizer metadata:");
    print_key("tokenizer.ggml.bos_token_id");
    print_key("tokenizer.ggml.eos_token_id");
    print_key("tokenizer.ggml.add_bos_token");

    // ── Resolved ModelConfig ───────────────────────────────────────────
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("cfg");
    println!("\nResolved ModelConfig:");
    println!("    architecture:   {}",     cfg.architecture);
    println!("    n_heads:        {}",     cfg.n_heads);
    println!("    n_kv_heads:     {}",     cfg.n_kv_heads);
    println!("    head_dim:       {}",     cfg.head_dim);
    println!("    hidden_dim:     {}",     cfg.hidden_dim);
    println!("    ffn_dim:        {}",     cfg.ffn_dim);
    println!("    vocab_size:     {}",     cfg.vocab_size);
    println!("    rope_freq_base: {}",     cfg.rope_freq_base);
    println!("    has_rope_freqs: {}",     cfg.has_rope_freqs);
    println!("    is_gqa:         {}",     cfg.is_gqa);
    println!("    rms_norm_eps:   {}",     cfg.rms_norm_eps);
    println!("    context_length: {}",     cfg.context_length);

    // ── Embedding + LM-head + rope_freqs tensor types + shapes ────────
    for t in gguf.tensors() {
        if t.name == "token_embd.weight"
            || t.name == "output.weight"
            || t.name == "rope_freqs.weight"
        {
            println!("\n    {} :: {:?}, shape={:?}, file_offset={}, byte_size={}",
                     t.name, t.ggml_type, t.shape, t.file_offset, t.byte_size);
        }
    }

    // ── (5) Read the rope_freqs.weight values — are they sensible? ────
    if let Some(rf) = gguf.tensors().iter().find(|t| t.name == "rope_freqs.weight") {
        let bytes = gguf.tensor_data_full(rf).expect("read rope_freqs");
        let n_elem = (bytes.len() / 4).min(8);
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
        };
        println!("\n(5) rope_freqs.weight first {n_elem} floats + range:");
        print!("    values[0..{n_elem}] =");
        for v in &floats[..n_elem] { print!(" {:.6}", v); }
        println!();
        let min = floats.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = floats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let total_len = floats.len();
        println!("    range: [{:.6}, {:.6}]  total_len: {}", min, max, total_len);
        // Sanity: expected half = head_dim/2 = 64. llama.cpp applies
        // Llama-3.1 rope_scaling.factor=8 with low/high freq factors.
        // Values should be > 1.0 for the low-freq tail (scaling down
        // the frequency), ~1.0 for the high-freq head.
    }
}

#[test]
fn diag_llama31_vs_qwen3() {
    let home = dirs::home_dir().expect("HOME").join("models");
    probe(&home.join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), "Llama-3.1-8B-Q4_K_M");
    probe(&home.join("Qwen3-8B-Q4_K_M.gguf"), "Qwen3-8B-Q4_K_M");
}
