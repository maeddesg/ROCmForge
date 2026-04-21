//! `--list-tensors` diagnostic command.
//!
//! Parses a GGUF file, prints its tensor inventory, metadata summary,
//! and quant-type distribution. CPU-only — no GPU allocation, no VRAM
//! upload. Intended for the smoke path when inspecting a new model.

use std::path::Path;

use crate::v1::core::gguf::GGUFFile;
use crate::v1::core::model_config::ModelConfig;
use crate::v1::core::tensor_info::{validate_quant_types, GgmlType};

pub fn run(path: impl AsRef<Path>) -> Result<(), String> {
    let gguf = GGUFFile::open(path).map_err(|e| format!("open failed: {e}"))?;

    println!("GGUF v{}", gguf.header().version);
    println!(
        "  Tensors: {}, Metadata entries: {}",
        gguf.tensor_count(),
        gguf.metadata_count()
    );

    match ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()) {
        Ok(cfg) => {
            println!();
            println!("Architecture: {}", cfg.architecture);
            println!("  n_layers={}  n_heads={}  n_kv_heads={}  GQA={}", cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.is_gqa);
            println!(
                "  hidden={}  ffn={}  head_dim={}  vocab={}",
                cfg.hidden_dim, cfg.ffn_dim, cfg.head_dim, cfg.vocab_size
            );
            println!("  rope_freq_base={}  rms_norm_eps={}", cfg.rope_freq_base, cfg.rms_norm_eps);
            println!(
                "  has_qk_norm={}  has_rope_freqs={}  has_attention_bias={}",
                cfg.has_qk_norm, cfg.has_rope_freqs, cfg.has_attention_bias
            );
            println!();
        }
        Err(e) => println!("(ModelConfig parse skipped: {e})\n"),
    }

    println!(
        "{:<48} {:<22} {:<6} {:>12}",
        "Tensor", "Shape", "Type", "Size"
    );
    println!("{}", "-".repeat(92));
    for t in gguf.tensors() {
        let shape = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let size_mb = t.byte_size as f64 / 1_048_576.0;
        let size_str = if size_mb >= 1.0 {
            format!("{size_mb:.2} MB")
        } else {
            format!("{:.2} KB", t.byte_size as f64 / 1024.0)
        };
        println!(
            "{:<48} {:<22} {:<6} {:>12}",
            truncate(&t.name, 48),
            truncate(&shape, 22),
            format!("{}", t.ggml_type),
            size_str
        );
    }

    println!();
    let total_bytes: u64 = gguf.tensors().iter().map(|t| t.byte_size).sum();
    println!(
        "Total: {} tensors, {:.2} GB",
        gguf.tensor_count(),
        total_bytes as f64 / 1e9
    );

    let mut by_type: std::collections::HashMap<GgmlType, usize> = Default::default();
    for t in gguf.tensors() {
        *by_type.entry(t.ggml_type).or_insert(0) += 1;
    }
    let mut types: Vec<(GgmlType, usize)> = by_type.into_iter().collect();
    types.sort_by_key(|(t, _)| *t as u32);
    let type_summary = types
        .iter()
        .map(|(t, n)| format!("{t} ({n} tensors)"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("Quant types: {type_summary}");

    let warnings = validate_quant_types(gguf.tensors());
    if !warnings.is_empty() {
        println!();
        println!("Warnings ({}):", warnings.len());
        for w in &warnings {
            println!("  {w}");
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
