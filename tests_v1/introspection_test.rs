//! Phase 1 / Schritt 1.13 — Model-Introspection tests.
//!
//! Runs the introspection scan on the three Phase-1 target models and
//! verifies:
//!   * The scan produces sensible structural output.
//!   * The known-risky Llama-3.1 special-token IDs (128006..=128009)
//!     land in `critical_embedding_tokens`.
//!   * The scan stays inside the 5-second budget specified by the
//!     Arch-Doc §2.2 cost target.
//!
//! A note on "safe" vs. "risky": empirically *all* three models have
//! some low-L2 embedding rows (byte-fallback + reserved slots), so
//! the SNR risk score drops below the doc's 2.0 threshold on each
//! of them. What differs is whether the critical rows are hit on
//! the hot path — Llama-3.1's chat-template tokens (128006..=128009)
//! are, Qwen3's reserved byte-fallback slots are not. The scan
//! cannot distinguish hot from cold usage; Phase-2 FP32-overlay is
//! the layer that decides which critical rows get upgraded.

#![cfg(feature = "v1")]

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::introspection::{introspect, PrecisionHint, TokenId};

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir().expect("HOME set").join("models").join(name)
}

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA31: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

// ── Structural tests ─────────────────────────────────────────────────────

#[test]
fn test_introspection_qwen3_structure() {
    let gguf = GGUFFile::open(model_path(QWEN3)).expect("open qwen3");
    let profile = introspect(&gguf);

    let (min_l2, max_l2) = profile.embedding_magnitude_range;
    assert!(
        min_l2 >= 0.0 && max_l2 > min_l2,
        "invalid embedding L2 range: ({min_l2}, {max_l2})"
    );
    assert!(max_l2 > 0.1, "some embedding row should carry signal");
    assert!(
        !profile.layer_magnitude_stats.is_empty(),
        "layer Stichprobe produced zero rows"
    );
    assert!(
        profile.quantization_noise_estimate > 0.0,
        "Q4_K noise estimate should be positive"
    );
    assert_eq!(
        profile.layer_magnitude_stats.len(),
        profile.precision_recommendation.len(),
        "one precision hint per layer-stats row"
    );
    profile.print_summary();
}

#[test]
fn test_introspection_llama31_special_tokens_flagged() {
    let gguf = GGUFFile::open(model_path(LLAMA31)).expect("open llama-3.1");
    let profile = introspect(&gguf);

    assert!(
        !profile.critical_embedding_tokens.is_empty(),
        "llama-3.1 must surface critical embedding tokens"
    );

    // The chat-template hot-path special tokens are the canonical
    // example from Arch-Doc §2.2. At least one of 128006..=128009
    // must land in the critical list — that's the whole reason
    // Model Introspection exists.
    let has_chat_special = profile
        .critical_embedding_tokens
        .iter()
        .any(|&t: &TokenId| (128006..=128009).contains(&t));
    assert!(
        has_chat_special,
        "llama-3.1 chat-template tokens 128006..=128009 should be critical; \
         got {} critical tokens, first 16: {:?}",
        profile.critical_embedding_tokens.len(),
        &profile
            .critical_embedding_tokens
            .iter()
            .take(16)
            .collect::<Vec<_>>()
    );
    profile.print_summary();
}

// ── Cost budget ──────────────────────────────────────────────────────────

#[test]
fn test_introspection_under_5_seconds() {
    let gguf = GGUFFile::open(model_path(QWEN3)).expect("open qwen3");
    let start = std::time::Instant::now();
    let _ = introspect(&gguf);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "introspection took {:.2}s, budget is 5s",
        elapsed.as_secs_f64()
    );
    println!("qwen3 introspection: {:.2}s", elapsed.as_secs_f64());
}

// ── Precision recommendation ─────────────────────────────────────────────

#[test]
fn test_introspection_precision_recommendation_populated() {
    let gguf = GGUFFile::open(model_path(QWEN3)).expect("open qwen3");
    let profile = introspect(&gguf);

    assert!(!profile.precision_recommendation.is_empty());
    let fp8 = profile
        .precision_recommendation
        .iter()
        .filter(|h| matches!(h, PrecisionHint::Fp8E4M3))
        .count();
    let bf16 = profile
        .precision_recommendation
        .iter()
        .filter(|h| matches!(h, PrecisionHint::Bf16Scales))
        .count();
    let fp32 = profile
        .precision_recommendation
        .iter()
        .filter(|h| matches!(h, PrecisionHint::Fp32Scales))
        .count();
    // The three tiers must cover 100 % of recorded layers — no lost
    // entries.
    assert_eq!(
        fp8 + bf16 + fp32,
        profile.precision_recommendation.len(),
        "FP8/BF16/FP32 must account for every recommendation"
    );
    println!(
        "qwen3 recommendation breakdown: FP8={fp8} BF16={bf16} FP32={fp32}"
    );
}

// ── Print summary doesn't crash ──────────────────────────────────────────

#[test]
fn test_introspection_print_summary() {
    let gguf = GGUFFile::open(model_path(QWEN3)).expect("open qwen3");
    let profile = introspect(&gguf);
    // Exercises both the summary frame and the critical-id preview.
    profile.print_summary();
    // Secondary invariant: `has_risk` is internally consistent.
    if profile.snr_risk_score < 2.0 {
        assert!(profile.has_risk());
    }
}
