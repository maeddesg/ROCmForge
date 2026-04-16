#![cfg(feature = "gpu")]

//! Correctness test: greedy speculative decoding vs greedy baseline.
//!
//! Runs the binary twice with the same prompt:
//!   A) target model only (greedy baseline)
//!   B) target + draft model (speculative decoding, greedy)
//!
//! Checks how many initial tokens match. Batched verify attention uses a
//! different float reduction order than sequential per-token attention, so
//! numerical divergence is expected at some point. The test documents the
//! divergence position and asserts a minimum prefix match.
//!
//! Requires: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 and actual model files.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const TARGET_MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
const DRAFT_MODEL: &str = "/home/maeddes/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const PROMPT: &str = "The Fibonacci sequence starts with 0 and 1. Each subsequent number is the sum of the two preceding ones. The first ten numbers are";
const MAX_TOKENS: &str = "50";

/// Minimum number of initial characters that must match before we accept
/// numerical divergence. This catches gross bugs (wrong tokens, off-by-one)
/// while tolerating float-reduction differences.
const MIN_PREFIX_MATCH: usize = 20;

fn binary_path() -> String {
    let path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("rocmforge");
    if path.exists() {
        return path.to_string_lossy().to_string();
    }
    "target/release/rocmforge".to_string()
}

fn skip_unless_enabled() -> bool {
    std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS").unwrap_or_default() != "1"
}

fn skip_if_models_missing() -> bool {
    !Path::new(TARGET_MODEL).exists() || !Path::new(DRAFT_MODEL).exists()
}

fn run_rocmforge(args: &[&str]) -> (String, String, bool) {
    let output = Command::new(binary_path())
        .args(args)
        .output()
        .expect("Failed to execute rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.success(),
    )
}

fn dump_divergence(baseline: &str, spec: &str) {
    let out_dir = Path::new("tests/out");
    std::fs::create_dir_all(out_dir).ok();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let baseline_chars: Vec<char> = baseline.chars().collect();
    let spec_chars: Vec<char> = spec.chars().collect();
    let mut diverge_pos = baseline_chars.len().min(spec_chars.len());
    for i in 0..diverge_pos {
        if baseline_chars[i] != spec_chars[i] {
            diverge_pos = i;
            break;
        }
    }

    let context_start = diverge_pos.saturating_sub(20);
    let context_end_b = (diverge_pos + 30).min(baseline.len());
    let context_end_s = (diverge_pos + 30).min(spec.len());

    let dump = serde_json::json!({
        "diverge_char_pos": diverge_pos,
        "matching_prefix_len": diverge_pos,
        "baseline_output": baseline,
        "spec_output": spec,
        "baseline_around_divergence": &baseline[context_start..context_end_b],
        "spec_around_divergence": &spec[context_start..context_end_s],
        "baseline_len": baseline.len(),
        "spec_len": spec.len(),
    });

    let path = out_dir.join(format!("diverge_{}.json", timestamp));
    std::fs::write(&path, serde_json::to_string_pretty(&dump).unwrap())
        .expect("Failed to write divergence dump");
    eprintln!("Divergence dump written to {}", path.display());
}

/// Find the number of matching prefix characters between two strings.
fn prefix_match_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(ca, cb)| ca == cb)
        .count()
}

fn run_spec_vs_baseline(spec_depth: &str) {
    if skip_unless_enabled() {
        eprintln!("Skipping: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to run");
        return;
    }
    if skip_if_models_missing() {
        eprintln!("Skipping: model files not found at {} and/or {}", TARGET_MODEL, DRAFT_MODEL);
        return;
    }

    eprintln!("=== Run A: Greedy baseline (no draft) ===");
    let (baseline_stdout, baseline_stderr, baseline_ok) = run_rocmforge(&[
        "--model", TARGET_MODEL,
        "--prompt", PROMPT,
        "--max-tokens", MAX_TOKENS,
        "--gpu",
    ]);
    assert!(baseline_ok, "Baseline run failed:\n{}", baseline_stderr);

    eprintln!("=== Run B: Greedy speculative (depth={}) ===", spec_depth);
    let (spec_stdout, spec_stderr, spec_ok) = run_rocmforge(&[
        "--model", TARGET_MODEL,
        "--draft-model", DRAFT_MODEL,
        "--prompt", PROMPT,
        "--max-tokens", MAX_TOKENS,
        "--spec-depth", spec_depth,
        "--gpu",
    ]);
    assert!(spec_ok, "Speculative run failed:\n{}", spec_stderr);

    let baseline_text = baseline_stdout.trim();
    let spec_text = spec_stdout.trim();

    eprintln!("Baseline ({} chars): {:?}", baseline_text.len(), baseline_text);
    eprintln!("Spec     ({} chars): {:?}", spec_text.len(), spec_text);

    let match_len = prefix_match_len(baseline_text, spec_text);

    if baseline_text == spec_text {
        eprintln!("PASS: Outputs are byte-identical ({} chars)", baseline_text.len());
    } else {
        dump_divergence(baseline_text, spec_text);
        eprintln!(
            "DIVERGENCE at char {}: batched verify produces different float reduction \
             than sequential attention. First {} chars match.",
            match_len, match_len
        );
        assert!(
            match_len >= MIN_PREFIX_MATCH,
            "Prefix match too short ({} < {} chars). This suggests a logic bug, \
             not just numerical divergence.\n\
             Baseline: {:?}\n\
             Spec:     {:?}",
            match_len, MIN_PREFIX_MATCH,
            &baseline_text[..baseline_text.len().min(80)],
            &spec_text[..spec_text.len().min(80)],
        );
        eprintln!(
            "PASS: {} chars match (>= {} minimum). Divergence is expected due to \
             batched vs sequential attention numerics.",
            match_len, MIN_PREFIX_MATCH
        );
    }
}

#[test]
#[serial]
#[ignore] // Requires real models + ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1
fn test_spec_greedy_vs_baseline_depth3() {
    run_spec_vs_baseline("3");
}

#[test]
#[serial]
#[ignore]
fn test_spec_greedy_vs_baseline_depth1() {
    run_spec_vs_baseline("1");
}

#[test]
#[serial]
#[ignore]
fn test_spec_greedy_vs_baseline_depth5() {
    run_spec_vs_baseline("5");
}
