#![cfg(feature = "gpu")]

//! Correctness test: batched lm_head vs sequential lm_head in verify.
//!
//! Runs speculative decoding twice with the same prompt/seed/greedy config:
//!   A) ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1 (sequential, reference)
//!   B) ROCMFORGE_DISABLE_BATCHED_LM_HEAD=0 (batched, under test)
//!
//! Asserts that the output tokens are byte-identical.
//! Both paths use the same GEMV kernels and float accumulation order,
//! so there should be zero numerical divergence (unlike batched-vs-sequential
//! attention which has different reduction order).
//!
//! Requires: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 and actual model files.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const TARGET_MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
const DRAFT_MODEL: &str = "/home/maeddes/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const PROMPT: &str = "Write a function that computes the nth Fibonacci number using dynamic programming. Include edge cases for n=0 and n=1.";
const MAX_TOKENS: &str = "50";

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

fn run_with_lm_head_flag(spec_depth: &str, disable_batched: &str) -> (String, String, bool) {
    let output = Command::new(binary_path())
        .args([
            "--model", TARGET_MODEL,
            "--draft-model", DRAFT_MODEL,
            "--prompt", PROMPT,
            "--max-tokens", MAX_TOKENS,
            "--spec-depth", spec_depth,
            "--gpu",
        ])
        .env("ROCMFORGE_DISABLE_BATCHED_LM_HEAD", disable_batched)
        .output()
        .expect("Failed to execute rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.success(),
    )
}

fn dump_divergence(depth: &str, sequential: &str, batched: &str) {
    let out_dir = Path::new("tests/out");
    std::fs::create_dir_all(out_dir).ok();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let seq_chars: Vec<char> = sequential.chars().collect();
    let bat_chars: Vec<char> = batched.chars().collect();
    let min_len = seq_chars.len().min(bat_chars.len());
    let mut diverge_pos = min_len;
    for i in 0..min_len {
        if seq_chars[i] != bat_chars[i] {
            diverge_pos = i;
            break;
        }
    }

    let dump = serde_json::json!({
        "test": format!("batched_lm_head_depth_{}", depth),
        "diverge_char_pos": diverge_pos,
        "sequential_output": sequential,
        "batched_output": batched,
        "sequential_len": sequential.len(),
        "batched_len": batched.len(),
    });

    let path = out_dir.join(format!("lm_head_diverge_{}_{}.json", depth, timestamp));
    std::fs::write(&path, serde_json::to_string_pretty(&dump).unwrap())
        .expect("Failed to write divergence dump");
    eprintln!("Divergence dump written to {}", path.display());
}

fn run_batched_vs_sequential(spec_depth: &str) {
    if skip_unless_enabled() {
        eprintln!("Skipping: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to run");
        return;
    }
    if skip_if_models_missing() {
        eprintln!(
            "Skipping: model files not found at {} and/or {}",
            TARGET_MODEL, DRAFT_MODEL
        );
        return;
    }

    eprintln!(
        "=== depth={}: Run A (sequential lm_head, DISABLE=1) ===",
        spec_depth
    );
    let (seq_stdout, seq_stderr, seq_ok) = run_with_lm_head_flag(spec_depth, "1");
    assert!(seq_ok, "Sequential run failed:\n{}", seq_stderr);

    eprintln!(
        "=== depth={}: Run B (batched lm_head, DISABLE=0) ===",
        spec_depth
    );
    let (bat_stdout, bat_stderr, bat_ok) = run_with_lm_head_flag(spec_depth, "0");
    assert!(bat_ok, "Batched run failed:\n{}", bat_stderr);

    let seq_text = seq_stdout.trim();
    let bat_text = bat_stdout.trim();

    eprintln!(
        "Sequential ({} chars): {:?}",
        seq_text.len(),
        &seq_text[..seq_text.len().min(80)]
    );
    eprintln!(
        "Batched    ({} chars): {:?}",
        bat_text.len(),
        &bat_text[..bat_text.len().min(80)]
    );

    if seq_text == bat_text {
        eprintln!(
            "PASS (depth={}): Outputs are byte-identical ({} chars)",
            spec_depth,
            seq_text.len()
        );
    } else {
        dump_divergence(spec_depth, seq_text, bat_text);
        panic!(
            "FAIL (depth={}): Batched lm_head produced different output than sequential!\n\
             Sequential: {:?}\n\
             Batched:    {:?}\n\
             This is a correctness bug — batched and sequential paths use the same \
             GEMV kernels and should produce identical results.",
            spec_depth,
            &seq_text[..seq_text.len().min(100)],
            &bat_text[..bat_text.len().min(100)],
        );
    }
}

#[test]
#[serial]
#[ignore] // Requires real models + ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1
fn test_batched_lm_head_depth_1() {
    run_batched_vs_sequential("1");
}

#[test]
#[serial]
#[ignore]
fn test_batched_lm_head_depth_3() {
    run_batched_vs_sequential("3");
}

#[test]
#[serial]
#[ignore]
fn test_batched_lm_head_depth_5() {
    run_batched_vs_sequential("5");
}

// ── Edge case: --spec-depth 9 should error ──────────────────────────────────────

#[test]
#[serial]
#[ignore]
fn test_spec_depth_exceeds_max() {
    if skip_unless_enabled() {
        eprintln!("Skipping: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to run");
        return;
    }

    let output = Command::new(binary_path())
        .args([
            "--model", TARGET_MODEL,
            "--draft-model", DRAFT_MODEL,
            "--prompt", "test",
            "--max-tokens", "10",
            "--spec-depth", "9",
            "--gpu",
        ])
        .output()
        .expect("Failed to execute rocmforge binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "Expected --spec-depth 9 to fail, but it succeeded"
    );
    assert!(
        stderr.contains("--spec-depth maximum is 8"),
        "Expected informative error message about max depth, got:\n{}",
        stderr
    );
    eprintln!("PASS: --spec-depth 9 correctly rejected with error message");
}

// ── Edge case: --spec-depth 8 should work ───────────────────────────────────────

#[test]
#[serial]
#[ignore]
fn test_spec_depth_8_max() {
    if skip_unless_enabled() {
        eprintln!("Skipping: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to run");
        return;
    }
    if skip_if_models_missing() {
        eprintln!(
            "Skipping: model files not found at {} and/or {}",
            TARGET_MODEL, DRAFT_MODEL
        );
        return;
    }

    eprintln!("=== depth=8: Should run without errors ===");
    let (stdout, stderr, ok) = run_with_lm_head_flag("8", "0");
    assert!(ok, "depth=8 run failed:\n{}", stderr);
    let text = stdout.trim();
    assert!(
        !text.is_empty(),
        "depth=8 produced no output. stderr:\n{}",
        stderr
    );
    eprintln!(
        "PASS: depth=8 completed successfully ({} chars output)",
        text.len()
    );
}
