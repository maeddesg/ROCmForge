#![cfg(feature = "gpu")]

//! Phase 3.1 correctness — WMMA prefill path with zero-padding for
//! arbitrary `seq_len ≥ 64`. Runs the binary on several unaligned
//! prompt lengths, once with WMMA disabled (scalar fallback) and once
//! with WMMA enabled (padded). The first decoded token must match —
//! the most sensitive signal for a wrong stride, wrong mask, or a
//! padding row that leaked into the output.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
const MAX_TOKENS: &str = "3";

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

fn skip_if_model_missing() -> bool {
    !Path::new(MODEL).exists()
}

fn build_prompt(n_words: usize) -> String {
    let mut s = String::with_capacity(n_words * 5);
    for i in 0..n_words {
        if i > 0 {
            s.push(' ');
        }
        s.push_str("word");
    }
    s
}

fn run_prompt(prompt: &str, disable_wmma: bool) -> (String, bool) {
    let mut cmd = Command::new(binary_path());
    cmd.args([
        "--model",
        MODEL,
        "--prompt",
        prompt,
        "--max-tokens",
        MAX_TOKENS,
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--no-template",
        "--gpu",
    ]);
    if disable_wmma {
        cmd.env("ROCMFORGE_DISABLE_WMMA_ATTENTION", "1");
        cmd.env("ROCMFORGE_DISABLE_WMMA_PREFILL", "1");
    } else {
        cmd.env_remove("ROCMFORGE_DISABLE_WMMA_ATTENTION");
        cmd.env_remove("ROCMFORGE_DISABLE_WMMA_PREFILL");
    }
    let output = cmd.output().expect("failed to launch rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        output.status.success(),
    )
}

fn extract_generation(stdout: &str) -> String {
    let mut generating = false;
    let mut out = String::new();
    for line in stdout.lines() {
        if !generating {
            if line.starts_with("Prompt:") {
                generating = true;
            }
            continue;
        }
        if line.contains("tokens in") && line.contains("tok/s") {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

fn check_seq_len(n_words: usize) {
    let prompt = build_prompt(n_words);
    let (out_ref, ok_ref) = run_prompt(&prompt, true);
    let (out_wmma, ok_wmma) = run_prompt(&prompt, false);
    assert!(ok_ref, "scalar reference run failed at n={}", n_words);
    assert!(ok_wmma, "WMMA padded run failed at n={}", n_words);

    let gen_ref = extract_generation(&out_ref);
    let gen_wmma = extract_generation(&out_wmma);

    let first_ref = gen_ref.split_whitespace().next().unwrap_or("");
    let first_wmma = gen_wmma.split_whitespace().next().unwrap_or("");

    if first_ref != first_wmma {
        eprintln!("=== scalar (n={}) ===\n{}", n_words, gen_ref);
        eprintln!("=== WMMA padded (n={}) ===\n{}", n_words, gen_wmma);
        panic!(
            "first decoded token differs at n={}: scalar={:?} wmma={:?}",
            n_words, first_ref, first_wmma
        );
    }
}

#[test]
#[serial]
fn padding_seq_65() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(65);
}

#[test]
#[serial]
fn padding_seq_100() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(100);
}

#[test]
#[serial]
fn padding_seq_200() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(200);
}

#[test]
#[serial]
fn padding_seq_300() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(300);
}

#[test]
#[serial]
fn padding_aligned_seq_64_is_noop() {
    // Already 64-aligned: padding logic must not change the output.
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(64);
}

#[test]
#[serial]
fn padding_aligned_seq_128_is_noop() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    check_seq_len(128);
}
