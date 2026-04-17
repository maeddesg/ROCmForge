#![cfg(feature = "gpu")]

//! Phase 3d end-to-end correctness — WMMA prefill attention (GQA + causal)
//! vs the scalar per-head `flash_attn_prefill_strided` kernel.
//!
//! Runs the binary twice with the same greedy config on a prompt long
//! enough to engage the WMMA attention path (seq_len ≥ 64, 64-aligned):
//!
//!   A) ROCMFORGE_DISABLE_WMMA_ATTENTION=1   (scalar reference)
//!   B) default                              (WMMA GQA + causal)
//!
//! We don't require byte-identity — the WMMA kernel reorders FP32
//! accumulations across matrix cores, and the online-softmax pass
//! handles numerator/denominator differently than the scalar kernel.
//! The first decoded token is the most sensitive signal for a real
//! logic bug (wrong stride, wrong mask, wrong head mapping); later
//! tokens can drift by FP rounding alone.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
const PROMPT_WORDS: usize = 80;
const MAX_TOKENS: &str = "5";

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

fn build_prompt() -> String {
    let mut s = String::new();
    for i in 0..PROMPT_WORDS {
        if i > 0 {
            s.push(' ');
        }
        s.push_str("word");
    }
    s
}

fn run_with_flag(disable_wmma_attention: bool) -> (String, bool) {
    let mut cmd = Command::new(binary_path());
    let prompt = build_prompt();
    cmd.args([
        "--model",
        MODEL,
        "--prompt",
        &prompt,
        "--max-tokens",
        MAX_TOKENS,
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--no-template",
        "--gpu",
    ]);
    if disable_wmma_attention {
        cmd.env("ROCMFORGE_DISABLE_WMMA_ATTENTION", "1");
    } else {
        cmd.env_remove("ROCMFORGE_DISABLE_WMMA_ATTENTION");
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

#[test]
#[serial]
fn wmma_attention_matches_scalar_first_token() {
    if skip_unless_enabled() {
        eprintln!("SKIP: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if skip_if_model_missing() {
        eprintln!("SKIP: model not found at {}", MODEL);
        return;
    }

    let (out_ref, ok_ref) = run_with_flag(true);
    let (out_wmma, ok_wmma) = run_with_flag(false);
    assert!(ok_ref, "scalar reference run failed");
    assert!(ok_wmma, "WMMA attention run failed");

    let gen_ref = extract_generation(&out_ref);
    let gen_wmma = extract_generation(&out_wmma);

    let first_ref = gen_ref.split_whitespace().next().unwrap_or("");
    let first_wmma = gen_wmma.split_whitespace().next().unwrap_or("");

    if first_ref != first_wmma {
        eprintln!("=== scalar reference ===\n{}", gen_ref);
        eprintln!("=== WMMA attention    ===\n{}", gen_wmma);
        panic!(
            "first decoded token differs: scalar={:?} wmma={:?}",
            first_ref, first_wmma
        );
    }
}
