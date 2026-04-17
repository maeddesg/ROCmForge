#![cfg(feature = "gpu")]

//! Phase 2d correctness test — WMMA Q4_0 prefill path vs the hipBLAS
//! reference that Phase 1 landed.
//!
//! Runs the binary twice with the same greedy config on a prompt long
//! enough to trigger the WMMA path (seq_len ≥ 64 and 64-aligned):
//!
//!   A) ROCMFORGE_DISABLE_WMMA_PREFILL=1   (hipBLAS FP16 Hgemm reference)
//!   B) default                            (WMMA Q4_0 inline-dequant)
//!
//! We don't require full byte-for-byte equality — hipBLAS accumulates in
//! FP16, the WMMA kernel accumulates in FP32, so later decoded tokens
//! can drift on low-confidence positions. We only assert that the first
//! decoded token matches, which is the most sensitive signal for a
//! logic bug (wrong weight layout, wrong dispatch path, wrong lane
//! mapping). FP16 drift shows up after several tokens, not the first.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
// Prompt of ~64 tokens so the WMMA path triggers. "word "-repeated
// prompts tokenise to roughly one token per word, so 60 copies lands
// comfortably over the M ≥ 64 threshold after the tokenizer adds/strips
// specials.
const PROMPT_WORDS: usize = 80;
const MAX_TOKENS: &str = "8";

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

fn run_with_flag(disable_wmma: bool) -> (String, bool) {
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
    if disable_wmma {
        cmd.env("ROCMFORGE_DISABLE_WMMA_PREFILL", "1");
    } else {
        cmd.env_remove("ROCMFORGE_DISABLE_WMMA_PREFILL");
    }
    let output = cmd.output().expect("failed to launch rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        output.status.success(),
    )
}

fn extract_generation(stdout: &str) -> String {
    let mut lines = stdout.lines().peekable();
    let mut generating = false;
    let mut out = String::new();
    while let Some(line) = lines.next() {
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
fn wmma_prefill_matches_hipblas_first_token() {
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
    assert!(ok_ref, "hipBLAS reference run failed");
    assert!(ok_wmma, "WMMA run failed");

    let gen_ref = extract_generation(&out_ref);
    let gen_wmma = extract_generation(&out_wmma);

    let first_ref = gen_ref.split_whitespace().next().unwrap_or("");
    let first_wmma = gen_wmma.split_whitespace().next().unwrap_or("");

    if first_ref != first_wmma {
        eprintln!("=== hipBLAS reference ===\n{}", gen_ref);
        eprintln!("=== WMMA                ===\n{}", gen_wmma);
        panic!(
            "first decoded token differs: hipblas={:?} wmma={:?}",
            first_ref, first_wmma
        );
    }
}
