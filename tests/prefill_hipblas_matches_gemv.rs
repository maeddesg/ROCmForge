#![cfg(feature = "gpu")]

//! Correctness test: hipBLAS-backed prefill path vs. the custom GEMV
//! reference path.
//!
//! Runs the binary twice with the same fixed prompt and greedy config:
//!   A) ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 (custom GEMM / GEMV reference)
//!   B) ROCMFORGE_DISABLE_HIPBLAS_PREFILL unset (hipBLAS Hgemm path)
//!
//! FP16 accumulation inside hipBLAS introduces single-ULP differences
//! at the logit level, so we do not require byte-identical output.
//! The test checks the generated text and fails only if the two paths
//! diverge at the *first* decoded token — which is the one most sensitive
//! to any logic bug in the matrix layout or the dispatch.
//!
//! Requires: ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 plus the 7B model.

use serial_test::serial;
use std::path::Path;
use std::process::Command;

const MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";
// Prompt long enough to trigger the hipBLAS path (seq_len ≥ 32).
const PROMPT: &str = "Explain the concept of speculative decoding used in modern large language model inference systems. Include the roles of the draft model and the target model, how verification works, and when it is a net win. Cover acceptance rate, batched verify, and what determines profitability in concrete terms.";
const MAX_TOKENS: &str = "16";

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

fn run_with_flag(disable_hipblas: bool) -> (String, bool) {
    let mut cmd = Command::new(binary_path());
    cmd.args([
        "--model",
        MODEL,
        "--prompt",
        PROMPT,
        "--max-tokens",
        MAX_TOKENS,
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--no-template",
        "--gpu",
    ]);
    if disable_hipblas {
        cmd.env("ROCMFORGE_DISABLE_HIPBLAS_PREFILL", "1");
    } else {
        cmd.env_remove("ROCMFORGE_DISABLE_HIPBLAS_PREFILL");
    }
    let output = cmd.output().expect("failed to launch rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        output.status.success(),
    )
}

/// Strip the startup banner and trailing `N tokens in …` line to leave
/// only the generated text.
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
fn hipblas_prefill_matches_gemv_first_token() {
    if skip_unless_enabled() {
        eprintln!(
            "SKIP: set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 to run this test"
        );
        return;
    }
    if skip_if_model_missing() {
        eprintln!("SKIP: model not found at {}", MODEL);
        return;
    }

    let (out_ref, ok_ref) = run_with_flag(true);
    let (out_hip, ok_hip) = run_with_flag(false);

    assert!(ok_ref, "GEMV reference run failed: {}", out_ref);
    assert!(ok_hip, "hipBLAS run failed: {}", out_hip);

    let gen_ref = extract_generation(&out_ref);
    let gen_hip = extract_generation(&out_hip);

    // Compare the first decoded token (first whitespace-separated fragment).
    // FP16 accumulation inside hipBLAS can cause later-token divergence,
    // so we do not require full-string equality.
    let first_ref = gen_ref.split_whitespace().next().unwrap_or("");
    let first_hip = gen_hip.split_whitespace().next().unwrap_or("");

    if first_ref != first_hip {
        eprintln!("=== GEMV reference (full) ===\n{}", gen_ref);
        eprintln!("=== hipBLAS (full)         ===\n{}", gen_hip);
        panic!(
            "first decoded token differs: reference={:?} hipblas={:?}",
            first_ref, first_hip
        );
    }
}
