//! Correctness test: AVX-512 VNNI Q4_0 GEMV vs the AVX2 reference path.
//!
//! Runs the CPU decode path twice with the same prompt/greedy config:
//!   A) ROCMFORGE_DISABLE_AVX512=1 (AVX2/scalar reference)
//!   B) ROCMFORGE_DISABLE_AVX512 unset (AVX-512 VNNI under test)
//!
//! Asserts that the two runs produce byte-identical output. The AVX-512
//! kernel uses different reduction order than AVX2 (vpdpbusd vs
//! maddubs+madd), so tiny float differences are theoretically possible —
//! but Q4_0 × Q8_0 is integer dot product with one final float multiply,
//! so results should be identical.

use std::path::Path;
use std::process::Command;

const MODEL: &str = "/home/maeddes/models/qwen2.5-0.5b-instruct-q4_0.gguf";
const PROMPT: &str = "Write a short poem about autumn leaves.";
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

fn skip_if_model_missing() -> bool {
    !Path::new(MODEL).exists()
}

fn run_with_flag(disable_avx512: bool) -> (String, bool) {
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
    ]);
    if disable_avx512 {
        cmd.env("ROCMFORGE_DISABLE_AVX512", "1");
    } else {
        cmd.env_remove("ROCMFORGE_DISABLE_AVX512");
    }
    let output = cmd.output().expect("failed to launch rocmforge binary");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        output.status.success(),
    )
}

/// Strip the runtime banner lines so that only the generated text is
/// compared. Banner content (timings, cache info) is run-dependent and
/// not a correctness signal.
fn extract_generation(stdout: &str) -> String {
    // Find the line "Prompt: N tokens" and take everything between it and
    // the final "N tokens in ... tok/s" line.
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
    out
}

#[test]
fn avx512_greedy_matches_avx2() {
    if skip_if_model_missing() {
        eprintln!("SKIP: model not found at {}", MODEL);
        return;
    }

    let (out_avx2, ok_avx2) = run_with_flag(true);
    let (out_avx512, ok_avx512) = run_with_flag(false);

    assert!(ok_avx2, "AVX2 reference run failed");
    assert!(ok_avx512, "AVX-512 run failed");

    let gen_avx2 = extract_generation(&out_avx2);
    let gen_avx512 = extract_generation(&out_avx512);

    if gen_avx2 != gen_avx512 {
        eprintln!("=== AVX2 (reference) ===\n{}", gen_avx2);
        eprintln!("=== AVX-512 ===\n{}", gen_avx512);
        panic!("AVX-512 greedy decode diverged from AVX2 reference");
    }
}
