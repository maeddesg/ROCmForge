#![cfg(feature = "gpu")]

//! Phase 5 Step 2 — chat template + single-turn inference correctness.
//!
//! Exercises the `rocmforge chat` subcommand end-to-end via the release
//! binary (simpler than building an in-process harness around
//! `ChatSession`). Tests are driven by piping stdin and parsing the
//! banner / reply / metrics the CLI prints.

use serial_test::serial;
use std::path::Path;
use std::process::{Command, Stdio};
use std::io::Write;

const MODEL: &str = "/home/maeddes/models/Qwen2.5-7B-Instruct-Q4_0.gguf";

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

/// Drive one chat turn over stdin. Returns the full stdout blob.
fn run_chat_turn(user_input: &str, max_tokens: usize) -> String {
    let mut child = Command::new(binary_path())
        .args([
            "chat",
            "--model",
            MODEL,
            "--max-tokens",
            &max_tokens.to_string(),
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("launch rocmforge chat");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        writeln!(stdin, "{}", user_input).unwrap();
        writeln!(stdin, "/quit").unwrap();
    }
    let out = child.wait_with_output().expect("wait for chat");
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// Extract the reply line between the banner and the "[TTFT:" metrics line.
fn extract_reply(stdout: &str) -> String {
    let marker = "Type /help for commands";
    let after_banner = match stdout.find(marker) {
        Some(p) => &stdout[p..],
        None => stdout,
    };
    let lines: Vec<&str> = after_banner.lines().collect();
    // Find the assistant line: the one right after "  > user_input" and before "[TTFT:"
    for (i, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("[TTFT:") {
            if i >= 1 {
                return lines[i - 1].trim().to_string();
            }
        }
    }
    String::new()
}

/// Pull metrics tuple (ttft_ms, tps, tokens) out of the "[TTFT:" line.
fn extract_metrics(stdout: &str) -> Option<(f64, f64, usize)> {
    for line in stdout.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("[TTFT: ") {
            // Format: "<ms>ms | <tps> tok/s | <n> tokens | <reason>]"
            let parts: Vec<&str> = rest.splitn(4, " | ").collect();
            if parts.len() < 4 {
                continue;
            }
            let ms: f64 = parts[0].trim_end_matches("ms").parse().ok()?;
            let tps: f64 = parts[1].trim_end_matches(" tok/s").parse().ok()?;
            let n: usize = parts[2].trim_end_matches(" tokens").parse().ok()?;
            return Some((ms, tps, n));
        }
    }
    None
}

#[test]
fn chat_template_round_trips_via_tokenizer() {
    // Independently-defined copy of the Qwen2.5 ChatML template — the unit
    // test in src/cli/template.rs verifies the generator produces this same
    // string. Here we care about the tokeniser round-trip.
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let formatted = "<|im_start|>system\nTest.<|im_end|>\n\
                     <|im_start|>user\nHello<|im_end|>\n\
                     <|im_start|>assistant\n";
    let file = rocmforge::loader::GgufFile::open(MODEL).expect("open model");
    let tok = rocmforge::tokenizer::BpeTokenizer::from_gguf(file.tokenizer_data());
    let ids = tok.encode(formatted, false);
    assert!(
        ids.len() >= 10,
        "tokenised template unexpectedly short: {} tokens",
        ids.len()
    );
    // The ChatML special tokens should round-trip through decode.
    let decoded = tok.decode(&ids, false);
    assert!(
        decoded.contains("system"),
        "round-trip lost 'system' marker: {:?}",
        decoded
    );
    assert!(
        decoded.contains("Hello"),
        "round-trip lost user input: {:?}",
        decoded
    );
}

#[test]
#[serial]
fn chat_turn_produces_valid_output() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let stdout = run_chat_turn("Hello", 20);
    let reply = extract_reply(&stdout);
    assert!(
        !reply.is_empty(),
        "chat produced no reply line; stdout was:\n{}",
        stdout
    );
    assert!(
        !reply.contains("<|im_start|>") && !reply.contains("<|im_end|>"),
        "chat reply contains unstripped template tokens: {:?}",
        reply
    );
    let metrics = extract_metrics(&stdout).expect("metrics line present");
    let (_ttft, tps, n) = metrics;
    assert!(
        n > 0,
        "chat produced zero tokens; stdout was:\n{}",
        stdout
    );
    assert!(tps > 0.0, "tok/s must be positive, got {}", tps);
}

#[test]
#[serial]
fn chat_is_deterministic_at_temperature_zero() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    // Short prompt and a small token budget to bound runtime.
    let a = run_chat_turn("What is 2 plus 2?", 8);
    let b = run_chat_turn("What is 2 plus 2?", 8);
    let reply_a = extract_reply(&a);
    let reply_b = extract_reply(&b);
    assert!(!reply_a.is_empty(), "run A produced no reply");
    assert_eq!(
        reply_a, reply_b,
        "chat not deterministic at temp=0:\n  A: {:?}\n  B: {:?}",
        reply_a, reply_b
    );
}
