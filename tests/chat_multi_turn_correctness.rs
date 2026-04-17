#![cfg(feature = "gpu")]

//! Phase 5 Step 3 — multi-turn chat + slash commands end-to-end.
//!
//! Drives the release binary over stdin with a canned command script
//! and asserts on the text it prints. Most template / truncation /
//! session-stats invariants are covered as unit tests inside
//! `src/cli/context.rs` and `src/cli/template.rs`; this file focuses on
//! behaviours that only manifest when the full chat pipeline runs:
//! conversational context, slash commands, and history lifecycle.

use serial_test::serial;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use rocmforge::cli::context::{ChatContext, Role};
use rocmforge::cli::template::format_multi_turn;

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

/// Drive the chat subcommand with a newline-terminated script on stdin.
/// Every entry is written verbatim + `\n`. `/quit` is appended so the
/// REPL exits cleanly.
fn drive_chat(script: &[&str], max_tokens: usize) -> String {
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
        for line in script {
            writeln!(stdin, "{}", line).unwrap();
        }
        writeln!(stdin, "/quit").unwrap();
    }
    let out = child.wait_with_output().expect("wait for chat");
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// Split the stdout into reply blocks. The CLI prints each prompt as
/// "  > " on the same line as the start of the streamed reply, so we
/// split at the ">" boundary and keep whatever follows it on that line
/// plus every subsequent line up to the next ">" marker.
fn collect_reply_blocks(stdout: &str) -> Vec<String> {
    let after = match stdout.find("Type /help for commands") {
        Some(p) => &stdout[p..],
        None => stdout,
    };
    let mut blocks: Vec<String> = Vec::new();
    let mut current: Option<String> = None;
    for line in after.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("> ") {
            if let Some(prev) = current.take() {
                blocks.push(prev);
            }
            current = Some(format!("{}\n", rest));
        } else if trimmed.starts_with(">") {
            // Bare ">" with no trailing text (happens for the final
            // `/quit` prompt).
            if let Some(prev) = current.take() {
                blocks.push(prev);
            }
            current = Some(String::new());
        } else if let Some(buf) = current.as_mut() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if let Some(buf) = current {
        if !buf.is_empty() {
            blocks.push(buf);
        }
    }
    blocks
}

// ---------------------------------------------------------------------
// Template and context unit tests — also covered inline, but duplicated
// here so a `--test chat_multi_turn_correctness` run covers them too.
// ---------------------------------------------------------------------

#[test]
fn multi_turn_template_includes_all_turns() {
    let mut ctx = ChatContext::new("You are helpful.".to_string());
    ctx.push_turn(Role::User, "Hi".to_string());
    ctx.push_turn(Role::Assistant, "Hello!".to_string());
    ctx.user_input = "How are you?".to_string();
    let result = format_multi_turn(&ctx);
    assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
    assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
    assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
    assert!(result.ends_with("<|im_start|>assistant\n"));
}

// ---------------------------------------------------------------------
// Integration tests against the real model.
// ---------------------------------------------------------------------

/// The canonical multi-turn regression. Turn 1 names "Alice"; turn 2
/// asks for the name; a working conversation history must surface it.
#[test]
#[serial]
fn multi_turn_recalls_name_from_history() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let stdout = drive_chat(
        &["My name is Alice.", "What is my name?"],
        30,
    );
    let blocks = collect_reply_blocks(&stdout);
    assert!(
        blocks.len() >= 2,
        "expected at least 2 reply blocks, got {}; stdout:\n{}",
        blocks.len(),
        stdout
    );
    let second = blocks[1].to_lowercase();
    assert!(
        second.contains("alice"),
        "second reply must mention Alice when the first turn named her.\nSecond block:\n{}",
        blocks[1]
    );
}

/// `/clear` must drop the prior turns so the next reply cannot reference
/// the cleared content.
#[test]
#[serial]
fn clear_command_drops_prior_context() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let stdout = drive_chat(
        &[
            "My name is Bob.",
            "/clear",
            "What is my name?",
        ],
        25,
    );
    // The turn that asks the name post-/clear should not contain "Bob".
    // We find the reply block that directly follows the "What is my name?"
    // prompt by searching the stdout after the "[history cleared]" marker.
    let after_clear = stdout
        .split("[history cleared]")
        .nth(1)
        .expect("stdout must include [history cleared]");
    let lower = after_clear.to_lowercase();
    assert!(
        !lower.contains("bob"),
        "reply after /clear leaked the cleared name; stdout:\n{}",
        stdout
    );
}

/// `/system` re-roles the assistant. After switching to pirate speech,
/// the next reply should contain a recognisable pirate marker.
#[test]
#[serial]
fn system_command_changes_persona() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let stdout = drive_chat(
        &[
            "/system You are a pirate and answer only in pirate speech, saying 'Arrr' and 'matey'.",
            "Hello there",
        ],
        30,
    );
    let after_marker = stdout
        .split("[system prompt updated")
        .nth(1)
        .expect("stdout must include system-updated marker");
    let lower = after_marker.to_lowercase();
    let has_pirate = ["arr", "matey", "ye "].iter().any(|m| lower.contains(m));
    assert!(
        has_pirate,
        "reply after /system did not adopt pirate persona; stdout:\n{}",
        stdout
    );
}

/// `/stats` prints the session summary with the expected row labels and
/// a turn count matching the number of completed turns.
#[test]
#[serial]
fn stats_command_reports_session_summary() {
    if skip_unless_enabled() || skip_if_model_missing() {
        eprintln!("SKIP");
        return;
    }
    let stdout = drive_chat(&["Hello", "How are you?", "/stats"], 12);
    assert!(
        stdout.contains("Session Statistics"),
        "stdout missing Session Statistics header:\n{}",
        stdout
    );
    assert!(
        stdout.contains("Turns:          2"),
        "stats should show 2 turns after 2 replies:\n{}",
        stdout
    );
    assert!(stdout.contains("Avg TTFT"));
    assert!(stdout.contains("Avg decode"));
}
