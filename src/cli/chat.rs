//! Interactive chat driver.
//!
//! Phase 5 Step 3: multi-turn conversation, `/clear` `/stats` `/system`
//! commands, and Ctrl+C interrupt handling that aborts the in-flight
//! generation but keeps the REPL open. Use `/quit` to leave.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::ModelConfig;
use crate::loader::GgufFile;
use crate::tokenizer::BpeTokenizer;

#[cfg(feature = "gpu")]
use crate::cpu::cache::CpuForwardScratch;
#[cfg(feature = "gpu")]
use crate::cpu::sampler::cpu_sample_top_p;
#[cfg(feature = "gpu")]
use crate::cpu::weights::CpuModelWeights;
#[cfg(feature = "gpu")]
use crate::gpu;

use super::context::{ChatContext, Role, SessionStats};
use super::template::{self, STOP_MARKERS};
use super::validate;

/// KV-cache budget per session; also the truncation threshold for the
/// formatted prompt. Comfortably covers multi-turn conversations without
/// overflowing the 7B VRAM envelope.
const CHAT_MAX_SEQ: usize = 4096;

/// Reserve this many tokens in the context window for the assistant's
/// response when deciding whether to truncate the history.
const RESPONSE_HEADROOM: usize = 512;

/// Global Ctrl+C flag. Set by the signal handler, polled by the decode
/// loop. An `OnceLock`-style init keeps us safe against `set_handler`
/// panicking on a second registration if `run()` is ever re-invoked.
static INTERRUPT: once_cell::sync::Lazy<Arc<AtomicBool>> =
    once_cell::sync::Lazy::new(|| Arc::new(AtomicBool::new(false)));
static INTERRUPT_INSTALLED: AtomicBool = AtomicBool::new(false);

fn install_interrupt_handler() {
    if INTERRUPT_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    let flag = Arc::clone(&INTERRUPT);
    // Ignore an error here — worst case Ctrl+C falls through to the
    // default handler and kills the process, which is still acceptable
    // behavior for an interactive CLI.
    let _ = ctrlc::set_handler(move || {
        flag.store(true, Ordering::SeqCst);
    });
}

fn interrupt_flag() -> Arc<AtomicBool> {
    Arc::clone(&INTERRUPT)
}

pub struct ChatArgs {
    pub model: String,
    pub system: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub draft_model: Option<String>,
    pub spec_depth: usize,
}

fn usage() -> ! {
    eprintln!("rocmforge chat — interactive chat loop");
    eprintln!();
    eprintln!("Usage: rocmforge chat --model <path> [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --model <path>         Path to GGUF model file");
    eprintln!();
    eprintln!("Optional:");
    eprintln!("  --system <text>        System prompt [default: \"You are a helpful assistant.\"]");
    eprintln!("  --max-tokens N         Max tokens per reply [default: 512]");
    eprintln!("  --temperature F        Sampling temperature [default: 0.0]");
    eprintln!("  --top-p F              Nucleus sampling threshold [default: 1.0]");
    eprintln!("  --draft-model <path>   Draft model for speculative decoding [recognised but ignored]");
    eprintln!("  --spec-depth N         Speculation depth [default: 5]");
    std::process::exit(1);
}

fn parse_chat_args(args: &[String]) -> ChatArgs {
    let mut model: Option<String> = None;
    let mut system = String::from("You are a helpful assistant.");
    let mut max_tokens = 512usize;
    let mut temperature = 0.0f32;
    let mut top_p = 1.0f32;
    let mut draft_model: Option<String> = None;
    let mut spec_depth = 5usize;

    let mut it = args.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "-m" | "--model" => model = Some(it.next().cloned().unwrap_or_else(|| usage())),
            "--system" => system = it.next().cloned().unwrap_or_else(|| usage()),
            "--max-tokens" => {
                max_tokens = it
                    .next()
                    .cloned()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-t" | "--temp" | "--temperature" => {
                temperature = it
                    .next()
                    .cloned()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--top-p" => {
                top_p = it
                    .next()
                    .cloned()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--draft-model" => {
                draft_model = Some(it.next().cloned().unwrap_or_else(|| usage()))
            }
            "--spec-depth" => {
                spec_depth = it
                    .next()
                    .cloned()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("Unknown flag: {}", other);
                usage();
            }
        }
    }

    ChatArgs {
        model: model.unwrap_or_else(|| usage()),
        system,
        max_tokens,
        temperature,
        top_p,
        draft_model,
        spec_depth,
    }
}

#[cfg(feature = "gpu")]
pub struct ChatSession {
    pub args: ChatArgs,
    pub file: GgufFile,
    pub config: ModelConfig,
    pub tokenizer: BpeTokenizer,
    pub cpu_weights: CpuModelWeights,
    pub gpu_weights: gpu::GpuModelWeights,
    pub device: gpu::GpuDevice,
    pub kv: gpu::GpuKvCache,
    pub gpu_scratch: gpu::GpuForwardScratch,
    pub host_scratch: CpuForwardScratch,
    pub max_seq: usize,
    pub ctx: ChatContext,
    pub stats: SessionStats,
}

pub fn run(args: &[String]) -> Result<(), String> {
    install_interrupt_handler();
    let args = parse_chat_args(args);

    if args.draft_model.is_some() || args.spec_depth != 5 {
        eprintln!("  [note] Speculative decoding will be available in a future update; --draft-model / --spec-depth are ignored.");
    }
    if !(0.0..=2.0).contains(&args.temperature) {
        return Err(format!(
            "--temperature must be in [0.0, 2.0], got {}",
            args.temperature
        ));
    }
    if !(0.0..=1.0).contains(&args.top_p) {
        return Err(format!("--top-p must be in [0.0, 1.0], got {}", args.top_p));
    }

    let (mut info, file, config) = validate::validate_before_load(&args.model)?;

    #[cfg(feature = "gpu")]
    {
        let tokenizer = BpeTokenizer::from_gguf(file.tokenizer_data());

        eprint!("  Loading CPU weights... ");
        let t = Instant::now();
        let cpu_weights = CpuModelWeights::load(&file, &config)
            .map_err(|e| format!("cpu weight load: {}", e))?;
        eprintln!("done in {:.1}s", t.elapsed().as_secs_f64());

        eprint!("  Loading GPU weights... ");
        let t = Instant::now();
        let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
            .map_err(|e| format!("gpu weight load: {}", e))?;
        eprintln!("done in {:.1}s", t.elapsed().as_secs_f64());

        let gpu_caps = gpu::detect()
            .ok_or_else(|| "GPU disappeared after detect()".to_string())?;
        let device = gpu::GpuDevice::init(gpu_caps.device_id)
            .map_err(|e| format!("gpu init: {}", e))?;

        let max_seq = config.max_seq_len.min(CHAT_MAX_SEQ);
        let kv = gpu::GpuKvCache::new(&config, max_seq)
            .map_err(|e| format!("gpu kv cache: {}", e))?;
        let gpu_scratch = gpu::GpuForwardScratch::new(&config)
            .map_err(|e| format!("gpu scratch: {}", e))?;
        let host_scratch = CpuForwardScratch::new(&config);

        validate::refresh_vram_usage(&mut info);
        validate::print_banner(&info);

        let chat_ctx = ChatContext::new(args.system.clone());
        let mut session = ChatSession {
            args,
            file,
            config,
            tokenizer,
            cpu_weights,
            gpu_weights,
            device,
            kv,
            gpu_scratch,
            host_scratch,
            max_seq,
            ctx: chat_ctx,
            stats: SessionStats::new(),
        };
        return input_loop(&mut session);
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = file;
        let _ = config;
        validate::print_banner(&info);
        cpu_only_loop()
    }
}

fn print_help() {
    println!("  Commands:");
    println!("    /clear            Clear conversation history");
    println!("    /stats            Show session statistics");
    println!("    /system <text>    Change system prompt (clears history)");
    println!("    /quit, /exit      Exit the chat");
    println!("    /help             Show this help");
    println!();
}

#[cfg(feature = "gpu")]
fn print_stats(stats: &SessionStats) {
    let dur = stats.session_duration();
    let total_s = dur.as_secs();
    let minutes = total_s / 60;
    let seconds = total_s % 60;
    println!();
    println!("  Session Statistics");
    println!("  ─────────────────────────────────");
    println!("  Turns:          {}", stats.turn_count);
    println!(
        "  Total tokens:   {} prompt + {} generated",
        stats.total_prompt_tokens, stats.total_generated_tokens
    );
    println!("  Avg TTFT:       {:.0} ms", stats.avg_ttft_ms());
    println!("  Avg decode:     {:.1} tok/s", stats.avg_decode_tps());
    println!("  Session time:   {}m {:02}s", minutes, seconds);
    println!("  ─────────────────────────────────");
    println!();
}

#[cfg(feature = "gpu")]
enum SlashOutcome {
    Continue,
    Quit,
    NotACommand,
}

#[cfg(feature = "gpu")]
fn handle_slash_command(input: &str, session: &mut ChatSession) -> SlashOutcome {
    if !input.starts_with('/') {
        return SlashOutcome::NotACommand;
    }
    let mut parts = input.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let rest = parts.next().map(|s| s.trim()).unwrap_or("");

    match cmd {
        "/quit" | "/exit" => {
            println!("  Goodbye.");
            SlashOutcome::Quit
        }
        "/help" => {
            print_help();
            SlashOutcome::Continue
        }
        "/clear" => {
            session.ctx.clear_history();
            println!("  [history cleared]");
            println!();
            SlashOutcome::Continue
        }
        "/stats" => {
            print_stats(&session.stats);
            SlashOutcome::Continue
        }
        "/system" => {
            if rest.is_empty() {
                println!("  Usage: /system <new system prompt>");
                println!();
            } else {
                session.ctx.set_system_prompt(rest.to_string());
                session.args.system = rest.to_string();
                println!("  [system prompt updated — history cleared to apply new system prompt]");
                println!();
            }
            SlashOutcome::Continue
        }
        _ => {
            println!("  Unknown command: {}. Type /help for the list.", cmd);
            println!();
            SlashOutcome::Continue
        }
    }
}

#[cfg(feature = "gpu")]
fn input_loop(session: &mut ChatSession) -> Result<(), String> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    loop {
        // Clear any stale interrupt from a previous turn.
        interrupt_flag().store(false, Ordering::SeqCst);

        print!("  > ");
        io::stdout().flush().ok();

        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => {
                println!();
                println!("  Goodbye.");
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(format!("read stdin: {}", e)),
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match handle_slash_command(input, session) {
            SlashOutcome::Quit => return Ok(()),
            SlashOutcome::Continue => continue,
            SlashOutcome::NotACommand => {}
        }

        if let Err(e) = run_turn(session, input) {
            eprintln!("  [error] {}", e);
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn cpu_only_loop() -> Result<(), String> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    println!("  (GPU feature disabled — chat loop runs without inference.)");
    loop {
        print!("  > ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => {
                println!();
                println!("  Goodbye.");
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(format!("read stdin: {}", e)),
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/quit" | "/exit" => {
                println!("  Goodbye.");
                return Ok(());
            }
            "/help" => {
                println!("  Commands: /quit, /exit, /help (GPU feature required for full command set)");
            }
            _ => {
                println!("  [GPU feature required for inference]");
            }
        }
    }
}

/// One chat turn: truncate history if needed, format, tokenise, run
/// prefill + streaming decode, append to history, record stats.
#[cfg(feature = "gpu")]
pub fn run_turn(session: &mut ChatSession, user_input: &str) -> Result<TurnOutcome, String> {
    session.ctx.user_input = user_input.to_string();

    // Truncate to fit the formatted prompt + response headroom inside
    // the KV cache.
    let budget = session.max_seq.saturating_sub(RESPONSE_HEADROOM).max(512);
    let arch = session.config.architecture.clone();
    let dropped = {
        let tok = &session.tokenizer;
        let arch_ref = arch.as_str();
        super::context::truncate_if_needed(
            &mut session.ctx,
            budget,
            |c| template::format_multi_turn_for(arch_ref, c),
            |s| tok.encode(s, true).len(),
        )
    };
    if dropped > 0 {
        let remaining = session.ctx.conversation_history.len() / 2;
        tracing::debug!(
            dropped_pairs = dropped,
            remaining_pairs = remaining,
            budget_tokens = budget,
            "History truncated to fit KV-cache budget"
        );
        println!(
            "  [context truncated: dropped {} oldest turn{}, {} turn{} remaining]",
            dropped,
            if dropped == 1 { "" } else { "s" },
            remaining,
            if remaining == 1 { "" } else { "s" },
        );
    }

    let prompt = template::format_multi_turn_for(&arch, &session.ctx);
    let prompt_tokens = session.tokenizer.encode(&prompt, true);
    if prompt_tokens.is_empty() {
        return Err("prompt tokenised to zero tokens".to_string());
    }
    if prompt_tokens.len() >= session.max_seq {
        return Err(format!(
            "prompt is {} tokens but chat KV cache is sized for {}",
            prompt_tokens.len(),
            session.max_seq
        ));
    }

    let use_greedy = session.args.top_p >= 1.0 && session.args.temperature == 0.0;
    let logits_mode = if use_greedy {
        gpu::GpuLogitsMode::GreedyArgmax
    } else {
        gpu::GpuLogitsMode::DownloadToHost
    };

    let int_flag = interrupt_flag();
    int_flag.store(false, Ordering::SeqCst);

    let prefill_start = Instant::now();
    let mut prefill_scratch = gpu::GpuPrefillScratch::new(&session.config, prompt_tokens.len())
        .map_err(|e| format!("prefill scratch: {}", e))?;
    let first_token = gpu::gpu_prefill_forward_hybrid(
        &session.device,
        &session.gpu_weights,
        &session.cpu_weights,
        &mut session.kv,
        &mut prefill_scratch,
        &mut session.gpu_scratch,
        &mut session.host_scratch,
        &prompt_tokens,
        0,
        &session.config,
        logits_mode,
    )
    .map_err(|e| format!("gpu prefill: {}", e))?;
    let ttft = prefill_start.elapsed();

    drop(prefill_scratch);

    let mut seed: u64 = 0xdeadbeef;
    let mut next_token = resolve_token_after_forward(
        first_token,
        &session.host_scratch,
        &session.gpu_scratch,
        &session.device,
        use_greedy,
        session.args.temperature,
        session.args.top_p,
        &mut seed,
        session.config.vocab_size,
    )?;

    let decode_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut generated = 0usize;
    let max_pos = session.max_seq;
    let max_tokens = session.args.max_tokens;
    let mut accumulated = String::new();
    let mut stopped_by = StopReason::MaxTokens;

    print!("  ");
    io::stdout().flush().ok();

    loop {
        if int_flag.load(Ordering::SeqCst) {
            stopped_by = StopReason::Interrupted;
            break;
        }
        if session.tokenizer.is_eog(next_token) {
            stopped_by = StopReason::EogToken;
            break;
        }
        if generated >= max_tokens {
            stopped_by = StopReason::MaxTokens;
            break;
        }
        if pos >= max_pos {
            stopped_by = StopReason::ContextFull;
            break;
        }

        let text = session.tokenizer.decode_token(next_token);
        if !text.is_empty() {
            print!("{}", text);
            io::stdout().flush().ok();
            accumulated.push_str(&text);
            if STOP_MARKERS.iter().any(|m| accumulated.ends_with(m)) {
                // Strip the stop marker from the visible reply.
                for m in STOP_MARKERS {
                    if accumulated.ends_with(m) {
                        let cut = accumulated.len() - m.len();
                        accumulated.truncate(cut);
                        break;
                    }
                }
                stopped_by = StopReason::StopMarker;
                break;
            }
        }
        generated += 1;

        gpu::gpu_embed_token_hybrid(
            &session.device,
            next_token,
            &session.gpu_weights,
            &session.cpu_weights,
            &mut session.gpu_scratch,
            &mut session.host_scratch,
            &session.config,
        )
        .map_err(|e| format!("gpu embed: {}", e))?;
        let decoded = gpu::gpu_full_forward_hybrid(
            &session.device,
            &session.gpu_weights,
            &session.cpu_weights,
            &mut session.kv,
            &mut session.gpu_scratch,
            &mut session.host_scratch,
            pos,
            &session.config,
            logits_mode,
        )
        .map_err(|e| format!("gpu decode: {}", e))?;
        pos += 1;

        next_token = resolve_token_after_forward(
            decoded,
            &session.host_scratch,
            &session.gpu_scratch,
            &session.device,
            use_greedy,
            session.args.temperature,
            session.args.top_p,
            &mut seed,
            session.config.vocab_size,
        )?;
    }

    if stopped_by == StopReason::Interrupted {
        println!();
        println!("  [interrupted]");
    } else {
        println!();
    }

    let decode_time = decode_start.elapsed();
    let decode_s = decode_time.as_secs_f64();
    let tps = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    println!(
        "  [TTFT: {:.0}ms | {:.1} tok/s | {} tokens | {}]",
        ttft.as_millis(),
        tps,
        generated,
        stopped_by.label()
    );
    println!();

    // Record history: append the user turn plus whatever the assistant
    // produced (even if interrupted — partial output is still relevant
    // context).
    session
        .ctx
        .push_turn(Role::User, user_input.to_string());
    session
        .ctx
        .push_turn(Role::Assistant, accumulated.clone());

    session.stats.record(
        prompt_tokens.len(),
        generated,
        ttft.as_secs_f64() * 1000.0,
        decode_s,
    );

    Ok(TurnOutcome {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        decode_tps: tps,
        stopped_by,
        response: accumulated,
        dropped_history_pairs: dropped,
    })
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn resolve_token_after_forward(
    returned: Option<u32>,
    host_scratch: &CpuForwardScratch,
    gpu_scratch: &gpu::GpuForwardScratch,
    device: &gpu::GpuDevice,
    use_greedy: bool,
    temperature: f32,
    top_p: f32,
    seed: &mut u64,
    vocab_size: usize,
) -> Result<u32, String> {
    if let Some(tok) = returned {
        return Ok(tok);
    }
    device
        .synchronize()
        .map_err(|e| format!("gpu sync: {}", e))?;
    if use_greedy {
        let slot = gpu_scratch.argmax_result_index.as_slice::<i32>()[0];
        if slot < 0 || (slot as usize) >= vocab_size {
            return Err(format!("gpu argmax returned out-of-range index {}", slot));
        }
        Ok(slot as u32)
    } else {
        *seed = seed.wrapping_add(1);
        Ok(cpu_sample_top_p(
            &host_scratch.logits,
            temperature,
            top_p,
            *seed,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EogToken,
    StopMarker,
    MaxTokens,
    ContextFull,
    Interrupted,
}

impl StopReason {
    fn label(&self) -> &'static str {
        match self {
            StopReason::EogToken => "EOS",
            StopReason::StopMarker => "stop",
            StopReason::MaxTokens => "max tokens",
            StopReason::ContextFull => "ctx full",
            StopReason::Interrupted => "interrupted",
        }
    }
}

#[cfg(feature = "gpu")]
pub struct TurnOutcome {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub ttft_ms: f64,
    pub decode_tps: f64,
    pub stopped_by: StopReason,
    pub response: String,
    pub dropped_history_pairs: usize,
}
