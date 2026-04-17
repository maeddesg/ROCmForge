//! Interactive chat driver.
//!
//! Phase 5 Step 2: single-turn inference with token-level streaming.
//! The user's line is wrapped in the Qwen2.5 ChatML template, tokenised
//! (the tokenizer recognises the literal `<|im_start|>` / `<|im_end|>`
//! markers as special tokens), fed through the same `gpu_prefill_forward_hybrid`
//! + `gpu_full_forward_hybrid` pair that the legacy `--prompt` path uses,
//! and the sampled tokens stream to stdout one at a time with `print!` +
//! `flush`. Each turn is independent — the KV cache is reused but
//! rewritten from position 0 (multi-turn history lands in Step 3).

use std::io::{self, BufRead, Write};
use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

#[cfg(feature = "gpu")]
use rocmforge::cpu::cache::CpuForwardScratch;
#[cfg(feature = "gpu")]
use rocmforge::cpu::sampler::cpu_sample_top_p;
#[cfg(feature = "gpu")]
use rocmforge::cpu::weights::CpuModelWeights;
#[cfg(feature = "gpu")]
use rocmforge::gpu;

use super::template::{self, ChatContext, STOP_MARKERS};
use super::validate;

/// Cap on the KV-cache size we allocate for single-turn chat. Keeps VRAM
/// overhead modest (~230 MB at 7B / 4096 positions / FP16 KV) while
/// comfortably covering prompt + reply for the Step 2 scope.
const CHAT_MAX_SEQ: usize = 4096;

/// Chat-specific CLI args parsed from the raw argv tail.
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
    eprintln!("  --draft-model <path>   Draft model for speculative decoding [recognised but ignored in Step 2]");
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
}

pub fn run(args: &[String]) -> Result<(), String> {
    let args = parse_chat_args(args);

    if args.draft_model.is_some() || args.spec_depth != 5 {
        eprintln!("  [note] Speculative decoding will be available in a future update; --draft-model / --spec-depth are ignored in Step 2.");
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

#[cfg(feature = "gpu")]
fn input_loop(session: &mut ChatSession) -> Result<(), String> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
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
                println!("  Commands:");
                println!("    /quit, /exit  — leave the chat");
                println!("    /help         — show this help");
                println!();
            }
            _ => {
                if let Err(e) = run_turn(session, input) {
                    eprintln!("  [error] {}", e);
                }
            }
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
                println!("  Commands: /quit, /exit, /help");
            }
            _ => {
                println!("  [GPU feature required for inference]");
            }
        }
    }
}

/// Run one chat turn: template → tokenise → prefill → streaming decode.
/// Public for integration tests; the input loop calls it per line.
#[cfg(feature = "gpu")]
pub fn run_turn(session: &mut ChatSession, user_input: &str) -> Result<TurnOutcome, String> {
    let ctx = ChatContext {
        system_prompt: session.args.system.clone(),
        user_input: user_input.to_string(),
    };
    let prompt = template::format_single_turn(&ctx);

    // `false` for add_special: the template already emits the special tokens
    // as literal strings, which the BPE tokenizer resolves to their IDs via
    // split_by_special_tokens.
    let prompt_tokens = session.tokenizer.encode(&prompt, false);
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

    println!();
    let decode_time = decode_start.elapsed();
    let tps = if decode_time.as_secs_f64() > 0.0 {
        generated as f64 / decode_time.as_secs_f64()
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

    Ok(TurnOutcome {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        decode_tps: tps,
        stopped_by,
        response: accumulated,
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
}

impl StopReason {
    fn label(&self) -> &'static str {
        match self {
            StopReason::EogToken => "EOS",
            StopReason::StopMarker => "stop",
            StopReason::MaxTokens => "max tokens",
            StopReason::ContextFull => "ctx full",
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
}
