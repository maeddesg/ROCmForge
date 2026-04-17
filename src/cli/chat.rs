//! Interactive chat driver.
//!
//! Phase 5 Step 1: argument parsing, startup validation, model load
//! (weights on GPU so the banner reports real VRAM usage), and an input
//! loop that recognises `/help` and `/quit` but still echoes every other
//! line back. Inference wiring lands in Step 2.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use rocmforge::config::ModelConfig;
use rocmforge::loader::GgufFile;

#[cfg(feature = "gpu")]
use rocmforge::gpu;

use super::validate::{self, StartupInfo};

/// Subset of the full `Args` struct, tuned to what the chat loop needs.
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
    eprintln!("  --draft-model <path>   Draft model for speculative decoding [Step 1: recognised but ignored]");
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

/// Holds everything the chat loop needs at runtime. Parked here for Step 2;
/// in Step 1 we only materialise the model weights so that the banner can
/// report realistic VRAM figures.
#[allow(dead_code)]
pub struct ChatSession {
    pub args: ChatArgs,
    pub file: GgufFile,
    pub config: ModelConfig,
    #[cfg(feature = "gpu")]
    pub gpu_weights: gpu::GpuModelWeights,
}

pub fn run(args: &[String]) -> Result<(), String> {
    let args = parse_chat_args(args);

    if args.draft_model.is_some() || args.spec_depth != 5 {
        eprintln!("  [note] Speculative decoding will be available in a future update; --draft-model / --spec-depth are ignored in Step 1.");
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
        eprint!("  Loading model weights to GPU... ");
        let t = Instant::now();
        let gpu_weights = gpu::GpuModelWeights::load(&file, &config)
            .map_err(|e| format!("gpu weight load: {}", e))?;
        eprintln!("done in {:.1}s", t.elapsed().as_secs_f64());
        validate::refresh_vram_usage(&mut info);
        validate::print_banner(&info);
        let _session = ChatSession {
            args,
            file,
            config,
            gpu_weights,
        };
        return input_loop();
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = file;
        let _ = config;
        validate::print_banner(&info);
        input_loop()
    }
}

fn input_loop() -> Result<(), String> {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    loop {
        print!("  > ");
        io::stdout().flush().ok();

        let mut line = String::new();
        match handle.read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl-D)
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
                println!(
                    "  [received {} chars, inference arrives in Step 2]",
                    input.len()
                );
                println!();
            }
        }
    }
}
