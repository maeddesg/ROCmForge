//! `rocmforge-v1` — v1.0 inference CLI entry point.
//!
//! Phase 1 commands:
//! - `--list-tensors`: CPU-only GGUF inventory / diagnostics.
//! - `--prompt <text>`: single-shot generation with greedy sampling.
//! - `--inference-test`: 15-prompt validation suite, Markdown report.
//! - `--interactive`: REPL-style conversation loop (no context carryover
//!   between turns in Phase 1 — KV-cache is reset each turn).

#[cfg(feature = "v1")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut list_tensors = false;
    let mut prompt: Option<String> = None;
    let mut max_tokens: usize = 256;
    let mut inference_test = false;
    let mut suite_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut interactive = false;
    let mut show_introspection = false;
    let mut show_quality = false;
    let mut show_tuning = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--model requires a path");
                    std::process::exit(2);
                }
                model = Some(args[i].clone());
            }
            "--list-tensors" => list_tensors = true,
            "--prompt" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--prompt requires text");
                    std::process::exit(2);
                }
                prompt = Some(args[i].clone());
            }
            "--max-tokens" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--max-tokens requires a number");
                    std::process::exit(2);
                }
                max_tokens = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("--max-tokens must be a positive integer");
                    std::process::exit(2);
                });
            }
            "--inference-test" => inference_test = true,
            "--suite" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--suite requires a path");
                    std::process::exit(2);
                }
                suite_path = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--output requires a path");
                    std::process::exit(2);
                }
                output_path = Some(args[i].clone());
            }
            "--interactive" => interactive = true,
            "--show-introspection" => show_introspection = true,
            "--show-quality" => show_quality = true,
            "--show-tuning" => show_tuning = true,
            "--show-all" => {
                show_introspection = true;
                show_quality = true;
                show_tuning = true;
            }
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let Some(path) = model else {
        eprintln!("--model <path> is required");
        print_help();
        std::process::exit(2);
    };

    if list_tensors {
        if let Err(e) = rocmforge::v1::cli::list_tensors::run(&path) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
        return;
    }

    #[cfg(feature = "gpu")]
    {
        if inference_test {
            let suite = suite_path
                .unwrap_or_else(|| "benches_v1/inference_test_prompts_15.json".to_string());
            let out = output_path
                .map(std::path::PathBuf::from)
                .unwrap_or_else(rocmforge::v1::cli::inference_test::default_output_path);
            if let Err(e) = rocmforge::v1::cli::inference_test::run(&path, &suite, &out) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
            return;
        }
        let show = rocmforge::v1::cli::inference_test::ShowFlags {
            introspection: show_introspection,
            quality: show_quality,
            tuning: show_tuning,
        };
        if let Some(p) = prompt {
            if let Err(e) =
                rocmforge::v1::cli::inference_test::run_single_prompt(&path, &p, max_tokens, show)
            {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
            return;
        }
        if interactive {
            if let Err(e) = rocmforge::v1::cli::inference_test::run_interactive(
                &path,
                max_tokens,
                show,
            ) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
            return;
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        if inference_test || prompt.is_some() || interactive {
            eprintln!(
                "--prompt / --inference-test / --interactive require the `gpu` feature. \
                 Build with: cargo build --release --features v1,gpu --bin rocmforge-v1"
            );
            std::process::exit(2);
        }
    }

    eprintln!("No command given. Try --list-tensors, --prompt, --inference-test, or --interactive.");
    std::process::exit(2);
}

// Interactive REPL now lives in the library — see
// `rocmforge::v1::cli::inference_test::run_interactive`. It builds
// the pipeline once at session start and reuses it per turn,
// which the old binary-side loop did not (it re-built the whole
// pipeline for every line of user input).

#[cfg(feature = "v1")]
fn print_help() {
    eprintln!(
        "Usage: rocmforge-v1 --model <path.gguf> [command]

Commands:
  --list-tensors                     GGUF tensor inventory (CPU only)
  --prompt <text>                    Generate response for a single prompt
  --inference-test                   Run 15-prompt validation suite
  --interactive                      REPL loop (one-shot per turn)

Options:
  --max-tokens <N>                   Generation cap (default 256)
  --suite <path>                     Suite JSON for --inference-test
                                     (default: benches_v1/inference_test_prompts_15.json)
  --output <path>                    Report file for --inference-test
                                     (default: results/inference_test_<date>.md)
  --show-introspection               Print the ModelProfile summary
  --show-quality                     Calibrate + print the Quality Monitor report
  --show-tuning                      Attach the self-tuning runtime and print
                                     the Bandit convergence report
  --show-all                         Equivalent to --show-introspection
                                     --show-quality --show-tuning"
    );
}

#[cfg(not(feature = "v1"))]
fn main() {
    eprintln!("rocmforge-v1 requires the `v1` feature: cargo run --features v1 --bin rocmforge-v1");
    std::process::exit(2);
}
