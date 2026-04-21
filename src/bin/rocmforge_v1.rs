//! `rocmforge-v1` — v1.0 inference CLI entry point.
//!
//! Phase 1 supports a single diagnostic command: `--list-tensors`. GPU
//! loading is exercised via the integration-test harness in
//! `tests_v1/gguf_test.rs`.

#[cfg(feature = "v1")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut list_tensors = false;
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

    eprintln!("Phase 1 only supports --list-tensors; nothing else to do.");
    std::process::exit(2);
}

#[cfg(feature = "v1")]
fn print_help() {
    eprintln!(
        "Usage: rocmforge-v1 --model <path.gguf> [--list-tensors]

Phase 1 commands:
  --list-tensors  Print tensor inventory, metadata summary, quant breakdown."
    );
}

#[cfg(not(feature = "v1"))]
fn main() {
    eprintln!("rocmforge-v1 requires the `v1` feature: cargo run --features v1 --bin rocmforge-v1");
    std::process::exit(2);
}
