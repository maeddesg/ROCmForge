//! `--inference-test` — run the 15-prompt validation suite.
//!
//! Loads the model once, builds one `InferencePipeline`, and iterates
//! over every prompt in `benches_v1/inference_test_prompts_15.json`.
//! `pipe.reset()` is called before every prompt so the KV-cache starts
//! clean — essential for comparable per-prompt metrics.
//!
//! Writes a Markdown report to the requested `--output` path with
//! human-fillable evaluation columns.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::v1::backend::gpu::device::GpuDevice;
use crate::v1::core::gguf::GGUFFile;
use crate::v1::core::inference::{GenerationResult, InferencePipeline};
use crate::v1::core::model_config::ModelConfig;
use crate::v1::core::model_loader::LoadedModel;
use crate::v1::core::sampling::SamplingConfig;
use crate::v1::core::tensor_info::{group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole};
use crate::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use crate::v1::introspection::ModelProfile;

/// Pick a sampling config appropriate for the model. When the
/// introspection scan flags the model as SNR-degraded
/// (`snr_risk_score < 2.0`, e.g. Llama-3.1-Q4_K_M with its ~182
/// special tokens dequantising below the noise floor), we add a
/// greedy `repeat_penalty=1.1` to stop the Q4_K-induced repetition
/// loops. High-SNR models stay on pure greedy for reproducible
/// output.
fn sampling_for(profile: &ModelProfile) -> SamplingConfig {
    if profile.snr_risk_score < 2.0 {
        eprintln!(
            "  ⚠ SNR-Risk {:.3} — using greedy + repeat_penalty=1.1 for stability",
            profile.snr_risk_score
        );
        SamplingConfig {
            repeat_penalty: 1.1,
            ..SamplingConfig::greedy()
        }
    } else {
        SamplingConfig::greedy()
    }
}

#[derive(Debug, Deserialize)]
struct PromptSuite {
    version: String,
    description: String,
    model_target: String,
    prompts: Vec<PromptEntry>,
}

#[derive(Debug, Deserialize)]
struct PromptEntry {
    id: u32,
    category: String,
    difficulty: String,
    name: String,
    max_tokens: usize,
    prompt: String,
    expected_behavior: String,
    quality_check: String,
}

struct PromptOutcome {
    entry: PromptEntry,
    result: GenerationResult,
}

pub fn run(
    model_path: impl AsRef<Path>,
    suite_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), String> {
    let model_path = model_path.as_ref().to_path_buf();
    let suite_path = suite_path.as_ref().to_path_buf();
    let output_path = output_path.as_ref().to_path_buf();

    let suite = load_suite(&suite_path)?;
    println!(
        "Suite: {} (v{}, {} prompts)\nModel: {}\n",
        suite.description,
        suite.version,
        suite.prompts.len(),
        model_path.display()
    );

    let device = GpuDevice::detect(0).map_err(|e| format!("GPU detect: {e}"))?;
    let model = LoadedModel::load(&model_path, &device)
        .map_err(|e| format!("model load: {e}"))?;
    let gguf = GGUFFile::open(&model_path).map_err(|e| format!("gguf reopen: {e}"))?;
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors())
        .map_err(|e| format!("model config: {e}"))?;

    let layers = group_tensors_by_layer(gguf.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf.tensors() {
        let (role, li) = parse_tensor_name(&t.name);
        if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
            globals.insert(role, t);
        }
    }
    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: globals,
    };
    let graph = GraphBuilder::build(&ctx).map_err(|e| format!("graph build: {e}"))?;
    let plan = BufferPlan::plan_phase1(&graph);

    // Max prompt in the suite needs max_seq ≥ longest prefill + longest
    // decode. The 1024-token prompts + 1024 max_tokens pushes this to
    // ~2048; give 3072 headroom.
    let max_seq = suite
        .prompts
        .iter()
        .map(|p| p.max_tokens + 512)
        .max()
        .unwrap_or(1024)
        .max(3072);

    let mut pipe = InferencePipeline::new(graph, plan, &model, &gguf, max_seq)
        .map_err(|e| format!("pipeline: {e}"))?;

    // Attach the Self-Tuning Runtime so Q4_K GEMVs go through the
    // Bandit. On the first few layers the Bandit explores both
    // variants; once q8_inline proves faster the rest of the run
    // stays on it. Kept unconditional so the Phase-1 validation
    // suite reflects the runtime users actually hit.
    pipe.executor.attach_runtime(crate::v1::runtime::Runtime::new(
        crate::v1::runtime::VariantRegistry::new(),
    ));

    // Calibrate the Quality Monitor — installs the mean-abs /
    // max-abs band for OUTPUT_HIDDEN so the per-32-token drift
    // check is actually meaningful. Phase 1 only logs drift
    // signals; Phase 2 uses them for precision escalation.
    pipe.calibrate_monitor().map_err(|e| format!("calibrate: {e}"))?;

    // Sampling is profile-aware: high-SNR models (Qwen3) run pure
    // greedy with the standing 1.05 safety-net penalty, low-SNR
    // models (Llama-3.1-Q4_K_M at SNR 0.023) fall back to the
    // sampling_for() path which escalates to repeat_penalty=1.1.
    let sampling = if pipe.profile.snr_risk_score < 2.0 {
        sampling_for(&pipe.profile)
    } else {
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.05,
            seed: 0,
        }
    };
    let mut outcomes: Vec<PromptOutcome> = Vec::with_capacity(suite.prompts.len());

    for (idx, entry) in suite.prompts.into_iter().enumerate() {
        pipe.reset().map_err(|e| format!("kv reset: {e}"))?;
        println!(
            "[{:2}/{}] {} — {} ({}/{})",
            idx + 1,
            outcomes.capacity(),
            entry.name,
            entry.category,
            entry.difficulty,
            entry.max_tokens
        );
        let result = pipe
            .generate(&entry.prompt, entry.max_tokens, &sampling, true)
            .map_err(|e| format!("generate prompt {}: {e}", entry.id))?;
        println!(
            "      prefill {:>3} tok in {:>6.0} ms ({:>5.1} tok/s) | decode {:>3} tok in {:>6.0} ms ({:>5.1} tok/s) | {}",
            result.prompt_tokens,
            result.prefill_ms,
            result.prefill_tok_s,
            result.generated_tokens,
            result.decode_ms,
            result.decode_tok_s,
            if result.hit_eos { "EOS" } else { "cap" },
        );
        outcomes.push(PromptOutcome { entry, result });
    }

    write_report(&output_path, &model_path, &suite.model_target, &outcomes)?;
    if let Some(rt) = pipe.executor.runtime() {
        rt.print_tuning_report();
    }
    // Quality Monitor end-of-suite summary — Phase 1 just
    // reports counts; caller can inspect `pipe.monitor.revision_log`
    // if it needs the raw events.
    println!(
        "Monitor events across suite: {}",
        pipe.monitor.revision_log.len()
    );
    for ev in pipe.monitor.revision_log.iter().take(8) {
        println!(
            "  token {} node {:?} — {:?}",
            ev.token_index, ev.node_id, ev.signal.reason
        );
    }
    println!("\nReport: {}", output_path.display());
    Ok(())
}

fn load_suite(path: &Path) -> Result<PromptSuite, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn write_report(
    out: &Path,
    model_path: &Path,
    suite_model_target: &str,
    outcomes: &[PromptOutcome],
) -> Result<(), String> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
    }
    let mut md = String::new();
    md.push_str("# ROCmForge v1.0 — Inference Validation Report\n\n");
    md.push_str(&format!(
        "- **Date:** {}\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    md.push_str(&format!("- **Model file:** `{}`\n", model_path.display()));
    md.push_str(&format!("- **Suite target:** `{}`\n", suite_model_target));
    md.push_str("- **Sampling:** greedy (temperature=0)\n");
    md.push_str("- **KV-cache:** reset between every prompt\n");
    md.push_str(&format!("- **Prompts:** {}\n\n", outcomes.len()));

    // Aggregate metrics.
    let total_prompt_tokens: usize = outcomes.iter().map(|o| o.result.prompt_tokens).sum();
    let total_decode_tokens: usize = outcomes.iter().map(|o| o.result.generated_tokens).sum();
    let total_prefill_ms: f64 = outcomes.iter().map(|o| o.result.prefill_ms).sum();
    let total_decode_ms: f64 = outcomes.iter().map(|o| o.result.decode_ms).sum();
    let total_ms: f64 = outcomes.iter().map(|o| o.result.total_ms).sum();
    let agg_prefill_tok_s = if total_prefill_ms > 0.0 {
        total_prompt_tokens as f64 / (total_prefill_ms / 1000.0)
    } else {
        0.0
    };
    let agg_decode_tok_s = if total_decode_ms > 0.0 {
        total_decode_tokens as f64 / (total_decode_ms / 1000.0)
    } else {
        0.0
    };
    md.push_str("## Aggregate\n\n");
    md.push_str(&format!(
        "| Prefill tok | Prefill ms | Prefill tok/s | Decode tok | Decode ms | Decode tok/s | Wallclock ms |\n"
    ));
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|\n");
    md.push_str(&format!(
        "| {} | {:.0} | {:.1} | {} | {:.0} | {:.1} | {:.0} |\n\n",
        total_prompt_tokens,
        total_prefill_ms,
        agg_prefill_tok_s,
        total_decode_tokens,
        total_decode_ms,
        agg_decode_tok_s,
        total_ms,
    ));

    // Per-prompt metrics table.
    md.push_str("## Per-prompt metrics\n\n");
    md.push_str("| # | Name | Category | Prefill tok | Decode tok | Prefill tok/s | Decode tok/s | Total ms | EOS |\n");
    md.push_str("|---:|---|---|---:|---:|---:|---:|---:|:-:|\n");
    for o in outcomes {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.0} | {} |\n",
            o.entry.id,
            o.entry.name,
            o.entry.category,
            o.result.prompt_tokens,
            o.result.generated_tokens,
            o.result.prefill_tok_s,
            o.result.decode_tok_s,
            o.result.total_ms,
            if o.result.hit_eos { "yes" } else { "no" },
        ));
    }
    md.push('\n');

    // Human evaluation sheet — one row per prompt, filled in manually.
    md.push_str("## Human evaluation (fill in manually)\n\n");
    md.push_str("Rating scale: **korrekt** / **teilweise** / **falsch** / **müll**.\n");
    md.push_str("Pass threshold: 12/15 korrekt for Phase-1 acceptance.\n\n");
    md.push_str("| # | Name | Rating | Notes |\n");
    md.push_str("|---:|---|:-:|---|\n");
    for o in outcomes {
        md.push_str(&format!("| {} | {} |   |   |\n", o.entry.id, o.entry.name));
    }
    md.push('\n');

    // Detailed outputs, one section each.
    md.push_str("## Outputs\n\n");
    for o in outcomes {
        md.push_str(&format!("### {}. {}\n\n", o.entry.id, o.entry.name));
        md.push_str(&format!(
            "- **Category:** {} / **difficulty:** {} / **max_tokens:** {}\n",
            o.entry.category, o.entry.difficulty, o.entry.max_tokens
        ));
        md.push_str(&format!(
            "- **Prefill:** {} tok in {:.0} ms ({:.1} tok/s)\n",
            o.result.prompt_tokens, o.result.prefill_ms, o.result.prefill_tok_s
        ));
        md.push_str(&format!(
            "- **Decode:** {} tok in {:.0} ms ({:.1} tok/s){}\n",
            o.result.generated_tokens,
            o.result.decode_ms,
            o.result.decode_tok_s,
            if o.result.hit_eos { " — hit EOS" } else { " — hit max_tokens" },
        ));
        md.push_str(&format!("- **Expected:** {}\n", o.entry.expected_behavior));
        md.push_str(&format!("- **Quality check:** {}\n\n", o.entry.quality_check));
        md.push_str("**Prompt:**\n\n");
        md.push_str("```\n");
        md.push_str(&o.entry.prompt);
        md.push_str("\n```\n\n");
        md.push_str("**Output:**\n\n");
        md.push_str("```\n");
        md.push_str(&o.result.output);
        md.push_str("\n```\n\n");
    }

    std::fs::write(out, md).map_err(|e| format!("write {}: {e}", out.display()))?;
    Ok(())
}

/// Opt-in diagnostics for `run_single_prompt`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShowFlags {
    /// After load, re-print the ModelProfile summary (redundant
    /// with the pipeline's own auto-print but useful as an
    /// explicit cue in scripts that pipe stdout).
    pub introspection: bool,
    /// Run the monitor calibration so the quality-monitor band
    /// is populated, and print the revision log at generation
    /// end — zero events on a clean run, one line per event
    /// otherwise.
    pub quality: bool,
    /// Attach the self-tuning runtime so the Q4_K GEMV bandit
    /// is exercised; print the per-shape convergence report at
    /// the end. Roughly doubles decode throughput vs. the
    /// fixed-kernel fallback.
    pub tuning: bool,
    /// Phase 2.4 — pass `<think>…</think>` tags through to the user
    /// (default: filter them out live). Only meaningful for Qwen3;
    /// other architectures never emit those tags.
    pub show_think: bool,
}

/// Startup banner — one line per model load so the user can see
/// at a glance what they're running. Called after the pipeline is
/// constructed but before any generation.
pub fn print_banner(
    model_path: &Path,
    config: &ModelConfig,
    device: &GpuDevice,
) {
    let model_name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.display().to_string());
    println!(
        "ROCmForge v1.0-dev | {} | {} ({}) | arch={} | {} layers | hidden={} heads={} vocab={}",
        model_name,
        device.name,
        device.gcn_arch_name,
        config.architecture,
        config.n_layers,
        config.hidden_dim,
        config.n_heads,
        config.vocab_size,
    );
}

/// Helper for the single-prompt CLI path. Builds a pipeline, runs one
/// generation, prints the output.
pub fn run_single_prompt(
    model_path: impl AsRef<Path>,
    prompt: &str,
    max_tokens: usize,
    show: ShowFlags,
) -> Result<(), String> {
    let model_path = model_path.as_ref().to_path_buf();
    let device = GpuDevice::detect(0).map_err(|e| format!("GPU detect: {e}"))?;
    let model = LoadedModel::load(&model_path, &device)
        .map_err(|e| format!("model load: {e}"))?;
    let gguf = GGUFFile::open(&model_path).map_err(|e| format!("gguf reopen: {e}"))?;
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors())
        .map_err(|e| format!("model config: {e}"))?;

    print_banner(&model_path, &cfg, &device);

    let layers = group_tensors_by_layer(gguf.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf.tensors() {
        let (role, li) = parse_tensor_name(&t.name);
        if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
            globals.insert(role, t);
        }
    }
    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: globals,
    };
    let graph = GraphBuilder::build(&ctx).map_err(|e| format!("graph build: {e}"))?;
    let plan = BufferPlan::plan_phase1(&graph);

    let max_seq = (max_tokens + 512).max(1024);
    let mut pipe = InferencePipeline::new(graph, plan, &model, &gguf, max_seq)
        .map_err(|e| format!("pipeline: {e}"))?;

    if show.introspection {
        pipe.profile.print_summary();
    }

    if show.tuning {
        pipe.executor
            .attach_runtime(crate::v1::runtime::Runtime::new(
                crate::v1::runtime::VariantRegistry::new(),
            ));
    }
    if show.quality {
        pipe.calibrate_monitor()
            .map_err(|e| format!("calibrate: {e}"))?;
    }

    pipe.reset_conversation()
        .map_err(|e| format!("reset: {e}"))?;

    let sampling = sampling_for(&pipe.profile);

    // Phase 2.4 — stream tokens live to stdout. Users see text as it
    // generates instead of waiting for the full output at the end.
    // `<think>…</think>` blocks are filtered by the StreamingEmitter
    // (default; `--show-think` turns filtering off — TODO wire when
    // the flag lands in the top-level CLI parser).
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let result = pipe
        .generate_turn_streaming(
            prompt,
            max_tokens,
            &sampling,
            /*apply_chat_template=*/ true,
            /*filter_think=*/ !show.show_think,
            |piece| {
                print!("{piece}");
                let _ = stdout.flush();
            },
        )
        .map_err(|e| format!("generate: {e}"))?;
    // Terminate the streaming line before the metrics summary.
    println!();
    eprintln!(
        "--- {} prompt tok, {} decode tok, {:.1} tok/s decode, {:.0} ms total ---",
        result.prompt_tokens,
        result.generated_tokens,
        result.decode_tok_s,
        result.total_ms
    );

    if show.tuning {
        if let Some(rt) = pipe.executor.runtime() {
            rt.print_tuning_report();
        }
    }
    if show.quality {
        println!(
            "Quality Monitor: {} event(s)",
            pipe.monitor.revision_log.len()
        );
        for ev in pipe.monitor.revision_log.iter().take(8) {
            println!(
                "  token {} node {:?} — {:?}",
                ev.token_index, ev.node_id, ev.signal.reason
            );
        }
    }
    Ok(())
}

/// REPL loop for the `--interactive` CLI path. **Builds the
/// pipeline exactly once** at session start; turns share the
/// KV cache so the model remembers previous exchanges.
///
/// Phase 2.4 upgrade:
///   * KV-cache persistence across turns (multi-turn chat works —
///     the Alice-test is the standing regression gate).
///   * Live streaming output: tokens appear as they generate.
///   * `<think>…</think>` blocks are filtered from the stream by
///     default (Qwen3 only; `--show-think` disables filtering).
///   * Commands: `/reset` starts a fresh conversation,
///     `/quit` or `/exit` leaves.
///
/// Max sequence length is bumped up for interactive use so a
/// handful of turns fit — Phase-1 sized the cache tightly for
/// single-shot runs.
pub fn run_interactive(
    model_path: impl AsRef<Path>,
    max_tokens: usize,
    show: ShowFlags,
) -> Result<(), String> {
    use std::io::{BufRead, Write};

    let model_path = model_path.as_ref().to_path_buf();
    let device = GpuDevice::detect(0).map_err(|e| format!("GPU detect: {e}"))?;
    let model = LoadedModel::load(&model_path, &device)
        .map_err(|e| format!("model load: {e}"))?;
    let gguf = GGUFFile::open(&model_path).map_err(|e| format!("gguf reopen: {e}"))?;
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors())
        .map_err(|e| format!("model config: {e}"))?;

    print_banner(&model_path, &cfg, &device);

    let layers = group_tensors_by_layer(gguf.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf.tensors() {
        let (role, li) = parse_tensor_name(&t.name);
        if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
            globals.insert(role, t);
        }
    }
    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: globals,
    };
    let graph = GraphBuilder::build(&ctx).map_err(|e| format!("graph build: {e}"))?;
    let plan = BufferPlan::plan_phase1(&graph);

    // Phase 2.4 — size the KV cache for a real conversation. Default
    // aims for ~10 typical turns (each ≈ 256 tokens); the ceiling is
    // bounded by the attention kernel's LDS budget (48 KiB of scores
    // → seq_len ≤ 12 288). If a user hits the cap mid-chat they get
    // a clean error and a prompt to `/reset`.
    let max_seq = ((max_tokens.saturating_mul(10)).max(4096)).min(8192);
    let mut pipe = InferencePipeline::new(graph, plan, &model, &gguf, max_seq)
        .map_err(|e| format!("pipeline: {e}"))?;

    if show.introspection {
        pipe.profile.print_summary();
    }
    if show.tuning {
        pipe.executor
            .attach_runtime(crate::v1::runtime::Runtime::new(
                crate::v1::runtime::VariantRegistry::new(),
            ));
    }
    if show.quality {
        pipe.calibrate_monitor()
            .map_err(|e| format!("calibrate: {e}"))?;
    }

    println!(
        "rocmforge-v1 chat mode. Commands: /reset (fresh chat), /quit or /exit (leave)."
    );
    println!(
        "Multi-turn: model remembers previous turns. KV cache size: {} tokens.\n",
        max_seq
    );

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let sampling = sampling_for(&pipe.profile);

    loop {
        print!("> ");
        stdout.flush().ok();
        let mut line = String::new();
        let read = stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        if read == 0 {
            println!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" || line == "quit" || line == "exit" {
            break;
        }
        if line == "/reset" {
            match pipe.reset_conversation() {
                Ok(_) => println!("(fresh chat started)"),
                Err(e) => eprintln!("reset failed: {e}"),
            }
            continue;
        }

        // Multi-turn generation: DO NOT reset between turns. The KV
        // cache retains the prior conversation; `generate_turn_streaming`
        // uses the continuation template starting with turn 2.
        match pipe.generate_turn_streaming(
            line,
            max_tokens,
            &sampling,
            /*apply_chat_template=*/ true,
            /*filter_think=*/ !show.show_think,
            |piece| {
                print!("{piece}");
                let _ = std::io::stdout().flush();
            },
        ) {
            Ok(result) => {
                println!();
                eprintln!(
                    "--- turn {} | {} prompt tok, {} decode tok, {:.1} tok/s decode, {:.0} ms | kv_pos={}/{} ---",
                    pipe.turn_count,
                    result.prompt_tokens,
                    result.generated_tokens,
                    result.decode_tok_s,
                    result.total_ms,
                    pipe.kv_pos,
                    pipe.max_seq,
                );
            }
            Err(e) => eprintln!("error: {e}"),
        }
        println!();
    }

    // Session summary — report bandit convergence and monitor
    // event count once at the end, not per turn.
    if show.tuning {
        if let Some(rt) = pipe.executor.runtime() {
            rt.print_tuning_report();
        }
    }
    if show.quality {
        println!(
            "Quality Monitor: {} event(s) across the session",
            pipe.monitor.revision_log.len()
        );
        for ev in pipe.monitor.revision_log.iter().take(8) {
            println!(
                "  token {} node {:?} — {:?}",
                ev.token_index, ev.node_id, ev.signal.reason
            );
        }
    }
    Ok(())
}

/// Default output path: `results/inference_test_YYYYMMDD.md`.
pub fn default_output_path() -> PathBuf {
    let date = chrono::Local::now().format("%Y%m%d");
    PathBuf::from(format!("results/inference_test_{date}.md"))
}
