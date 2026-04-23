//! Phase 2 / Schritt 2.1.3 Block E — Performance-Analyse + Regression-Check.
//!
//! Block D endete mit einem Null-Ergebnis auf End-to-End Decode:
//! 33.2 tok/s bei 50 Tokens vs. 40.7 tok/s in der 2.0.3-Baseline bei
//! 100 Tokens. Block E klärt ob das (a) ein Token-Count-Warmup-
//! Artefakt, (b) ein Hook-Dispatch-Overhead, oder (c) eine echte
//! Block-D-Regression am statischen Pfad ist.
//!
//! Der Haupttest `test_block_e_regression_benchmark` misst in einem
//! Prozess mit EINEM Model-Load vier Varianten hintereinander, damit
//! thermale + Bandit-Startup-Unterschiede zwischen den Läufen
//! minimal sind:
//!
//!   Run A  — Static, 100 Tokens, Mutex-Prompt   (≙ 2.0.3 Baseline)
//!   Run A' — Static, 50 Tokens,  Mutex-Prompt   (≙ Block D Setup)
//!   Run B  — Hook (w=8, c=4), 100 Tokens        (Hook-Overhead @ Default)
//!   Run C  — Hook (GA-Winner), 100 Tokens       (volle Tuned-Version)
//!
//! Bandit wird in jedem Run via `attach_runtime` aktiviert — genau
//! so wie die CLI `--show-tuning` die 2.0.3-Messung gefahren hat.
//!
//! Diese Tests sind hinter `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1`
//! und dem Qwen3-8B-Q4_K_M GGUF in `~/models/` gegatet.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::backend::gpu::wrappers::{reset_sync_count, sync_count};
use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::inference::{GenerationResult, InferencePipeline};
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::model_loader::LoadedModel;
use rocmforge::v1::core::sampling::SamplingConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
};
use rocmforge::v1::ga::block_d::{
    block_d_default_config, make_gate_up_hook, BlockDGenome, DynamicKernelCache2D,
};
use rocmforge::v1::ga::engine::KernelGa;
use rocmforge::v1::ga::logger::GaLogger;
use rocmforge::v1::ga::types::{KernelShape, PrecisionLevel};
use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use rocmforge::v1::ir::formats::q4_k;
use rocmforge::v1::runtime::{Runtime, VariantRegistry};
use serial_test::serial;
use std::sync::Arc;

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
const MUTEX_PROMPT: &str = "Explain what a mutex is in one paragraph.";

fn model_path() -> std::path::PathBuf {
    dirs::home_dir().expect("HOME").join("models").join(QWEN3)
}

fn real_model_tests_enabled() -> bool {
    std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}

/// Load the inference pipeline with the Self-Tuning Runtime (Bandit)
/// attached — matches the CLI's `--show-tuning` path, i.e. the 2.0.3
/// baseline conditions. Uses `Box::leak` for the `'static` reference
/// needed by the pipeline lifetime (memory reclaimed at process
/// exit).
fn load_pipeline_with_bandit() -> InferencePipeline<'static> {
    let device = GpuDevice::detect(0).expect("gpu");
    let path = model_path();
    let model = LoadedModel::load(&path, &device).expect("load model");
    let gguf = GGUFFile::open(&path).expect("reopen gguf");
    let model_static: &'static LoadedModel = Box::leak(Box::new(model));
    let gguf_static: &'static GGUFFile = Box::leak(Box::new(gguf));
    let cfg =
        ModelConfig::from_metadata(gguf_static.metadata(), gguf_static.tensors()).expect("cfg");
    let layers = group_tensors_by_layer(gguf_static.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf_static.tensors() {
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
    let graph = GraphBuilder::build(&ctx).expect("build graph");
    let plan = BufferPlan::plan_phase1(&graph);
    let mut pipe =
        InferencePipeline::new(graph, plan, model_static, gguf_static, 512).expect("pipeline");
    pipe.executor
        .attach_runtime(Runtime::new(VariantRegistry::new()));
    pipe.calibrate_monitor().expect("calibrate");
    pipe
}

/// Synthetic Q4_K weights for GA tuning at a given `(k, n)` shape.
fn gen_q4k_weights(n_rows: usize, ncols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(n_rows % 256, 0);
    let blocks_per_col = n_rows / 256;
    let total = ncols * blocks_per_col;
    let mut buf = vec![0u8; total * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..16 {
            buf[b * 144 + i] = rng.u8(..) & 0x3F;
        }
        for i in 16..144 {
            buf[b * 144 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_input(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Run a short GA to get a winner genome at the model's actual
/// `(hidden_dim, ffn_dim)` shape.
fn ga_winner_for(
    hidden_dim: usize,
    ffn_dim: usize,
) -> (BlockDGenome, Arc<rocmforge::v1::ga::dynamic::DynamicKernel>) {
    let parity = KernelShape::new(1, 512, 4096);
    let bench = KernelShape::new(1, ffn_dim, hidden_dim);
    let bwg = gen_q4k_weights(bench.k, bench.n, 0xBD1);
    let bwu = gen_q4k_weights(bench.k, bench.n, 0xBD2);
    let bi = gen_input(bench.k, 0xBD3);
    let pwg = gen_q4k_weights(parity.k, parity.n, 0xBE1);
    let pwu = gen_q4k_weights(parity.k, parity.n, 0xBE2);
    let pi = gen_input(parity.k, 0xBE3);
    let mut cache = DynamicKernelCache2D::new();
    let mut logger = GaLogger::capturing("block-e-ga".to_string());
    let mut ga = KernelGa::new(block_d_default_config(42));
    let r = ga.run_num_waves_and_multi_row_cols(
        &parity,
        &bench,
        &q4_k(),
        PrecisionLevel::Fp16,
        432.8,
        &bwg,
        &bwu,
        &bi,
        &pwg,
        &pwu,
        &pi,
        &mut cache,
        &mut logger,
    );
    let winner = r.all_candidates[0].genome;
    let kernel = cache.get_or_compile(winner).expect("winner recompile");
    (winner, kernel)
}

/// Compile a specific `(w, c)` kernel for the model's FFN shape —
/// used by Run B where we need the Phase-1 default `(w=8, c=4)`
/// going through the dynamic-dispatch path.
fn compile_kernel_for(genome: BlockDGenome) -> Arc<rocmforge::v1::ga::dynamic::DynamicKernel> {
    let mut cache = DynamicKernelCache2D::new();
    cache.get_or_compile(genome).expect("compile specific")
}

/// Pretty-print a single generate() result row.
fn print_run(label: &str, max_tok: usize, r: &GenerationResult) {
    println!(
        "  {:<32}  {:>3} tok  prefill {:>5.1} tok/s  decode {:>5.1} tok/s  total {:>5.0} ms  eos={}",
        label,
        r.generated_tokens,
        r.prefill_tok_s,
        r.decode_tok_s,
        r.total_ms,
        r.hit_eos
    );
    let _ = max_tok;
}

// ── Main regression benchmark — 4 runs in one process ─────────────────────

#[test]
#[serial]
fn test_block_e_regression_benchmark() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        eprintln!("skipping — {} not found", model_path().display());
        return;
    }

    let mut pipe = load_pipeline_with_bandit();
    let hidden_dim = pipe.config.hidden_dim;
    let ffn_dim = pipe.config.ffn_dim;
    println!(
        "=== Block E regression benchmark — Qwen3-8B (hidden={}, ffn={}, layers={}) ===",
        hidden_dim, ffn_dim, pipe.config.n_layers
    );

    let sampling = SamplingConfig::greedy();

    // Pre-compile everything we'll need so the clocks during the
    // timed runs aren't affected by hipcc spawns. This takes a few
    // seconds but isolates the timing concern cleanly.
    let phase1_kernel = compile_kernel_for(BlockDGenome::phase1_default());
    let (ga_genome, ga_kernel) = ga_winner_for(hidden_dim, ffn_dim);

    // ── Run A: Static, 100 Tokens, Mutex ───────────────────────────
    pipe.executor.set_gate_up_swiglu_dynamic_kernel(None);
    assert!(pipe.executor.gate_up_swiglu_dynamic_kernel().is_none());
    pipe.reset().expect("reset A");
    let run_a = pipe
        .generate(MUTEX_PROMPT, 100, &sampling, true)
        .expect("run A");

    // ── Run A': Static, 50 Tokens, Mutex — token-count effect ───────
    pipe.reset().expect("reset A'");
    let run_a_50 = pipe
        .generate(MUTEX_PROMPT, 50, &sampling, true)
        .expect("run A50");

    // ── Run B: Hook + Phase-1-Default (w=8, c=4), 100 Tokens ───────
    pipe.executor
        .set_gate_up_swiglu_dynamic_kernel(Some(make_gate_up_hook(
            phase1_kernel.clone(),
            hidden_dim,
            ffn_dim,
        )));
    pipe.reset().expect("reset B");
    let run_b = pipe
        .generate(MUTEX_PROMPT, 100, &sampling, true)
        .expect("run B");

    // ── Run C: Hook + GA-Winner, 100 Tokens ────────────────────────
    pipe.executor
        .set_gate_up_swiglu_dynamic_kernel(Some(make_gate_up_hook(
            ga_kernel.clone(),
            hidden_dim,
            ffn_dim,
        )));
    pipe.reset().expect("reset C");
    let run_c = pipe
        .generate(MUTEX_PROMPT, 100, &sampling, true)
        .expect("run C");

    // ── Bonus: Static, 100 Tokens AGAIN to check warm-steady-state ─
    // This is *after* the hook runs — we clear the hook and re-run
    // to confirm the static path doesn't degrade across repeated
    // invocations (a regression signal a single Run A couldn't
    // catch).
    pipe.executor.set_gate_up_swiglu_dynamic_kernel(None);
    pipe.reset().expect("reset A2");
    let run_a2 = pipe
        .generate(MUTEX_PROMPT, 100, &sampling, true)
        .expect("run A2");

    // Remove hook once more to leave the pipeline clean.
    pipe.executor.set_gate_up_swiglu_dynamic_kernel(None);

    // ── Report ─────────────────────────────────────────────────────
    println!("\nResults (all runs, Mutex prompt):");
    print_run("A  — Static,   100 tok", 100, &run_a);
    print_run("A' — Static,    50 tok", 50, &run_a_50);
    print_run("B  — Hook w8c4, 100 tok", 100, &run_b);
    print_run("C  — Hook GA,   100 tok", 100, &run_c);
    print_run("A2 — Static,   100 tok (repeat)", 100, &run_a2);
    println!("\nGA winner used for Run C: {:?}", ga_genome);

    let hook_overhead_b = 1.0 - (run_b.decode_tok_s / run_a.decode_tok_s);
    let hook_overhead_c = 1.0 - (run_c.decode_tok_s / run_a.decode_tok_s);
    let token_count_ratio = run_a_50.decode_tok_s / run_a.decode_tok_s;
    println!(
        "Hook overhead (B vs A, same config):  {:+.1}% ({} → {} tok/s)",
        hook_overhead_b * 100.0,
        run_a.decode_tok_s as u32,
        run_b.decode_tok_s as u32
    );
    println!(
        "Tuned delta   (C vs A, GA winner):    {:+.1}% ({} → {} tok/s)",
        hook_overhead_c * 100.0,
        run_a.decode_tok_s as u32,
        run_c.decode_tok_s as u32
    );
    println!(
        "Token-count effect (50tok / 100tok):  {:.3}× ({} → {} tok/s)",
        token_count_ratio, run_a_50.decode_tok_s as u32, run_a.decode_tok_s as u32
    );
    println!(
        "Static repeatability (A vs A2):       {:.1}% drift",
        (run_a2.decode_tok_s - run_a.decode_tok_s).abs() / run_a.decode_tok_s * 100.0
    );

    // ── Assertions ─────────────────────────────────────────────────
    // Static 100-token path must not regress more than 10% below the
    // 2.0.3 reference (40.7 tok/s). Thermal + Bandit-cold-start noise
    // typically lands within 5%; anything >10% points to a real
    // Block-D code regression on the static path.
    assert!(
        run_a.decode_tok_s >= 36.0,
        "REGRESSION: static 100-tok decode ({} tok/s) far below 2.0.3 baseline 40.7",
        run_a.decode_tok_s
    );
    // Hook overhead at same kernel must be < 10% — a single Option
    // field + match arm on dispatch shouldn't cost much.
    assert!(
        hook_overhead_b < 0.10,
        "Hook dispatch overhead too high: {:.1}%",
        hook_overhead_b * 100.0
    );
    // Tuned run must produce output (no crash/NaN).
    assert!(
        run_c.generated_tokens > 0 && !run_c.output.trim().is_empty(),
        "tuned run C produced no output"
    );
    // Static path is stable across repeats.
    let repeat_drift = (run_a2.decode_tok_s - run_a.decode_tok_s).abs() / run_a.decode_tok_s;
    assert!(
        repeat_drift < 0.10,
        "Static path drifted {:.1}% between A and A2 — non-idempotent?",
        repeat_drift * 100.0
    );
}

// ── Sync count ────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_block_e_sync_count_under_p0_gate() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        return;
    }

    let mut pipe = load_pipeline_with_bandit();
    pipe.reset().expect("reset");

    reset_sync_count();
    let r = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("generate");
    let syncs = sync_count();
    println!(
        "Sync count for 100-tok Mutex run: {} (reference 2.0.3: 129, P0 gate: < 200)",
        syncs
    );
    println!(
        "  generated {} tokens, decode {:.1} tok/s, {:.0} ms total",
        r.generated_tokens, r.decode_tok_s, r.total_ms
    );
    assert!(syncs < 200, "P0 sync-count regression: {} ≥ 200", syncs);
}

// ── Short 5-prompt smoke on both paths ────────────────────────────────────

#[test]
#[serial]
fn test_block_e_short_suite_static_and_tuned() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        return;
    }

    // Hand-picked subset of the 15-prompt suite: two smoke, one
    // code-gen, one conversational, one math. Keeps wall-clock
    // down while still covering breadth.
    let prompts: &[(&str, usize)] = &[
        ("Hallo", 64),
        ("Zähle von 1 bis 10", 64),
        (
            "Write a function in Python that returns the nth Fibonacci number.",
            160,
        ),
        ("Erzähle mir einen kurzen Witz.", 96),
        ("Was ist 17 mal 23?", 32),
    ];

    let mut pipe = load_pipeline_with_bandit();
    let hidden_dim = pipe.config.hidden_dim;
    let ffn_dim = pipe.config.ffn_dim;
    let (_, ga_kernel) = ga_winner_for(hidden_dim, ffn_dim);

    let sampling = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.05,
        seed: 0,
    };

    println!("=== Static path (no hook) ===");
    pipe.executor.set_gate_up_swiglu_dynamic_kernel(None);
    let mut static_decode_sum = 0.0;
    let mut static_tokens_sum = 0;
    for (p, cap) in prompts {
        pipe.reset().expect("reset");
        let r = pipe.generate(p, *cap, &sampling, true).expect("gen");
        println!(
            "  [{}...]  {} tok / {:.1} tok/s",
            p.chars().take(24).collect::<String>(),
            r.generated_tokens,
            r.decode_tok_s
        );
        assert!(r.generated_tokens > 0, "static: no tokens for {p:?}");
        assert!(
            !r.output.trim().is_empty(),
            "static: empty output for {p:?}"
        );
        static_decode_sum += r.decode_ms;
        static_tokens_sum += r.generated_tokens;
    }
    let static_agg = (static_tokens_sum as f64) / (static_decode_sum / 1000.0);

    println!("\n=== Tuned path (GA-winner hook) ===");
    pipe.executor
        .set_gate_up_swiglu_dynamic_kernel(Some(make_gate_up_hook(
            ga_kernel.clone(),
            hidden_dim,
            ffn_dim,
        )));
    let mut tuned_decode_sum = 0.0;
    let mut tuned_tokens_sum = 0;
    for (p, cap) in prompts {
        pipe.reset().expect("reset");
        let r = pipe.generate(p, *cap, &sampling, true).expect("gen");
        println!(
            "  [{}...]  {} tok / {:.1} tok/s",
            p.chars().take(24).collect::<String>(),
            r.generated_tokens,
            r.decode_tok_s
        );
        assert!(r.generated_tokens > 0, "tuned: no tokens for {p:?}");
        assert!(!r.output.trim().is_empty(), "tuned: empty output for {p:?}");
        tuned_decode_sum += r.decode_ms;
        tuned_tokens_sum += r.generated_tokens;
    }
    let tuned_agg = (tuned_tokens_sum as f64) / (tuned_decode_sum / 1000.0);
    pipe.executor.set_gate_up_swiglu_dynamic_kernel(None);

    println!(
        "\nAggregate decode tok/s:  static={:.1}  tuned={:.1}  delta={:+.1}%",
        static_agg,
        tuned_agg,
        (tuned_agg / static_agg - 1.0) * 100.0
    );
}
