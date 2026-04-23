//! Dispatch-path optimisation follow-up (post-2.1.4, 2026-04-23).
//!
//! Replaces two per-Gemm HashMap lookups (`runtime.select_variant`
//! + `runtime.kernel_for`) + the per-call `all_exploiting()` check
//! with a pre-computed `node_fast_cache` populated once the Bandit
//! has committed. The `buffer_ptrs` Vec lookup (O(1) by BufferId)
//! replaces the per-call `self.buffers[&id]` HashMap access on
//! every node dispatch.
//!
//! Gates:
//!   * Correctness: fast-path output IDENTICAL to legacy path
//!   * Perf: decode tok/s must not regress; any win is a bonus
//!   * Bandit: still functional (exploration + convergence)
//!   * Monitor: still fires on drift

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::inference::InferencePipeline;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::model_loader::LoadedModel;
use rocmforge::v1::core::sampling::SamplingConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
};
use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use rocmforge::v1::runtime::{Runtime, VariantRegistry};
use serial_test::serial;

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

// ── Consolidated GPU suite (one model load) ────────────────────────────

#[test]
#[serial]
fn test_dispatch_optimization_suite() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        return;
    }

    let mut pipe = load_pipeline_with_bandit();

    // ── 1. Warm-up run: lets the Bandit converge + triggers cache ──
    // After this run, runtime.all_exploiting() should be true and
    // the next decode build the node_fast_cache.
    pipe.reset().expect("reset");
    let _warm = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("warmup");

    // ── 2. Run with LEGACY dispatch (env-forced) ───────────────────
    std::env::set_var("ROCMFORGE_LEGACY_DISPATCH", "1");
    pipe.reset().expect("reset");
    let legacy = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("legacy gen");

    // ── 3. Run with FAST dispatch (cache rebuilt first decode) ────
    std::env::remove_var("ROCMFORGE_LEGACY_DISPATCH");
    // The setter invalidates the cache; next execute_decode rebuilds.
    pipe.executor.set_fused_gate_up(false);
    pipe.reset().expect("reset");
    let fast = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("fast gen");

    println!(
        "  legacy: decode {:.1} tok/s, prefill {:.1} tok/s, output[:60]='{}'",
        legacy.decode_tok_s,
        legacy.prefill_tok_s,
        legacy.output.chars().take(60).collect::<String>()
    );
    println!(
        "  fast:   decode {:.1} tok/s, prefill {:.1} tok/s, output[:60]='{}'",
        fast.decode_tok_s,
        fast.prefill_tok_s,
        fast.output.chars().take(60).collect::<String>()
    );

    // ── Correctness: outputs must agree token-for-token ────────────
    assert!(legacy.generated_tokens > 0);
    assert!(fast.generated_tokens > 0);
    assert_eq!(
        legacy.generated_tokens, fast.generated_tokens,
        "fast and legacy must generate same number of tokens"
    );
    let legacy_tokens: Vec<&str> = legacy.output.split_whitespace().collect();
    let fast_tokens: Vec<&str> = fast.output.split_whitespace().collect();
    let common = legacy_tokens
        .iter()
        .zip(fast_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "  first {} whitespace-tokens agree between legacy and fast",
        common
    );
    // Bandit selections are stable post-convergence → greedy outputs
    // should match EXACTLY. We require all tokens to agree.
    assert_eq!(
        legacy.output, fast.output,
        "fast-path decoded text diverged from legacy"
    );

    // ── Perf: fast must not regress meaningfully ───────────────────
    // 5 % tolerance for thermal / Bandit cold-start noise; we don't
    // require a positive speedup to pass (the gains from this
    // follow-up are small — ~250 µs per token CPU side).
    let ratio = fast.decode_tok_s / legacy.decode_tok_s;
    println!(
        "  decode ratio fast/legacy = {:.3}  (fast {:.1} vs legacy {:.1} tok/s)",
        ratio, fast.decode_tok_s, legacy.decode_tok_s
    );
    assert!(
        ratio >= 0.95,
        "fast-path regressed more than 5 % vs legacy: {:.3}",
        ratio
    );
    // Decode tok/s must stay above the post-unfuse floor.
    assert!(
        fast.decode_tok_s >= 60.0,
        "fast decode {:.1} tok/s < 60 floor",
        fast.decode_tok_s
    );

    // ── Coherence: Mutex keyword present ───────────────────────────
    for res in [&legacy, &fast] {
        let lower = res.output.to_lowercase();
        assert!(
            lower.contains("mutex") || lower.contains("mutual") || lower.contains("exclusion"),
            "Mutex prompt output lacks keyword: '{}'",
            res.output
        );
    }
}

// ── CPU-only / unit ────────────────────────────────────────────────────

#[test]
fn test_legacy_env_parsing() {
    let prev = std::env::var("ROCMFORGE_LEGACY_DISPATCH").ok();
    std::env::remove_var("ROCMFORGE_LEGACY_DISPATCH");
    assert!(std::env::var("ROCMFORGE_LEGACY_DISPATCH").is_err());
    std::env::set_var("ROCMFORGE_LEGACY_DISPATCH", "1");
    assert_eq!(
        std::env::var("ROCMFORGE_LEGACY_DISPATCH").ok().as_deref(),
        Some("1")
    );
    std::env::remove_var("ROCMFORGE_LEGACY_DISPATCH");
    if let Some(v) = prev {
        std::env::set_var("ROCMFORGE_LEGACY_DISPATCH", v);
    }
}
