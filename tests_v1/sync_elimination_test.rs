//! Phase 2 / Schritt 2.0.1 — Bandit sync-elimination tests.
//!
//! Checks the P0-gate from `ga_tuning_spec.md §8.1`: per-kernel
//! `hipStreamSynchronize` calls have been replaced by HIP-event
//! pairs drained once per token. Target: ≤ 200 syncs per 100
//! decode tokens (v0.x reference: 114).

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::backend::gpu::wrappers::{reset_sync_count, sync_count};
use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::inference::InferencePipeline;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::model_loader::LoadedModel;
use rocmforge::v1::core::sampling::SamplingConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
};
use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use rocmforge::v1::runtime::{Runtime, ShapeBandit, VariantId, VariantRegistry};
use serial_test::serial;

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir().expect("HOME set").join("models").join(name)
}

fn load_pipeline_qwen3_with_tuning() -> InferencePipeline<'static> {
    let device = GpuDevice::detect(0).expect("gpu");
    let path = model_path("Qwen3-8B-Q4_K_M.gguf");
    let model = LoadedModel::load(&path, &device).expect("load model");
    let gguf = GGUFFile::open(&path).expect("reopen gguf");
    let model_static: &'static LoadedModel = Box::leak(Box::new(model));
    let gguf_static: &'static GGUFFile = Box::leak(Box::new(gguf));

    let cfg = ModelConfig::from_metadata(gguf_static.metadata(), gguf_static.tensors())
        .expect("cfg");
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
    let mut pipe = InferencePipeline::new(graph, plan, model_static, gguf_static, 256)
        .expect("pipeline");
    pipe.executor
        .attach_runtime(Runtime::new(VariantRegistry::new()));
    pipe
}

// ── CPU-only Bandit phase flag ───────────────────────────────────────────

#[test]
fn test_bandit_is_exploiting_threshold() {
    // Boundary case: exactly at the threshold documented in Arch-Doc
    // §2.5 — every arm ≥ 5 pulls AND total > 50.
    let mut b = ShapeBandit::new(&[VariantId(0), VariantId(1)]);
    for _ in 0..49 {
        let v = b.select();
        b.record(v, 100.0);
    }
    assert!(
        !b.is_exploiting(),
        "bandit with total_pulls=49 should still explore"
    );
    b.record(b.select(), 100.0);
    b.record(b.select(), 100.0);
    // One more pull pushes total_pulls to 51, both arms ≥ 5 by now.
    assert!(
        b.is_exploiting(),
        "bandit with total_pulls>50 and all arms ≥5 should exploit"
    );
}

#[test]
fn test_runtime_all_exploiting_empty_registry() {
    // A registry with no multi-variant shapes has no bandits at all.
    // `all_exploiting` is vacuously true so the executor can skip
    // event recording from the first token.
    let runtime = Runtime::new(VariantRegistry::new());
    assert!(runtime.all_exploiting());
}

// ── P0 Gate ──────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_p0_gate_sync_count_under_200() {
    let mut pipe = load_pipeline_qwen3_with_tuning();
    pipe.reset().expect("reset");

    // Zero the counter after pipeline construction — the warm-up
    // path (model upload, kv reset, etc.) has legitimate syncs that
    // aren't part of the decode hot loop.
    reset_sync_count();

    let result = pipe
        .generate(
            "Explain what a mutex is in one paragraph.",
            100,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    assert!(result.generated_tokens > 0);

    let sync_total = sync_count();
    println!(
        "sync count for 100-token decode (post-reset): {}",
        sync_total
    );
    assert!(
        sync_total < 200,
        "P0 gate: expected < 200 HipStream::synchronize calls per 100 tokens, got {}",
        sync_total
    );
}

// ── Bandit still converges on q8_inline ──────────────────────────────────

#[test]
#[serial]
fn test_bandit_still_converges_with_events() {
    let mut pipe = load_pipeline_qwen3_with_tuning();
    pipe.reset().expect("reset");
    let result = pipe
        .generate(
            "Write a detailed essay about CPU architecture.",
            128,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    assert!(result.generated_tokens > 0);

    let rt = pipe.executor.runtime().expect("runtime attached");
    assert!(
        rt.all_exploiting(),
        "all bandits should be in exploitation after 128 decode tokens"
    );
    // Every multi-variant shape should have picked `q4_k_q8_inline`
    // as its dominant arm (>55% of pulls).
    for (shape, bandit) in &rt.bandits {
        let winner = bandit.best_variant();
        let winner_pulls = bandit
            .arms
            .iter()
            .find(|a| a.variant_id == winner)
            .map(|a| a.n_pulls)
            .unwrap_or(0);
        let pct = winner_pulls as f64 / bandit.total_pulls.max(1) as f64;
        assert!(
            pct > 0.55,
            "shape {:?}: winner {:?} only {:.0}% — convergence broke",
            shape,
            winner,
            pct * 100.0
        );
    }
}

// ── Decode quality + speed ───────────────────────────────────────────────

#[test]
#[serial]
fn test_decode_quality_unchanged_and_faster() {
    let mut pipe = load_pipeline_qwen3_with_tuning();
    pipe.reset().expect("reset");
    let result = pipe
        .generate(
            "Explain what a mutex is in one paragraph.",
            100,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    assert!(!result.output.is_empty());
    // Output must still mention the canonical terms — a sanity
    // check that the sync removal didn't silently race buffers.
    let lower = result.output.to_lowercase();
    assert!(
        lower.contains("mutex") || lower.contains("mutual"),
        "output should contain 'mutex'/'mutual', got: {:?}",
        result.output
    );
    // Phase-1 baseline on this prompt with tuning was 30.6 tok/s.
    // With sync-elimination we expect ≥ 33 tok/s (doc estimate
    // 5-10 %); test threshold is conservative at 32.
    assert!(
        result.decode_tok_s > 32.0,
        "decode should exceed 32 tok/s after sync-elim, got {:.1}",
        result.decode_tok_s
    );
    println!(
        "decode {:.1} tok/s (phase-1 baseline 30.6)",
        result.decode_tok_s
    );
}
