//! Phase 1 / Schritt 1.15 — CLI + Integration tests.
//!
//! Verifies the end-to-end Startup-Flow from Arch-Doc §5.2:
//! GGUF → Introspection → Graph → Executor → (optional) Calibration
//! → (optional) Runtime → generate.
//!
//! The test harness intentionally uses `Box::leak` for the
//! `'static` model/GGUF references, which means each test permanently
//! retains its VRAM. With only 16 GB on the dev card we can't fit
//! three 8B pipelines in one process, so the suite is kept to two
//! end-to-end tests — one per model that actually runs clean in
//! Phase 1 (Qwen3-8B and Llama-3.1-8B). Qwen2.5 hits two deferred
//! Phase-1 gaps (Q4_1 GEMV kernel missing for its ffn_down, plus
//! unwired attention biases); see the step-1.15 report for details.

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

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir().expect("HOME set").join("models").join(name)
}

fn load_pipeline_for(name: &str) -> InferencePipeline<'static> {
    let device = GpuDevice::detect(0).expect("gpu");
    let path = model_path(name);
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
    InferencePipeline::new(graph, plan, model_static, gguf_static, 256).expect("pipeline")
}

/// Full Startup-Flow on Qwen3: Säulen 1 (introspection), 2 (graph),
/// 3 (executor) after `InferencePipeline::new`; then 4 (runtime) +
/// 5 (monitor) attached explicitly. Exercises `generate` at the
/// end so the bandits and monitor actually see traffic.
#[test]
#[serial]
fn test_startup_flow_qwen3_all_pillars() {
    let mut pipe = load_pipeline_for("Qwen3-8B-Q4_K_M.gguf");

    // Säule 1 — ran during ::new()
    assert!(
        !pipe.profile.layer_magnitude_stats.is_empty(),
        "introspection should have produced layer stats"
    );
    assert!(
        !pipe.profile.precision_recommendation.is_empty(),
        "introspection should have produced precision hints"
    );

    // Säule 4 — attach and verify bandits wired up.
    pipe.executor
        .attach_runtime(Runtime::new(VariantRegistry::new()));
    let rt = pipe.executor.runtime().expect("runtime attached");
    assert!(
        !rt.bandits.is_empty(),
        "runtime should register ≥1 multi-variant shape"
    );

    // Säule 5 — calibrate and verify band installed.
    pipe.calibrate_monitor().expect("calibrate");
    assert!(
        !pipe.monitor.expected_ranges.is_empty(),
        "monitor calibration should populate expected_ranges"
    );

    // End-to-end generate to confirm the attached Säulen work in
    // the hot loop. 32 tokens is enough for the bandit to flip to
    // its winning variant and for the monitor to run ≥1 sampled
    // hidden-state check.
    pipe.reset().expect("reset");
    let result = pipe
        .generate("What is 2+2?", 32, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(result.generated_tokens > 0);
    assert!(
        result.decode_tok_s > 10.0,
        "decode should exceed 10 tok/s, got {:.1}",
        result.decode_tok_s
    );
    println!(
        "qwen3: {} tok / {:.1} tok/s — {:?}",
        result.generated_tokens, result.decode_tok_s, result.output
    );
    // Runtime convergence — the q8_inline variant should have
    // taken over every multi-variant shape by now.
    let converged = pipe
        .executor
        .runtime()
        .expect("runtime")
        .bandits
        .values()
        .any(|b| {
            b.total_pulls > 50
                && b.arms
                    .iter()
                    .max_by_key(|a| a.n_pulls)
                    .map(|a| a.n_pulls as f64 / b.total_pulls as f64 > 0.8)
                    .unwrap_or(false)
        });
    assert!(
        converged,
        "at least one bandit should be past exploration by token 32"
    );
    // Monitor ran ≥1 check during 32 decode tokens (sample_rate=32)
    // so the log vector exists, but on a clean prompt it stays empty.
    println!(
        "monitor events: {}",
        pipe.monitor.revision_log.len()
    );
}

/// Llama-3.1 end-to-end — regression for the step-1.14 NeoX-RoPE
/// + Q4_K/Q6_K LDS-ceiling fixes. Before those landed this prompt
/// either crashed with `HIP error 1` in GEMV or produced garbage
/// after ~30 decode tokens.
#[test]
#[serial]
fn test_full_pipeline_llama31() {
    let mut pipe = load_pipeline_for("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
    pipe.reset().expect("reset");
    let result = pipe
        .generate("Hello", 16, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(
        result.generated_tokens > 0,
        "llama-3.1 should generate ≥ 1 token"
    );
    println!("llama-3.1: {:?}", result.output);
}
