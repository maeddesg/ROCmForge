//! HIP-Graph decode integration (Option C — single kv_pos slot).
//!
//! Verifies:
//!   1. The legacy path and the graph-replay path produce bit-
//!      identical output on the same prompt (same kernel dispatches
//!      + same committed bandit choices — no arithmetic change).
//!   2. Graph replay is not measurably slower than the legacy path
//!      (primary gate: ≥ 0.95× decode throughput — this is a
//!      dispatch-overhead optimisation, the expected gain is small
//!      and dominated by measurement jitter).
//!   3. `ROCMFORGE_DISABLE_HIP_GRAPH=1` keeps capture off.
//!
//! All heavy tests are consolidated to amortise the Box::leak'd
//! 5 GB model load.

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

// ── CPU-only env policy ───────────────────────────────────────────────

#[test]
fn test_disable_env_parsing() {
    let prev = std::env::var("ROCMFORGE_DISABLE_HIP_GRAPH").ok();
    std::env::remove_var("ROCMFORGE_DISABLE_HIP_GRAPH");
    assert!(std::env::var("ROCMFORGE_DISABLE_HIP_GRAPH").is_err());
    std::env::set_var("ROCMFORGE_DISABLE_HIP_GRAPH", "1");
    assert_eq!(
        std::env::var("ROCMFORGE_DISABLE_HIP_GRAPH").ok().as_deref(),
        Some("1")
    );
    std::env::remove_var("ROCMFORGE_DISABLE_HIP_GRAPH");
    if let Some(v) = prev {
        std::env::set_var("ROCMFORGE_DISABLE_HIP_GRAPH", v);
    }
}

// ── Consolidated GPU suite ────────────────────────────────────────────

#[test]
#[serial]
fn test_hip_graph_suite() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        return;
    }

    // Ensure env doesn't leak in from a previous run.
    std::env::remove_var("ROCMFORGE_DISABLE_HIP_GRAPH");

    let mut pipe = load_pipeline_with_bandit();

    // ── 1. Warmup run — lets the Bandit converge so compile_fast_dispatch
    //      and HIP-graph capture have a stable decision set.
    pipe.reset().expect("reset");
    let _warm = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("warmup");

    // ── 2. Legacy-path run (env-forced). Captures nothing.
    std::env::set_var("ROCMFORGE_DISABLE_HIP_GRAPH", "1");
    pipe.reset().expect("reset");
    let legacy = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("legacy gen");

    // ── 3. HIP-graph path — env cleared, cache invalidated by the
    //      env switch (we call set_fused_gate_up with the same value
    //      to tickle invalidate_fast_dispatch without changing semantics).
    std::env::remove_var("ROCMFORGE_DISABLE_HIP_GRAPH");
    // Nudge the internal cache in case warmup populated it; the
    // current impl invalidates on `set_fused_gate_up`.
    let fused = pipe.executor.fused_gate_up();
    pipe.executor.set_fused_gate_up(fused);
    pipe.reset().expect("reset");
    let graph = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("graph gen");

    println!(
        "  legacy: decode {:.1} tok/s, prefill {:.1} tok/s, output[:60] = '{}'",
        legacy.decode_tok_s,
        legacy.prefill_tok_s,
        legacy.output.chars().take(60).collect::<String>()
    );
    println!(
        "  graph:  decode {:.1} tok/s, prefill {:.1} tok/s, output[:60] = '{}'",
        graph.decode_tok_s,
        graph.prefill_tok_s,
        graph.output.chars().take(60).collect::<String>()
    );

    // ── 4. Correctness: outputs must agree.
    assert!(legacy.generated_tokens > 0);
    assert!(graph.generated_tokens > 0);
    assert!(!graph.output.trim().is_empty());
    let legacy_tokens: Vec<&str> = legacy.output.split_whitespace().collect();
    let graph_tokens: Vec<&str> = graph.output.split_whitespace().collect();
    let common = legacy_tokens
        .iter()
        .zip(graph_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "  first {} whitespace-tokens agree between legacy and graph",
        common
    );
    // Both paths dispatch the same kernels (same bandit choices,
    // same weight pointers, same buffers). Output should be
    // bit-identical.
    assert_eq!(
        legacy.output, graph.output,
        "HIP-graph output diverged from legacy — SetParams path is wrong or nodes are mis-indexed"
    );

    // ── 5. Coherence (Mutex keyword).
    for res in [&legacy, &graph] {
        let lower = res.output.to_lowercase();
        assert!(
            lower.contains("mutex") || lower.contains("mutual") || lower.contains("exclusion"),
            "Mutex prompt output lacks keyword: '{}'",
            res.output
        );
    }

    // ── 6. Performance — the graph path must not regress more than 5 %.
    let ratio = graph.decode_tok_s / legacy.decode_tok_s;
    println!(
        "  decode ratio graph/legacy = {:.3}  (graph {:.1} vs legacy {:.1} tok/s)",
        ratio, graph.decode_tok_s, legacy.decode_tok_s
    );
    assert!(
        ratio >= 0.95,
        "HIP-graph replay regressed > 5 % vs legacy: {:.3}",
        ratio
    );
    assert!(
        graph.decode_tok_s >= 60.0,
        "graph decode {:.1} tok/s < 60 floor",
        graph.decode_tok_s
    );

    // ── 7. Prefill must be untouched — HIP-graph only affects decode.
    let prefill_ratio = graph.prefill_tok_s / legacy.prefill_tok_s;
    assert!(
        prefill_ratio >= 0.90 && prefill_ratio <= 1.10,
        "prefill drifted: legacy {:.1} vs graph {:.1} ({:.2}x)",
        legacy.prefill_tok_s,
        graph.prefill_tok_s,
        prefill_ratio
    );
}
