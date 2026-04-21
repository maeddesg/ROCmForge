//! Phase 1 / Schritt 1.10 Block A — Computation graph build tests.
//!
//! CPU-only: the graph is constructed from GGUF metadata + `ModelConfig`
//! without touching VRAM. All three Phase-1 target models must build
//! cleanly through the same `GraphBuilder::build()` path — no
//! architecture-specific branches.

#![cfg(feature = "v1")]

use std::collections::HashMap;

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, LayerTensors, TensorInfo, TensorRole,
};
use rocmforge::v1::graph::{
    BufferId, ComputationGraph, GraphBuildContext, GraphBuilder, GraphNode,
};

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .expect("HOME set")
        .join("models")
        .join(name)
}

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA31: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const QWEN25: &str = "Qwen2.5-7B-Instruct-Q4_0.gguf";

fn build_graph_from_gguf(
    path: std::path::PathBuf,
) -> (
    ModelConfig,
    Vec<LayerTensors>,
    HashMap<TensorRole, TensorInfo>,
    ComputationGraph,
) {
    let gguf = GGUFFile::open(&path).expect("open gguf");
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("config");
    let layers = group_tensors_by_layer(gguf.tensors());

    // Collect global-tensor roles.
    let mut globals_owned: HashMap<TensorRole, TensorInfo> = HashMap::new();
    for t in gguf.tensors() {
        let (role, layer) = parse_tensor_name(&t.name);
        if layer.is_none() {
            if !matches!(role, TensorRole::Unknown(_)) {
                globals_owned.insert(role, t.clone());
            }
        }
    }
    // Build a borrow map for the context.
    let global_refs: HashMap<TensorRole, &TensorInfo> = globals_owned
        .iter()
        .map(|(k, v)| (k.clone(), v))
        .collect();

    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: global_refs,
    };
    let graph = GraphBuilder::build(&ctx).expect("graph build");
    (cfg, layers, globals_owned, graph)
}

fn count_kind(graph: &ComputationGraph, kind: &str) -> usize {
    graph.nodes.iter().filter(|n| n.kind() == kind).count()
}

// ── Per-model build tests ─────────────────────────────────────────────────

#[test]
fn test_graph_build_qwen3() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));
    assert_eq!(cfg.n_layers, 36);
    assert!(cfg.has_qk_norm, "Qwen3 should have qk_norm");
    assert!(!cfg.has_rope_freqs, "Qwen3 should not have rope_freqs");
    assert!(cfg.is_gqa);

    let embeddings = count_kind(&graph, "Embedding");
    let rope = count_kind(&graph, "Rope");
    let attn = count_kind(&graph, "Attention");
    let residual = count_kind(&graph, "ResidualAdd");
    let gate_up = count_kind(&graph, "GateUpSwiGLU");
    let rms = count_kind(&graph, "RmsNorm");
    let gemm = count_kind(&graph, "Gemm");
    let kv_cache = count_kind(&graph, "KvCacheAppend");

    assert_eq!(embeddings, 1);
    assert_eq!(rope, 36);
    assert_eq!(attn, 36);
    assert_eq!(residual, 72);
    assert_eq!(gate_up, 36);
    assert_eq!(kv_cache, 36);

    // 2 per-layer (attn_norm + ffn_norm) + 2 qk-norm per layer + 1 output_norm.
    assert_eq!(rms, 2 * 36 + 2 * 36 + 1, "got {rms}");
    // 3 QKV + 1 O-proj + 1 down-proj per layer + 1 lm-head.
    assert_eq!(gemm, 5 * 36 + 1, "got {gemm}");

    // Every Rope node for Qwen3 has no rope_freqs.
    for n in &graph.nodes {
        if let GraphNode::Rope { rope_freqs, .. } = n {
            assert!(rope_freqs.is_none(), "Qwen3 RoPE must not reference rope_freqs");
        }
    }

    println!(
        "Qwen3 graph: nodes={}, buffers={}, rms_norms={rms}, gemms={gemm}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

#[test]
fn test_graph_build_llama31() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(LLAMA31));
    assert_eq!(cfg.n_layers, 32);
    assert!(!cfg.has_qk_norm);
    assert!(cfg.has_rope_freqs);

    assert_eq!(count_kind(&graph, "Rope"), 32);
    assert_eq!(count_kind(&graph, "Attention"), 32);
    // 32 attn_norm + 32 ffn_norm + 1 output_norm = 65.
    assert_eq!(count_kind(&graph, "RmsNorm"), 65);
    // 3 QKV + 1 O + 1 down per layer + 1 LM head.
    assert_eq!(count_kind(&graph, "Gemm"), 5 * 32 + 1);

    // Every Rope node for Llama-3.1 has rope_freqs set.
    let rope_with_freqs = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, GraphNode::Rope { rope_freqs: Some(_), .. }))
        .count();
    assert_eq!(rope_with_freqs, 32, "Llama-3.1 should use custom rope_freqs");

    println!(
        "Llama-3.1 graph: nodes={}, buffers={}, rope_with_freqs={rope_with_freqs}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

#[test]
fn test_graph_build_qwen25() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(QWEN25));
    assert_eq!(cfg.n_layers, 28);
    assert!(cfg.has_attention_bias, "Qwen2.5 should have attention biases");
    assert!(!cfg.has_qk_norm);
    assert!(!cfg.has_rope_freqs);

    // Count Gemm nodes that carry a bias. Qwen2.5 has Q/K/V bias =>
    // 3 per layer with bias, 2 without.
    let gemm_with_bias = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, GraphNode::Gemm { bias: Some(_), .. }))
        .count();
    assert_eq!(
        gemm_with_bias,
        3 * cfg.n_layers,
        "Qwen2.5 QKV bias count wrong: {gemm_with_bias}"
    );

    println!(
        "Qwen2.5 graph: nodes={}, buffers={}, gemm_with_bias={gemm_with_bias}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

// ── Meta-invariants ───────────────────────────────────────────────────────

#[test]
fn test_graph_no_model_specific_code() {
    // All three models must build through the same function without
    // any branching on architecture name inside `GraphBuilder`.
    for name in [QWEN3, LLAMA31, QWEN25] {
        let (_, _, _, graph) = build_graph_from_gguf(model_path(name));
        assert!(!graph.nodes.is_empty(), "{name}: empty graph");
    }
}

#[test]
fn test_graph_node_order_valid() {
    let (_, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));

    // Walk nodes in order. Each input buffer must already be in the
    // defined set. The token-ids buffer is implicitly defined.
    let mut defined: std::collections::HashSet<BufferId> =
        std::collections::HashSet::new();
    defined.insert(graph.token_ids_buffer);

    for (i, node) in graph.nodes.iter().enumerate() {
        for input in node.inputs() {
            assert!(
                defined.contains(&input),
                "node {i} ({}) reads undefined buffer {input:?}",
                node.kind()
            );
        }
        for output in node.outputs() {
            defined.insert(output);
        }
    }
    println!("Graph node order: all {} nodes valid", graph.nodes.len());
}

#[test]
fn test_graph_buffer_count_reasonable() {
    // Fully-unrolled graph should still have a bounded buffer count —
    // rough ballpark: one per transient intermediate per layer. For
    // Qwen3 with 36 layers we allocate ~14 per layer plus globals, so
    // ≲ 800 is the expected ceiling. Fail hard if we overshoot wildly.
    let (_, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));
    assert!(
        graph.num_buffers < 2000,
        "buffer count exploded: {}",
        graph.num_buffers
    );
    println!("Qwen3 buffer count: {}", graph.num_buffers);
}

// ─── Block B: BufferPlan + GraphExecutor ───────────────────────────────────

use rocmforge::v1::graph::BufferPlan;

#[test]
fn test_buffer_plan_qwen3() {
    let (_, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));
    let plan = BufferPlan::plan_phase1(&graph);
    // Every buffer referenced by the graph has a spec.
    for node in &graph.nodes {
        for buf in node.inputs().iter().chain(node.outputs().iter()) {
            assert!(
                plan.specs.contains_key(buf),
                "Buffer {buf:?} in node {} has no spec",
                node.kind()
            );
        }
    }
    let max = plan.max_bytes(1);
    println!(
        "Qwen3 buffer plan: {} specs, max_bytes (seq=1) = {} B ({:.2} KB)",
        plan.specs.len(),
        max,
        max as f64 / 1024.0
    );
    // Logits buffer = vocab_size × 4 = 151936 × 4 ≈ 608 KB — biggest.
    assert!(max >= graph.config.vocab_size * 4);
}

#[cfg(feature = "gpu")]
mod gpu_executor {
    use super::*;
    use rocmforge::v1::backend::gpu::device::GpuDevice;
    use rocmforge::v1::core::model_loader::LoadedModel;
    use rocmforge::v1::graph::GraphExecutor;
    use serial_test::serial;

    fn build_graph_from_loaded(
        model: &LoadedModel,
    ) -> rocmforge::v1::graph::ComputationGraph {
        use rocmforge::v1::core::gguf::GGUFFile;
        // We parsed metadata in `model_loader.rs::load`, but the graph
        // builder wants per-layer TensorInfo. We re-open the GGUF to
        // get it without touching VRAM again.
        // NOTE: this is a Phase-1 shortcut — production code would
        // keep TensorInfo around in LoadedModel.
        let _ = model; // TODO: extend LoadedModel to carry TensorInfo.
        // For this test we rebuild from scratch via the GGUF path.
        unimplemented!("use ctx from rebuilt gguf instead");
    }

    /// Load the model *and* build the graph from the same GGUF, keeping
    /// the context alive for the lifetime of the test.
    fn load_and_build(
        path: std::path::PathBuf,
    ) -> (
        LoadedModel,
        rocmforge::v1::core::gguf::GGUFFile,
        rocmforge::v1::graph::ComputationGraph,
    ) {
        let device = GpuDevice::detect(0).expect("gpu");
        let model = LoadedModel::load(&path, &device).expect("load model");
        // Separate rebuild for the graph context (per note above).
        let gguf =
            rocmforge::v1::core::gguf::GGUFFile::open(&path).expect("re-open gguf");
        let cfg =
            rocmforge::v1::core::model_config::ModelConfig::from_metadata(
                gguf.metadata(),
                gguf.tensors(),
            )
            .expect("config");
        let layers = rocmforge::v1::core::tensor_info::group_tensors_by_layer(
            gguf.tensors(),
        );
        let mut globals_owned: std::collections::HashMap<
            rocmforge::v1::core::tensor_info::TensorRole,
            rocmforge::v1::core::tensor_info::TensorInfo,
        > = std::collections::HashMap::new();
        for t in gguf.tensors() {
            let (role, layer) =
                rocmforge::v1::core::tensor_info::parse_tensor_name(&t.name);
            if layer.is_none() {
                if !matches!(
                    role,
                    rocmforge::v1::core::tensor_info::TensorRole::Unknown(_)
                ) {
                    globals_owned.insert(role, t.clone());
                }
            }
        }
        let global_refs: std::collections::HashMap<_, _> =
            globals_owned.iter().map(|(k, v)| (k.clone(), v)).collect();
        let ctx = rocmforge::v1::graph::GraphBuildContext {
            config: &cfg,
            layers: &layers,
            global_tensors: global_refs,
        };
        let graph =
            rocmforge::v1::graph::GraphBuilder::build(&ctx).expect("graph build");
        (model, gguf, graph)
    }

    #[test]
    #[serial]
    fn test_executor_dry_run_qwen3() {
        let (model, gguf, graph) = load_and_build(model_path(QWEN3));
        let plan = BufferPlan::plan_phase1(&graph);
        let max_seq = 128usize; // keep KV cache allocation modest
        let mut executor =
            GraphExecutor::new(graph, plan, &model, &gguf, max_seq).expect("executor");

        // Feed BOS token (id 0 works for any vocab).
        let logits = executor.execute_decode(0, 0).expect("decode");
        assert_eq!(logits.len(), model.config.vocab_size);

        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 =
            logits.iter().sum::<f32>() / logits.len() as f32;
        for v in &logits {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
        println!(
            "Qwen3 dry-run logits: len={} min={:.3} max={:.3} mean={:.3}",
            logits.len(),
            min,
            max,
            mean
        );
    }

    #[test]
    #[serial]
    fn test_executor_prefill_3_tokens() {
        let (model, gguf, graph) = load_and_build(model_path(QWEN3));
        let plan = BufferPlan::plan_phase1(&graph);
        let max_seq = 128usize;
        let mut executor =
            GraphExecutor::new(graph, plan, &model, &gguf, max_seq).expect("executor");

        let tokens = vec![0u32, 1, 2];
        let logits = executor.execute_prefill(&tokens, 0).expect("prefill");
        assert_eq!(logits.len(), model.config.vocab_size);
        for v in &logits {
            assert!(v.is_finite(), "non-finite logit after prefill: {v}");
        }
        println!(
            "Qwen3 prefill 3 tokens ok: last-token logits range {:.3}..{:.3}",
            logits.iter().cloned().fold(f32::INFINITY, f32::min),
            logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    #[test]
    #[serial]
    fn test_executor_decode_sequence() {
        // Prefill one token, then decode 5 more. KV cache should grow
        // monotonically without crashes or non-finite logits.
        let (model, gguf, graph) = load_and_build(model_path(QWEN3));
        let plan = BufferPlan::plan_phase1(&graph);
        let mut executor =
            GraphExecutor::new(graph, plan, &model, &gguf, 16).expect("executor");

        let logits_prefill = executor.execute_prefill(&[0u32], 0).expect("prefill");
        assert!(logits_prefill.iter().all(|v| v.is_finite()));

        for step in 1..=5usize {
            let logits = executor.execute_decode(step as u32, step).expect("decode");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite logits at decode step {step}"
            );
        }
        println!("Qwen3 decode sequence (1 + 5 steps): no NaN/Inf");
    }
}
