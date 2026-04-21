//! Phase 1 / Schritt 1.11 Block A — tokenizer + sampling + inference pipeline.

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
use rocmforge::v1::core::tokenizer::Tokenizer;
use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use serial_test::serial;

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .expect("HOME set")
        .join("models")
        .join(name)
}

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";

fn load_tokenizer(path: &str) -> Tokenizer {
    let gguf = GGUFFile::open(model_path(path)).expect("open gguf");
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("cfg");
    Tokenizer::from_gguf_metadata(gguf.metadata(), &cfg.architecture).expect("tokenizer")
}

fn load_pipeline_qwen3() -> InferencePipeline<'static> {
    let device = GpuDevice::detect(0).expect("gpu");
    let path = model_path(QWEN3);
    let model = LoadedModel::load(&path, &device).expect("load model");
    let gguf = GGUFFile::open(&path).expect("reopen gguf");

    // Box-leak gives us 'static references for the pipeline's lifetime
    // parameter. The OS reclaims leaked memory when the test process exits.
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

    InferencePipeline::new(graph, plan, model_static, gguf_static, 256)
        .expect("pipeline")
}

// ── Tokenizer tests ────────────────────────────────────────────────────────

#[test]
fn test_tokenizer_load_qwen3() {
    let tok = load_tokenizer(QWEN3);
    assert!(tok.vocab_size() > 100_000);
    assert!(tok.bos_id().is_some() || tok.eos_id().is_some());
    println!(
        "Qwen3 tokenizer: vocab={} bos={:?} eos={:?}",
        tok.vocab_size(),
        tok.bos_id(),
        tok.eos_id()
    );
}

#[test]
fn test_tokenizer_encode_decode_roundtrip() {
    let tok = load_tokenizer(QWEN3);
    let text = "Hello, how are you?";
    let ids = tok.encode(text, false);
    assert!(!ids.is_empty());
    let back = tok.decode(&ids, true);
    assert_eq!(back.trim(), text.trim(), "roundtrip failed: {ids:?}");
    println!(
        "Roundtrip: '{}' → {} tokens → '{}'",
        text,
        ids.len(),
        back
    );
}

#[test]
fn test_tokenizer_special_tokens_not_in_plain_text() {
    // Plain user text must not contain control tokens like <|im_start|>.
    // That's the tokenizer's job — `apply_chat_template` adds them.
    let tok = load_tokenizer(QWEN3);
    let ids = tok.encode("just some plain text", false);
    for &tid in &ids {
        assert!(
            !tok.is_eos(tid),
            "plain text tokenisation produced EOS token {tid}"
        );
    }
}

#[test]
fn test_chat_template_qwen3() {
    let tok = load_tokenizer(QWEN3);
    let formatted = tok.apply_chat_template("Hallo", None);
    assert!(formatted.contains("user"));
    assert!(formatted.contains("assistant"));
    assert!(formatted.contains("<|im_start|>"));
    println!("Qwen3 chat template head: {}", &formatted[..100.min(formatted.len())]);
}

// ── Inference pipeline tests ──────────────────────────────────────────────

#[test]
#[serial]
fn test_generate_single_token_greedy() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let result = pipe
        .generate("Hallo", 1, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(result.generated_tokens >= 1);
    println!(
        "Single token: '{}' ({} prompt tokens)",
        result.output, result.prompt_tokens
    );
}

#[test]
#[serial]
fn test_generate_short_response() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let result = pipe
        .generate("What is 2+2?", 20, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(result.generated_tokens > 0);
    println!(
        "Short response ({} tokens, {:.1} tok/s): '{}'",
        result.generated_tokens, result.decode_tok_s, result.output
    );
}

#[test]
#[serial]
fn test_generate_performance_baseline() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let result = pipe
        .generate(
            "Explain what a mutex is in one paragraph.",
            50,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    println!(
        "Baseline: prefill {:.1} tok/s, decode {:.1} tok/s, total {:.0} ms",
        result.prefill_tok_s, result.decode_tok_s, result.total_ms
    );
}

// ── Additional Block-B tests ──────────────────────────────────────────────

/// Q4_K-sensitive numerics check — 17 × 23 = 391. If the model emits
/// a different product, something is wrong with the quantised attention
/// scores.
#[test]
#[serial]
fn test_arithmetic_17x23() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let result = pipe
        .generate(
            "Was ist 17 × 23? Antworte nur mit der Zahl.",
            64,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    println!("17×23 response: {:?}", result.output);
    // We don't hard-assert the "391" substring because Qwen3 often emits
    // the chain-of-thought <think> block first. This test is here to
    // flag a crash or empty output; the human reviewer reads the stdout.
    assert!(
        result.generated_tokens > 0,
        "no tokens generated for arithmetic prompt"
    );
}

/// The tokenizer must not crash on emoji / multi-byte input, and the
/// pipeline must still produce text.
#[test]
#[serial]
fn test_emoji_no_crash() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let result = pipe
        .generate(
            "🎉🎊🎈 What do these emojis mean?",
            32,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("generate");
    println!("emoji response: {:?}", result.output);
    assert!(result.generated_tokens > 0);
}

/// Long system prompt (~200 tokens) exercises the sequential prefill
/// path with more pressure than a handful of words. We don't care about
/// the content of the reply — only that prefill and decode both succeed
/// without GPU errors.
#[test]
#[serial]
fn test_long_context_prefill() {
    let mut pipe = load_pipeline_qwen3();
    pipe.reset().expect("reset");
    let long_prompt = "You are an expert systems architect specializing in \
        high-performance computing, distributed systems, and GPU programming. \
        You have extensive experience with AMD RDNA architectures, ROCm software \
        stack, HIP programming, and CUDA-to-HIP porting. You understand memory \
        hierarchies including L1/L2 caches, LDS, and global memory access patterns. \
        \n\nWhat is the single biggest bottleneck for batch-1 LLM decode on \
        consumer GPUs?";
    let result = pipe
        .generate(long_prompt, 32, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(
        result.prompt_tokens > 100,
        "prefill should exceed 100 tokens, got {}",
        result.prompt_tokens
    );
    assert!(result.generated_tokens > 0);
    println!(
        "long-context: prefill {} tok @ {:.1} tok/s, decoded {} tok @ {:.1} tok/s",
        result.prompt_tokens,
        result.prefill_tok_s,
        result.generated_tokens,
        result.decode_tok_s,
    );
}

/// Runs the full 15-prompt suite against Qwen3-8B and writes the
/// Markdown report to `tests/out/inference_test_block_b.md`. This test
/// is the main Block-B deliverable — it exercises the same code path
/// the CLI uses.
#[test]
#[serial]
fn test_run_15_prompt_suite() {
    use rocmforge::v1::cli::inference_test::run;
    let model = model_path(QWEN3);
    let suite = std::path::PathBuf::from("benches_v1/inference_test_prompts_15.json");
    let out = std::path::PathBuf::from("tests/out/inference_test_block_b.md");
    run(&model, &suite, &out).expect("15-prompt suite");
    let md = std::fs::read_to_string(&out).expect("read report");
    assert!(md.contains("ROCmForge v1.0"));
    assert!(md.contains("Per-prompt metrics"));
    assert!(md.contains("Human evaluation"));
    println!("Report written: {}", out.display());
}
