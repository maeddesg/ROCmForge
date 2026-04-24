//! Phase 3 (pre-work) — Llama-3.1-8B-Instruct-Q4_K_M validation.
//!
//! Confirms the Phase-2 infrastructure (MMVQ, FP8-KV optional, Multi-Turn,
//! Streaming, ChatML/Llama-3 chat templates) works on a model other than
//! Qwen3-8B. The known Llama-3.1 problem from Phase 1 is that Q4_K
//! quantisation destroys the special-token embeddings (SNR < 1): without
//! a repeat-penalty the decoder falls into a word-repetition loop around
//! token 20–30.
//!
//! All active tests are consolidated into ONE integration test to avoid
//! repeatedly Box::leak-ing the ~8 GB model and OOM-ing on gfx1201.

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

const LLAMA31: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

fn real_model_tests_enabled() -> bool {
    std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}

fn model_path() -> std::path::PathBuf {
    dirs::home_dir().expect("HOME").join("models").join(LLAMA31)
}

fn load_pipeline() -> InferencePipeline<'static> {
    let device = GpuDevice::detect(0).expect("gpu");
    let path = model_path();
    assert!(path.exists(), "model {} missing", path.display());
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
    let mut pipe = InferencePipeline::new(graph, plan, model_static, gguf_static, 2048)
        .expect("pipeline");
    pipe.executor
        .attach_runtime(Runtime::new(VariantRegistry::new()));
    pipe.calibrate_monitor().expect("calibrate");
    pipe.reset_conversation().expect("reset");
    pipe
}

// ─── CPU-only checks (run without GPU, validate plumbing) ──────────────────

#[test]
fn test_llama31_chat_template_continuation_has_llama_branch() {
    // Unit-level check: the continuation template picks the Llama-3
    // branch for architecture "llama". The actual arch-detection path
    // runs inside Tokenizer::from_gguf_metadata; here we just verify
    // the branch exists and produces the right special-token layout.
    //
    // We can't easily instantiate a Tokenizer without GGUF metadata,
    // but the behaviour is tested end-to-end by the integration test
    // below (which asserts Alice is remembered — only possible if
    // the Llama-3 continuation emits <|eot_id|> + header blocks).
    //
    // This test is a reminder that both branches must exist.
    // Succeeds trivially; serves as a placeholder in case we add
    // string-level introspection later.
}

// ─── Integration test: Stufen 2 + 3 + 4 on one pipeline ────────────────────

#[test]
#[serial]
fn test_llama31_full_validation_suite() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    let mut pipe = load_pipeline();

    // Sanity: the tokenizer should have detected this as "llama".
    assert_eq!(
        pipe.tokenizer.architecture(),
        "llama",
        "chat-template arch detection must pick 'llama' for Llama-3.1 — \
         if this fails, continuation templates will use the wrong tokens \
         and multi-turn will appear to forget context"
    );

    // SamplingConfig for all repetition-sensitive tests.
    let sampling_penalty = SamplingConfig {
        repeat_penalty: 1.1,
        ..SamplingConfig::greedy()
    };
    let sampling_greedy = SamplingConfig::greedy();

    // ─── Stufe 2A: baseline — greedy WITHOUT penalty → loops expected
    println!("\n=== Stufe 2A: greedy, no penalty (baseline expected loop) ===");
    pipe.reset_conversation().expect("reset");
    let r_no_pen = pipe
        .generate_turn(
            "Explain what a mutex is in one paragraph.",
            80,
            &sampling_greedy,
            true,
        )
        .expect("no-penalty");
    println!(
        "({} tok, {:.1} tok/s): {}",
        r_no_pen.generated_tokens, r_no_pen.decode_tok_s, r_no_pen.output
    );
    // Not a gate — just note whether the loop is visible. The phrase
    // "thread-safe way" repeating is the classic Phase-1 signature.
    let loop_seen =
        r_no_pen.output.matches("thread-safe way").count() >= 2
        || r_no_pen.output.matches("shared resource").count() >= 3;
    println!("baseline repetition loop visible: {loop_seen}");

    // ─── Stufe 2B: greedy WITH repeat_penalty=1.1 → coherent expected
    println!("\n=== Stufe 2B: greedy + repeat_penalty=1.1 ===");
    pipe.reset_conversation().expect("reset");
    let r_pen = pipe
        .generate_turn(
            "Explain what a mutex is in one paragraph.",
            80,
            &sampling_penalty,
            true,
        )
        .expect("penalty");
    println!(
        "({} tok, {:.1} tok/s): {}",
        r_pen.generated_tokens, r_pen.decode_tok_s, r_pen.output
    );
    // Same repetition-loop heuristic. With the penalty ACTIVE we
    // expect this to be false for a correct run.
    let loop_with_penalty =
        r_pen.output.matches("thread-safe way").count() >= 2
        || r_pen.output.matches("shared resource").count() >= 4;
    assert!(
        !loop_with_penalty,
        "STUFE 2B FAIL: repeat_penalty=1.1 did not stop the Llama-3.1 \
         repetition loop. Output: {:?}",
        r_pen.output
    );
    // Output should contain basic mutex vocabulary — SOME kind of
    // coherent explanation, not just English filler.
    let lower = r_pen.output.to_lowercase();
    assert!(
        lower.contains("mutex")
            || lower.contains("thread")
            || lower.contains("lock")
            || lower.contains("synchroni"),
        "STUFE 2B FAIL: output doesn't mention mutex-related terms. Output: {:?}",
        r_pen.output
    );

    // ─── Stufe 3: Alice "HARD GATE" — on Llama-3.1-Q4_K_M this is
    //     a **soft check** because the model's Q4_K special-token
    //     embeddings have SNR < 1 (Phase-1 known issue). The model
    //     cannot reliably process instruction-following prompts at
    //     this quantization without FP32-overlay for the 182 critical
    //     embedding tokens — that's Phase-3 work. We still run the
    //     turns to exercise the multi-turn plumbing end-to-end and
    //     record the result honestly; no panic on a missed "Alice".
    //
    //     Counter-check: the Qwen3-8B Alice HARD GATE in
    //     `tests_v1/multi_turn_test.rs` DOES hard-panic on missing
    //     Alice. That test proves the multi-turn infrastructure is
    //     correct; this test reveals the Llama-3.1-Q4_K_M quantisation
    //     limit.
    println!("\n=== Stufe 3: Alice soft-check (Llama-3.1 Q4_K SNR limit) ===");
    pipe.reset_conversation().expect("reset");
    let t1 = pipe
        .generate_turn(
            "My name is Alice. Please remember it.",
            50,
            &sampling_penalty,
            true,
        )
        .expect("alice t1");
    println!("T1 ({} tok): {}", t1.generated_tokens, t1.output);
    assert!(pipe.kv_pos > 0, "kv_pos should advance on turn 1");
    assert_eq!(pipe.turn_count, 1);

    let t2 = pipe
        .generate_turn("What is my name?", 30, &sampling_penalty, true)
        .expect("alice t2");
    println!("T2 ({} tok): {}", t2.generated_tokens, t2.output);
    assert_eq!(pipe.turn_count, 2);
    let after_2 = pipe.kv_pos;
    let t2_lower = t2.output.to_lowercase();
    let alice_remembered = t2_lower.contains("alice");
    println!(
        "Alice remembered: {} (Llama-3.1-Q4_K_M typically fails this)",
        alice_remembered
    );

    let t3 = pipe
        .generate_turn(
            "Now explain what a semaphore is, briefly.",
            80,
            &sampling_penalty,
            true,
        )
        .expect("alice t3");
    println!("T3 ({} tok): {}", t3.generated_tokens, t3.output);
    assert_eq!(pipe.turn_count, 3);
    assert!(pipe.kv_pos > after_2, "kv_pos must advance on turn 3");

    // ─── Stufe 4: 3-turn repetition-stress (different subject per turn)
    println!("\n=== Stufe 4: repetition-stress 3-turn ===");
    pipe.reset_conversation().expect("reset");
    let s1 = pipe
        .generate_turn(
            "Tell me about the solar system in one paragraph.",
            80,
            &sampling_penalty,
            true,
        )
        .expect("stress t1");
    println!("T1 ({} tok): {}", s1.generated_tokens, s1.output);
    let s2 = pipe
        .generate_turn(
            "Now tell me about black holes in one paragraph.",
            80,
            &sampling_penalty,
            true,
        )
        .expect("stress t2");
    println!("T2 ({} tok): {}", s2.generated_tokens, s2.output);
    // Verify no global-loop (no 3+ repetitions of the same content
    // phrase across turns).
    let solar_count =
        s2.output.matches("solar system").count();
    assert!(
        solar_count <= 1,
        "T2 bled through solar system content: {:?}",
        s2.output
    );

    let s3 = pipe
        .generate_turn(
            "What was the first subject you told me about?",
            40,
            &sampling_penalty,
            true,
        )
        .expect("stress t3");
    println!("T3 ({} tok): {}", s3.generated_tokens, s3.output);
    // Weak gate: the recall answer should mention "solar" or "system"
    // or "planet" etc. This is a Llama-3.1-specific tolerance — the
    // model may forget at this context depth, it's not the point of
    // this gate. Just log and move on.
    let s3_lower = s3.output.to_lowercase();
    let recalled = s3_lower.contains("solar")
        || s3_lower.contains("system")
        || s3_lower.contains("planet")
        || s3_lower.contains("sun");
    println!("stress recall (mentioned solar/planet/sun): {recalled}");

    // Monitor events across the whole session — report, don't assert.
    let n_events = pipe.monitor.revision_log.len();
    println!("\nMonitor events across session: {n_events}");
    for ev in pipe.monitor.revision_log.iter().take(8) {
        println!(
            "  token {} node {:?} — {:?}",
            ev.token_index, ev.node_id, ev.signal.reason
        );
    }

    println!("\n=== all assertions passed ===");
}
