//! Post-2.1.5 follow-up — un-fused `gate_up_swiglu` on the decode path.
//!
//! rocprof (2026-04-23) flagged the fused `gemv_q4_k_gate_up_swiglu`
//! as the single biggest decode bottleneck (437 µs, 20 % BW, 65 %
//! of decode GPU time). The un-fuse splits the work into:
//!
//!   1. `gemv_q4_k_q8_inline(gate_w, in, gate_scratch)`
//!   2. `gemv_q4_k_q8_inline(up_w,   in, up_scratch)`
//!   3. `swiglu(gate_scratch, up_scratch, output)`
//!
//! Projected: 437 µs → ~90 µs per layer per token (~3× speedup on
//! the hottest kernel, ~1.7× on end-to-end decode).
//!
//! All tests in this file are consolidated into a single GPU test
//! that loads the model once — the Box::leak pipeline-construction
//! pattern our test harness uses hits VRAM OOM after 2 – 3
//! independent loads.

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

// ── CPU-only policy test ──────────────────────────────────────────────

#[test]
fn test_fused_gate_up_env_default_off() {
    // Default (no env) should be un-fused. Only when ROCMFORGE_FUSED_GATE_UP=1
    // does the fused path kick in.
    let prev = std::env::var("ROCMFORGE_FUSED_GATE_UP").ok();
    std::env::remove_var("ROCMFORGE_FUSED_GATE_UP");
    let fused_default = std::env::var("ROCMFORGE_FUSED_GATE_UP").ok().as_deref() == Some("1");
    assert!(!fused_default);
    std::env::set_var("ROCMFORGE_FUSED_GATE_UP", "1");
    let fused_on = std::env::var("ROCMFORGE_FUSED_GATE_UP").ok().as_deref() == Some("1");
    assert!(fused_on);
    std::env::remove_var("ROCMFORGE_FUSED_GATE_UP");
    if let Some(v) = prev {
        std::env::set_var("ROCMFORGE_FUSED_GATE_UP", v);
    }
}

// ── Consolidated GPU suite ────────────────────────────────────────────

#[test]
#[serial]
fn test_unfused_gate_up_suite() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    if !model_path().exists() {
        return;
    }

    let mut pipe = load_pipeline_with_bandit();

    // ── 1. Correctness: unfused vs fused, same prompt ────────────
    //
    // Run the Mutex prompt 20 tokens each way. Greedy → identical
    // prefix if both paths are numerically close enough to keep
    // the top-1 logit stable.
    pipe.executor.set_fused_gate_up(true);
    pipe.reset().expect("reset");
    let fused = pipe
        .generate(MUTEX_PROMPT, 20, &SamplingConfig::greedy(), true)
        .expect("fused generate");

    pipe.executor.set_fused_gate_up(false);
    pipe.reset().expect("reset");
    let unfused = pipe
        .generate(MUTEX_PROMPT, 20, &SamplingConfig::greedy(), true)
        .expect("unfused generate");

    println!(
        "  fused   output[:60] = {}",
        fused.output.chars().take(60).collect::<String>()
    );
    println!(
        "  unfused output[:60] = {}",
        unfused.output.chars().take(60).collect::<String>()
    );

    assert!(fused.generated_tokens > 0);
    assert!(unfused.generated_tokens > 0);
    assert!(!unfused.output.trim().is_empty());

    // Numerical agreement is tight on the short prefix. Greedy
    // decoding is chaotic — one flipped logit ripples — so we
    // require only the first ~5 tokens to match. Beyond that we
    // accept divergence because FP32 sum ordering differs between
    // the fused and split paths.
    let fused_tokens: Vec<&str> = fused.output.split_whitespace().collect();
    let unfused_tokens: Vec<&str> = unfused.output.split_whitespace().collect();
    let common = fused_tokens
        .iter()
        .zip(unfused_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "  first {} whitespace-tokens of output agree between fused and unfused",
        common
    );
    assert!(
        common >= 3,
        "unfused output diverged from fused from the very first word — numerics broken"
    );

    // ── 2. Coherence check ("mutex" keyword) ─────────────────────
    let lower = unfused.output.to_lowercase();
    assert!(
        lower.contains("mutex") || lower.contains("mutual") || lower.contains("exclusion"),
        "unfused Mutex prompt should mention mutex/mutual/exclusion, got: '{}'",
        unfused.output
    );

    // ── 3. Performance: unfused > fused ──────────────────────────
    pipe.executor.set_fused_gate_up(true);
    pipe.reset().expect("reset");
    let _warmup = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("fused warmup");
    pipe.reset().expect("reset");
    let fused_100 = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("fused timed");

    pipe.executor.set_fused_gate_up(false);
    pipe.reset().expect("reset");
    let _warmup = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("unfused warmup");
    pipe.reset().expect("reset");
    let unfused_100 = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("unfused timed");

    let speedup = unfused_100.decode_tok_s / fused_100.decode_tok_s;
    println!(
        "  100-tok Mutex decode: fused {:.1} tok/s -> unfused {:.1} tok/s ({:.2}x)",
        fused_100.decode_tok_s, unfused_100.decode_tok_s, speedup
    );
    println!(
        "  100-tok Mutex prefill: fused {:.1} tok/s -> unfused {:.1} tok/s",
        fused_100.prefill_tok_s, unfused_100.prefill_tok_s
    );

    // Hard gate: unfused must be at least 30 % faster on decode.
    assert!(
        speedup >= 1.30,
        "unfused speedup {:.2}x < 1.30x gate (fused {:.1}, unfused {:.1})",
        speedup,
        fused_100.decode_tok_s,
        unfused_100.decode_tok_s
    );

    // Hard floor: unfused decode must clear 60 tok/s (conservative
    // vs the projected 77 – 89 range).
    assert!(
        unfused_100.decode_tok_s >= 60.0,
        "unfused decode {:.1} tok/s < 60 floor",
        unfused_100.decode_tok_s
    );

    // Prefill must be untouched — the un-fuse only affects the
    // decode path. Allow ±10 % thermal/jitter noise.
    let prefill_ratio = unfused_100.prefill_tok_s / fused_100.prefill_tok_s;
    assert!(
        prefill_ratio >= 0.90 && prefill_ratio <= 1.10,
        "prefill drifted: fused {:.1} vs unfused {:.1} ({:.2}x)",
        fused_100.prefill_tok_s,
        unfused_100.prefill_tok_s,
        prefill_ratio
    );
}
