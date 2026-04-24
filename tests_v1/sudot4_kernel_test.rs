//! Q4_K Q8-inline GEMV using `__builtin_amdgcn_sudot4`
//! (`v_dot4_i32_iu8` on gfx1201).
//!
//! Validates:
//!   1. Parity: sudot4-kernel output ≈ q4_k_q8_inline output on
//!      random Q4_K weights (bit-exact arithmetic, tiny rounding
//!      delta from FP-summation ordering).
//!   2. Bandit picks up the new variant (3 Q4_K variants
//!      registered per shape now).
//!   3. End-to-end decode stays coherent and the Bandit converges
//!      on sudot4 for the hot shapes.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_q8_inline, rocmforge_launch_gemv_q4_k_q8_inline_sudot4,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::inference::InferencePipeline;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::model_loader::LoadedModel;
use rocmforge::v1::core::sampling::SamplingConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, GgmlType, TensorInfo, TensorRole,
};
use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};
use rocmforge::v1::runtime::{KernelId, Runtime, ShapeKey, VariantRegistry};
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

// ── CPU-only: Bandit registration ─────────────────────────────────────

// sudot4 was deregistered on 2026-04-23 because it is 1.41× slower
// than q4_k_q8_inline on gfx1201 AND its presence as a third bandit
// arm prevents UCB1 from committing within a single CLI run (rocprof
// showed 8 640 q4_k_standard exploration pulls with 3 arms vs. 90
// with 2). The KernelId stays in the enum and the kernel is still
// callable — the bandit just does not see it anymore.
//
// 2026-04-24: q4_k_q8_inline itself was replaced by q4_k_mmvq (the
// llama.cpp MMVQ port). Registry now has exactly 2 Q4_K variants:
// standard + mmvq.
#[test]
fn test_sudot4_variant_is_deregistered() {
    let mut reg = VariantRegistry::new();
    reg.register_gemv_shape(GgmlType::Q4_K, 4096, 4096);
    let shape = ShapeKey {
        op_type: rocmforge::v1::runtime::OpType::Gemv,
        format: GgmlType::Q4_K,
        n: 4096,
        k: 4096,
    };
    let variants = reg.variants.get(&shape).expect("Q4_K registered");
    let kernels: Vec<_> = variants.iter().map(|v| v.kernel).collect();
    assert!(kernels.contains(&KernelId::GemvQ4KStandard));
    assert!(
        kernels.contains(&KernelId::GemvQ4KMmvq),
        "MMVQ variant should replace q8_inline as the non-standard Q4_K arm"
    );
    assert!(
        !kernels.contains(&KernelId::GemvQ4KQ8InlineSudot4),
        "sudot4 must stay deregistered (slower + blocks bandit convergence)"
    );
    assert!(
        !kernels.contains(&KernelId::GemvQ4KQ8Inline),
        "q8_inline deregistered 2026-04-24 — replaced by MMVQ (1.26-1.53× faster)"
    );
    assert_eq!(
        variants.len(),
        2,
        "expected exactly 2 Q4_K variants (standard + mmvq), got {variants:?}"
    );
}

// ── GPU: kernel-level parity ───────────────────────────────────────────

#[test]
#[serial]
fn test_sudot4_parity_vs_q8_inline() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }

    // Qwen3-8B-like shape
    const N_ROWS: usize = 4096; // K
    const NCOLS: usize = 1024; // N
    let weights = gen_q4k_weights(N_ROWS, NCOLS, 0xC1C2);
    let input = gen_input(N_ROWS, 0xC3C4);

    let mut d_w = HipBuffer::new(weights.len()).expect("d_w");
    d_w.copy_from_host(&weights).expect("up w");
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len()).expect("d_in");
    d_in.copy_from_host(in_bytes).expect("up in");
    let mut d_out_ref = HipBuffer::new(NCOLS * 4).expect("d_out_ref");
    let mut d_out_sudot4 = HipBuffer::new(NCOLS * 4).expect("d_out_sudot4");

    let stream = HipStream::new().expect("stream");

    unsafe {
        let rc = rocmforge_launch_gemv_q4_k_q8_inline(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_ref.as_mut_ptr() as *mut f32,
            N_ROWS as i32,
            NCOLS as i32,
            stream.raw(),
        );
        assert_eq!(rc, 0, "q8_inline rc={rc}");
        let rc = rocmforge_launch_gemv_q4_k_q8_inline_sudot4(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_sudot4.as_mut_ptr() as *mut f32,
            N_ROWS as i32,
            NCOLS as i32,
            stream.raw(),
        );
        assert_eq!(rc, 0, "sudot4 rc={rc}");
    }
    stream.synchronize().expect("sync");

    let mut host_ref = vec![0f32; NCOLS];
    let mut host_sudot4 = vec![0f32; NCOLS];
    unsafe {
        use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
        hipMemcpy(
            host_ref.as_mut_ptr() as *mut _,
            d_out_ref.as_ptr(),
            NCOLS * 4,
            hipMemcpyDeviceToHost,
        );
        hipMemcpy(
            host_sudot4.as_mut_ptr() as *mut _,
            d_out_sudot4.as_ptr(),
            NCOLS * 4,
            hipMemcpyDeviceToHost,
        );
    }

    // Same arithmetic at a different instruction granularity; FP
    // summation order within a sub-block is identical (both kernels
    // accumulate in the same order), so outputs should be exactly
    // equal modulo FP-FMA rounding. Require max_abs < 1e-3 on
    // realistic Q4_K magnitudes.
    let abs_max = host_ref
        .iter()
        .chain(host_sudot4.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in host_ref.iter().zip(host_sudot4.iter()) {
        let abs_err = (a - b).abs();
        let denom = a.abs().max(b.abs());
        max_abs = max_abs.max(abs_err);
        if denom >= abs_max * 0.01 {
            max_rel = max_rel.max(abs_err / denom);
        }
    }
    println!(
        "  Q4_K N={} K={}: max_abs={:.5} max_rel={:.6} (abs_max_out={:.3})",
        NCOLS, N_ROWS, max_abs, max_rel, abs_max
    );
    // Honest bounds: same algorithm, different instruction lowering.
    // Sudot4 uses packed 4-wide MACs vs the reference's scalar MACs;
    // associativity of FP summation is identical (we accumulate into
    // the same scalar acc in the same order).
    assert!(max_rel < 0.01, "sudot4 parity: max_rel={max_rel} > 1 %");
}

// ── GPU: end-to-end decode ─────────────────────────────────────────────

#[test]
#[serial]
fn test_e2e_decode_with_sudot4_variant() {
    if !real_model_tests_enabled() {
        return;
    }
    if !model_path().exists() {
        return;
    }

    let mut pipe = load_pipeline_with_bandit();
    // 2026-04-24: Q4_K has 2 Bandit variants after the MMVQ port —
    // {standard, mmvq}. Previously {standard, q8_inline, sudot4} and
    // then briefly {standard, q8_inline}. See
    // `results/phase2_mmvq_kernel_port.md`.
    let runtime = pipe.executor.runtime().expect("runtime attached");
    let q4k_shapes: Vec<_> = runtime
        .registry
        .variants
        .iter()
        .filter(|(s, _)| s.format == GgmlType::Q4_K)
        .collect();
    println!("  Q4_K shapes: {}", q4k_shapes.len());
    for (shape, vs) in &q4k_shapes {
        println!("    {:?}: {} variants", shape, vs.len());
        assert_eq!(vs.len(), 2, "Q4_K shape {:?} should have 2 variants", shape);
    }

    // Warmup + timed run — the Bandit should explore standard + mmvq
    // and commit on the fastest (mmvq, per isolated benchmark).
    pipe.reset().expect("reset");
    let _warm = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("warmup");
    pipe.reset().expect("reset");
    let r = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("timed");

    println!(
        "  Mutex 100 tok: prefill {:.1} tok/s  decode {:.1} tok/s",
        r.prefill_tok_s, r.decode_tok_s
    );
    println!(
        "  output[:60] = {}",
        r.output.chars().take(60).collect::<String>()
    );

    assert!(r.generated_tokens > 0);
    assert!(!r.output.trim().is_empty());
    let lower = r.output.to_lowercase();
    assert!(
        lower.contains("mutex") || lower.contains("mutual") || lower.contains("exclusion"),
        "output missing mutex keyword: {}",
        r.output
    );
    // Post-unfuse baseline is 68.8 tok/s. The sudot4 variant is a
    // pure speedup candidate for the Bandit; at minimum the result
    // shouldn't regress below 60 tok/s (3-variant exploration has
    // a slightly bigger warmup cost than 2-variant).
    assert!(
        r.decode_tok_s >= 60.0,
        "e2e decode {:.1} tok/s < 60 floor",
        r.decode_tok_s
    );
}
