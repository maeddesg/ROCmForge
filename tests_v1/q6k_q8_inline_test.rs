//! Phase 2 Schritt 2.1.4 — Q6_K Q8-inline GEMV variant.
//!
//! Exercises the new `rocmforge_launch_gemv_q6_k_q8_inline` kernel
//! at three levels:
//!   1. Kernel-level parity — same weights + input through
//!      q6_k_standard and q6_k_q8_inline must produce numerically
//!      close output. Int8 vs FP32 arithmetic so the tolerance is
//!      relative, not bit-exact.
//!   2. Bandit integration — VariantRegistry now returns TWO
//!      variants for Q6_K shapes.
//!   3. End-to-end decode — coherent text and unchanged Monitor
//!      metrics when the Bandit is allowed to pick q8_inline.
//!
//! The heavy model-loading tests are consolidated into one GPU
//! test so Box::leak doesn't OOM the 16 GB card.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::device::GpuDevice;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q6_k_q8_inline, rocmforge_launch_gemv_q6_k_standard,
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
use std::time::Instant;

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

/// Fill a Q6_K super-block with deterministic but representative
/// content (realistic scale magnitudes, varied nibbles/qh bits).
/// Block layout (§3.7):
///   0..128   ql (128 B)
///   128..192 qh (64 B)
///   192..208 sc (16 signed int8)
///   208..210 d  (fp16)
fn fill_q6k_block(dst: &mut [u8], seed: u64) {
    assert_eq!(dst.len(), 210);
    // ql nibbles — pseudo-random but seeded.
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut nxt = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u32
    };
    for i in 0..128 {
        dst[i] = (nxt() & 0xFF) as u8;
    }
    for i in 128..192 {
        dst[i] = (nxt() & 0xFF) as u8;
    }
    // sub-scales: small signed int8 range, never zero
    for i in 192..208 {
        let v = ((nxt() & 0x7F) as i8).saturating_add(4);
        dst[i] = v as u8;
    }
    // d: small positive fp16 so (q6 - 32) × d stays in range
    let d_f32 = 0.05f32;
    let d_h = half::f16::from_f32(d_f32).to_bits().to_le_bytes();
    dst[208] = d_h[0];
    dst[209] = d_h[1];
}

// ── CPU-only ──────────────────────────────────────────────────────────

// q6_k_q8_inline was deregistered on 2026-04-23 because it is
// 1.5-1.9× slower than q6_k_standard on gfx1201 AND its presence as
// a second bandit arm prevents UCB1 from committing the LM-head
// Q6_K shape within a 15-prompt run (~6 K pulls, q6_k convergence
// needs >10 K). The KernelId stays in the enum and the kernel is
// still callable — the bandit just does not see it anymore.
#[test]
fn test_q6k_q8_inline_is_deregistered() {
    let mut reg = VariantRegistry::new();
    reg.register_gemv_shape(GgmlType::Q6_K, 4096, 4096);
    let shape = ShapeKey {
        op_type: rocmforge::v1::runtime::OpType::Gemv,
        format: GgmlType::Q6_K,
        n: 4096,
        k: 4096,
    };
    let variants = reg.variants.get(&shape).expect("q6_k registered");
    let names: Vec<_> = variants.iter().map(|v| v.name).collect();
    assert!(names.contains(&"q6_k_standard"));
    assert!(
        !names.contains(&"q6_k_q8_inline"),
        "q6_k_q8_inline must stay deregistered (slower + blocks bandit convergence)"
    );
    assert_eq!(
        variants.len(),
        1,
        "expected exactly 1 Q6_K variant, got {variants:?}"
    );
    let kernels: Vec<_> = variants.iter().map(|v| v.kernel).collect();
    assert!(kernels.contains(&KernelId::GemvQ6KStandard));
    assert!(!kernels.contains(&KernelId::GemvQ6KQ8Inline));
}

// ── GPU: kernel-level parity + perf ──────────────────────────────────

#[test]
#[serial]
fn test_q6k_q8_inline_kernel_parity_vs_standard() {
    if !real_model_tests_enabled() {
        eprintln!("skipping — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }

    // Synthetic Q6_K weight matrix, N=256 rows × K=512 cols.
    // Constraints: N multiple of 32 (WMMA-safe), K multiple of 256.
    // Shape picked to fit the Q6_K_Q8_SHARED_MEM_LIMIT and be small
    // enough to compute on both kernels in <1 ms.
    const N_ROWS: usize = 512; // K
    const NCOLS: usize = 256; // N
    let blocks_per_col = N_ROWS / 256;
    let bytes_per_col = blocks_per_col * 210;
    let total_bytes = NCOLS * bytes_per_col;

    let mut weights = vec![0u8; total_bytes];
    for c in 0..NCOLS {
        for b in 0..blocks_per_col {
            let off = (c * blocks_per_col + b) * 210;
            fill_q6k_block(&mut weights[off..off + 210], (c * 31 + b) as u64);
        }
    }

    let input: Vec<f32> = (0..N_ROWS)
        .map(|i| (i as f32 * 0.013 - 3.3).sin())
        .collect();

    let mut d_w = HipBuffer::new(total_bytes).expect("d_w");
    d_w.copy_from_host(&weights).expect("up w");
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len()).expect("d_in");
    d_in.copy_from_host(in_bytes).expect("up in");
    let mut d_out_std = HipBuffer::new(NCOLS * 4).expect("d_out_std");
    let mut d_out_q8 = HipBuffer::new(NCOLS * 4).expect("d_out_q8");

    let stream = HipStream::new().expect("stream");

    unsafe {
        let rc = rocmforge_launch_gemv_q6_k_standard(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_std.as_mut_ptr() as *mut f32,
            N_ROWS as i32,
            NCOLS as i32,
            stream.raw(),
        );
        assert_eq!(rc, 0, "q6_k_standard rc={rc}");
        let rc = rocmforge_launch_gemv_q6_k_q8_inline(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_q8.as_mut_ptr() as *mut f32,
            N_ROWS as i32,
            NCOLS as i32,
            stream.raw(),
        );
        assert_eq!(rc, 0, "q6_k_q8_inline rc={rc}");
    }
    stream.synchronize().expect("sync");

    let mut host_std = vec![0f32; NCOLS];
    let mut host_q8 = vec![0f32; NCOLS];
    unsafe {
        use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
        let rc = hipMemcpy(
            host_std.as_mut_ptr() as *mut _,
            d_out_std.as_ptr(),
            NCOLS * 4,
            hipMemcpyDeviceToHost,
        );
        assert_eq!(rc, 0);
        let rc = hipMemcpy(
            host_q8.as_mut_ptr() as *mut _,
            d_out_q8.as_ptr(),
            NCOLS * 4,
            hipMemcpyDeviceToHost,
        );
        assert_eq!(rc, 0);
    }

    // Per-element relative error — Int8 quantisation of the
    // activation adds ≈ 1/254 = 0.4 % noise per FMA. Accumulated
    // over K=512 the worst case is ~5 %, but in practice the errors
    // cancel and we see << 1 % mean.
    // Synthetic test data (random ql/qh/sc) produces many outputs
    // near zero; the small-denominator rows blow up relative error
    // without indicating real inaccuracy. Filter to outputs ≥ 10 %
    // of the per-vector max magnitude before averaging.
    let abs_max = host_std
        .iter()
        .chain(host_q8.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let mag_gate = (abs_max * 0.1).max(1e-3);

    let mut max_rel = 0.0f32;
    let mut sum_rel = 0.0f64;
    let mut n_significant = 0usize;
    for (a, b) in host_std.iter().zip(host_q8.iter()) {
        let denom = a.abs().max(b.abs());
        if denom < mag_gate {
            continue;
        }
        let rel = (a - b).abs() / denom;
        max_rel = max_rel.max(rel);
        sum_rel += rel as f64;
        n_significant += 1;
    }
    let mean_rel = sum_rel / (n_significant as f64).max(1.0);
    println!(
        "  Q6_K parity: max_rel={:.4} mean_rel={:.4} over {} significant elements (mag_gate={:.3})",
        max_rel, mean_rel, n_significant, mag_gate
    );
    // Q8-activation introduces ~0.4 % per-element noise; accumulated
    // over K=512 it stays below 5 % max / 2 % mean on non-degenerate
    // outputs. Synthetic data has near-uniform sub-scale sign so
    // real-model numerics are tighter than this bar.
    assert!(
        max_rel < 0.10,
        "q6_k_q8_inline max_rel={max_rel} > 10 % on significant elements"
    );
    assert!(
        mean_rel < 0.03,
        "q6_k_q8_inline mean_rel={mean_rel} > 3 % on significant elements"
    );
}

// ── GPU: end-to-end decode + 15-prompt decision ──────────────────────

#[test]
#[serial]
fn test_e2e_decode_with_q6k_variants_available() {
    if !real_model_tests_enabled() {
        return;
    }
    if !model_path().exists() {
        return;
    }

    let mut pipe = load_pipeline_with_bandit();

    // Inspect the bandit — confirm Q6_K shapes have 2 variants now.
    let runtime = pipe.executor.runtime().expect("runtime attached");
    let q6k_shapes: Vec<_> = runtime
        .registry
        .variants
        .iter()
        .filter(|(s, _)| s.format == GgmlType::Q6_K)
        .collect();
    println!("  Q6_K shapes in registry: {}", q6k_shapes.len());
    for (shape, vs) in &q6k_shapes {
        println!("    shape={:?} variants={}", shape, vs.len());
        assert_eq!(vs.len(), 2, "Q6_K shape {:?} should have 2 variants", shape);
    }

    // Warmup + timed Mutex run.
    pipe.reset().expect("reset");
    let _warmup = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("warmup");
    pipe.reset().expect("reset");
    let t0 = Instant::now();
    let result = pipe
        .generate(MUTEX_PROMPT, 100, &SamplingConfig::greedy(), true)
        .expect("mutex gen");
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  Mutex 100 tok: prefill {:.1} tok/s | decode {:.1} tok/s | wall {:.0} ms",
        result.prefill_tok_s, result.decode_tok_s, wall_ms
    );
    println!(
        "  output[:60] = {}",
        result.output.chars().take(60).collect::<String>()
    );
    assert!(result.generated_tokens > 0);
    assert!(!result.output.trim().is_empty());
    let low = result.output.to_lowercase();
    assert!(
        low.contains("mutex") || low.contains("mutual") || low.contains("exclusion"),
        "Mutex prompt output missing keyword: {}",
        result.output
    );
    // Floor chosen ABOVE the pre-unfuse baseline (40.6 tok/s) but
    // slightly below the post-unfuse best (68.8). In practice adding
    // a second Q6_K variant can cost a few tok/s during Bandit
    // exploration (every shape explores both arms) and the LM-head
    // Q6_K (n=151 936) is already at 95 % BW with standard — the
    // bandit should correctly keep standard there but pays an
    // exploration tax in the first ~20 tokens.
    assert!(
        result.decode_tok_s >= 55.0,
        "e2e decode {:.1} tok/s < 55 floor (post-unfuse was 68.8)",
        result.decode_tok_s
    );
}
