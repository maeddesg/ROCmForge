//! Phase 2 Schritt 1/3 — llama.cpp MMVQ kernel port.
//!
//! Validates the standalone `quantize_q8_1` HIP kernel:
//!   1. Block layout is bit-identical to llama.cpp's `block_q8_1`
//!      (4-byte `half2 ds` header + 32 `int8_t qs`).
//!   2. Quantization is numerically correct on known inputs.
//!   3. Q8_1.qs values and scale `d` match what Q8_0 would produce on
//!      the same input — same quantization algorithm, just with an
//!      extra `s` field. Lets the Bandit run bit-parity comparisons
//!      against the existing q4_k_q8_inline kernel once MMVQ lands.
//!   4. Performance: hidden_dim=4096 quantization runs in single-digit
//!      microseconds (the whole point of moving the quant out of the
//!      GEMV hot loop is that it's negligible as a pre-pass).

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::quantize::{
    rocmforge_launch_quantize_q8_1, BlockQ81, QK8_1,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipStream};
use serial_test::serial;
use std::ffi::c_void;
use std::mem::size_of;

// ─── Struct layout tests (CPU-only) ─────────────────────────────────────────

#[test]
fn test_block_q8_1_size() {
    assert_eq!(
        size_of::<BlockQ81>(),
        36,
        "block_q8_1 must be 36 bytes to match llama.cpp layout"
    );
}

#[test]
fn test_block_q8_1_field_offsets() {
    let b = BlockQ81 {
        ds: [0, 0],
        qs: [0; 32],
    };
    let base = (&b) as *const _ as usize;
    let ds_off = (&b.ds) as *const _ as usize - base;
    let qs_off = (&b.qs) as *const _ as usize - base;
    assert_eq!(ds_off, 0, "ds must sit at offset 0");
    assert_eq!(qs_off, 4, "qs must start at offset 4 (after half2)");
}

#[test]
fn test_qk8_1_constant() {
    assert_eq!(QK8_1, 32);
}

// ─── GPU helpers ────────────────────────────────────────────────────────────

fn quantize_on_gpu(input: &[f32]) -> Vec<BlockQ81> {
    assert!(input.len() % QK8_1 == 0, "input must be a multiple of 32");
    let n_blocks = input.len() / QK8_1;

    let stream = HipStream::new().expect("stream");

    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    let mut d_in = HipBuffer::new(input_bytes.len()).expect("alloc input");
    d_in.copy_from_host(input_bytes).expect("h2d input");

    let mut d_out =
        HipBuffer::new(n_blocks * size_of::<BlockQ81>()).expect("alloc output");

    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut c_void,
            input.len() as i32,
            stream.raw(),
        )
    };
    assert_eq!(rc, 0, "quantize_q8_1 returned hipError {rc}");
    stream.synchronize().expect("sync");

    let mut readback = vec![0u8; n_blocks * size_of::<BlockQ81>()];
    d_out.copy_to_host(&mut readback).expect("d2h output");

    readback
        .chunks_exact(size_of::<BlockQ81>())
        .map(|chunk| {
            let mut b = BlockQ81 {
                ds: [0, 0],
                qs: [0; 32],
            };
            b.ds[0] = u16::from_le_bytes([chunk[0], chunk[1]]);
            b.ds[1] = u16::from_le_bytes([chunk[2], chunk[3]]);
            for i in 0..32 {
                b.qs[i] = chunk[4 + i] as i8;
            }
            b
        })
        .collect()
}

/// CPU Q8_0 reference — same quantization algorithm as Q8_1 modulo the
/// extra sum. Lets us check bit-parity on `qs` and the scale `d`.
/// Returns `(d, qs)` per 32-element block.
fn q8_0_reference(input: &[f32]) -> Vec<(f32, [i8; 32])> {
    input
        .chunks_exact(QK8_1)
        .map(|block| {
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 127.0;
            let mut qs = [0i8; 32];
            if amax != 0.0 {
                for (i, v) in block.iter().enumerate() {
                    qs[i] = (v / d).round().max(-127.0).min(127.0) as i8;
                }
            }
            // Round-trip d through f16 so the reference matches the
            // kernel's half-precision storage exactly.
            let d_half = f16::from_f32(d).to_f32();
            (d_half, qs)
        })
        .collect()
}

// ─── Quantisierungs-Korrektheit ─────────────────────────────────────────────

#[test]
#[serial]
fn test_quantize_q8_1_basic() {
    // Known input 1.0..=32.0 → amax = 32.0, d = 32/127 ≈ 0.252.
    let input: Vec<f32> = (1..=32).map(|x| x as f32).collect();
    let out = quantize_on_gpu(&input);
    assert_eq!(out.len(), 1);

    let d = f16::from_bits(out[0].ds[0]).to_f32();
    let expected_d = 32.0f32 / 127.0f32;
    let d_ref = f16::from_f32(expected_d).to_f32();
    assert!(
        (d - d_ref).abs() < 1e-6,
        "d = {d}, expected ≈ {d_ref}"
    );

    // qs[31] must round to 127 (xi=32, xi/d=127.0).
    assert_eq!(out[0].qs[31], 127, "qs[31] expected 127, got {}", out[0].qs[31]);
    // qs[0] must round to 4 (xi=1, xi/d ≈ 3.968 → round → 4).
    assert_eq!(out[0].qs[0], 4, "qs[0] expected 4, got {}", out[0].qs[0]);
    // All qs must be monotone since input is monotone.
    for i in 1..32 {
        assert!(
            out[0].qs[i] >= out[0].qs[i - 1],
            "qs must be monotone non-decreasing, qs[{i}]={} < qs[{}]={}",
            out[0].qs[i],
            i - 1,
            out[0].qs[i - 1]
        );
    }
}

#[test]
#[serial]
fn test_quantize_q8_1_zeros() {
    let input = vec![0.0f32; 32];
    let out = quantize_on_gpu(&input);
    assert_eq!(out.len(), 1);
    // amax == 0 → d stored as half(0) == 0, all qs == 0, sum == 0.
    assert_eq!(f16::from_bits(out[0].ds[0]).to_f32(), 0.0);
    assert_eq!(f16::from_bits(out[0].ds[1]).to_f32(), 0.0);
    for (i, q) in out[0].qs.iter().enumerate() {
        assert_eq!(*q, 0, "qs[{i}] must be 0 for zero input");
    }
}

#[test]
#[serial]
fn test_quantize_q8_1_negative() {
    // Symmetric signed input around 0 → sum ≈ 0.
    let input: Vec<f32> = (0..32)
        .map(|i| if i < 16 { -(i as f32) } else { (i - 16) as f32 })
        .collect();
    let out = quantize_on_gpu(&input);
    assert_eq!(out.len(), 1);

    let d = f16::from_bits(out[0].ds[0]).to_f32();
    let s = f16::from_bits(out[0].ds[1]).to_f32();

    // amax = 15, d ≈ 15/127.
    let expected_d = f16::from_f32(15.0 / 127.0).to_f32();
    assert!((d - expected_d).abs() < 1e-5, "d = {d}, expected ≈ {expected_d}");

    // sum = Σ xi = -(0+1+..+15) + (0+1+..+15) = 0.
    assert!(s.abs() < 1e-3, "symmetric sum should be ~0, got {s}");

    // qs values: i<16 negative, i>=16 non-negative.
    for i in 0..16 {
        assert!(out[0].qs[i] <= 0, "qs[{i}] = {} expected ≤ 0", out[0].qs[i]);
    }
    for i in 16..32 {
        assert!(out[0].qs[i] >= 0, "qs[{i}] = {} expected ≥ 0", out[0].qs[i]);
    }
}

#[test]
#[serial]
fn test_quantize_q8_1_hidden_dim_4096() {
    let mut rng = fastrand::Rng::with_seed(0xD0D0_C0DE);
    let input: Vec<f32> = (0..4096).map(|_| rng.f32() * 4.0 - 2.0).collect();

    let out = quantize_on_gpu(&input);
    assert_eq!(out.len(), 4096 / QK8_1);

    // Every block must have finite d/sum and qs in range.
    for (bi, block) in out.iter().enumerate() {
        let d = f16::from_bits(block.ds[0]).to_f32();
        let s = f16::from_bits(block.ds[1]).to_f32();
        assert!(d.is_finite() && d >= 0.0, "block {bi}: d={d} not finite/nonneg");
        assert!(s.is_finite(), "block {bi}: s={s} not finite");
        for (i, q) in block.qs.iter().enumerate() {
            // i8 already caps the range to [-128, 127]; we only reject -128
            // because the kernel clamps to [-127, 127] via round(xi/d).
            assert!(*q >= -127i8, "block {bi}.qs[{i}] = {q} below -127");
        }
    }

    // sum ≈ d · Σqs (what the ggml struct comment actually says; the
    // kernel stores warp-reduced Σxi, which matches up to rounding).
    for (bi, block) in out.iter().enumerate() {
        let d = f16::from_bits(block.ds[0]).to_f32();
        let s = f16::from_bits(block.ds[1]).to_f32();
        let sigma_qs: i32 = block.qs.iter().map(|q| *q as i32).sum();
        let d_times_sigma = d * sigma_qs as f32;
        // Tolerance: each qs contributes at most 0.5 * d of rounding error,
        // 32 elements → |error| ≤ 16*d, plus half-precision rounding on
        // both d and s → 2% slack for safety.
        let tol = 16.0 * d + 0.02 * s.abs() + 1e-3;
        assert!(
            (s - d_times_sigma).abs() <= tol,
            "block {bi}: s={s}, d*Σqs={d_times_sigma}, tol={tol}"
        );
    }
}

// ─── Parity gegen Q8_0 ──────────────────────────────────────────────────────

#[test]
#[serial]
fn test_q8_1_vs_q8_0_qs_and_scale_identical() {
    let mut rng = fastrand::Rng::with_seed(0xFEED_F00D);
    let input: Vec<f32> = (0..256).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let gpu = quantize_on_gpu(&input);
    let cpu = q8_0_reference(&input);
    assert_eq!(gpu.len(), cpu.len());

    for (bi, (g, (d_ref, qs_ref))) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d_gpu = f16::from_bits(g.ds[0]).to_f32();
        assert!(
            (d_gpu - d_ref).abs() < 1e-6,
            "block {bi}: d_gpu={d_gpu} vs d_ref={d_ref}"
        );
        for i in 0..32 {
            assert_eq!(
                g.qs[i], qs_ref[i],
                "block {bi}.qs[{i}]: gpu={} ref={} (xi={})",
                g.qs[i],
                qs_ref[i],
                input[bi * 32 + i]
            );
        }
    }
}

// ─── Performance ────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_quantize_q8_1_fast_4096() {
    let mut rng = fastrand::Rng::with_seed(42);
    let input: Vec<f32> = (0..4096).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let stream = HipStream::new().expect("stream");
    let input_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(input_bytes.len()).expect("alloc in");
    d_in.copy_from_host(input_bytes).expect("h2d");
    let mut d_out = HipBuffer::new((input.len() / QK8_1) * size_of::<BlockQ81>())
        .expect("alloc out");

    // Warm-up.
    for _ in 0..3 {
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut c_void,
                input.len() as i32,
                stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stream.synchronize().expect("sync warmup");

    // Timed: 100 launches, report median.
    let start = HipEvent::new().expect("ev start");
    let stop = HipEvent::new().expect("ev stop");
    let runs = 100;
    start.record(&stream).expect("rec start");
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut c_void,
                input.len() as i32,
                stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stop.record(&stream).expect("rec stop");
    stop.synchronize().expect("sync");

    let total_ms = HipEvent::elapsed_ms(&start, &stop).expect("elapsed");
    let per_call_us = (total_ms as f64 / runs as f64) * 1000.0;

    println!(
        "quantize_q8_1 on 4096 floats: {:.3} µs/call (mean over {} runs)",
        per_call_us, runs
    );

    // Budget: 10 µs. The kernel is launch-bound and a single launch
    // on gfx1201 is ~3 µs; 10 µs leaves headroom for system jitter.
    assert!(
        per_call_us < 10.0,
        "quantize_q8_1 too slow: {per_call_us:.3} µs > 10 µs budget"
    );
}

// ─── Lifecycle / reusability ────────────────────────────────────────────────

#[test]
#[serial]
fn test_q8_1_buffer_reusable_across_launches() {
    // A single persistent device buffer reused across multiple launches
    // with different inputs — validates that the kernel writes the full
    // block every time and leaves nothing stale.
    let stream = HipStream::new().expect("stream");
    let mut d_in = HipBuffer::new(128 * 4).expect("alloc in");
    let mut d_out = HipBuffer::new(4 * size_of::<BlockQ81>()).expect("alloc out");

    for seed in 0u64..5 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let input: Vec<f32> = (0..128).map(|_| rng.f32() * 10.0 - 5.0).collect();
        let input_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
        };
        d_in.copy_from_host(input_bytes).expect("h2d");

        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut c_void,
                input.len() as i32,
                stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        stream.synchronize().expect("sync");

        let mut bytes = vec![0u8; 4 * size_of::<BlockQ81>()];
        d_out.copy_to_host(&mut bytes).expect("d2h");
        // Parse and compare to Q8_0 reference.
        let cpu_ref = q8_0_reference(&input);
        for (bi, chunk) in bytes.chunks_exact(size_of::<BlockQ81>()).enumerate() {
            let d_gpu = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            let (d_ref, qs_ref) = &cpu_ref[bi];
            assert!(
                (d_gpu - d_ref).abs() < 1e-6,
                "seed {seed} block {bi}: d_gpu={d_gpu} vs d_ref={d_ref}"
            );
            for i in 0..32 {
                assert_eq!(chunk[4 + i] as i8, qs_ref[i], "seed {seed} block {bi}.qs[{i}]");
            }
        }
    }
}

// ─── Edge cases ─────────────────────────────────────────────────────────────

// ─── Decode smoke test — no regression ──────────────────────────────────────

fn real_model_tests_enabled() -> bool {
    std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}

/// Decode the same prompt twice — once with Q8_1 pre-quantize disabled,
/// once enabled — and confirm the produced tokens are identical. Q8_1
/// is write-only in this step (no kernel reads the buffer), so enabling
/// it must not change any logit. Also reports the wall-time delta.
#[test]
#[serial]
fn test_decode_with_q8_1_prequant_no_regression() {
    if !real_model_tests_enabled() {
        eprintln!(
            "ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS not set; skipping real-model decode smoke"
        );
        return;
    }

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

    const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
    let path = dirs::home_dir().expect("HOME").join("models").join(QWEN3);
    assert!(path.exists(), "model {} must be present", path.display());

    let device = GpuDevice::detect(0).expect("gpu");
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
    let mut pipe =
        InferencePipeline::new(graph, plan, model_static, gguf_static, 512).expect("pipeline");
    pipe.executor
        .attach_runtime(Runtime::new(VariantRegistry::new()));
    pipe.calibrate_monitor().expect("calibrate");

    let prompt = "Explain what a mutex is in one paragraph.";
    let n_tokens = 20usize;
    let sampling = SamplingConfig::greedy();

    // Run 1: prequant DISABLED (default).
    pipe.executor.set_q8_1_prequant(false);
    pipe.reset().expect("reset 1");
    let t0 = std::time::Instant::now();
    let result_off = pipe
        .generate(prompt, n_tokens, &sampling, true)
        .expect("generate off");
    let wall_off_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Run 2: prequant ENABLED — buffer is filled but no kernel reads it.
    pipe.executor.set_q8_1_prequant(true);
    pipe.reset().expect("reset 2");
    let t0 = std::time::Instant::now();
    let result_on = pipe
        .generate(prompt, n_tokens, &sampling, true)
        .expect("generate on");
    let wall_on_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Correctness: greedy output MUST be identical because the Q8_1
    // buffer has no reader in this step.
    assert_eq!(
        result_off.generated_tokens, result_on.generated_tokens,
        "token count diverged: off={} on={}",
        result_off.generated_tokens, result_on.generated_tokens
    );
    assert_eq!(
        result_off.output, result_on.output,
        "output text diverged (q8_1 prequant should be pure side-effect):\n  off: {:?}\n  on:  {:?}",
        result_off.output, result_on.output
    );

    // Performance: loose regression gate. The quantize kernel itself
    // is ~4 µs; call-site overhead is negligible vs a full decode step
    // (~17 ms/token on Qwen3-8B). Allow 5 % headroom for measurement
    // noise; anything beyond that is a real regression.
    let regression = (wall_on_ms - wall_off_ms) / wall_off_ms * 100.0;
    println!(
        "decode {} tokens: off={:.1} ms, on={:.1} ms (Δ {:+.2} %)",
        n_tokens, wall_off_ms, wall_on_ms, regression
    );
    assert!(
        regression < 5.0,
        "prequant active regressed decode by {:.2} % (> 5 % budget)",
        regression
    );
}

#[test]
#[serial]
fn test_quantize_q8_1_rejects_bad_sizes() {
    let stream = HipStream::new().expect("stream");
    let d_in = HipBuffer::new(32 * 4).expect("alloc in");
    let mut d_out = HipBuffer::new(size_of::<BlockQ81>()).expect("alloc out");

    // n_elements == 0
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut c_void,
            0,
            stream.raw(),
        )
    };
    assert_ne!(rc, 0, "zero n_elements must fail");

    // Not a multiple of 32
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut c_void,
            31,
            stream.raw(),
        )
    };
    assert_ne!(rc, 0, "non-QK8_1 multiple must fail");
}
