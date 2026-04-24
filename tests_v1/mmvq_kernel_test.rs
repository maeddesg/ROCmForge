//! Phase 2 Schritt 2/3 — llama.cpp MMVQ kernel port.
//!
//! Tests:
//!   1. Correctness against a CPU FP32 reference (dequantize Q4_K → FP32
//!      dot product against FP32 input) on all 4 Q4_K shapes that Qwen3-8B
//!      actually dispatches. Output must be within ~2 % relative error of
//!      the CPU reference (Q4_K inherent quantization precision). Note:
//!      comparing MMVQ directly against `gemv_q4_k_q8_inline` is NOT a
//!      good ground-truth test — the two kernels use different Q8
//!      quantizers (MMVQ uses llama.cpp's `roundf(xi/d)`, q8_inline uses
//!      `static_cast<int8_t>(truncate(xi/d))` with round-tripped d),
//!      so their outputs diverge by ~1 % even though both are correct.
//!      In practice MMVQ is CLOSER to the CPU reference than q8_inline
//!      (verified in `tests_v1/mmvq_debug_test.rs::dbg_vs_cpu_n1_k4096`:
//!      MMVQ 0.06 % rel err vs q8_inline 0.85 %).
//!   2. Isolated kernel benchmark on the same shapes: mmvq µs (incl.
//!      quantize_q8_1 prep) vs q8_inline µs. This is the **decision**
//!      input — if mmvq doesn't win on Qwen3-8B shapes, we stop and
//!      report honestly instead of registering a slower kernel with
//!      the Bandit.
//!
//! Bandit registration is **not** done here. That is a follow-up step
//! driven by the benchmark output; see `results/phase2_mmvq_kernel_port.md`.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_mmvq, rocmforge_launch_gemv_q4_k_q8_inline,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::quantize::{rocmforge_launch_quantize_q8_1, BlockQ81, QK8_1};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipStream};
use serial_test::serial;
use std::ffi::c_void;
use std::mem::size_of;

// ─── Test data generators ──────────────────────────────────────────────────

/// Row-major Q4_K matrix: N output rows × (K / 256) super-blocks each.
/// Values: `d` and `dmin` in ±0.01, scales 6-bit masked, nibbles random.
/// Matches the layout used by both `gemv_q4_k_q8_inline` and MMVQ.
fn gen_q4k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0, "K must be a multiple of Q4_K super-block size 256");
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row;
    let mut buf = vec![0u8; total * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4]
            .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        // 6-bit packed scales+mins in bytes 4..16 (uint8_t scales[12]).
        // Mask to 6-bit to stay in representable range after extraction.
        for i in 4..16 {
            buf[b * 144 + i] = rng.u8(..) & 0x3F;
        }
        // Nibble bytes 16..144 (128 bytes = 256 nibbles).
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

// ─── GPU launch helpers ─────────────────────────────────────────────────────

struct Bufs {
    d_w: HipBuffer,
    d_input_fp: HipBuffer,
    d_q8_1: HipBuffer,
    d_out: HipBuffer,
    stream: HipStream,
}

fn prepare_bufs(weights: &[u8], input: &[f32], n: usize) -> Bufs {
    let mut d_w = HipBuffer::new(weights.len()).expect("d_w");
    d_w.copy_from_host(weights).expect("up w");

    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_input_fp = HipBuffer::new(in_bytes.len()).expect("d_input_fp");
    d_input_fp.copy_from_host(in_bytes).expect("up in");

    let n_q8_blocks = input.len() / QK8_1;
    let d_q8_1 = HipBuffer::new(n_q8_blocks * size_of::<BlockQ81>()).expect("d_q8_1");

    let d_out = HipBuffer::new(n * 4).expect("d_out");

    let stream = HipStream::new().expect("stream");
    Bufs {
        d_w,
        d_input_fp,
        d_q8_1,
        d_out,
        stream,
    }
}

fn launch_q8_inline(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_q8_inline(
            bufs.d_w.as_ptr() as *const u8,
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "q8_inline rc={rc}");
    bufs.stream.synchronize().expect("sync q8i");

    let mut out = vec![0f32; n];
    unsafe {
        hipMemcpy(
            out.as_mut_ptr() as *mut _,
            bufs.d_out.as_ptr(),
            n * 4,
            hipMemcpyDeviceToHost,
        );
    }
    out
}

fn launch_mmvq_with_prequant(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    // Step 1: quantize the FP32 input into Q8_1.
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_q8_1.as_mut_ptr(),
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "quantize_q8_1 rc={rc}");

    // Step 2: MMVQ over Q4_K × Q8_1.
    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_mmvq(
            bufs.d_w.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_out.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "mmvq rc={rc}");
    bufs.stream.synchronize().expect("sync mmvq");

    let mut out = vec![0f32; n];
    unsafe {
        hipMemcpy(
            out.as_mut_ptr() as *mut _,
            bufs.d_out.as_ptr(),
            n * 4,
            hipMemcpyDeviceToHost,
        );
    }
    out
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let abs_max = a
        .iter()
        .chain(b.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let abs_err = (x - y).abs();
        let denom = x.abs().max(y.abs());
        max_abs = max_abs.max(abs_err);
        if denom >= abs_max * 0.01 {
            max_rel = max_rel.max(abs_err / denom);
        }
    }
    (max_abs, max_rel)
}

// ─── CPU FP32 reference (ground truth) ─────────────────────────────────────

fn dequant_q4k_block(bytes: &[u8]) -> [f32; 256] {
    assert_eq!(bytes.len(), 144);
    let d = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([bytes[2], bytes[3]])).to_f32();
    let scales = &bytes[4..16];
    let qs = &bytes[16..144];
    let mut out = [0.0f32; 256];
    for j in 0..8 {
        let (sc, mn): (u8, u8) = if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        };
        let pair_base = (j >> 1) * 32;
        let is_upper = (j & 1) != 0;
        for i in 0..32 {
            let byte = qs[pair_base + i];
            let nib = if is_upper { byte >> 4 } else { byte & 0x0F };
            out[j * 32 + i] = d * (sc as f32) * (nib as f32) - dmin * (mn as f32);
        }
    }
    out
}

fn cpu_fp32_gemv(weights: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / 256;
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let row_offset = row * blocks_per_row * 144;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let bb = &weights[row_offset + b * 144..row_offset + (b + 1) * 144];
            let dequant = dequant_q4k_block(bb);
            for i in 0..256 {
                acc += dequant[i] * input[b * 256 + i];
            }
        }
        out[row] = acc;
    }
    out
}

fn compare_rel_stats(ref_: &[f32], obs: &[f32]) -> (f32, f32, f32) {
    let mag_max = ref_.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut rels: Vec<f32> = ref_
        .iter()
        .zip(obs.iter())
        .map(|(r, o)| {
            let denom = r.abs().max(o.abs()).max(mag_max * 0.01);
            (r - o).abs() / denom
        })
        .collect();
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = rels[rels.len() / 2];
    let p99 = rels[(rels.len() * 99) / 100];
    let max = *rels.last().unwrap();
    (median, p99, max)
}

// ─── Correctness: MMVQ vs CPU FP32, on Qwen3-8B Q4_K shapes ────────────────

fn mmvq_vs_cpu_at(n: usize, k: usize, seed: u64, median_tol: f32) {
    let weights = gen_q4k_weights(n, k, seed);
    let input = gen_input(k, seed.wrapping_add(1));
    let mut bufs = prepare_bufs(&weights, &input, n);

    let cpu_ref = cpu_fp32_gemv(&weights, &input, n, k);
    let out_mmvq = launch_mmvq_with_prequant(&mut bufs, n, k);
    let out_inline = launch_q8_inline(&mut bufs, n, k);

    let (m_mmvq, p99_mmvq, max_mmvq) = compare_rel_stats(&cpu_ref, &out_mmvq);
    let (m_inline, p99_inline, _max_inline) = compare_rel_stats(&cpu_ref, &out_inline);

    println!(
        "  N={:5} K={:5}: MMVQ vs CPU: med={:.5} p99={:.4} max={:.4}  | q8_inline vs CPU: med={:.5} p99={:.4}",
        n, k, m_mmvq, p99_mmvq, max_mmvq, m_inline, p99_inline
    );

    assert!(
        m_mmvq < median_tol,
        "MMVQ median error at N={n} K={k}: {m_mmvq} > tol={median_tol}"
    );
}

#[test]
#[serial]
fn test_mmvq_vs_cpu_q_proj_shape_4096x4096() {
    mmvq_vs_cpu_at(4096, 4096, 0xA1A1, 0.02);
}

#[test]
#[serial]
fn test_mmvq_vs_cpu_kv_proj_shape_1024x4096() {
    mmvq_vs_cpu_at(1024, 4096, 0xB2B2, 0.02);
}

#[test]
#[serial]
fn test_mmvq_vs_cpu_ffn_gate_up_shape_12288x4096() {
    mmvq_vs_cpu_at(12288, 4096, 0xC3C3, 0.02);
}

#[test]
#[serial]
fn test_mmvq_vs_cpu_ffn_down_shape_4096x12288() {
    mmvq_vs_cpu_at(4096, 12288, 0xD4D4, 0.02);
}

// ─── Isolated kernel benchmark — the decision input ────────────────────────

/// Returns (q8_inline_us, mmvq_us_incl_quant, speedup, bw_pct_mmvq).
fn benchmark_at(n: usize, k: usize, seed: u64) -> (f64, f64, f64, f64) {
    let weights = gen_q4k_weights(n, k, seed);
    let input = gen_input(k, seed.wrapping_add(1));
    let mut bufs = prepare_bufs(&weights, &input, n);

    // Warm-up.
    for _ in 0..5 {
        let _ = launch_q8_inline(&mut bufs, n, k);
        let _ = launch_mmvq_with_prequant(&mut bufs, n, k);
    }

    let runs = 50usize;

    // Time q8_inline.
    let start = HipEvent::new().expect("ev");
    let stop = HipEvent::new().expect("ev");
    start.record(&bufs.stream).expect("rec");
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_q8_inline(
                bufs.d_w.as_ptr() as *const u8,
                bufs.d_input_fp.as_ptr() as *const f32,
                bufs.d_out.as_mut_ptr() as *mut f32,
                k as i32,
                n as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stop.record(&bufs.stream).expect("rec");
    stop.synchronize().expect("sync");
    let q8i_ms = HipEvent::elapsed_ms(&start, &stop).expect("elapsed") as f64;
    let q8i_us = (q8i_ms / runs as f64) * 1000.0;

    // Time MMVQ incl. per-call Q8_1 quantize (the fair comparison — both
    // kernels replace the same work: input activation + Q4_K dot product).
    let start = HipEvent::new().expect("ev");
    let stop = HipEvent::new().expect("ev");
    start.record(&bufs.stream).expect("rec");
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                bufs.d_input_fp.as_ptr() as *const f32,
                bufs.d_q8_1.as_mut_ptr(),
                k as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_mmvq(
                bufs.d_w.as_ptr(),
                bufs.d_q8_1.as_ptr(),
                bufs.d_out.as_mut_ptr(),
                n as i32,
                k as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stop.record(&bufs.stream).expect("rec");
    stop.synchronize().expect("sync");
    let mmvq_ms = HipEvent::elapsed_ms(&start, &stop).expect("elapsed") as f64;
    let mmvq_us = (mmvq_ms / runs as f64) * 1000.0;

    let speedup = q8i_us / mmvq_us;

    // BW% — weight traffic dominates at batch-1 decode. 144 bytes per
    // Q4_K super-block × (N × K / 256) super-blocks = 0.5625 × N × K bytes.
    let weight_bytes = 0.5625 * (n as f64) * (k as f64);
    let bw_gb_s_mmvq = weight_bytes / (mmvq_us * 1e-6) / 1e9;
    let peak_gb_s = 640.0;
    let bw_pct = (bw_gb_s_mmvq / peak_gb_s) * 100.0;

    println!(
        "  N={:5} K={:5}: q8_inline={:6.2} µs  mmvq={:6.2} µs  speedup={:.2}×  BW(mmvq)={:.1}%",
        n, k, q8i_us, mmvq_us, speedup, bw_pct
    );

    (q8i_us, mmvq_us, speedup, bw_pct)
}

#[test]
#[serial]
fn test_mmvq_benchmark_all_shapes() {
    println!("\n=== MMVQ isolated kernel benchmark (all Qwen3-8B Q4_K shapes) ===");
    println!("Each row reports mean over 50 runs after 5-iter warmup.");
    println!("mmvq timing INCLUDES the per-call quantize_q8_1 prep step.\n");

    let shapes: &[(usize, usize, u64, &str)] = &[
        (4096, 4096, 0x1111, "Q-proj (N=4096, K=4096)"),
        (1024, 4096, 0x2222, "K/V-proj (N=1024, K=4096)"),
        (12288, 4096, 0x3333, "FFN gate/up (N=12288, K=4096)"),
        (4096, 12288, 0x4444, "FFN down (N=4096, K=12288)"),
    ];

    let mut results = Vec::new();
    for (n, k, seed, label) in shapes {
        println!("{}", label);
        results.push((*label, benchmark_at(*n, *k, *seed)));
    }

    println!("\n=== Summary ===");
    let mut wins = 0;
    let mut losses = 0;
    for (label, (q8i, mmvq, speedup, bw)) in &results {
        let verdict = if *speedup >= 1.0 { "WIN" } else { "LOSS" };
        if *speedup >= 1.0 {
            wins += 1;
        } else {
            losses += 1;
        }
        println!(
            "  [{}] {:<30} q8i={:6.2}µs mmvq={:6.2}µs ({:.2}×, {:.1}% BW)",
            verdict, label, q8i, mmvq, speedup, bw
        );
    }

    println!("\nMMVQ wins: {}/{}", wins, wins + losses);
    // Do NOT fail on "loss" — the whole point of this benchmark is to
    // gather evidence for the Bandit-registration decision. A loss here
    // means: don't register MMVQ, stay on q8_inline. Decision is made
    // outside this test in the report.
}

// ─── Struct sanity (ensures kernel's hip-side structs line up) ─────────────

#[test]
fn test_block_q8_1_still_36_bytes() {
    // MMVQ depends on the 36-byte layout — guard against drift.
    assert_eq!(std::mem::size_of::<BlockQ81>(), 36);
}
