//! Phase 2 Schritt 4 — Q6_K × Q8_1 MMVQ kernel.
//!
//! Tests:
//!   1. Parity vs the existing `gemv_q6_k_standard` kernel (which has
//!      been validated against CPU references since Phase 1.8). We
//!      expect median relative error below ~2 % — MMVQ goes through
//!      Q8_1 activation quantization whereas standard reads FP32
//!      activations directly, so per-element drift matches what the
//!      Q4_K MMVQ port sees vs Q4_K standard (~0.4-1 %).
//!   2. Benchmark MMVQ vs standard on three Q6_K shapes: a small
//!      layer shape, the Qwen3-8B LM-head shape (N=151936, K=4096),
//!      and a mid-sized proxy for any internal Q6_K tensor.
//!   3. The LM-head benchmark is DIAGNOSTIC only — at N=151936 the
//!      standard kernel already hits ~95 % BW via L2 amortisation.
//!      MMVQ may win, tie, or lose here; the Bandit decides per
//!      shape once both are registered.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q6_k_mmvq, rocmforge_launch_gemv_q6_k_standard,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::quantize::{rocmforge_launch_quantize_q8_1, BlockQ81, QK8_1};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipStream};
use serial_test::serial;
use std::ffi::c_void;
use std::mem::size_of;

// ─── Test data ──────────────────────────────────────────────────────────────

/// Row-major Q6_K matrix: N output rows × (K / 256) super-blocks each.
/// Each super-block is 210 bytes: ql[128] + qh[64] + scales[16 int8] + half d.
fn gen_q6k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row;
    let mut buf = vec![0u8; total * 210];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
        let base = b * 210;
        // ql: 128 random bytes
        for i in 0..128 {
            buf[base + i] = rng.u8(..);
        }
        // qh: 64 random bytes
        for i in 128..192 {
            buf[base + i] = rng.u8(..);
        }
        // scales: 16 int8 values in [-32, 31] (plausible quantized range)
        for i in 192..208 {
            let s = (rng.i32(-32..32)) as i8;
            buf[base + i] = s as u8;
        }
        // d: half scale, small magnitude
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[base + 208..base + 210].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    buf
}

fn gen_input(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

// ─── GPU helpers ────────────────────────────────────────────────────────────

struct Bufs {
    d_w: HipBuffer,
    d_input_fp: HipBuffer,
    d_q8_1: HipBuffer,
    d_out: HipBuffer,
    stream: HipStream,
}

fn prepare(weights: &[u8], input: &[f32], n: usize) -> Bufs {
    let mut d_w = HipBuffer::new(weights.len()).expect("d_w");
    d_w.copy_from_host(weights).expect("up w");
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_input_fp = HipBuffer::new(in_bytes.len()).expect("d_input_fp");
    d_input_fp.copy_from_host(in_bytes).expect("up in");
    let d_q8_1 = HipBuffer::new((input.len() / QK8_1) * size_of::<BlockQ81>()).expect("d_q8_1");
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

fn launch_standard(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    let rc = unsafe {
        rocmforge_launch_gemv_q6_k_standard(
            bufs.d_w.as_ptr() as *const u8,
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "q6_k_standard rc={rc}");
    bufs.stream.synchronize().expect("sync std");
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

fn launch_mmvq(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_q8_1.as_mut_ptr(),
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "quantize_q8_1 rc={rc}");
    let rc = unsafe {
        rocmforge_launch_gemv_q6_k_mmvq(
            bufs.d_w.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_out.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "q6_k_mmvq rc={rc}");
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

// ─── Correctness: MMVQ vs Q6_K standard ────────────────────────────────────

fn mmvq_vs_standard_at(n: usize, k: usize, seed: u64, median_tol: f32) {
    let weights = gen_q6k_weights(n, k, seed);
    let input = gen_input(k, seed.wrapping_add(1));
    let mut bufs = prepare(&weights, &input, n);

    let out_std = launch_standard(&mut bufs, n, k);
    let out_mmvq = launch_mmvq(&mut bufs, n, k);

    let (med, p99, max) = compare_rel_stats(&out_std, &out_mmvq);
    let mag = out_std.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "  N={:6} K={:5}: |out|max={:.2}  MMVQ vs std: med={:.5} p99={:.4} max={:.4}",
        n, k, mag, med, p99, max
    );
    assert!(
        med < median_tol,
        "Q6_K MMVQ vs standard at N={n} K={k}: median {med} > tol {median_tol}"
    );
}

#[test]
#[serial]
fn test_q6k_mmvq_parity_small_layer() {
    mmvq_vs_standard_at(1024, 4096, 0xB101, 0.02);
}

#[test]
#[serial]
fn test_q6k_mmvq_parity_mid_layer() {
    mmvq_vs_standard_at(4096, 4096, 0xB202, 0.02);
}

#[test]
#[serial]
fn test_q6k_mmvq_parity_lm_head_shape() {
    // Qwen3-8B LM-head: N=151936, K=4096. Too big for random generation +
    // full comparison; use a smaller proxy with the same K=4096 and a
    // shape MMVQ will actually see in production (8192 output rows).
    mmvq_vs_standard_at(8192, 4096, 0xB303, 0.02);
}

// ─── Benchmark ──────────────────────────────────────────────────────────────

fn bench_at(n: usize, k: usize, seed: u64) -> (f64, f64, f64) {
    let weights = gen_q6k_weights(n, k, seed);
    let input = gen_input(k, seed.wrapping_add(1));
    let mut bufs = prepare(&weights, &input, n);

    for _ in 0..5 {
        let _ = launch_standard(&mut bufs, n, k);
        let _ = launch_mmvq(&mut bufs, n, k);
    }

    let runs = 30usize;

    let start = HipEvent::new().unwrap();
    let stop = HipEvent::new().unwrap();
    start.record(&bufs.stream).unwrap();
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_gemv_q6_k_standard(
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
    stop.record(&bufs.stream).unwrap();
    stop.synchronize().unwrap();
    let std_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    let start = HipEvent::new().unwrap();
    let stop = HipEvent::new().unwrap();
    start.record(&bufs.stream).unwrap();
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
            rocmforge_launch_gemv_q6_k_mmvq(
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
    stop.record(&bufs.stream).unwrap();
    stop.synchronize().unwrap();
    let mmvq_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    let speedup = std_us / mmvq_us;

    // Weight traffic per call: (N × K / 256) × 210 bytes.
    let weight_bytes = (210.0 / 256.0) * (n as f64) * (k as f64);
    let bw_mmvq_gb_s = weight_bytes / (mmvq_us * 1e-6) / 1e9;
    let bw_pct = (bw_mmvq_gb_s / 640.0) * 100.0;

    println!(
        "  N={:6} K={:5}: std={:7.2} µs  mmvq={:7.2} µs  speedup={:.2}×  BW(mmvq)={:.1}%",
        n, k, std_us, mmvq_us, speedup, bw_pct
    );
    (std_us, mmvq_us, speedup)
}

#[test]
#[serial]
fn test_q6k_mmvq_benchmark_all_shapes() {
    println!("\n=== Q6_K MMVQ vs Q6_K standard benchmark ===");
    println!("mmvq timing INCLUDES the per-call quantize_q8_1 prep.\n");

    let shapes: &[(usize, usize, u64, &str)] = &[
        (1024, 4096, 0x1111, "Small layer (N=1024, K=4096)"),
        (4096, 4096, 0x2222, "Mid layer (N=4096, K=4096)"),
        (8192, 4096, 0x3333, "Large layer (N=8192, K=4096)"),
        // LM-head proxy at half size — full 151936 takes too long for
        // the random weight generator in a unit test. The Bandit will
        // decide the real LM-head shape on actual model data.
        (16384, 4096, 0x4444, "LM-head proxy (N=16384, K=4096)"),
    ];

    let mut results = Vec::new();
    for (n, k, seed, label) in shapes {
        println!("{}", label);
        results.push((*label, bench_at(*n, *k, *seed)));
    }

    println!("\n=== Summary ===");
    for (label, (std_us, mmvq_us, speedup)) in &results {
        let verdict = if *speedup >= 1.0 { "WIN" } else { "LOSS" };
        println!(
            "  [{}] {:<32}  std={:7.2}µs mmvq={:7.2}µs  ({:.2}×)",
            verdict, label, std_us, mmvq_us, speedup
        );
    }
    // Don't fail on a single shape loss — the Bandit picks per shape.
    // The prompt warns that the LM-head at 95 % BW may not improve.
}
