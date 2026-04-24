//! Phase 2 Schritt 3 — MMVQ fused Gate + Up + SwiGLU.
//!
//! Tests:
//!   1. Correctness: `mmvq_fused(W_gate, W_up, q8_1)` matches the
//!      composite `silu(mmvq(W_gate, q8_1)) * mmvq(W_up, q8_1)` within
//!      FP-reassociation tolerance (same ops, same rounding order per
//!      lane — the only difference is the epilog's intra-register mul
//!      vs. an external SwiGLU kernel).
//!   2. Vs CPU FP32 reference on the gate/up shape (N=12288, K=4096
//!      for Qwen3-8B).
//!   3. **Critical gate:** isolated benchmark of fused vs unfused —
//!      mmvq_fused µs must beat (quantize + 2×mmvq + swiglu) µs.
//!      If it doesn't, the kernel repeats the Phase-2 gate_up failure
//!      and we DO NOT register it.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::elementwise::rocmforge_launch_swiglu;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_mmvq, rocmforge_launch_gemv_q4_k_mmvq_fused,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::quantize::{rocmforge_launch_quantize_q8_1, BlockQ81, QK8_1};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipEvent, HipStream};
use serial_test::serial;
use std::ffi::c_void;
use std::mem::size_of;

// ─── Test data ──────────────────────────────────────────────────────────────

fn gen_q4k(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row;
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

fn dequant_q4k_block(bytes: &[u8]) -> [f32; 256] {
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn cpu_fused_reference(
    w_gate: &[u8],
    w_up: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let gate = cpu_fp32_gemv(w_gate, input, n, k);
    let up = cpu_fp32_gemv(w_up, input, n, k);
    gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect()
}

// ─── GPU helpers ────────────────────────────────────────────────────────────

struct Bufs {
    d_w_gate: HipBuffer,
    d_w_up: HipBuffer,
    d_input_fp: HipBuffer,
    d_q8_1: HipBuffer,
    d_gate_scratch: HipBuffer,
    d_up_scratch: HipBuffer,
    d_out: HipBuffer,
    stream: HipStream,
}

fn prepare(w_gate: &[u8], w_up: &[u8], input: &[f32], n: usize) -> Bufs {
    let mut d_w_gate = HipBuffer::new(w_gate.len()).unwrap();
    d_w_gate.copy_from_host(w_gate).unwrap();
    let mut d_w_up = HipBuffer::new(w_up.len()).unwrap();
    d_w_up.copy_from_host(w_up).unwrap();

    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_input_fp = HipBuffer::new(in_bytes.len()).unwrap();
    d_input_fp.copy_from_host(in_bytes).unwrap();

    let d_q8_1 = HipBuffer::new((input.len() / QK8_1) * size_of::<BlockQ81>()).unwrap();
    let d_gate_scratch = HipBuffer::new(n * 4).unwrap();
    let d_up_scratch = HipBuffer::new(n * 4).unwrap();
    let d_out = HipBuffer::new(n * 4).unwrap();

    let stream = HipStream::new().unwrap();
    Bufs {
        d_w_gate,
        d_w_up,
        d_input_fp,
        d_q8_1,
        d_gate_scratch,
        d_up_scratch,
        d_out,
        stream,
    }
}

fn launch_fused(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
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
        rocmforge_launch_gemv_q4_k_mmvq_fused(
            bufs.d_w_gate.as_ptr(),
            bufs.d_w_up.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_out.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "mmvq_fused rc={rc}");
    bufs.stream.synchronize().unwrap();

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

fn launch_unfused(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    // quantize_q8_1
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1(
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_q8_1.as_mut_ptr(),
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0);
    // mmvq(gate)
    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_mmvq(
            bufs.d_w_gate.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_gate_scratch.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0);
    // mmvq(up)
    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_mmvq(
            bufs.d_w_up.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_up_scratch.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0);
    // swiglu(gate, up) → out
    let rc = unsafe {
        rocmforge_launch_swiglu(
            bufs.d_gate_scratch.as_ptr() as *const f32,
            bufs.d_up_scratch.as_ptr() as *const f32,
            bufs.d_out.as_mut_ptr() as *mut f32,
            n as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0);
    bufs.stream.synchronize().unwrap();

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

// ─── Correctness tests ──────────────────────────────────────────────────────

#[test]
#[serial]
fn test_fused_vs_unfused_parity() {
    // Qwen3-8B gate/up shape.
    const N: usize = 12288;
    const K: usize = 4096;
    let w_gate = gen_q4k(N, K, 0xCAFE);
    let w_up = gen_q4k(N, K, 0xBABE);
    let input = gen_input(K, 0xF00D);
    let mut bufs = prepare(&w_gate, &w_up, &input, N);

    let out_unfused = launch_unfused(&mut bufs, N, K);
    let out_fused = launch_fused(&mut bufs, N, K);

    let (med, p99, max) = compare_rel_stats(&out_unfused, &out_fused);
    let mag = out_unfused.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "  N={} K={}: |out|max={:.2} fused vs unfused med={:.5} p99={:.4} max={:.4}",
        N, K, mag, med, p99, max
    );
    // The fused kernel computes the same dots in the same FP-summation
    // order (both tmp_up and tmp_gate accumulate lane-by-lane); only
    // the SwiGLU mul happens in register instead of after a roundtrip.
    // Expect median ~0, some p99/max on near-zero outputs.
    assert!(med < 0.01, "fused vs unfused median: {med} > 1%");
}

#[test]
#[serial]
fn test_fused_vs_cpu_reference() {
    const N: usize = 12288;
    const K: usize = 4096;
    let w_gate = gen_q4k(N, K, 0x1234);
    let w_up = gen_q4k(N, K, 0x5678);
    let input = gen_input(K, 0x9ABC);
    let mut bufs = prepare(&w_gate, &w_up, &input, N);

    let cpu_ref = cpu_fused_reference(&w_gate, &w_up, &input, N, K);
    let out_fused = launch_fused(&mut bufs, N, K);

    let (med, p99, max) = compare_rel_stats(&cpu_ref, &out_fused);
    let mag = cpu_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "  N={} K={}: |out|max={:.2} fused vs CPU med={:.5} p99={:.4} max={:.4}",
        N, K, mag, med, p99, max
    );
    assert!(med < 0.02, "fused vs CPU median: {med} > 2%");
}

// ─── Performance — THE critical gate ────────────────────────────────────────

#[test]
#[serial]
fn test_fused_beats_unfused_benchmark() {
    const N: usize = 12288;
    const K: usize = 4096;
    let w_gate = gen_q4k(N, K, 0xAA01);
    let w_up = gen_q4k(N, K, 0xAA02);
    let input = gen_input(K, 0xAA03);
    let mut bufs = prepare(&w_gate, &w_up, &input, N);

    // Warmup.
    for _ in 0..5 {
        let _ = launch_unfused(&mut bufs, N, K);
        let _ = launch_fused(&mut bufs, N, K);
    }

    let runs = 50usize;

    // Time unfused (5 kernels).
    let start = HipEvent::new().unwrap();
    let stop = HipEvent::new().unwrap();
    start.record(&bufs.stream).unwrap();
    for _ in 0..runs {
        // quantize
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                bufs.d_input_fp.as_ptr() as *const f32,
                bufs.d_q8_1.as_mut_ptr(),
                K as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_mmvq(
                bufs.d_w_gate.as_ptr(),
                bufs.d_q8_1.as_ptr(),
                bufs.d_gate_scratch.as_mut_ptr(),
                N as i32,
                K as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_mmvq(
                bufs.d_w_up.as_ptr(),
                bufs.d_q8_1.as_ptr(),
                bufs.d_up_scratch.as_mut_ptr(),
                N as i32,
                K as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            rocmforge_launch_swiglu(
                bufs.d_gate_scratch.as_ptr() as *const f32,
                bufs.d_up_scratch.as_ptr() as *const f32,
                bufs.d_out.as_mut_ptr() as *mut f32,
                N as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stop.record(&bufs.stream).unwrap();
    stop.synchronize().unwrap();
    let unfused_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    // Time fused (2 kernels).
    let start = HipEvent::new().unwrap();
    let stop = HipEvent::new().unwrap();
    start.record(&bufs.stream).unwrap();
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_quantize_q8_1(
                bufs.d_input_fp.as_ptr() as *const f32,
                bufs.d_q8_1.as_mut_ptr(),
                K as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_mmvq_fused(
                bufs.d_w_gate.as_ptr(),
                bufs.d_w_up.as_ptr(),
                bufs.d_q8_1.as_ptr(),
                bufs.d_out.as_mut_ptr(),
                N as i32,
                K as i32,
                bufs.stream.raw(),
            )
        };
        assert_eq!(rc, 0);
    }
    stop.record(&bufs.stream).unwrap();
    stop.synchronize().unwrap();
    let fused_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    let speedup = unfused_us / fused_us;

    // BW% for fused: weights read = 2 × (N × K / 256) × 144 bytes.
    let weight_bytes = 2.0 * 0.5625 * (N as f64) * (K as f64);
    let bw_gb_s = weight_bytes / (fused_us * 1e-6) / 1e9;
    let bw_pct = (bw_gb_s / 640.0) * 100.0;

    println!(
        "\n=== MMVQ fused vs unfused (N={}, K={}) — {} runs ===",
        N, K, runs
    );
    println!(
        "  unfused (5 kernels): {:.2} µs",
        unfused_us
    );
    println!(
        "  fused   (2 kernels): {:.2} µs  ({:.2}× speedup, {:.1}% BW)",
        fused_us, speedup, bw_pct
    );

    // This is the CRITICAL gate. If fused is slower, we hit the
    // Phase-2 gate_up failure mode and must NOT ship this kernel.
    assert!(
        speedup > 1.0,
        "fused kernel MUST beat unfused — repeat of gate_up_swiglu regression risk"
    );
}
