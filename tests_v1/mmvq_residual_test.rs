//! Phase 2 Schritt 2b — MMVQ-residual kernel (fused GEMV + residual-add epilog).
//!
//! Tests:
//!   1. Parity: `mmvq_residual(w, q8_1, residual=r, out=r)` must equal
//!      `mmvq(w, q8_1, out=tmp)` + `r += tmp` — the residual-add is
//!      a free epilog at lane 0, so the composite is bit-exact to FP
//!      summation order (same ops, same order, same rounding).
//!   2. Correctness vs CPU FP32 reference on the two Qwen3-8B shapes
//!      that actually dispatch through FusedGemmResidual: O-proj
//!      (N=4096 K=4096) and FFN-down (N=4096 K=12288).
//!   3. Benchmark vs `q4_k_q8_inline_residual` (the kernel it replaces)
//!      on the same shapes.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_mmvq, rocmforge_launch_gemv_q4_k_mmvq_residual,
    rocmforge_launch_gemv_q4_k_q8_inline_residual,
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

fn gen_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
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

fn cpu_gemv_plus_residual(
    weights: &[u8],
    input: &[f32],
    residual: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
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
        out[row] = acc + residual[row];
    }
    out
}

// ─── GPU launch helpers ─────────────────────────────────────────────────────

struct Bufs {
    d_w: HipBuffer,
    d_input_fp: HipBuffer,
    d_residual: HipBuffer,
    d_q8_1: HipBuffer,
    d_out: HipBuffer,
    stream: HipStream,
}

fn prepare(weights: &[u8], input: &[f32], residual: &[f32], n: usize) -> Bufs {
    let mut d_w = HipBuffer::new(weights.len()).expect("d_w");
    d_w.copy_from_host(weights).expect("up w");

    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_input_fp = HipBuffer::new(in_bytes.len()).expect("d_input_fp");
    d_input_fp.copy_from_host(in_bytes).expect("up in");

    let res_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(residual.as_ptr() as *const u8, residual.len() * 4) };
    let mut d_residual = HipBuffer::new(res_bytes.len()).expect("d_residual");
    d_residual.copy_from_host(res_bytes).expect("up residual");

    let d_q8_1 = HipBuffer::new((input.len() / QK8_1) * size_of::<BlockQ81>()).expect("d_q81");
    let d_out = HipBuffer::new(n * 4).expect("d_out");

    let stream = HipStream::new().expect("stream");
    Bufs {
        d_w,
        d_input_fp,
        d_residual,
        d_q8_1,
        d_out,
        stream,
    }
}

fn launch_mmvq_residual(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    // Copy residual into d_out (in-place pattern: output = residual read AND written)
    let mut residual_host = vec![0u8; bufs.d_residual.size()];
    bufs.d_residual.copy_to_host(&mut residual_host).unwrap();
    bufs.d_out.copy_from_host(&residual_host).unwrap();

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
        rocmforge_launch_gemv_q4_k_mmvq_residual(
            bufs.d_w.as_ptr(),
            bufs.d_q8_1.as_ptr(),
            bufs.d_out.as_ptr(), // residual = current output buffer
            bufs.d_out.as_mut_ptr(),
            n as i32,
            k as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "mmvq_residual rc={rc}");
    bufs.stream.synchronize().expect("sync mmvq_res");

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

fn launch_mmvq_plus_cpu_add(bufs: &mut Bufs, residual: &[f32], n: usize, k: usize) -> Vec<f32> {
    // Reference: run MMVQ (no residual) then add residual on CPU.
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
    for i in 0..n {
        out[i] += residual[i];
    }
    out
}

fn launch_q8_inline_residual(bufs: &mut Bufs, n: usize, k: usize) -> Vec<f32> {
    // Reset d_out to current residual
    let mut residual_host = vec![0u8; bufs.d_residual.size()];
    bufs.d_residual.copy_to_host(&mut residual_host).unwrap();
    bufs.d_out.copy_from_host(&residual_host).unwrap();

    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_q8_inline_residual(
            bufs.d_w.as_ptr() as *const u8,
            bufs.d_input_fp.as_ptr() as *const f32,
            bufs.d_out.as_ptr() as *const f32,
            bufs.d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            bufs.stream.raw(),
        )
    };
    assert_eq!(rc, 0, "q8_inline_residual rc={rc}");
    bufs.stream.synchronize().expect("sync q8ir");

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

// ─── Parity: mmvq_residual vs (mmvq + CPU add) ──────────────────────────────

fn parity_composite_at(n: usize, k: usize, seed: u64) {
    let w = gen_q4k(n, k, seed);
    let input = gen_vec(k, seed.wrapping_add(1), 1.0);
    let residual = gen_vec(n, seed.wrapping_add(2), 0.5);

    let mut bufs = prepare(&w, &input, &residual, n);

    // Reference: mmvq WITHOUT residual → add residual on CPU.
    let out_composite = launch_mmvq_plus_cpu_add(&mut bufs, &residual, n, k);
    // Candidate: fused mmvq_residual.
    let out_fused = launch_mmvq_residual(&mut bufs, n, k);

    // The fused kernel does `tmp + residual[row]` at lane 0, exactly
    // matching the composite. Expect bit-identity modulo FP FMA rounding.
    let (med, p99, max) = compare_rel_stats(&out_composite, &out_fused);
    let mag = out_composite
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    println!(
        "  N={:5} K={:5}: |out|max={:.2} fused vs composite med={:.6} p99={:.5} max={:.5}",
        n, k, mag, med, p99, max
    );
    assert!(
        max < 1e-3,
        "fused vs composite at N={n} K={k}: max rel = {max}"
    );
}

#[test]
#[serial]
fn test_mmvq_residual_o_proj_parity_composite() {
    parity_composite_at(4096, 4096, 0xE1E1);
}

#[test]
#[serial]
fn test_mmvq_residual_ffn_down_parity_composite() {
    parity_composite_at(4096, 12288, 0xE2E2);
}

// ─── Correctness: mmvq_residual vs CPU FP32 reference ───────────────────────

fn mmvq_residual_vs_cpu_at(n: usize, k: usize, seed: u64, tol_median: f32) {
    let w = gen_q4k(n, k, seed);
    let input = gen_vec(k, seed.wrapping_add(1), 1.0);
    let residual = gen_vec(n, seed.wrapping_add(2), 0.5);

    let cpu_ref = cpu_gemv_plus_residual(&w, &input, &residual, n, k);

    let mut bufs = prepare(&w, &input, &residual, n);
    let out = launch_mmvq_residual(&mut bufs, n, k);

    let (med, p99, max) = compare_rel_stats(&cpu_ref, &out);
    let mag = cpu_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!(
        "  N={:5} K={:5}: |out|max={:.2} vs CPU med={:.5} p99={:.4} max={:.4}",
        n, k, mag, med, p99, max
    );
    assert!(
        med < tol_median,
        "MMVQ-residual median vs CPU at N={n} K={k}: {med} > {tol_median}"
    );
}

#[test]
#[serial]
fn test_mmvq_residual_vs_cpu_o_proj() {
    mmvq_residual_vs_cpu_at(4096, 4096, 0xF1F1, 0.02);
}

#[test]
#[serial]
fn test_mmvq_residual_vs_cpu_ffn_down() {
    mmvq_residual_vs_cpu_at(4096, 12288, 0xF2F2, 0.02);
}

// ─── Benchmark: mmvq_residual vs q8_inline_residual ─────────────────────────

fn bench_at(n: usize, k: usize, seed: u64) -> (f64, f64, f64) {
    let w = gen_q4k(n, k, seed);
    let input = gen_vec(k, seed.wrapping_add(1), 1.0);
    let residual = gen_vec(n, seed.wrapping_add(2), 0.5);
    let mut bufs = prepare(&w, &input, &residual, n);

    // Warm-up.
    for _ in 0..5 {
        let _ = launch_q8_inline_residual(&mut bufs, n, k);
        let _ = launch_mmvq_residual(&mut bufs, n, k);
    }

    let runs = 50usize;

    // Time q8_inline_residual (single kernel, inline quant).
    let start = HipEvent::new().unwrap();
    let stop = HipEvent::new().unwrap();
    start.record(&bufs.stream).unwrap();
    for _ in 0..runs {
        let rc = unsafe {
            rocmforge_launch_gemv_q4_k_q8_inline_residual(
                bufs.d_w.as_ptr() as *const u8,
                bufs.d_input_fp.as_ptr() as *const f32,
                bufs.d_out.as_ptr() as *const f32,
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
    let q8ir_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    // Time mmvq_residual incl. quantize_q8_1 prep.
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
            rocmforge_launch_gemv_q4_k_mmvq_residual(
                bufs.d_w.as_ptr(),
                bufs.d_q8_1.as_ptr(),
                bufs.d_out.as_ptr(),
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
    let mmvqr_us = (HipEvent::elapsed_ms(&start, &stop).unwrap() as f64 / runs as f64) * 1000.0;

    let speedup = q8ir_us / mmvqr_us;
    println!(
        "  N={:5} K={:5}: q8_inline_res={:6.2} µs  mmvq_res={:6.2} µs  speedup={:.2}×",
        n, k, q8ir_us, mmvqr_us, speedup
    );
    (q8ir_us, mmvqr_us, speedup)
}

#[test]
#[serial]
fn test_mmvq_residual_benchmark_both_shapes() {
    println!("\n=== MMVQ-residual vs q8_inline_residual benchmark ===");
    println!("mmvq_residual timing INCLUDES the per-call quantize_q8_1.\n");
    let (q_o, m_o, s_o) = bench_at(4096, 4096, 0x1001);
    let (q_d, m_d, s_d) = bench_at(4096, 12288, 0x1002);
    println!("\n=== Summary ===");
    println!("  O-proj   (N=4096, K=4096):  q8ir={:6.2}µs mmvqr={:6.2}µs ({:.2}×)", q_o, m_o, s_o);
    println!("  FFN-down (N=4096, K=12288): q8ir={:6.2}µs mmvqr={:6.2}µs ({:.2}×)", q_d, m_d, s_d);
}
