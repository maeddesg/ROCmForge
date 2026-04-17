//! Phase 2b throughput benchmark.
//!
//! Compares:
//!   (a) hipBLAS Hgemm (VALU fallback on gfx1201) — Phase 1 baseline
//!   (b) Phase 2a: FP16-WMMA kernel with a separate Q4_0 → FP16 dequant
//!   (c) Phase 2b: WMMA with inline Q4_0 dequant (the kernel under test)
//!
//! For (b) we time WMMA only and separately time the dequant kernel so
//! we can add them as "(b) total = WMMA + dequant" — that's the
//! apples-to-apples comparison against (c).
//!
//! All timings are medians of 100 measured iterations with HIP events.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_event_create, hip_event_destroy, hip_event_elapsed_time, hip_event_record,
    hip_event_synchronize, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::hipblas_ffi::{f16_bits_from_f32, hgemm, hipblasOperation_t, HipBlasHandle};
use rocmforge::gpu::kernels::wmma::{launch_wmma_gemm_q4_0, launch_wmma_gemm_tiled};
use rocmforge::gpu::prefill_gemm::dequantize_q4_0_to_f16_on_stream;
use rocmforge::gpu::weights::GpuBuffer;
use serde_json::json;
use std::time::SystemTime;

const FP16_TFLOPS: f64 = 49.0e12;
const Q4_0_BLOCK_BYTES: usize = 18;
const QK4_0: usize = 32;

fn seeded_halfs(n: usize, seed: u64) -> Vec<f16> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (s >> 33) as u32;
            f16::from_f32(((r & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5)
        })
        .collect()
}

fn seeded_q4_0(n_rows: usize, k_cols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k_cols % QK4_0, 0);
    let nb = n_rows * k_cols / QK4_0;
    let mut buf = vec![0u8; nb * Q4_0_BLOCK_BYTES];
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for blk in 0..nb {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let f = ((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let scale = f16::from_f32(0.002 + 0.018 * f).to_bits().to_le_bytes();
        buf[blk * 18] = scale[0];
        buf[blk * 18 + 1] = scale[1];
        for i in 0..16 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[blk * 18 + 2 + i] = (s >> 33) as u8;
        }
    }
    buf
}

fn time_kernel<F: FnMut()>(warmup: usize, iters: usize, mut call: F) -> Vec<f64> {
    let start = hip_event_create().unwrap();
    let stop = hip_event_create().unwrap();
    for _ in 0..warmup {
        call();
    }
    hip_stream_synchronize(hipStream_t::null()).unwrap();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        hip_event_record(start, hipStream_t::null()).unwrap();
        call();
        hip_event_record(stop, hipStream_t::null()).unwrap();
        hip_event_synchronize(stop).unwrap();
        samples.push(hip_event_elapsed_time(start, stop).unwrap() as f64 * 1000.0);
    }
    hip_event_destroy(start).ok();
    hip_event_destroy(stop).ok();
    samples
}

fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut v: Vec<f64> = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = v[v.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95 = v[(v.len() as f64 * 0.95) as usize];
    (mean, median, p95)
}

struct Row {
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
    hipblas: (f64, f64, f64),
    phase2a_wmma: (f64, f64, f64),
    phase2a_dequant: (f64, f64, f64),
    phase2b: (f64, f64, f64),
    peak_us: f64,
}

fn bench(label: &'static str, m: usize, n: usize, k: usize) -> Row {
    let warmup = 10;
    let iters = 100;

    // Inputs
    let a_halfs = seeded_halfs(m * k, 1);
    let a_f32: Vec<f32> = a_halfs.iter().map(|x| x.to_f32()).collect();
    let w_q4 = seeded_q4_0(n, k, 2);
    let b_halfs = seeded_halfs(k * n, 3); // dummy for FP16 path — shape matches

    let d_a_f32 = GpuBuffer::alloc(m * k * 4).unwrap();
    let d_a_f16 = GpuBuffer::alloc(m * k * 2).unwrap();
    let d_b_f16 = GpuBuffer::alloc(k * n * 2).unwrap();
    let d_w_q4  = GpuBuffer::alloc(w_q4.len()).unwrap();
    let d_w_f16 = GpuBuffer::alloc(n * k * 2).unwrap();
    let d_d_f32 = GpuBuffer::alloc(m * n * 4).unwrap();
    let d_d_f16 = GpuBuffer::alloc(m * n * 2).unwrap();

    unsafe {
        hip_memcpy_h2d(d_a_f32.as_ptr(), a_f32.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_a_f16.as_ptr(), a_halfs.as_ptr() as *const u8, m * k * 2).unwrap();
        hip_memcpy_h2d(d_b_f16.as_ptr(), b_halfs.as_ptr() as *const u8, k * n * 2).unwrap();
        hip_memcpy_h2d(d_w_q4.as_ptr(), w_q4.as_ptr() as *const u8, w_q4.len()).unwrap();
    }

    // (a) hipBLAS Hgemm reference — row-major D = A @ B
    let handle = HipBlasHandle::create().unwrap();
    handle.set_stream(hipStream_t::null()).unwrap();
    let alpha = f16_bits_from_f32(1.0);
    let beta = f16_bits_from_f32(0.0);
    let hipblas_samples = time_kernel(warmup, iters, || unsafe {
        hgemm(
            &handle,
            hipblasOperation_t::HIPBLAS_OP_N,
            hipblasOperation_t::HIPBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            alpha,
            d_b_f16.as_ptr() as *const u16,
            n as i32,
            d_a_f16.as_ptr() as *const u16,
            k as i32,
            beta,
            d_d_f16.as_ptr() as *mut u16,
            n as i32,
        )
        .unwrap();
    });

    // (b) Phase 2a WMMA isolated (B_f16 already resident, no dequant in timing)
    let phase2a_wmma_samples = time_kernel(warmup, iters, || {
        launch_wmma_gemm_tiled(
            d_a_f16.as_ptr() as *const u16,
            d_b_f16.as_ptr() as *const u16,
            d_d_f32.as_ptr() as *mut f32,
            m, n, k,
            hipStream_t::null(),
        )
        .unwrap();
    });

    // (b') Separate Q4_0 → FP16 dequant timing (to build "phase 2a total")
    let phase2a_dequant_samples = time_kernel(warmup, iters, || {
        dequantize_q4_0_to_f16_on_stream(
            d_w_q4.as_ptr(),
            d_w_f16.as_ptr(),
            n * k,
            hipStream_t::null(),
        )
        .unwrap();
    });

    // (c) Phase 2b inline Q4_0 + WMMA
    let phase2b_samples = time_kernel(warmup, iters, || {
        launch_wmma_gemm_q4_0(
            d_a_f32.as_ptr() as *const f32,
            d_w_q4.as_ptr() as *const u8,
            d_d_f32.as_ptr() as *mut f32,
            m, n, k,
            hipStream_t::null(),
        )
        .unwrap();
    });

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let peak_us = flops / FP16_TFLOPS * 1.0e6;

    Row {
        label, m, n, k,
        hipblas: stats(&hipblas_samples),
        phase2a_wmma: stats(&phase2a_wmma_samples),
        phase2a_dequant: stats(&phase2a_dequant_samples),
        phase2b: stats(&phase2b_samples),
        peak_us,
    }
}

fn main() {
    let shapes = [
        ("QKV/O",   256,  3584,  3584),
        ("Gate/Up", 256, 18944,  3584),
        ("Down",    256,  3584, 18944),
    ];

    let mut rows = Vec::new();
    for (label, m, n, k) in shapes {
        eprintln!("benchmarking {} (M={}, N={}, K={})", label, m, n, k);
        let r = bench(label, m, n, k);
        let (_, wm, _) = r.hipblas;
        let (_, p2a_w, _) = r.phase2a_wmma;
        let (_, p2a_d, _) = r.phase2a_dequant;
        let (_, p2b, _) = r.phase2b;
        let p2a_total = p2a_w + p2a_d;
        eprintln!(
            "  hipblas {:7.1} µs  |  p2a (wmma {:6.1} + deq {:5.1} = {:6.1})  |  p2b {:6.1} µs  |  speedup vs hipblas {:4.2}×  vs p2a-total {:4.2}×  peak {:5.1} ({:4.1}% of peak)",
            wm, p2a_w, p2a_d, p2a_total, p2b, wm / p2b, p2a_total / p2b, r.peak_us, 100.0 * r.peak_us / p2b
        );
        rows.push(r);
    }

    let entries: Vec<_> = rows
        .iter()
        .map(|r| {
            let p2a_total_median = r.phase2a_wmma.1 + r.phase2a_dequant.1;
            json!({
                "label": r.label,
                "M": r.m, "N": r.n, "K": r.k,
                "hipblas_us":           { "mean": r.hipblas.0,        "median": r.hipblas.1,        "p95": r.hipblas.2 },
                "phase2a_wmma_us":      { "mean": r.phase2a_wmma.0,   "median": r.phase2a_wmma.1,   "p95": r.phase2a_wmma.2 },
                "phase2a_dequant_us":   { "mean": r.phase2a_dequant.0,"median": r.phase2a_dequant.1,"p95": r.phase2a_dequant.2 },
                "phase2a_total_median_us": p2a_total_median,
                "phase2b_us":           { "mean": r.phase2b.0,        "median": r.phase2b.1,        "p95": r.phase2b.2 },
                "speedup_vs_hipblas": r.hipblas.1 / r.phase2b.1,
                "speedup_vs_phase2a_total": p2a_total_median / r.phase2b.1,
                "theoretical_peak_us": r.peak_us,
                "fraction_of_peak": r.peak_us / r.phase2b.1,
            })
        })
        .collect();

    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let j = json!({
        "ts": ts,
        "hardware": "RX 9070 XT (gfx1201, RDNA 4)",
        "theoretical_peak_fp16_tflops": FP16_TFLOPS / 1.0e12,
        "shapes": entries,
    });
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
}
