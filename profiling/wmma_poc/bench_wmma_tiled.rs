//! Throughput benchmark for the Phase 2a Step 2 tiled WMMA GEMM.
//!
//! Builds as a standalone example binary inside the POC directory so it
//! can be committed without adding to the main test/bench matrix.
//!
//! Measures WMMA-tiled and hipBLAS Hgemm wall-clock times at the three
//! Qwen2.5-7B prefill shape classes at pp=256. Reports median µs per
//! call and speedup against hipBLAS, plus the FP16 theoretical-peak
//! reference at 49 TFLOPS.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_event_create, hip_event_destroy, hip_event_elapsed_time, hip_event_record,
    hip_event_synchronize, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::hipblas_ffi::{f16_bits_from_f32, hgemm, hipblasOperation_t, HipBlasHandle};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_tiled;
use rocmforge::gpu::weights::GpuBuffer;
use serde_json::json;
use std::time::SystemTime;

const FP16_TFLOPS: f64 = 49.0e12; // RDNA 4 RX 9070 XT spec FP16 peak

fn seeded_halfs(n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (state >> 33) as u32;
        let normalised = ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5;
        out.push(f16::from_f32(normalised));
    }
    out
}

fn time_kernel<F: FnMut()>(warmup: usize, iters: usize, mut call: F) -> Vec<f64> {
    let start = hip_event_create().expect("start");
    let stop = hip_event_create().expect("stop");

    for _ in 0..warmup {
        call();
    }
    hip_stream_synchronize(hipStream_t::null()).expect("warmup sync");

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        hip_event_record(start, hipStream_t::null()).expect("start rec");
        call();
        hip_event_record(stop, hipStream_t::null()).expect("stop rec");
        hip_event_synchronize(stop).expect("stop sync");
        let ms = hip_event_elapsed_time(start, stop).expect("elapsed");
        samples.push(ms as f64 * 1000.0); // → µs
    }

    hip_event_destroy(start).ok();
    hip_event_destroy(stop).ok();
    samples
}

fn summary(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p95_idx = (sorted.len() as f64 * 0.95) as usize;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];
    (mean, median, p95)
}

struct Result {
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
    wmma_us: (f64, f64, f64),
    hipblas_us: (f64, f64, f64),
    theoretical_peak_us: f64,
}

fn bench_shape(label: &'static str, m: usize, n: usize, k: usize) -> Result {
    let warmup = 10;
    let iters = 100;

    let a_host = seeded_halfs(m * k, 1);
    let b_host = seeded_halfs(k * n, 2);
    let d_a = GpuBuffer::alloc(m * k * 2).unwrap();
    let d_b = GpuBuffer::alloc(k * n * 2).unwrap();
    let d_d_wmma = GpuBuffer::alloc(m * n * 4).unwrap();
    let d_d_hipb = GpuBuffer::alloc(m * n * 2).unwrap();

    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a_host.as_ptr() as *const u8, m * k * 2).unwrap();
        hip_memcpy_h2d(d_b.as_ptr(), b_host.as_ptr() as *const u8, k * n * 2).unwrap();
    }

    // WMMA timing
    let wmma_samples = time_kernel(warmup, iters, || {
        launch_wmma_gemm_tiled(
            d_a.as_ptr() as *const u16,
            d_b.as_ptr() as *const u16,
            d_d_wmma.as_ptr() as *mut f32,
            m,
            n,
            k,
            hipStream_t::null(),
        )
        .unwrap();
    });

    // hipBLAS timing
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
            d_b.as_ptr() as *const u16,
            n as i32,
            d_a.as_ptr() as *const u16,
            k as i32,
            beta,
            d_d_hipb.as_ptr() as *mut u16,
            n as i32,
        )
        .unwrap();
    });

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let peak_us = (flops / FP16_TFLOPS) * 1.0e6;

    Result {
        label,
        m,
        n,
        k,
        wmma_us: summary(&wmma_samples),
        hipblas_us: summary(&hipblas_samples),
        theoretical_peak_us: peak_us,
    }
}

fn emit_json(results: &[Result]) -> serde_json::Value {
    let entries: Vec<_> = results
        .iter()
        .map(|r| {
            let (wm, wmed, wp95) = r.wmma_us;
            let (hm, hmed, hp95) = r.hipblas_us;
            let speedup = hmed / wmed;
            let peak_frac = r.theoretical_peak_us / wmed;
            json!({
                "label": r.label,
                "M": r.m, "N": r.n, "K": r.k,
                "wmma":   { "mean_us": wm, "median_us": wmed, "p95_us": wp95 },
                "hipblas":{ "mean_us": hm, "median_us": hmed, "p95_us": hp95 },
                "theoretical_peak_us": r.theoretical_peak_us,
                "speedup_vs_hipblas": speedup,
                "fraction_of_peak": peak_frac,
            })
        })
        .collect();
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    json!({
        "ts": ts,
        "hardware": "RX 9070 XT (gfx1201, RDNA 4)",
        "theoretical_peak_fp16_tflops": FP16_TFLOPS / 1.0e12,
        "shapes": entries,
    })
}

fn main() {
    let shapes = [
        ("QKV/O",   256,  3584,  3584),
        ("Gate/Up", 256, 18944,  3584),
        ("Down",    256,  3584, 18944),
    ];
    let mut results = Vec::new();
    for (label, m, n, k) in shapes {
        eprintln!("benchmarking {} (M={}, N={}, K={})", label, m, n, k);
        let r = bench_shape(label, m, n, k);
        let (_, wmed, _) = r.wmma_us;
        let (_, hmed, _) = r.hipblas_us;
        eprintln!(
            "  wmma median = {:7.1} µs  hipblas median = {:7.1} µs  speedup = {:5.2}×  peak µs = {:7.1} ({:4.1}%)",
            wmed,
            hmed,
            hmed / wmed,
            r.theoretical_peak_us,
            100.0 * r.theoretical_peak_us / wmed
        );
        results.push(r);
    }
    let j = emit_json(&results);
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
}
