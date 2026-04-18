//! Isolated timing of `launch_wmma_gemm_q4_k` on the real Phase 7
//! target shapes. Reports the median of 100 launches per shape.
//! Activation and weight buffers are allocated and warmed once per
//! shape so the measurement reflects kernel cost, not alloc/copy.

#![cfg(feature = "gpu")]

use half::f16;
use rocmforge::gpu::ffi::{
    hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_q4_k;
use rocmforge::gpu::weights::GpuBuffer;
use std::time::Instant;

const Q4_K_BLOCK_BYTES: usize = 144;

fn rand_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0) - 0.5) * 0.5
    }).collect()
}

fn rand_q4_k(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0);
    let blocks_per_row = k / 256;
    let mut buf = vec![0u8; n_rows * blocks_per_row * Q4_K_BLOCK_BYTES];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for block_idx in 0..(n_rows * blocks_per_row) {
        let base = block_idx * Q4_K_BLOCK_BYTES;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ds = ((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let d = f16::from_f32(0.001 + 0.010 * ds);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ms = ((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let dmin = f16::from_f32(0.001 + 0.010 * ms);
        let db = d.to_bits().to_le_bytes();
        let mb = dmin.to_bits().to_le_bytes();
        buf[base] = db[0]; buf[base + 1] = db[1];
        buf[base + 2] = mb[0]; buf[base + 3] = mb[1];
        for i in 0..140 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 4 + i] = (s >> 33) as u8;
        }
    }
    buf
}

fn bench_shape(label: &str, m: usize, n: usize, k: usize) {
    let a = rand_floats(m * k, 0xAABB);
    let w = rand_q4_k(n, k, 0xCCDD);
    let d_a = GpuBuffer::alloc(m * k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_d = GpuBuffer::alloc(m * n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    // Warmup
    for _ in 0..5 {
        launch_wmma_gemm_q4_k(
            d_a.as_ptr() as *const f32,
            d_w.as_ptr() as *const u8,
            d_d.as_ptr() as *mut f32,
            m, n, k,
            hipStream_t::null(),
        ).unwrap();
    }
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    // Timed runs
    let mut samples = Vec::with_capacity(100);
    for _ in 0..100 {
        let t0 = Instant::now();
        launch_wmma_gemm_q4_k(
            d_a.as_ptr() as *const f32,
            d_w.as_ptr() as *const u8,
            d_d.as_ptr() as *mut f32,
            m, n, k,
            hipStream_t::null(),
        ).unwrap();
        hip_stream_synchronize(hipStream_t::null()).unwrap();
        samples.push(t0.elapsed().as_micros() as u64);
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min_us = samples[0];
    println!("{:<38}  median={:>6} µs   min={:>6} µs", label, median, min_us);
}

fn main() {
    println!("Q4_K WMMA isolated timing (M={}, 100 launches, median):", 256);
    println!();
    bench_shape("256×4096×4096 (Qwen3 Q/O)",     256, 4096, 4096);
    bench_shape("256×12288×4096 (Qwen3 Gate/Up)", 256, 12288, 4096);
    bench_shape("256×4096×12288 (Qwen3 Down)",    256, 4096, 12288);
    bench_shape("256×14336×4096 (Llama Gate/Up)", 256, 14336, 4096);
    bench_shape("256×4096×14336 (Llama Down)",    256, 4096, 14336);
}
