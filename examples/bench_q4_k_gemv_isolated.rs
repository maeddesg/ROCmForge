//! Isolated timing of `gemv_q4_k_f32_on_stream` on the real Phase 7
//! projection shapes (M = 1). Reports the median of 200 launches per
//! shape. Buffers are allocated and warmed once per shape.

#![cfg(feature = "gpu")]

use half::f16;
use rocmforge::gpu::ffi::{
    hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::gemv_q4_k_f32_on_stream;
use rocmforge::gpu::weights::GpuBuffer;
use std::time::Instant;

const Q4_K_BLOCK_BYTES: usize = 144;

fn rand_q4_k(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
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

fn rand_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0) - 0.5) * 0.5
    }).collect()
}

fn bench_shape(label: &str, n: usize, k: usize) {
    let a = rand_floats(k, 0xAABB);
    let w = rand_q4_k(n, k, 0xCCDD);

    let d_a = GpuBuffer::alloc(k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_out = GpuBuffer::alloc(n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    for _ in 0..10 {
        gemv_q4_k_f32_on_stream(
            d_w.as_ptr() as *const u8,
            d_a.as_ptr() as *const f32,
            d_out.as_ptr() as *mut f32,
            k, n,
            hipStream_t::null(),
        ).unwrap();
    }
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut samples = Vec::with_capacity(200);
    for _ in 0..200 {
        let t0 = Instant::now();
        gemv_q4_k_f32_on_stream(
            d_w.as_ptr() as *const u8,
            d_a.as_ptr() as *const f32,
            d_out.as_ptr() as *mut f32,
            k, n,
            hipStream_t::null(),
        ).unwrap();
        hip_stream_synchronize(hipStream_t::null()).unwrap();
        samples.push(t0.elapsed().as_micros() as u64);
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min_us = samples[0];
    println!("{:<42}  median={:>5} µs   min={:>5} µs", label, median, min_us);
}

fn main() {
    println!("Q4_K GEMV isolated timing (M=1, 200 launches, median):");
    println!();
    bench_shape("N=4096  K=4096  (Qwen3 Q/O)",      4096, 4096);
    bench_shape("N=12288 K=4096  (Qwen3 Gate/Up)",  12288, 4096);
    bench_shape("N=4096  K=12288 (Qwen3 Down)",     4096, 12288);
    bench_shape("N=14336 K=4096  (Llama Gate/Up)",  14336, 4096);
    bench_shape("N=4096  K=14336 (Llama Down)",     4096, 14336);
}
