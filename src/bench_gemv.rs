//! CPU GEMV micro-benchmark.
//!
//! Compares Q4_0 × Q8_0 kernel at the shapes used by Qwen2.5-0.5B and -7B.
//! Toggled between AVX2 and AVX-512 VNNI via `ROCMFORGE_DISABLE_AVX512`.

use rocmforge::cpu::features::CpuFeatures;
use rocmforge::cpu::ops::gemv_q4_0_q8_0;
use std::time::Instant;

fn make_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let num_blocks = in_dim / 32;
    let mut w = vec![0u8; out_dim * num_blocks * 18];
    for row in 0..out_dim {
        for b in 0..num_blocks {
            let off = row * num_blocks * 18 + b * 18;
            // scale = 0.1 as f16
            let scale_bits = half::f16::from_f32(0.1).to_bits().to_le_bytes();
            w[off] = scale_bits[0];
            w[off + 1] = scale_bits[1];
            for j in 0..16 {
                let lo = ((row + b + j) % 16) as u8;
                let hi = ((row * 3 + b + j + 1) % 16) as u8;
                w[off + 2 + j] = lo | (hi << 4);
            }
        }
    }
    w
}

fn bench_shape(label: &str, out_dim: usize, in_dim: usize, iters: usize) {
    let w = make_weights(out_dim, in_dim);
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut y = vec![0.0f32; out_dim];
    let mut scratch = vec![0u8; (in_dim / 32) * 34];

    // Warmup
    for _ in 0..10 {
        gemv_q4_0_q8_0(&w, &x, &mut y, out_dim, in_dim, Some(&mut scratch));
    }

    let start = Instant::now();
    for _ in 0..iters {
        gemv_q4_0_q8_0(&w, &x, &mut y, out_dim, in_dim, Some(&mut scratch));
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / iters as f64;
    let ops_per_call = 2.0 * out_dim as f64 * in_dim as f64; // mul+add per pair
    let gops = (ops_per_call * iters as f64) / elapsed.as_secs_f64() / 1e9;
    let bytes_per_call = out_dim as f64 * (in_dim as f64 / 32.0) * 18.0;
    let bandwidth_gbs = (bytes_per_call * iters as f64) / elapsed.as_secs_f64() / 1e9;

    println!(
        "  {:<32} {:>7.1} µs  {:>6.1} GOPS  {:>6.1} GB/s weight read",
        label, us_per_call, gops, bandwidth_gbs
    );
}

fn main() {
    let features = CpuFeatures::get();
    let disable_avx512 = std::env::var_os("ROCMFORGE_DISABLE_AVX512").is_some();
    let effective_path = if disable_avx512 {
        "AVX2"
    } else if features.has_avx512 && features.has_avx512_vnni {
        "AVX-512 VNNI"
    } else if features.has_avx2 {
        "AVX2"
    } else {
        "Scalar"
    };

    println!("Detected CPU features: {}", features.description());
    println!("Effective kernel path: {}", effective_path);
    println!(
        "ROCMFORGE_DISABLE_AVX512 is {}",
        if disable_avx512 { "set" } else { "unset" }
    );
    println!();

    // Qwen2.5-0.5B shapes
    println!("Qwen2.5-0.5B (hidden=896, intermediate=4864):");
    bench_shape("QKV/O  (896 × 896)",      896,   896, 5000);
    bench_shape("Gate/Up (4864 × 896)",    4864,  896, 2000);
    bench_shape("Down   (896 × 4864)",     896,  4864, 2000);
    println!();

    // Qwen2.5-7B shapes
    println!("Qwen2.5-7B (hidden=3584, intermediate=18944):");
    bench_shape("QKV/O   (3584 × 3584)",   3584,  3584, 200);
    bench_shape("Gate/Up (18944 × 3584)", 18944,  3584,  50);
    bench_shape("Down   (3584 × 18944)",   3584, 18944,  50);
}
