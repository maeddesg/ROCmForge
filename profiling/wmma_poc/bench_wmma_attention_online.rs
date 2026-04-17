//! Phase 3b throughput — online-softmax WMMA attention at pp64, 128,
//! 256, 512 vs the existing scalar `flash_attn_prefill_strided_kernel`.
//!
//! Both paths process all 28 Q heads per measured iteration. HIP
//! events, 10 warmup + 100 measured, median µs.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_event_create, hip_event_destroy, hip_event_elapsed_time, hip_event_record,
    hip_event_synchronize, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::attention::flash_attn_prefill_strided;
use rocmforge::gpu::kernels::wmma::launch_wmma_attention_prefill_online;
use rocmforge::gpu::weights::GpuBuffer;
use serde_json::json;
use std::time::SystemTime;

const HEAD_DIM: usize = 128;
const NUM_HEADS: usize = 28;

fn seeded_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (s >> 33) as u32;
        ((r & 0x3F) as f32) / 256.0
    }).collect()
}

fn time_kernel<F: FnMut()>(warmup: usize, iters: usize, mut call: F) -> Vec<f64> {
    let start = hip_event_create().unwrap();
    let stop  = hip_event_create().unwrap();
    for _ in 0..warmup { call(); }
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

fn median(samples: &[f64]) -> f64 {
    let mut v: Vec<f64> = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_seq(seq_len: usize) -> (f64, f64) {
    let warmup = 10;
    let iters = 100;

    let stride = NUM_HEADS * HEAD_DIM;
    let total = seq_len * stride;

    let q_f32 = seeded_f32(total, 1);
    let k_f32 = seeded_f32(total, 2);
    let v_f32 = seeded_f32(total, 3);
    let q_f16: Vec<f16> = q_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let d_q_f32 = GpuBuffer::alloc(total * 4).unwrap();
    let d_k_f32 = GpuBuffer::alloc(total * 4).unwrap();
    let d_v_f32 = GpuBuffer::alloc(total * 4).unwrap();
    let d_o_f32 = GpuBuffer::alloc(total * 4).unwrap();

    let d_q_f16 = GpuBuffer::alloc(total * 2).unwrap();
    let d_k_f16 = GpuBuffer::alloc(total * 2).unwrap();
    let d_v_f16 = GpuBuffer::alloc(total * 2).unwrap();
    let d_o_wmma = GpuBuffer::alloc(total * 4).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q_f32.as_ptr(), q_f32.as_ptr() as *const u8, total * 4).unwrap();
        hip_memcpy_h2d(d_k_f32.as_ptr(), k_f32.as_ptr() as *const u8, total * 4).unwrap();
        hip_memcpy_h2d(d_v_f32.as_ptr(), v_f32.as_ptr() as *const u8, total * 4).unwrap();
        hip_memcpy_h2d(d_q_f16.as_ptr(), q_f16.as_ptr() as *const u8, total * 2).unwrap();
        hip_memcpy_h2d(d_k_f16.as_ptr(), k_f16.as_ptr() as *const u8, total * 2).unwrap();
        hip_memcpy_h2d(d_v_f16.as_ptr(), v_f16.as_ptr() as *const u8, total * 2).unwrap();
    }

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let existing_samples = time_kernel(warmup, iters, || {
        for h in 0..NUM_HEADS {
            flash_attn_prefill_strided(
                d_o_f32.as_ptr() as *mut f32,
                d_q_f32.as_ptr() as *const f32,
                d_k_f32.as_ptr() as *const f32,
                d_v_f32.as_ptr() as *const f32,
                seq_len, HEAD_DIM,
                stride, stride, stride,
                h * HEAD_DIM, h * HEAD_DIM, h * HEAD_DIM,
                scale,
            ).unwrap();
        }
    });

    let wmma_samples = time_kernel(warmup, iters, || {
        launch_wmma_attention_prefill_online(
            d_q_f16.as_ptr() as *const u16,
            d_k_f16.as_ptr() as *const u16,
            d_v_f16.as_ptr() as *const u16,
            d_o_wmma.as_ptr() as *mut f32,
            seq_len, NUM_HEADS, stride, scale,
            hipStream_t::null(),
        ).unwrap();
    });

    (median(&existing_samples), median(&wmma_samples))
}

fn main() {
    let seqs = [64, 128, 256, 512];
    let mut results = Vec::new();
    eprintln!("existing kernel vs WMMA online-softmax ({} heads, head_dim={}):", NUM_HEADS, HEAD_DIM);
    for &seq in &seqs {
        let (existing, wmma) = bench_seq(seq);
        eprintln!(
            "  seq={:4}:  existing {:9.1} µs   wmma {:8.1} µs   speedup {:6.2}×",
            seq, existing, wmma, existing / wmma
        );
        results.push((seq, existing, wmma));
    }

    // pp=256, 28-layer prefill extrapolation.
    let (_, _, wmma_256) = results.iter().find(|(s, _, _)| *s == 256).cloned().unwrap();
    let est_attention_ms = wmma_256 * 28.0 / 1000.0;
    eprintln!();
    eprintln!(
        "extrapolated WMMA attention for pp=256 × 28 layers: {:.1} ms  (baseline ≈ 2,330 ms)",
        est_attention_ms
    );

    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let entries: Vec<_> = results.iter().map(|(seq, existing, wmma)| json!({
        "seq_len": seq,
        "num_heads": NUM_HEADS,
        "existing_us_median": existing,
        "wmma_online_us_median": wmma,
        "speedup_median": existing / wmma,
    })).collect();
    let j = json!({
        "ts": ts,
        "hardware": "RX 9070 XT (gfx1201, RDNA 4)",
        "shapes": entries,
        "pp256_28layer_attention_estimate_ms": est_attention_ms,
        "pp256_28layer_attention_reference_ms": 2330.0,
    });
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
}
