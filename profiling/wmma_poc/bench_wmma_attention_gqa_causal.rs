//! Phase 3c throughput — GQA (28/4) + causal WMMA attention vs Phase 3b
//! (no GQA, no mask) vs the existing scalar kernel.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_event_create, hip_event_destroy, hip_event_elapsed_time, hip_event_record,
    hip_event_synchronize, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::attention::flash_attn_prefill_strided;
use rocmforge::gpu::kernels::wmma::{
    launch_wmma_attention_prefill_gqa_causal, launch_wmma_attention_prefill_online,
};
use rocmforge::gpu::weights::GpuBuffer;
use serde_json::json;
use std::time::SystemTime;

const HEAD_DIM: usize = 128;
const NUM_Q_HEADS: usize = 28;
const NUM_KV_HEADS: usize = 4;

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

fn median(s: &[f64]) -> f64 {
    let mut v: Vec<f64> = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(seq_len: usize) -> (f64, f64, f64) {
    let warmup = 10;
    let iters = 100;

    // FP32 inputs (for existing kernel), with GQA-sized K/V
    let q_stride  = NUM_Q_HEADS  * HEAD_DIM;
    let kv_stride = NUM_KV_HEADS * HEAD_DIM;
    let q_n  = seq_len * q_stride;
    let kv_n = seq_len * kv_stride;

    // The existing strided attention kernel we benchmark against does not
    // know about GQA; it assumes q_stride == kv_stride. To keep the
    // wall-clock comparison apples-to-apples with Phase 3b (which also
    // used full KV), we time the existing kernel at the non-GQA layout
    // (28 KV heads) but time Phase 3c with actual GQA (4 KV heads).

    // Non-GQA path — for existing kernel + Phase 3b comparison.
    let q_full_f32 = seeded_f32(q_n, 1);
    let k_full_f32 = seeded_f32(q_n, 2);
    let v_full_f32 = seeded_f32(q_n, 3);
    let q_full_f16: Vec<f16> = q_full_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let k_full_f16: Vec<f16> = k_full_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let v_full_f16: Vec<f16> = v_full_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // GQA path — Phase 3c.
    let q_gqa_f16 = q_full_f16.clone();
    let k_gqa_f16: Vec<f16> = seeded_f32(kv_n, 12).iter().map(|&x| f16::from_f32(x)).collect();
    let v_gqa_f16: Vec<f16> = seeded_f32(kv_n, 13).iter().map(|&x| f16::from_f32(x)).collect();

    // Allocate device buffers.
    let d_q_full_f32 = GpuBuffer::alloc(q_n * 4).unwrap();
    let d_k_full_f32 = GpuBuffer::alloc(q_n * 4).unwrap();
    let d_v_full_f32 = GpuBuffer::alloc(q_n * 4).unwrap();
    let d_o_f32      = GpuBuffer::alloc(q_n * 4).unwrap();

    let d_q_full_f16 = GpuBuffer::alloc(q_n * 2).unwrap();
    let d_k_full_f16 = GpuBuffer::alloc(q_n * 2).unwrap();
    let d_v_full_f16 = GpuBuffer::alloc(q_n * 2).unwrap();
    let d_o_p3b      = GpuBuffer::alloc(q_n * 4).unwrap();

    let d_q_gqa_f16  = GpuBuffer::alloc(q_n * 2).unwrap();
    let d_k_gqa_f16  = GpuBuffer::alloc(kv_n * 2).unwrap();
    let d_v_gqa_f16  = GpuBuffer::alloc(kv_n * 2).unwrap();
    let d_o_p3c      = GpuBuffer::alloc(q_n * 4).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q_full_f32.as_ptr(), q_full_f32.as_ptr() as *const u8, q_n * 4).unwrap();
        hip_memcpy_h2d(d_k_full_f32.as_ptr(), k_full_f32.as_ptr() as *const u8, q_n * 4).unwrap();
        hip_memcpy_h2d(d_v_full_f32.as_ptr(), v_full_f32.as_ptr() as *const u8, q_n * 4).unwrap();

        hip_memcpy_h2d(d_q_full_f16.as_ptr(), q_full_f16.as_ptr() as *const u8, q_n * 2).unwrap();
        hip_memcpy_h2d(d_k_full_f16.as_ptr(), k_full_f16.as_ptr() as *const u8, q_n * 2).unwrap();
        hip_memcpy_h2d(d_v_full_f16.as_ptr(), v_full_f16.as_ptr() as *const u8, q_n * 2).unwrap();

        hip_memcpy_h2d(d_q_gqa_f16.as_ptr(),  q_gqa_f16.as_ptr()  as *const u8, q_n * 2).unwrap();
        hip_memcpy_h2d(d_k_gqa_f16.as_ptr(),  k_gqa_f16.as_ptr()  as *const u8, kv_n * 2).unwrap();
        hip_memcpy_h2d(d_v_gqa_f16.as_ptr(),  v_gqa_f16.as_ptr()  as *const u8, kv_n * 2).unwrap();
    }

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Existing kernel (no GQA, implicit causal mask is built in).
    let existing_samples = time_kernel(warmup, iters, || {
        for h in 0..NUM_Q_HEADS {
            flash_attn_prefill_strided(
                d_o_f32.as_ptr() as *mut f32,
                d_q_full_f32.as_ptr() as *const f32,
                d_k_full_f32.as_ptr() as *const f32,
                d_v_full_f32.as_ptr() as *const f32,
                seq_len, HEAD_DIM,
                q_stride, q_stride, q_stride,
                h * HEAD_DIM, h * HEAD_DIM, h * HEAD_DIM,
                scale,
            ).unwrap();
        }
    });

    // Phase 3b (no GQA, no mask).
    let p3b_samples = time_kernel(warmup, iters, || {
        launch_wmma_attention_prefill_online(
            d_q_full_f16.as_ptr() as *const u16,
            d_k_full_f16.as_ptr() as *const u16,
            d_v_full_f16.as_ptr() as *const u16,
            d_o_p3b.as_ptr() as *mut f32,
            seq_len, NUM_Q_HEADS, q_stride, scale,
            hipStream_t::null(),
        ).unwrap();
    });

    // Phase 3c (GQA 28/4 + causal).
    let p3c_samples = time_kernel(warmup, iters, || {
        launch_wmma_attention_prefill_gqa_causal(
            d_q_gqa_f16.as_ptr() as *const u16,
            d_k_gqa_f16.as_ptr() as *const u16,
            d_v_gqa_f16.as_ptr() as *const u16,
            d_o_p3c.as_ptr() as *mut f32,
            seq_len, NUM_Q_HEADS, NUM_KV_HEADS,
            true, scale,
            hipStream_t::null(),
        ).unwrap();
    });

    (median(&existing_samples), median(&p3b_samples), median(&p3c_samples))
}

fn main() {
    let seqs = [64, 128, 256, 512];
    let mut rows = Vec::new();
    eprintln!("existing vs WMMA Phase 3b (no GQA/mask) vs WMMA Phase 3c (GQA 28/4 + causal):");
    for &seq in &seqs {
        let (ex, p3b, p3c) = bench(seq);
        eprintln!(
            "  seq={:4}:  existing {:9.1} µs   p3b {:8.1} µs   p3c {:8.1} µs   3c vs existing {:5.2}×   3c vs 3b {:5.2}×",
            seq, ex, p3b, p3c, ex / p3c, p3b / p3c
        );
        rows.push((seq, ex, p3b, p3c));
    }

    let (_, _, _, p3c_256) = rows.iter().find(|(s, _, _, _)| *s == 256).cloned().unwrap();
    let est_attention_ms  = p3c_256 * 28.0 / 1000.0;
    let est_prefill_ms    = 110.0 + est_attention_ms + 330.0;
    let est_tokps         = 256.0 / (est_prefill_ms / 1000.0);
    eprintln!();
    eprintln!(
        "extrapolated pp=256 × 28 layers with Phase 3c attention: {:.1} ms",
        est_attention_ms
    );
    eprintln!(
        "estimated full-prefill pp=256: GEMM 110 + attn {:.1} + overhead 330 ≈ {:.1} ms  →  {:.0} tok/s",
        est_attention_ms, est_prefill_ms, est_tokps
    );

    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let entries: Vec<_> = rows.iter().map(|(s, ex, p3b, p3c)| json!({
        "seq_len": s,
        "existing_us_median": ex,
        "phase3b_us_median": p3b,
        "phase3c_us_median": p3c,
        "speedup_vs_existing": ex / p3c,
        "speedup_vs_phase3b":  p3b / p3c,
    })).collect();
    let j = json!({
        "ts": ts,
        "hardware": "RX 9070 XT (gfx1201, RDNA 4)",
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "shapes": entries,
        "pp256_28layer_attention_estimate_ms": est_attention_ms,
        "estimated_pp256_prefill_ms": est_prefill_ms,
        "estimated_pp256_tokens_per_s": est_tokps,
    });
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
}
