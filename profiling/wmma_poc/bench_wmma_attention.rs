//! Phase 3a Step 2 — throughput comparison.
//!
//! `flash_attn_prefill_strided_kernel` (the existing scalar kernel in
//! `hip_kernels/attention.hip`) vs the new WMMA multi-head kernel, on
//! two sequence lengths. The existing kernel launches once per head;
//! the WMMA kernel launches once for all heads. We report wall-clock
//! per full forward (all 28 heads) in both cases so the numbers are
//! directly comparable.
//!
//! FP32 inputs are shared between the two paths (the FP16 kernel sees
//! the same values after a __float2half conversion, but that
//! conversion runs outside the timed window).

use half::f16;
use rocmforge::gpu::ffi::{
    hip_event_create, hip_event_destroy, hip_event_elapsed_time, hip_event_record,
    hip_event_synchronize, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::attention::flash_attn_prefill_strided;
use rocmforge::gpu::kernels::wmma::launch_wmma_attention_prefill_multihead;
use rocmforge::gpu::weights::GpuBuffer;
use serde_json::json;
use std::time::SystemTime;

const HEAD_DIM: usize = 128;
const NUM_HEADS: usize = 28;

fn seeded_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (s >> 33) as u32;
            ((r & 0x3F) as f32) / 256.0
        })
        .collect()
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
    seq_len: usize,
    existing: (f64, f64, f64),
    wmma: (f64, f64, f64),
}

fn bench(seq_len: usize) -> Row {
    let warmup = 10;
    let iters = 100;

    let stride = NUM_HEADS * HEAD_DIM;
    let total_elems = seq_len * stride;

    // FP32 inputs for the existing kernel
    let q_f32 = seeded_f32(total_elems, 1);
    let k_f32 = seeded_f32(total_elems, 2);
    let v_f32 = seeded_f32(total_elems, 3);

    // FP16 copies for the WMMA kernel (same bits after conversion)
    let q_f16: Vec<f16> = q_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let k_f16: Vec<f16> = k_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let v_f16: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let d_q_f32 = GpuBuffer::alloc(total_elems * 4).unwrap();
    let d_k_f32 = GpuBuffer::alloc(total_elems * 4).unwrap();
    let d_v_f32 = GpuBuffer::alloc(total_elems * 4).unwrap();
    let d_o_f32 = GpuBuffer::alloc(total_elems * 4).unwrap();

    let d_q_f16 = GpuBuffer::alloc(total_elems * 2).unwrap();
    let d_k_f16 = GpuBuffer::alloc(total_elems * 2).unwrap();
    let d_v_f16 = GpuBuffer::alloc(total_elems * 2).unwrap();
    let d_o_wmma = GpuBuffer::alloc(total_elems * 4).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q_f32.as_ptr(), q_f32.as_ptr() as *const u8, total_elems * 4).unwrap();
        hip_memcpy_h2d(d_k_f32.as_ptr(), k_f32.as_ptr() as *const u8, total_elems * 4).unwrap();
        hip_memcpy_h2d(d_v_f32.as_ptr(), v_f32.as_ptr() as *const u8, total_elems * 4).unwrap();
        hip_memcpy_h2d(d_q_f16.as_ptr(), q_f16.as_ptr() as *const u8, total_elems * 2).unwrap();
        hip_memcpy_h2d(d_k_f16.as_ptr(), k_f16.as_ptr() as *const u8, total_elems * 2).unwrap();
        hip_memcpy_h2d(d_v_f16.as_ptr(), v_f16.as_ptr() as *const u8, total_elems * 2).unwrap();
    }

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Existing kernel: one launch per head, total time = sum.
    let existing_samples = time_kernel(warmup, iters, || {
        for h in 0..NUM_HEADS {
            flash_attn_prefill_strided(
                d_o_f32.as_ptr() as *mut f32,
                d_q_f32.as_ptr() as *const f32,
                d_k_f32.as_ptr() as *const f32,
                d_v_f32.as_ptr() as *const f32,
                seq_len,
                HEAD_DIM,
                stride, // out_stride
                stride, // q_stride
                stride, // kv_stride
                h * HEAD_DIM,
                h * HEAD_DIM,
                h * HEAD_DIM,
                scale,
            )
            .unwrap();
        }
    });

    // WMMA kernel: single launch covers all heads.
    let wmma_samples = time_kernel(warmup, iters, || {
        launch_wmma_attention_prefill_multihead(
            d_q_f16.as_ptr() as *const u16,
            d_k_f16.as_ptr() as *const u16,
            d_v_f16.as_ptr() as *const u16,
            d_o_wmma.as_ptr() as *mut f32,
            seq_len,
            NUM_HEADS,
            stride,
            scale,
            hipStream_t::null(),
        )
        .unwrap();
    });

    Row {
        seq_len,
        existing: stats(&existing_samples),
        wmma: stats(&wmma_samples),
    }
}

fn main() {
    let mut rows = Vec::new();
    for seq in [64, 128] {
        eprintln!("benchmarking seq_len = {} ({} heads, head_dim={})", seq, NUM_HEADS, HEAD_DIM);
        let r = bench(seq);
        let (_, em, _) = r.existing;
        let (_, wm, _) = r.wmma;
        eprintln!(
            "  existing median = {:8.1} µs   wmma median = {:7.1} µs   speedup = {:5.2}×",
            em,
            wm,
            em / wm
        );
        rows.push(r);
    }

    // Extrapolate attention cost for pp256 (the prefill workload we care about).
    // Without online softmax our current kernel cannot be called directly at
    // seq=256 — so the value below is an EXTRAPOLATION assuming roughly
    // linear scaling with the number of query tiles.  For each query tile
    // (64 queries), the current kernel's time scales ~linearly with K
    // (number of keys), so for pp256:
    //
    //   time_per_query_tile(K=256) ≈ time(K=128) · (256/128)
    //   total = time_per_query_tile(K=256) · (256/64)
    //
    // This is a very rough upper bound — online softmax in Phase 3b
    // changes the constants.
    let wmma_seq128 = rows[1].wmma.1; // median µs for seq=128 (one full forward of 28 heads)
    let extrapolated_pp256_single_layer_us = wmma_seq128 * 2.0 * 2.0; // ×2 for K scaling, ×2 for query tiles
    let extrapolated_pp256_full_prefill_ms = extrapolated_pp256_single_layer_us * 28.0 / 1000.0;

    eprintln!();
    eprintln!(
        "extrapolated WMMA attention time for pp=256 full prefill (28 layers): {:.1} ms",
        extrapolated_pp256_full_prefill_ms
    );
    eprintln!("(reference: existing kernel measured at ~2330 ms for the same workload)");

    let ts = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
    let entries: Vec<_> = rows
        .iter()
        .map(|r| {
            json!({
                "seq_len": r.seq_len,
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
                "existing_us":  { "mean": r.existing.0, "median": r.existing.1, "p95": r.existing.2 },
                "wmma_us":      { "mean": r.wmma.0,     "median": r.wmma.1,     "p95": r.wmma.2 },
                "speedup_median": r.existing.1 / r.wmma.1,
            })
        })
        .collect();

    let j = json!({
        "ts": ts,
        "hardware": "RX 9070 XT (gfx1201, RDNA 4)",
        "shapes": entries,
        "extrapolated_pp256_prefill_attention_ms": extrapolated_pp256_full_prefill_ms,
        "reference_existing_pp256_prefill_attention_ms": 2330.0,
    });
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
}
