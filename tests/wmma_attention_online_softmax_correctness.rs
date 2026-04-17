#![cfg(feature = "gpu")]

//! Phase 3b — online-softmax WMMA attention correctness.
//!
//! Four diagnostics:
//!   1. seq=64, online vs Phase 3a global softmax — a single KV tile,
//!      no rescaling happens, so the two paths must match bit-identical.
//!   2. seq=128, online vs Phase 3a — two KV tiles, rescaling happens
//!      once; nonzero drift is allowed but expected ≪ tolerance.
//!   3. seq=256, online vs CPU FP32 reference — four tiles.
//!   4. seq=512, online vs CPU FP32 reference — eight tiles.
//!
//! The Phase 3a kernel is limited to seq ∈ {64, 128}, so for the two
//! larger sizes we fall back to a CPU reference only.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::{
    launch_wmma_attention_prefill_multihead,
    launch_wmma_attention_prefill_online,
};
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const HEAD_DIM: usize = 128;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn seeded_halfs(n: usize, seed: u64) -> Vec<f16> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (s >> 33) as u32;
            f16::from_f32((r & 0x3F) as f32 / 256.0)
        })
        .collect()
}

fn cpu_attention_multihead(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let stride = num_heads * HEAD_DIM;
    let mut o = vec![0.0f32; seq_len * stride];

    for h in 0..num_heads {
        let off = h * HEAD_DIM;
        let mut s = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut acc = 0.0f32;
                for d in 0..HEAD_DIM {
                    acc += q[i * stride + off + d].to_f32()
                         * k[j * stride + off + d].to_f32();
                }
                s[i * seq_len + j] = acc * scale;
            }
        }
        for i in 0..seq_len {
            let row = &mut s[i * seq_len..(i + 1) * seq_len];
            let mut m = row[0];
            for &x in row.iter().skip(1) {
                if x > m {
                    m = x;
                }
            }
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            for x in row.iter_mut() {
                *x /= sum;
            }
        }
        for i in 0..seq_len {
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += s[i * seq_len + j] * v[j * stride + off + d].to_f32();
                }
                o[i * stride + off + d] = acc;
            }
        }
    }
    o
}

fn run_online(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let stride = num_heads * HEAD_DIM;
    let elems = seq_len * stride;

    let d_q = GpuBuffer::alloc(elems * 2).unwrap();
    let d_k = GpuBuffer::alloc(elems * 2).unwrap();
    let d_v = GpuBuffer::alloc(elems * 2).unwrap();
    let d_o = GpuBuffer::alloc(elems * 4).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q.as_ptr(), q.as_ptr() as *const u8, elems * 2).unwrap();
        hip_memcpy_h2d(d_k.as_ptr(), k.as_ptr() as *const u8, elems * 2).unwrap();
        hip_memcpy_h2d(d_v.as_ptr(), v.as_ptr() as *const u8, elems * 2).unwrap();
    }

    launch_wmma_attention_prefill_online(
        d_q.as_ptr() as *const u16,
        d_k.as_ptr() as *const u16,
        d_v.as_ptr() as *const u16,
        d_o.as_ptr() as *mut f32,
        seq_len, num_heads, stride, scale,
        hipStream_t::null(),
    )
    .unwrap();
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; elems];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_o.as_ptr(), elems * 4).unwrap();
    }
    out
}

fn run_phase3a(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let stride = num_heads * HEAD_DIM;
    let elems = seq_len * stride;

    let d_q = GpuBuffer::alloc(elems * 2).unwrap();
    let d_k = GpuBuffer::alloc(elems * 2).unwrap();
    let d_v = GpuBuffer::alloc(elems * 2).unwrap();
    let d_o = GpuBuffer::alloc(elems * 4).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q.as_ptr(), q.as_ptr() as *const u8, elems * 2).unwrap();
        hip_memcpy_h2d(d_k.as_ptr(), k.as_ptr() as *const u8, elems * 2).unwrap();
        hip_memcpy_h2d(d_v.as_ptr(), v.as_ptr() as *const u8, elems * 2).unwrap();
    }

    launch_wmma_attention_prefill_multihead(
        d_q.as_ptr() as *const u16,
        d_k.as_ptr() as *const u16,
        d_v.as_ptr() as *const u16,
        d_o.as_ptr() as *mut f32,
        seq_len, num_heads, stride, scale,
        hipStream_t::null(),
    )
    .unwrap();
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; elems];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_o.as_ptr(), elems * 4).unwrap();
    }
    out
}

fn compare(label: &str, gpu: &[f32], ref_: &[f32], tol: f32) {
    let mut max_diff = 0.0f32;
    let mut sum = 0.0f64;
    let mut worst = 0usize;
    for i in 0..gpu.len() {
        let d = (gpu[i] - ref_[i]).abs();
        sum += d as f64;
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    let mean = sum / gpu.len() as f64;
    eprintln!(
        "{label}: max_abs_diff = {:.3e}  mean = {:.3e}  (tol {:.1e})",
        max_diff, mean, tol
    );
    assert!(
        max_diff <= tol,
        "{label}: max diff {:.3e} > tol {:.1e} at idx {} (online={} ref={})",
        max_diff,
        tol,
        worst,
        gpu[worst],
        ref_[worst]
    );
}

fn inputs(seq: usize, heads: usize) -> (Vec<f16>, Vec<f16>, Vec<f16>, f32) {
    let stride = heads * HEAD_DIM;
    let n = seq * stride;
    (
        seeded_halfs(n, 0xA1 ^ (seq * 7) as u64),
        seeded_halfs(n, 0xA2 ^ (seq * 11) as u64),
        seeded_halfs(n, 0xA3 ^ (seq * 13) as u64),
        1.0 / (HEAD_DIM as f32).sqrt(),
    )
}

#[test]
#[serial]
fn online_seq64_matches_phase3a() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(64, 28);
    let online = run_online(&q, &k, &v, 64, 28, scale);
    let ref_  = run_phase3a(&q, &k, &v, 64, 28, scale);
    // Even with a single KV tile and no rescaling, the two paths don't
    // produce bit-identical output: Phase 3a stores NORMALISED P in
    // FP16 (values ≤ ~1/64), Phase 3b stores UN-normalised P in FP16
    // (values up to ~1) and divides by the row sum at the end. The
    // FP16 quantisation grid lands at different absolute values, so
    // the downstream P·V GEMM rounds slightly differently. Tolerance
    // 1e-4 comfortably captures that while still catching any real
    // logic bug (wrong tile count, wrong output stride, missing init).
    compare("seq=64 online vs Phase3a", &online, &ref_, 1.0e-4);
}

#[test]
#[serial]
fn online_seq128_matches_phase3a() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(128, 28);
    let online = run_online(&q, &k, &v, 128, 28, scale);
    let ref_  = run_phase3a(&q, &k, &v, 128, 28, scale);
    // Two KV tiles — rescaling happens once, slight drift expected.
    compare("seq=128 online vs Phase3a", &online, &ref_, 1.0e-4);
}

#[test]
#[serial]
fn online_seq256_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(256, 28);
    let online = run_online(&q, &k, &v, 256, 28, scale);
    let cpu = cpu_attention_multihead(&q, &k, &v, 256, 28, scale);
    compare("seq=256 online vs CPU", &online, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn online_seq512_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(512, 28);
    let online = run_online(&q, &k, &v, 512, 28, scale);
    let cpu = cpu_attention_multihead(&q, &k, &v, 512, 28, scale);
    compare("seq=512 online vs CPU", &online, &cpu, 1.0e-2);
}
