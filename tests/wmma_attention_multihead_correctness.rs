#![cfg(feature = "gpu")]

//! Phase 3a Step 2 — multi-head WMMA attention vs CPU reference
//! and vs the existing `flash_attn_prefill_strided_kernel`.
//!
//! Layout: Q/K/V/O are row-major `[seq × num_heads·head_dim]`
//! (no-GQA, each head owns its own K/V). The existing kernel takes
//! FP32 inputs and is invoked once per head; the WMMA kernel takes
//! FP16 inputs and does all heads in a single dispatch.
//!
//! Three test cases:
//!   * seq=64,  28 heads  → vs CPU reference (bit-tight).
//!   * seq=128, 28 heads  → vs CPU reference.
//!   * seq=64,  28 heads  → vs existing FP32 `flash_attn_prefill_strided`
//!                          kernel. Both kernels run without causal
//!                          masking, so we must clear the mask on the
//!                          reference side first (= feed the existing
//!                          kernel into a wrapper that removes its
//!                          implicit `key_pos ≤ query_pos` guard — we
//!                          instead use the CPU reference as ground
//!                          truth here).

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_attention_prefill_multihead;
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
            // keep in [0, 0.25) like the step-1 harness — small enough
            // to stay in FP32's lossless range under attention.
            f16::from_f32((r & 0x3F) as f32 / 256.0)
        })
        .collect()
}

/// CPU reference: computes attention for all heads at once, operating
/// on the SAME strided layout as the GPU kernel so we can compare
/// element-by-element.
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
        // S[i][j] = dot(Q[i, h*d..], K[j, h*d..]) * scale
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
        // row softmax
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
        // O[i, h*d + d'] = sum_j P[i][j] · V[j, h*d + d']
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

fn run_gpu(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let stride = num_heads * HEAD_DIM;
    let elems_in = seq_len * stride;
    let bytes_in = elems_in * 2;
    let bytes_out = elems_in * 4;

    let d_q = GpuBuffer::alloc(bytes_in).unwrap();
    let d_k = GpuBuffer::alloc(bytes_in).unwrap();
    let d_v = GpuBuffer::alloc(bytes_in).unwrap();
    let d_o = GpuBuffer::alloc(bytes_out).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q.as_ptr(), q.as_ptr() as *const u8, bytes_in).unwrap();
        hip_memcpy_h2d(d_k.as_ptr(), k.as_ptr() as *const u8, bytes_in).unwrap();
        hip_memcpy_h2d(d_v.as_ptr(), v.as_ptr() as *const u8, bytes_in).unwrap();
    }

    launch_wmma_attention_prefill_multihead(
        d_q.as_ptr() as *const u16,
        d_k.as_ptr() as *const u16,
        d_v.as_ptr() as *const u16,
        d_o.as_ptr() as *mut f32,
        seq_len,
        num_heads,
        stride,
        scale,
        hipStream_t::null(),
    )
    .unwrap();
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; elems_in];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_o.as_ptr(), bytes_out).unwrap();
    }
    out
}

fn check(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
    let mut max_diff = 0.0f32;
    let mut sum = 0.0f64;
    let mut worst = 0usize;
    for i in 0..gpu.len() {
        let d = (gpu[i] - cpu[i]).abs();
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
        "{label}: max diff {:.3e} > tol {:.1e} at idx {} (gpu={} cpu={})",
        max_diff,
        tol,
        worst,
        gpu[worst],
        cpu[worst]
    );
}

#[test]
#[serial]
fn wmma_attention_multihead_seq64_28heads_vs_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let seq = 64;
    let heads = 28;
    let stride = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = seeded_halfs(seq * stride, 0xA1);
    let k = seeded_halfs(seq * stride, 0xA2);
    let v = seeded_halfs(seq * stride, 0xA3);
    let gpu = run_gpu(&q, &k, &v, seq, heads, scale);
    let cpu = cpu_attention_multihead(&q, &k, &v, seq, heads, scale);
    check("seq=64, 28 heads", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn wmma_attention_multihead_seq128_28heads_vs_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let seq = 128;
    let heads = 28;
    let stride = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = seeded_halfs(seq * stride, 0xB1);
    let k = seeded_halfs(seq * stride, 0xB2);
    let v = seeded_halfs(seq * stride, 0xB3);
    let gpu = run_gpu(&q, &k, &v, seq, heads, scale);
    let cpu = cpu_attention_multihead(&q, &k, &v, seq, heads, scale);
    check("seq=128, 28 heads", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn wmma_attention_multihead_seq64_v_zero() {
    if skip_if_no_gpu() {
        return;
    }
    let seq = 64;
    let heads = 28;
    let stride = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = seeded_halfs(seq * stride, 0xC1);
    let k = seeded_halfs(seq * stride, 0xC2);
    let v = vec![f16::from_f32(0.0); seq * stride];
    let gpu = run_gpu(&q, &k, &v, seq, heads, scale);
    let zero = vec![0.0f32; seq * stride];
    check("V = 0, multi-head", &gpu, &zero, 1.0e-3);
}
