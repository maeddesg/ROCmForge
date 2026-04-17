#![cfg(feature = "gpu")]

//! Phase 3c correctness — GQA + causal mask, isolated and combined.
//!
//! Test matrix (all on the Qwen2.5-7B attention shape: 28 Q heads,
//! 4 KV heads, head_dim=128):
//!
//!   * GQA only (causal=false):
//!       - seq=64   vs CPU GQA reference
//!       - seq=128  vs CPU GQA reference
//!
//!   * Causal only (num_kv_heads = num_q_heads, no GQA):
//!       - seq=64   vs CPU causal reference
//!       - seq=128  vs CPU causal reference
//!       - seq=256  vs CPU causal reference
//!       - first-token sanity: O[0] ≈ V[0] (position 0 attends to
//!         itself only)
//!
//!   * Both (the Qwen real-world combination):
//!       - seq=64   vs CPU GQA + causal reference
//!       - seq=128  vs CPU GQA + causal reference
//!       - seq=256  vs CPU GQA + causal reference

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_attention_prefill_gqa_causal;
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

/// CPU reference for GQA + optional causal prefill attention.
fn cpu_attention_ref(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let gqa = num_q_heads / num_kv_heads;
    let q_stride = num_q_heads * HEAD_DIM;
    let kv_stride = num_kv_heads * HEAD_DIM;

    let mut o = vec![0.0f32; seq_len * q_stride];

    for h in 0..num_q_heads {
        let kv_h = h / gqa;
        let q_off = h * HEAD_DIM;
        let k_off = kv_h * HEAD_DIM;

        let mut s = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    s[i * seq_len + j] = f32::NEG_INFINITY;
                    continue;
                }
                let mut acc = 0.0f32;
                for d in 0..HEAD_DIM {
                    acc += q[i * q_stride + q_off + d].to_f32()
                         * k[j * kv_stride + k_off + d].to_f32();
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
            if !m.is_finite() {
                for x in row.iter_mut() {
                    *x = 0.0;
                }
                continue;
            }
            let mut sum = 0.0f32;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            if sum > 0.0 {
                for x in row.iter_mut() {
                    *x /= sum;
                }
            }
        }
        for i in 0..seq_len {
            for d in 0..HEAD_DIM {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += s[i * seq_len + j] * v[j * kv_stride + k_off + d].to_f32();
                }
                o[i * q_stride + q_off + d] = acc;
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
    num_q_heads: usize,
    num_kv_heads: usize,
    causal: bool,
    scale: f32,
) -> Vec<f32> {
    let q_bytes  = seq_len * num_q_heads  * HEAD_DIM * 2;
    let kv_bytes = seq_len * num_kv_heads * HEAD_DIM * 2;
    let o_bytes  = seq_len * num_q_heads  * HEAD_DIM * 4;

    let d_q = GpuBuffer::alloc(q_bytes).unwrap();
    let d_k = GpuBuffer::alloc(kv_bytes).unwrap();
    let d_v = GpuBuffer::alloc(kv_bytes).unwrap();
    let d_o = GpuBuffer::alloc(o_bytes).unwrap();

    unsafe {
        hip_memcpy_h2d(d_q.as_ptr(), q.as_ptr() as *const u8, q_bytes).unwrap();
        hip_memcpy_h2d(d_k.as_ptr(), k.as_ptr() as *const u8, kv_bytes).unwrap();
        hip_memcpy_h2d(d_v.as_ptr(), v.as_ptr() as *const u8, kv_bytes).unwrap();
    }

    launch_wmma_attention_prefill_gqa_causal(
        d_q.as_ptr() as *const u16,
        d_k.as_ptr() as *const u16,
        d_v.as_ptr() as *const u16,
        d_o.as_ptr() as *mut f32,
        seq_len,
        num_q_heads,
        num_kv_heads,
        causal,
        scale,
        hipStream_t::null(),
    )
    .unwrap();
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let elems = seq_len * num_q_heads * HEAD_DIM;
    let mut out = vec![0.0f32; elems];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_o.as_ptr(), o_bytes).unwrap();
    }
    out
}

fn compare(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
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
        "{label}: max {:.3e} > tol {:.1e} at idx {} (gpu={} cpu={})",
        max_diff, tol, worst, gpu[worst], cpu[worst]
    );
}

fn inputs(seq: usize, n_q: usize, n_kv: usize) -> (Vec<f16>, Vec<f16>, Vec<f16>, f32) {
    (
        seeded_halfs(seq * n_q * HEAD_DIM,  0xE1 ^ (seq * 11 + n_q) as u64),
        seeded_halfs(seq * n_kv * HEAD_DIM, 0xE2 ^ (seq * 13 + n_kv) as u64),
        seeded_halfs(seq * n_kv * HEAD_DIM, 0xE3 ^ (seq * 17 + n_kv) as u64),
        1.0 / (HEAD_DIM as f32).sqrt(),
    )
}

// ===== GQA only, no mask =====
#[test]
#[serial]
fn gqa_only_seq64() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(64, 28, 4);
    let gpu = run_gpu(&q, &k, &v, 64, 28, 4, false, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 64, 28, 4, false, scale);
    compare("GQA 28/4, seq=64, no causal", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn gqa_only_seq128() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(128, 28, 4);
    let gpu = run_gpu(&q, &k, &v, 128, 28, 4, false, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 128, 28, 4, false, scale);
    compare("GQA 28/4, seq=128, no causal", &gpu, &cpu, 1.0e-2);
}

// ===== Causal only (num_kv_heads = num_q_heads → gqa_ratio = 1) =====
#[test]
#[serial]
fn causal_only_seq64() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(64, 28, 28);
    let gpu = run_gpu(&q, &k, &v, 64, 28, 28, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 64, 28, 28, true, scale);
    compare("causal, seq=64", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn causal_only_seq128() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(128, 28, 28);
    let gpu = run_gpu(&q, &k, &v, 128, 28, 28, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 128, 28, 28, true, scale);
    compare("causal, seq=128", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn causal_only_seq256() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(256, 28, 28);
    let gpu = run_gpu(&q, &k, &v, 256, 28, 28, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 256, 28, 28, true, scale);
    compare("causal, seq=256", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn causal_first_token_equals_v_of_first_token() {
    // At position 0, only key 0 is visible → softmax trivially yields
    // weight 1.0 on V[0]. So O[0, h, :] must equal V[0, h, :] for each
    // head.
    if skip_if_no_gpu() {
        return;
    }
    let seq = 64;
    let n_q = 28;
    let (q, k, v, scale) = inputs(seq, n_q, n_q);
    let gpu = run_gpu(&q, &k, &v, seq, n_q, n_q, true, scale);

    let stride = n_q * HEAD_DIM;
    let mut max_diff = 0.0f32;
    for h in 0..n_q {
        let off = h * HEAD_DIM;
        for d in 0..HEAD_DIM {
            let got = gpu[0 * stride + off + d];
            let want = v[0 * stride + off + d].to_f32();
            let diff = (got - want).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    eprintln!("O[0] vs V[0]: max_abs_diff = {:.3e}", max_diff);
    // FP16 inputs → FP32 accumulation round trip: ~1e-4 is ample.
    assert!(
        max_diff <= 1.0e-3,
        "first-token self-attention drifted too much: {:.3e}",
        max_diff
    );
}

// ===== GQA + causal (the real Qwen2.5-7B shape) =====
#[test]
#[serial]
fn gqa_causal_seq64() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(64, 28, 4);
    let gpu = run_gpu(&q, &k, &v, 64, 28, 4, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 64, 28, 4, true, scale);
    compare("GQA 28/4 + causal, seq=64", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn gqa_causal_seq128() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(128, 28, 4);
    let gpu = run_gpu(&q, &k, &v, 128, 28, 4, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 128, 28, 4, true, scale);
    compare("GQA 28/4 + causal, seq=128", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn gqa_causal_seq256() {
    if skip_if_no_gpu() {
        return;
    }
    let (q, k, v, scale) = inputs(256, 28, 4);
    let gpu = run_gpu(&q, &k, &v, 256, 28, 4, true, scale);
    let cpu = cpu_attention_ref(&q, &k, &v, 256, 28, 4, true, scale);
    compare("GQA 28/4 + causal, seq=256", &gpu, &cpu, 1.0e-2);
}
