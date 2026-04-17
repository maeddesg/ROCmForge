#![cfg(feature = "gpu")]

//! Phase 3a correctness test for the WMMA prefill-attention PoC.
//!
//! Same three diagnostic cases as the standalone HIP harness in
//! `profiling/wmma_poc/test_wmma_attention.hip`:
//!   1. deterministic Q/K/V vs a CPU FP32 reference (abs diff ≤ 1e-2).
//!   2. K = Q — S is symmetric; isolates K^T load-transpose bugs.
//!   3. V = 0 — output must be exactly zero; catches accumulator-init
//!      bugs or spurious writes in Phase 3.
//!
//! Runs inside the normal Rust test harness so `cargo test --features
//! gpu` exercises the kernel end-to-end via the regular FFI path.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_attention_prefill_64;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const SEQ: usize = 64;
const HEAD_DIM: usize = 128;
const ELEMENTS: usize = SEQ * HEAD_DIM;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn cpu_attention_reference(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    scale: f32,
) -> Vec<f32> {
    // S = Q · K^T / sqrt(head_dim)
    let mut s = vec![0.0f32; SEQ * SEQ];
    for i in 0..SEQ {
        for j in 0..SEQ {
            let mut acc = 0.0f32;
            for d in 0..HEAD_DIM {
                acc += q[i * HEAD_DIM + d].to_f32() * k[j * HEAD_DIM + d].to_f32();
            }
            s[i * SEQ + j] = acc * scale;
        }
    }
    // Row-wise softmax.
    for i in 0..SEQ {
        let row = &mut s[i * SEQ..(i + 1) * SEQ];
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
    // O = P · V
    let mut o = vec![0.0f32; ELEMENTS];
    for i in 0..SEQ {
        for d in 0..HEAD_DIM {
            let mut acc = 0.0f32;
            for j in 0..SEQ {
                acc += s[i * SEQ + j] * v[j * HEAD_DIM + d].to_f32();
            }
            o[i * HEAD_DIM + d] = acc;
        }
    }
    o
}

fn deterministic_halfs(n: usize, seed: usize) -> Vec<f16> {
    (0..n)
        .map(|i| {
            let v = ((i + seed) * 37) & 0x3F; // 0..63
            f16::from_f32(v as f32 / 256.0)
        })
        .collect()
}

fn zero_halfs(n: usize) -> Vec<f16> {
    vec![f16::from_f32(0.0); n]
}

fn run_on_gpu(q: &[f16], k: &[f16], v: &[f16], scale: f32) -> Vec<f32> {
    let bytes_f16 = ELEMENTS * std::mem::size_of::<u16>();
    let bytes_f32 = ELEMENTS * std::mem::size_of::<f32>();

    let d_q = GpuBuffer::alloc(bytes_f16).expect("alloc Q");
    let d_k = GpuBuffer::alloc(bytes_f16).expect("alloc K");
    let d_v = GpuBuffer::alloc(bytes_f16).expect("alloc V");
    let d_o = GpuBuffer::alloc(bytes_f32).expect("alloc O");

    unsafe {
        hip_memcpy_h2d(d_q.as_ptr(), q.as_ptr() as *const u8, bytes_f16).expect("h2d Q");
        hip_memcpy_h2d(d_k.as_ptr(), k.as_ptr() as *const u8, bytes_f16).expect("h2d K");
        hip_memcpy_h2d(d_v.as_ptr(), v.as_ptr() as *const u8, bytes_f16).expect("h2d V");
    }

    launch_wmma_attention_prefill_64(
        d_q.as_ptr() as *const u16,
        d_k.as_ptr() as *const u16,
        d_v.as_ptr() as *const u16,
        d_o.as_ptr() as *mut f32,
        scale,
        hipStream_t::null(),
    )
    .expect("attention launch");
    hip_stream_synchronize(hipStream_t::null()).expect("sync");

    let mut out = vec![0.0f32; ELEMENTS];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_o.as_ptr(), bytes_f32).expect("d2h O");
    }
    out
}

fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], tol: f32) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut worst = 0usize;
    for i in 0..ELEMENTS {
        let d = (gpu[i] - cpu[i]).abs();
        sum_diff += d as f64;
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    let mean_diff = sum_diff / ELEMENTS as f64;
    eprintln!(
        "{label}: max_abs_diff = {:.3e}  mean_abs_diff = {:.3e}  (tol {:.1e})",
        max_diff, mean_diff, tol
    );
    assert!(
        max_diff <= tol,
        "{label}: max diff {:.3e} exceeds tol {:.1e} at idx {} (gpu={} cpu={})",
        max_diff,
        tol,
        worst,
        gpu[worst],
        cpu[worst]
    );
}

#[test]
#[serial]
fn wmma_attention_deterministic_vs_cpu() {
    if skip_if_no_gpu() {
        eprintln!("SKIP: no HIP device available");
        return;
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = deterministic_halfs(ELEMENTS, 1);
    let k = deterministic_halfs(ELEMENTS, 7);
    let v = deterministic_halfs(ELEMENTS, 13);
    let gpu = run_on_gpu(&q, &k, &v, scale);
    let cpu = cpu_attention_reference(&q, &k, &v, scale);
    assert_close("deterministic", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn wmma_attention_identity_k_equals_q() {
    if skip_if_no_gpu() {
        return;
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = deterministic_halfs(ELEMENTS, 2);
    let k = q.clone();
    let v = deterministic_halfs(ELEMENTS, 17);
    let gpu = run_on_gpu(&q, &k, &v, scale);
    let cpu = cpu_attention_reference(&q, &k, &v, scale);
    assert_close("K = Q", &gpu, &cpu, 1.0e-2);
}

#[test]
#[serial]
fn wmma_attention_zero_v_yields_zero_o() {
    if skip_if_no_gpu() {
        return;
    }
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let q = deterministic_halfs(ELEMENTS, 3);
    let k = deterministic_halfs(ELEMENTS, 5);
    let v = zero_halfs(ELEMENTS);
    let gpu = run_on_gpu(&q, &k, &v, scale);
    let zero = vec![0.0f32; ELEMENTS];
    assert_close("V = 0", &gpu, &zero, 1.0e-3);
}
