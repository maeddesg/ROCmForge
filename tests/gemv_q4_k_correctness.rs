#![cfg(feature = "gpu")]

//! Phase 7 Step 4 correctness tests for `gemv_q4_k_f32_on_stream` and
//! its residual variant. Same three-layer validation as the WMMA Q4_K
//! Step 3 tests: byte-level smoke, then shape tests against a CPU
//! reference that is a byte-for-byte port of `ggml-quants.c`.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::{
    gemv_q4_k_f32_on_stream, gemv_q4_k_f32_residual_on_stream,
};
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_ELEMS_PER_BLOCK: usize = 256;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

// ─── CPU reference (copied from tests/wmma_q4_k_correctness.rs) ─────────

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k_block(block: &[u8]) -> [f32; 256] {
    assert_eq!(block.len(), 144);
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut out = [0.0f32; 256];
    let mut write = 0usize;
    let mut qs_off = 0usize;
    let mut is_ = 0usize;
    for _ in (0..256).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is_, scales);
        let d1 = d * (sc0 as f32);
        let m1 = dmin * (m0 as f32);
        let (sc1, m1v) = get_scale_min_k4(is_ + 1, scales);
        let d2 = d * (sc1 as f32);
        let m2 = dmin * (m1v as f32);
        for l in 0..32 {
            out[write] = d1 * ((qs[qs_off + l] & 0x0F) as f32) - m1;
            write += 1;
        }
        for l in 0..32 {
            out[write] = d2 * ((qs[qs_off + l] >> 4) as f32) - m2;
            write += 1;
        }
        qs_off += 32;
        is_ += 2;
    }
    out
}

fn gen_q4_k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % Q4_K_ELEMS_PER_BLOCK == 0);
    let blocks_per_row = k / Q4_K_ELEMS_PER_BLOCK;
    let mut buf = vec![0u8; n * blocks_per_row * Q4_K_BLOCK_BYTES];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for block_idx in 0..(n * blocks_per_row) {
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
        for i in 0..12 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 4 + i] = (s >> 33) as u8;
        }
        for i in 0..128 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 16 + i] = (s >> 33) as u8;
        }
    }
    buf
}

fn dequant_q4_k_matrix(buf: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / Q4_K_ELEMS_PER_BLOCK;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for b in 0..blocks_per_row {
            let base = (row * blocks_per_row + b) * Q4_K_BLOCK_BYTES;
            let block = &buf[base..base + Q4_K_BLOCK_BYTES];
            let decoded = dequant_q4_k_block(block);
            for i in 0..Q4_K_ELEMS_PER_BLOCK {
                out[row * k + b * Q4_K_ELEMS_PER_BLOCK + i] = decoded[i];
            }
        }
    }
    out
}

fn seeded_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((((s >> 33) as u32 & 0xFFFF) as f32 / 65535.0) - 0.5) * 0.5
    }).collect()
}

fn cpu_gemv(w_nk: &[f32], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    // output[c] = sum_k w_nk[c, k] * input[k]
    let mut out = vec![0.0f32; n];
    for c in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += w_nk[c * k + kk] * input[kk];
        }
        out[c] = acc;
    }
    out
}

fn check_shape(label: &str, n: usize, k: usize, tol: f32) {
    println!("== {}: N={} K={} ==", label, n, k);
    let a = seeded_floats(k, 0xAABB ^ (n * 11 + k * 7) as u64);
    let w = gen_q4_k_weights(n, k, 0xCCDD ^ (n * 11 + k * 7) as u64);
    let w_deq = dequant_q4_k_matrix(&w, n, k);

    let d_a = GpuBuffer::alloc(k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_out = GpuBuffer::alloc(n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    gemv_q4_k_f32_on_stream(
        d_w.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        k, n,
        hipStream_t::null(),
    ).expect("gemv_q4_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; n];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_out.as_ptr(), n * 4).unwrap();
    }
    let cpu_out = cpu_gemv(&w_deq, &a, n, k);

    let mut max_diff = 0.0f32;
    let mut worst = 0usize;
    for i in 0..n {
        let diff = (gpu_out[i] - cpu_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            worst = i;
        }
    }
    println!("  max_abs_diff = {:.3e} (tolerance {:.0e})", max_diff, tol);
    assert!(
        max_diff < tol,
        "{}: max abs diff {:.3e} > tol {:.0e} (gpu={} cpu={} at idx {})",
        label, max_diff, tol, gpu_out[worst], cpu_out[worst], worst
    );
}

// ─── Byte-level smoke ────────────────────────────────────────────────────

/// One hand-crafted super-block, input vector of all 1.0. Output column
/// `c` then equals sum of all 256 dequantised weights for that column,
/// which we can compute from the CPU reference and compare.
#[test]
#[serial]
fn gemv_q4_k_single_block_byte_level() {
    if skip_if_no_gpu() {
        return;
    }

    let mut block = [0u8; Q4_K_BLOCK_BYTES];
    let d_bytes = f16::from_f32(0.5).to_bits().to_le_bytes();
    let m_bytes = f16::from_f32(0.25).to_bits().to_le_bytes();
    block[0] = d_bytes[0]; block[1] = d_bytes[1];
    block[2] = m_bytes[0]; block[3] = m_bytes[1];
    // Set scales/mins for 8 sub-blocks. Keep bits 6..7 zero so all
    // sub-blocks 4..7 have scale/min with high bits = 0.
    block[4] = 3;  block[5] = 5;  block[6] = 7;  block[7] = 9;   // scale_0..3
    block[8] = 1;  block[9] = 2;  block[10] = 3; block[11] = 4;  // min_0..3
    block[12] = 0x00; block[13] = 0x11; block[14] = 0x22; block[15] = 0x33;
    for i in 0..128 {
        let lo = (i as u8) & 0x0F;
        let hi = ((i as u8 + 1) & 0x0F) as u8;
        block[16 + i] = (hi << 4) | lo;
    }

    let expected_vec = dequant_q4_k_block(&block);
    let expected: f32 = expected_vec.iter().sum();

    // N = 64 output columns, K = 256 (one super-block per column).
    // Input = all 1.0s → output[c] = sum over all 256 dequantised vals.
    const N: usize = 64;
    const K: usize = 256;
    let input = vec![1.0f32; K];
    let mut weights = Vec::with_capacity(N * Q4_K_BLOCK_BYTES);
    for _ in 0..N {
        weights.extend_from_slice(&block);
    }

    let d_a = GpuBuffer::alloc(K * 4).unwrap();
    let d_w = GpuBuffer::alloc(weights.len()).unwrap();
    let d_out = GpuBuffer::alloc(N * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), input.as_ptr() as *const u8, K * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), weights.as_ptr() as *const u8, weights.len()).unwrap();
    }

    gemv_q4_k_f32_on_stream(
        d_w.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        K, N,
        hipStream_t::null(),
    ).expect("gemv_q4_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; N];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_out.as_ptr(), N * 4).unwrap();
    }

    for c in 0..N {
        let diff = (out[c] - expected).abs();
        assert!(
            diff < 1e-1,
            "column {} mismatch: actual={} expected={} (diff={})",
            c, out[c], expected, diff
        );
    }
}

// ─── Shape tests ─────────────────────────────────────────────────────────

#[test]
#[serial]
fn gemv_q4_k_1x64_256() {
    if skip_if_no_gpu() { return; }
    check_shape("N=64 K=256 (minimal)", 64, 256, 5e-3);
}

#[test]
#[serial]
fn gemv_q4_k_qwen3_qkv_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("N=4096 K=4096 (Qwen3 Q/O)", 4096, 4096, 5e-2);
}

#[test]
#[serial]
fn gemv_q4_k_qwen3_gate_up_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("N=12288 K=4096 (Qwen3 Gate/Up)", 12288, 4096, 5e-2);
}

#[test]
#[serial]
fn gemv_q4_k_qwen3_down_shape() {
    if skip_if_no_gpu() { return; }
    // K=12288 → direct-global path (LDS budget exceeded)
    check_shape("N=4096 K=12288 (Qwen3 Down)", 4096, 12288, 1e-1);
}

#[test]
#[serial]
fn gemv_q4_k_llama31_gate_up_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("N=14336 K=4096 (Llama-3.1 Gate/Up)", 14336, 4096, 5e-2);
}

// ─── Residual variant ────────────────────────────────────────────────────

#[test]
#[serial]
fn gemv_q4_k_residual_qwen3_qkv() {
    if skip_if_no_gpu() { return; }
    const N: usize = 4096;
    const K: usize = 4096;
    let a = seeded_floats(K, 0xDEAD_BEEF);
    let r = seeded_floats(N, 0xCAFE_F00D);
    let w = gen_q4_k_weights(N, K, 0xFACE_B00C);
    let w_deq = dequant_q4_k_matrix(&w, N, K);

    let d_a = GpuBuffer::alloc(K * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_r = GpuBuffer::alloc(N * 4).unwrap();
    let d_out = GpuBuffer::alloc(N * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, K * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
        hip_memcpy_h2d(d_r.as_ptr(), r.as_ptr() as *const u8, N * 4).unwrap();
    }

    gemv_q4_k_f32_residual_on_stream(
        d_w.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_r.as_ptr() as *const f32,
        d_out.as_ptr() as *mut f32,
        K, N,
        hipStream_t::null(),
    ).expect("gemv_q4_k residual launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; N];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_out.as_ptr(), N * 4).unwrap();
    }

    let gemv = cpu_gemv(&w_deq, &a, N, K);
    let mut max_diff = 0.0f32;
    for c in 0..N {
        let expected = gemv[c] + r[c];
        let diff = (gpu_out[c] - expected).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("residual max_abs_diff = {:.3e}", max_diff);
    assert!(max_diff < 5e-2, "residual max diff {:.3e} > 5e-2", max_diff);
}
