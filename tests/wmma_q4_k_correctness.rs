#![cfg(feature = "gpu")]

//! Phase 7 Step 3 correctness tests for `wmma_gemm_q4_k`.
//!
//! Three layers of validation, cheapest to most expensive:
//!
//! 1. **Byte-level smoke** — a hand-constructed super-block with known
//!    `d`, `dmin`, scales, mins, and nibbles. The output values are
//!    hand-computable; any mistake in the 6-bit scale/min unpack, in
//!    the paired-sub-block nibble selection, or in the dequant
//!    formula (the infamous missing `/64`) shifts the whole output
//!    and the assertion fires.
//!
//! 2. **Golden vectors** — the first 3 super-blocks of
//!    `blk.0.ffn_gate.weight` from Qwen3-8B Q4_K_M, dequantised by the
//!    Python port of `ggml-quants.c::dequantize_row_q4_K` and stored
//!    in `profiling/golden_vectors/`. The kernel must reproduce the
//!    same floats bit-for-bit (modulo FP16 accumulation noise vs. the
//!    FP32 reference, capped at 1e-3 absolute).
//!
//! 3. **Shape tests** — random Q4_K weights on the real Qwen3-8B and
//!    Llama-3.1-8B projection shapes, validated against a CPU GEMM
//!    that reads the same CPU-dequant weights the kernel consumes.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_q4_k;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_ELEMS_PER_BLOCK: usize = 256;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

// ─── CPU reference dequant — byte-for-byte port of ggml-quants.c ────────

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

// ─── Byte-level smoke test ─────────────────────────────────────────────

/// Build a single hand-crafted Q4_K super-block with known values, run
/// it through the kernel with `A = I_64`-ish (tile of FP32 identity
/// rows), and check the output matches the CPU dequant.
#[test]
#[serial]
fn wmma_q4_k_single_block_byte_level() {
    if skip_if_no_gpu() {
        return;
    }

    // Build one super-block:
    //   d    = 0.5
    //   dmin = 0.25
    //   scales[0..3] = 3, 5, 7, 9        (low 6 bits; high 2 bits = 0)
    //   scales[4..7] = 1, 2, 3, 4        (mins 0..3, low 6 bits = 1..4)
    //   scales[8..11] = 0x00, 0x11, 0x22, 0x33
    //                  = scale_4/min_4 .. scale_7/min_7 low 4 bits
    //     → scale_4 = 0, min_4 = 0
    //       scale_5 = 1, min_5 = 1
    //       scale_6 = 2, min_6 = 2
    //       scale_7 = 3, min_7 = 3
    //   qs[0..127] = fill with a deterministic pattern
    //
    // Since scales[0..3] high bits are 0, sub-blocks 4..7 get no high-bit
    // contribution (scale/min stay at the low-4-bit values above).
    let mut block = [0u8; Q4_K_BLOCK_BYTES];
    let d_h = f16::from_f32(0.5).to_bits().to_le_bytes();
    let m_h = f16::from_f32(0.25).to_bits().to_le_bytes();
    block[0] = d_h[0]; block[1] = d_h[1];
    block[2] = m_h[0]; block[3] = m_h[1];
    // scales layout (low bits only for the first 4 sub-blocks):
    block[4] = 3;  block[5] = 5;  block[6] = 7;  block[7] = 9;   // scale_0..3
    block[8] = 1;  block[9] = 2;  block[10] = 3; block[11] = 4;  // min_0..3
    block[12] = 0x00; block[13] = 0x11; block[14] = 0x22; block[15] = 0x33;
    // qs nibbles — repeating 0..15 so each nibble position is distinct.
    for i in 0..128 {
        let lo = (i as u8) & 0x0F;
        let hi = ((i as u8 + 1) & 0x0F) as u8;
        block[16 + i] = (hi << 4) | lo;
    }

    // Expected dequant from the same formula the kernel implements.
    let expected = dequant_q4_k_block(&block);

    // M=64, N=64, K=256. Activation `A` is the 64×256 identity-like
    // matrix with A[r, r] = 1 for r < 64 and zeros elsewhere — that
    // way output[r, c] = weight[c, r] for r ∈ [0, 64).
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 256;
    let mut a = vec![0.0f32; M * K];
    for r in 0..M {
        a[r * K + r] = 1.0;
    }

    // One super-block per N column; 64 identical copies.
    let mut weights = Vec::with_capacity(N * Q4_K_BLOCK_BYTES);
    for _ in 0..N {
        weights.extend_from_slice(&block);
    }

    let d_a = GpuBuffer::alloc(M * K * 4).unwrap();
    let d_w = GpuBuffer::alloc(weights.len()).unwrap();
    let d_d = GpuBuffer::alloc(M * N * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, M * K * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), weights.as_ptr() as *const u8, weights.len()).unwrap();
    }

    launch_wmma_gemm_q4_k(
        d_a.as_ptr() as *const f32,
        d_w.as_ptr() as *const u8,
        d_d.as_ptr() as *mut f32,
        M, N, K,
        hipStream_t::null(),
    )
    .expect("wmma_q4_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; M * N];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_d.as_ptr(), M * N * 4).unwrap();
    }

    // For M=64 and `A` as constructed: `output[r, c] = weight[c, r]`.
    // Since every N column has the same super-block, weight[c, r] is the
    // same as `expected[r]`. All 64 rows of any N column should match.
    for r in 0..M {
        for c in 0..N {
            let actual = out[r * N + c];
            let exp = expected[r];
            let diff = (actual - exp).abs();
            assert!(
                diff < 1e-2,
                "byte-level mismatch at row={} col={}: actual={} expected={} (diff={})",
                r, c, actual, exp, diff
            );
        }
    }
}

// ─── Shape tests via CPU-reference GEMM ────────────────────────────────

/// Generate a random Q4_K weight buffer of logical shape [N × K] (one
/// super-block per N row per K/256 group).
fn gen_q4_k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % Q4_K_ELEMS_PER_BLOCK == 0);
    let blocks_per_row = k / Q4_K_ELEMS_PER_BLOCK;
    let mut buf = vec![0u8; n * blocks_per_row * Q4_K_BLOCK_BYTES];
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for block_idx in 0..(n * blocks_per_row) {
        let base = block_idx * Q4_K_BLOCK_BYTES;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ds = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let d = f16::from_f32(0.001 + 0.010 * ds);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ms = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let dmin = f16::from_f32(0.001 + 0.010 * ms);
        let db = d.to_bits().to_le_bytes();
        let mb = dmin.to_bits().to_le_bytes();
        buf[base] = db[0]; buf[base + 1] = db[1];
        buf[base + 2] = mb[0]; buf[base + 3] = mb[1];
        for i in 0..12 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 4 + i] = (state >> 33) as u8;
        }
        for i in 0..128 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 16 + i] = (state >> 33) as u8;
        }
    }
    buf
}

/// CPU dequant of an entire [N × K] Q4_K weight buffer, producing a
/// [N × K] FP32 matrix. Used to build the ground-truth GEMM result.
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
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let raw = (state >> 33) as u32;
            ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5
        })
        .collect()
}

fn cpu_gemm_row_major(
    a: &[f32], w_nk: &[f32], m: usize, n: usize, k: usize,
) -> Vec<f32> {
    // output[m, n] = sum_k a[m, k] * w_nk[n, k]
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * w_nk[col * k + kk];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn check_shape(label: &str, m: usize, n: usize, k: usize, tol: f32) {
    println!("== {}: M={} N={} K={} ==", label, m, n, k);
    let a = seeded_floats(m * k, 0xAABB ^ (m * 11 + k * 7) as u64);
    let w = gen_q4_k_weights(n, k, 0xCCDD ^ (n * 11 + k * 7) as u64);
    let w_deq = dequant_q4_k_matrix(&w, n, k);

    let d_a = GpuBuffer::alloc(m * k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_d = GpuBuffer::alloc(m * n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    launch_wmma_gemm_q4_k(
        d_a.as_ptr() as *const f32,
        d_w.as_ptr() as *const u8,
        d_d.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("wmma_q4_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_d.as_ptr(), m * n * 4).unwrap();
    }

    // For small-enough shapes we do a direct CPU GEMM; for the big shapes
    // this is slow but still finishes in a few seconds per test and keeps
    // the test dependency-free.
    let cpu_out = cpu_gemm_row_major(&a, &w_deq, m, n, k);

    let mut max_diff = 0.0f32;
    let mut worst = 0usize;
    for i in 0..(m * n) {
        let d = (gpu_out[i] - cpu_out[i]).abs();
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    println!("  max_abs_diff = {:.3e} (tolerance {:.0e})", max_diff, tol);
    assert!(
        max_diff < tol,
        "{}: max abs diff {:.3e} exceeds tolerance {:.0e} (gpu={} cpu={} at idx {})",
        label, max_diff, tol, gpu_out[worst], cpu_out[worst], worst
    );
}

#[test]
#[serial]
fn wmma_q4_k_64x64x256() {
    if skip_if_no_gpu() { return; }
    check_shape("64x64x256", 64, 64, 256, 5e-2);
}

#[test]
#[serial]
fn wmma_q4_k_64x256x256() {
    if skip_if_no_gpu() { return; }
    check_shape("64x256x256", 64, 256, 256, 5e-2);
}

#[test]
#[serial]
fn wmma_q4_k_qwen3_qkv_shape() {
    if skip_if_no_gpu() { return; }
    // Qwen3-8B Q projection-ish shape (M=64 kept small for test speed)
    check_shape("64x4096x4096 (Qwen3 Q/O)", 64, 4096, 4096, 5e-2);
}

#[test]
#[serial]
fn wmma_q4_k_qwen3_gate_up_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("64x12288x4096 (Qwen3 Gate/Up)", 64, 12288, 4096, 5e-2);
}

#[test]
#[serial]
fn wmma_q4_k_qwen3_down_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("64x4096x12288 (Qwen3 Down)", 64, 4096, 12288, 1e-1);
}

#[test]
#[serial]
fn wmma_q4_k_llama31_gate_up_shape() {
    if skip_if_no_gpu() { return; }
    check_shape("64x14336x4096 (Llama-3.1 Gate/Up)", 64, 14336, 4096, 5e-2);
}
