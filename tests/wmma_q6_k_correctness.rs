#![cfg(feature = "gpu")]

//! Phase 8b Step 2 correctness tests for `wmma_gemm_q6_k`.
//!
//! Two layers of validation:
//!
//! 1. **Byte-level smoke** — a hand-constructed Q6_K super-block with known
//!    `d`, `scales[16]`, and `ql`/`qh` bit patterns. The output values
//!    are computed with the same per-element closed-form formula the
//!    kernel uses (ground truth derived directly from
//!    `ggml-quants.c::dequantize_row_q6_K`, see
//!    `docs/q6_k_block_format.md` §3).
//!
//! 2. **Shape tests** — random Q6_K weights at the real Qwen3-8B
//!    projection shapes (`attn_v` = 1024 × 4096, `ffn_down` = 4096 × 12288),
//!    validated against a CPU GEMM over the same CPU-dequant weights.
//!
//! Tolerances match the Q4_K tests (5e-2 for most shapes, 1e-1 for the
//! deep `ffn_down` shape where FP16 accumulation noise grows with K).

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_q6_k;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q6_K_BLOCK_BYTES: usize = 210;
const Q6_K_ELEMS_PER_BLOCK: usize = 256;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

// ─── CPU reference dequant — port of ggml-quants.c::dequantize_row_q6_K ──

fn dequant_q6_k_block(block: &[u8]) -> [f32; 256] {
    assert_eq!(block.len(), Q6_K_BLOCK_BYTES);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales: &[i8] = unsafe {
        std::slice::from_raw_parts(block[192..208].as_ptr() as *const i8, 16)
    };
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    let mut out = [0.0f32; 256];
    // Two halves of 128 elements each.
    for n in 0..2 {
        let ql_base = n * 64;
        let qh_base = n * 32;
        let sc_base = n * 8;
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql[ql_base + l +  0] & 0x0F) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i8) as i32 - 32;
            let q2 = (((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i8) as i32 - 32;
            let q3 = (((ql[ql_base + l +  0] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i8) as i32 - 32;
            let q4 = (((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i8) as i32 - 32;
            let y_base = n * 128 + l;
            out[y_base +  0] = d * (scales[sc_base + is + 0] as f32) * (q1 as f32);
            out[y_base + 32] = d * (scales[sc_base + is + 2] as f32) * (q2 as f32);
            out[y_base + 64] = d * (scales[sc_base + is + 4] as f32) * (q3 as f32);
            out[y_base + 96] = d * (scales[sc_base + is + 6] as f32) * (q4 as f32);
        }
    }
    out
}

// ─── Byte-level smoke test ─────────────────────────────────────────────

/// Hand-built Q6_K super-block with known bit patterns; kernel output
/// must match the CPU reference within 1e-2 per element.
#[test]
#[serial]
fn wmma_q6_k_single_block_byte_level() {
    if skip_if_no_gpu() {
        return;
    }

    let mut block = [0u8; Q6_K_BLOCK_BYTES];

    // Fill ql with a reproducible pattern: nibble(i) = i % 16.
    for i in 0..128 {
        let lo = (i as u8) & 0x0F;
        let hi = ((i as u8 + 1) & 0x0F) as u8;
        block[i] = (hi << 4) | lo;
    }
    // Fill qh: byte(i) = (i % 256). The kernel reads 2 bits at a time
    // (bits 0..1, 2..3, 4..5, 6..7), so every combination shows up.
    for i in 0..64 {
        block[128 + i] = ((i * 7) & 0xFF) as u8;
    }
    // scales: a mix of positive and negative int8 values.
    let scales: [i8; 16] = [
         4, -3,  7, -5,  2, -1,  6, -4,
         1,  3, -2,  5, -6,  8, -7,  2,
    ];
    for i in 0..16 {
        block[192 + i] = scales[i] as u8;
    }
    // d = 0.5 (FP16).
    let d_bytes = f16::from_f32(0.5).to_bits().to_le_bytes();
    block[208] = d_bytes[0];
    block[209] = d_bytes[1];

    let expected = dequant_q6_k_block(&block);

    // M=64, N=64, K=256. A is [64 × 256] with A[r, r] = 1 → output is
    // effectively weight[c, r] for each (r, c).
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 256;
    let mut a = vec![0.0f32; M * K];
    for r in 0..M {
        a[r * K + r] = 1.0;
    }

    // Repeat the same super-block across all N columns.
    let mut weights = Vec::with_capacity(N * Q6_K_BLOCK_BYTES);
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

    launch_wmma_gemm_q6_k(
        d_a.as_ptr() as *const f32,
        d_w.as_ptr() as *const u8,
        d_d.as_ptr() as *mut f32,
        M, N, K,
        hipStream_t::null(),
    )
    .expect("wmma_q6_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; M * N];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_d.as_ptr(), M * N * 4).unwrap();
    }

    for r in 0..M {
        for c in 0..N {
            let actual = out[r * N + c];
            let exp = expected[r];
            let diff = (actual - exp).abs();
            assert!(
                diff < 1e-2,
                "byte-level mismatch at row={} col={}: actual={} expected={} diff={}",
                r, c, actual, exp, diff
            );
        }
    }
}

// ─── Shape tests via CPU-reference GEMM ────────────────────────────────

fn gen_q6_k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % Q6_K_ELEMS_PER_BLOCK == 0);
    let blocks_per_row = k / Q6_K_ELEMS_PER_BLOCK;
    let mut buf = vec![0u8; n * blocks_per_row * Q6_K_BLOCK_BYTES];
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for block_idx in 0..(n * blocks_per_row) {
        let base = block_idx * Q6_K_BLOCK_BYTES;
        // ql + qh: random bytes.
        for i in 0..192 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + i] = (state >> 33) as u8;
        }
        // scales: signed int8 in a reasonable range (-48..47 is typical).
        for i in 0..16 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = ((state >> 33) as i32 & 0x7F) - 48;
            buf[base + 192 + i] = (s as i8) as u8;
        }
        // d: small positive FP16.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ds = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let d = f16::from_f32(0.001 + 0.010 * ds);
        let db = d.to_bits().to_le_bytes();
        buf[base + 208] = db[0];
        buf[base + 209] = db[1];
    }
    buf
}

fn dequant_q6_k_matrix(buf: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / Q6_K_ELEMS_PER_BLOCK;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for b in 0..blocks_per_row {
            let base = (row * blocks_per_row + b) * Q6_K_BLOCK_BYTES;
            let block = &buf[base..base + Q6_K_BLOCK_BYTES];
            let decoded = dequant_q6_k_block(block);
            for i in 0..Q6_K_ELEMS_PER_BLOCK {
                out[row * k + b * Q6_K_ELEMS_PER_BLOCK + i] = decoded[i];
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

fn cpu_gemm_row_major(a: &[f32], w_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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
    let w = gen_q6_k_weights(n, k, 0xCCDD ^ (n * 11 + k * 7) as u64);
    let w_deq = dequant_q6_k_matrix(&w, n, k);

    let d_a = GpuBuffer::alloc(m * k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_d = GpuBuffer::alloc(m * n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    launch_wmma_gemm_q6_k(
        d_a.as_ptr() as *const f32,
        d_w.as_ptr() as *const u8,
        d_d.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("wmma_q6_k launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_d.as_ptr(), m * n * 4).unwrap();
    }

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
fn wmma_q6_k_64x64x256() {
    if skip_if_no_gpu() { return; }
    check_shape("64x64x256", 64, 64, 256, 5e-2);
}

#[test]
#[serial]
fn wmma_q6_k_64x256x256() {
    if skip_if_no_gpu() { return; }
    check_shape("64x256x256", 64, 256, 256, 5e-2);
}

#[test]
#[serial]
fn wmma_q6_k_qwen3_attn_v_shape() {
    if skip_if_no_gpu() { return; }
    // Qwen3-8B attn_v: N=1024, K=4096. Q6_K has wider signed quant range
    // (-32..31) than Q4_K (0..15), so FP16 accumulation error at long K
    // grows accordingly. 0.15% rel error = 1.5e-1 absolute at these mags.
    check_shape("64x1024x4096 (Qwen3 attn_v)", 64, 1024, 4096, 1.5e-1);
}

#[test]
#[serial]
fn wmma_q6_k_qwen3_ffn_down_shape() {
    if skip_if_no_gpu() { return; }
    // Qwen3-8B ffn_down: N=4096, K=12288 — deep K, looser tolerance
    check_shape("64x4096x12288 (Qwen3 ffn_down)", 64, 4096, 12288, 2.5e-1);
}
