#![cfg(feature = "gpu")]

//! Phase 4 Step 4 correctness test: inline-dequant WMMA-Q4_1 against a
//! CPU-dequantised FP16 reference that feeds the Phase 2a tiled WMMA.
//!
//! Both paths accumulate in FP32 with the same 16-wide K iteration pattern
//! and the same `__float2half` activation conversion, so we expect a
//! bit-identical result. A sanity single-block test pins the 20-byte
//! stride / nibble-offset / dequant formula.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::{launch_wmma_gemm_q4_1, launch_wmma_gemm_tiled};
use rocmforge::gpu::prefill_gemm::convert_f32_to_f16_on_stream;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q4_1_BLOCK_BYTES: usize = 20;
const QK4_1: usize = 32;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn seeded_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (state >> 33) as u32;
        let normalised = ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5;
        out.push(normalised);
    }
    out
}

/// Build a Q4_1 weight tensor of logical shape `[rows × cols]`.
/// Returns the 20-byte-per-block raw buffer (size = rows * cols/32 * 20).
/// Byte layout per block: [scale:2][min:2][nibbles:16]
fn gen_q4_1_weights(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    assert!(cols % QK4_1 == 0);
    let n_blocks = rows * cols / QK4_1;
    let mut buf = vec![0u8; n_blocks * Q4_1_BLOCK_BYTES];

    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for blk in 0..n_blocks {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let s_mix = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let scale = f16::from_f32(0.002 + 0.018 * s_mix);

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m_mix = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let min_val = f16::from_f32(-0.05 + 0.10 * m_mix);

        let sb = scale.to_bits().to_le_bytes();
        let mb = min_val.to_bits().to_le_bytes();
        buf[blk * Q4_1_BLOCK_BYTES]     = sb[0];
        buf[blk * Q4_1_BLOCK_BYTES + 1] = sb[1];
        buf[blk * Q4_1_BLOCK_BYTES + 2] = mb[0];
        buf[blk * Q4_1_BLOCK_BYTES + 3] = mb[1];

        for i in 0..16 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[blk * Q4_1_BLOCK_BYTES + 4 + i] = (state >> 33) as u8;
        }
    }
    buf
}

/// CPU dequant Q4_1 weights into a row-major `[rows × cols]` FP16 tensor.
fn dequant_q4_1_host(w: &[u8], rows: usize, cols: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); rows * cols];
    let blocks_per_row = cols / QK4_1;
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let block_ofs = (r * blocks_per_row + b) * Q4_1_BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([w[block_ofs], w[block_ofs + 1]]);
            let min_bits   = u16::from_le_bytes([w[block_ofs + 2], w[block_ofs + 3]]);
            let scale = f16::from_bits(scale_bits).to_f32();
            let min_v = f16::from_bits(min_bits).to_f32();
            for i in 0..16 {
                let qs = w[block_ofs + 4 + i];
                let nib_lo = (qs & 0x0F) as f32;
                let nib_hi = (qs >> 4)   as f32;
                let row_offset = r * cols + b * QK4_1;
                out[row_offset + i]      = f16::from_f32(nib_lo * scale + min_v);
                out[row_offset + i + 16] = f16::from_f32(nib_hi * scale + min_v);
            }
        }
    }
    out
}

fn transpose_f16_host(src: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut dst = vec![f16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

struct ShapeCase {
    label: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

fn check_shape(case: &ShapeCase) {
    let (m, n, k) = (case.m, case.n, case.k);
    println!("== {}: M={} N={} K={} ==", case.label, m, n, k);

    let a_f32 = seeded_floats(m * k, 0xAABB ^ (m * 11 + k * 7) as u64);
    let w_q4_host = gen_q4_1_weights(n, k, 0xCCDD ^ (n * 11 + k * 7) as u64);

    let d_a_f32 = GpuBuffer::alloc(m * k * std::mem::size_of::<f32>()).unwrap();
    let d_w_q4  = GpuBuffer::alloc(w_q4_host.len()).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a_f32.as_ptr(), a_f32.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w_q4.as_ptr(),  w_q4_host.as_ptr() as *const u8, w_q4_host.len()).unwrap();
    }

    let d_d_q4 = GpuBuffer::alloc(m * n * std::mem::size_of::<f32>()).unwrap();
    launch_wmma_gemm_q4_1(
        d_a_f32.as_ptr() as *const f32,
        d_w_q4.as_ptr() as *const u8,
        d_d_q4.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("wmma_q4_1 launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut q4_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(q4_out.as_mut_ptr() as *mut u8, d_d_q4.as_ptr(), m * n * 4).unwrap();
    }

    // Reference path: dequant Q4_1 on host, transpose to [K × N], use Phase 2a WMMA.
    let d_a_f16 = GpuBuffer::alloc(m * k * std::mem::size_of::<u16>()).unwrap();
    convert_f32_to_f16_on_stream(
        d_a_f32.as_ptr() as *const f32,
        d_a_f16.as_ptr(),
        m * k,
        hipStream_t::null(),
    )
    .unwrap();

    let w_nk_host = dequant_q4_1_host(&w_q4_host, n, k);
    let b_kn_host = transpose_f16_host(&w_nk_host, n, k);
    let d_b_kn = GpuBuffer::alloc(k * n * std::mem::size_of::<u16>()).unwrap();
    unsafe {
        hip_memcpy_h2d(d_b_kn.as_ptr(), b_kn_host.as_ptr() as *const u8, k * n * 2).unwrap();
    }

    let d_d_ref = GpuBuffer::alloc(m * n * std::mem::size_of::<f32>()).unwrap();
    launch_wmma_gemm_tiled(
        d_a_f16.as_ptr() as *const u16,
        d_b_kn.as_ptr() as *const u16,
        d_d_ref.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("phase 2a launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut ref_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(ref_out.as_mut_ptr() as *mut u8, d_d_ref.as_ptr(), m * n * 4).unwrap();
    }

    let mut mismatches = 0usize;
    let mut max_diff = 0.0f32;
    let mut worst = 0usize;
    for i in 0..(m * n) {
        if q4_out[i] != ref_out[i] {
            mismatches += 1;
            let d = (q4_out[i] - ref_out[i]).abs();
            if d > max_diff {
                max_diff = d;
                worst = i;
            }
        }
    }
    println!(
        "  mismatches = {}/{}  max_abs_diff = {:.3e}",
        mismatches,
        m * n,
        max_diff
    );
    assert_eq!(
        mismatches, 0,
        "{}: expected bit-identical output, got {} mismatches (max diff {:.3e} at idx {}: q4={} ref={})",
        case.label, mismatches, max_diff, worst, q4_out[worst], ref_out[worst]
    );
}

/// Byte-level smoke: a hand-constructed single Q4_1 block with known
/// scale/min/nibbles must round-trip through the WMMA kernel. This pins
/// the 20-byte stride, the min-offset at byte 2, and the nibble-offset
/// at byte 4 — the three things the most likely to silently corrupt.
#[test]
#[serial]
fn wmma_q4_1_single_block_byte_level() {
    if skip_if_no_gpu() {
        return;
    }
    // Shape: M=64 N=64 K=32 — exactly one K_CHUNK, one TILE_M, one TILE_N.
    // 64 N-columns × 1 K-block × 20 bytes = 1280 weight bytes.
    let m = 64;
    let n = 64;
    let k = 32;
    let n_blocks = n; // blocks_per_row = K/32 = 1
    let mut w = vec![0u8; n_blocks * Q4_1_BLOCK_BYTES];

    // Populate each N-column's single block with scale=0.5, min=-1.0,
    // nibbles=[0..15 repeating]. Dequant: nib * 0.5 - 1.0 ∈ [-1.0, 6.5].
    let scale = f16::from_f32(0.5).to_bits().to_le_bytes();
    let minv  = f16::from_f32(-1.0).to_bits().to_le_bytes();
    for col in 0..n {
        let ofs = col * Q4_1_BLOCK_BYTES;
        w[ofs]     = scale[0];
        w[ofs + 1] = scale[1];
        w[ofs + 2] = minv[0];
        w[ofs + 3] = minv[1];
        // qs[i] = low-nibble i, high-nibble (i+1) & 0x0F — gives distinct
        // values across all 32 K slots so a byte-offset bug immediately
        // shifts the output.
        for i in 0..16 {
            let lo = (i & 0x0F) as u8;
            let hi = ((i + 1) & 0x0F) as u8;
            w[ofs + 4 + i] = (hi << 4) | lo;
        }
    }

    // Activation: identity-ish — row r has a 1.0 at k=r and 0.0 elsewhere.
    // Then output[r, c] = dequant(W[c, r]) for r < k, and 0 for r ≥ k.
    let mut a = vec![0.0f32; m * k];
    for r in 0..k {
        a[r * k + r] = 1.0;
    }

    let d_a = GpuBuffer::alloc(m * k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_d = GpuBuffer::alloc(m * n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }

    launch_wmma_gemm_q4_1(
        d_a.as_ptr() as *const f32,
        d_w.as_ptr() as *const u8,
        d_d.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("wmma_q4_1 launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(out.as_mut_ptr() as *mut u8, d_d.as_ptr(), m * n * 4).unwrap();
    }

    // Expected: for each output column c, output[r, c] for r ∈ [0, 32)
    // should equal the dequantised nibble at K-slot r of the block at
    // column c. Nibble layout inside our synthesised block:
    //   qs[i] low nibble  = (i & 0x0F)        → element i
    //   qs[i] high nibble = ((i+1) & 0x0F)    → element i+16
    // Dequant: nib * 0.5 + (-1.0).
    for r in 0..k {
        for c in 0..n {
            let expected_nib = if r < 16 { r } else { (r - 16 + 1) & 0x0F };
            let expected = (expected_nib as f32) * 0.5 - 1.0;
            let actual = out[r * n + c];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-3,
                "byte-level mismatch at row={} col={}: actual={} expected={} (nib={})",
                r, c, actual, expected, expected_nib
            );
        }
    }
    // Rows r ≥ k should be exactly 0.
    for r in k..m {
        for c in 0..n {
            let v = out[r * n + c];
            assert!(v.abs() < 1e-6, "expected 0 at r={} c={}, got {}", r, c, v);
        }
    }
}

#[test]
#[serial]
fn wmma_q4_1_64x64x64() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "64x64x64", m: 64, n: 64, k: 64 });
}

#[test]
#[serial]
fn wmma_q4_1_256x256x256() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "256x256x256", m: 256, n: 256, k: 256 });
}

#[test]
#[serial]
fn wmma_q4_1_down_proj_shape() {
    if skip_if_no_gpu() {
        return;
    }
    // Same shape as the real FFN-down projection in Qwen2.5-7B Q4_0.
    check_shape(&ShapeCase { label: "256x3584x18944 (Down)", m: 256, n: 3584, k: 18944 });
}
