#![cfg(feature = "gpu")]

//! Phase 2b correctness test: WMMA with inline Q4_0 dequant vs.
//! Phase 2a's FP16 WMMA fed by the separate Q4_0 → FP16 dequant kernel.
//!
//! Both paths accumulate in FP32, run the same WMMA instruction in the
//! same K order (Phase 2b splits each Q4_0 block into two 16-wide
//! k_sub iterations; Phase 2a does the same 16-wide iterations), and
//! the input conversion `__float2half` rounds deterministically. A
//! bit-identical result is therefore expected.
//!
//! Weight generation: random 18-byte Q4_0 blocks with the scale bytes
//! overridden to a known non-extreme FP16 pattern so we never hit NaN
//! or Inf. Every byte of the nibble region is left random.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::{launch_wmma_gemm_q4_0, launch_wmma_gemm_tiled};
use rocmforge::gpu::prefill_gemm::{
    convert_f32_to_f16_on_stream, dequantize_q4_0_to_f16_on_stream,
};
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q4_0_BLOCK_BYTES: usize = 18;
const QK4_0: usize = 32;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn seeded_halfs(n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (state >> 33) as u32;
        let normalised = ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5;
        out.push(f16::from_f32(normalised));
    }
    out
}

fn seeded_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (state >> 33) as u32;
        // Keep activations in a modest range: [-0.5, 0.5]
        let normalised = ((raw & 0xFFFF) as f32 / 65535.0 - 0.5);
        out.push(normalised);
    }
    out
}

/// Build a Q4_0 weight tensor of logical shape `[rows × cols]` (row-major).
/// Returns the 18-byte-per-block raw buffer (size = rows * cols/32 * 18).
fn gen_q4_0_weights(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    assert!(cols % QK4_0 == 0);
    let n_blocks = rows * cols / QK4_0;
    let mut buf = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];

    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for blk in 0..n_blocks {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Scale in a safe FP16 range, e.g. [0.002, 0.02] — far from Inf/NaN.
        let s_mix = ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0;
        let scale = f16::from_f32(0.002 + 0.018 * s_mix);
        let sb = scale.to_bits().to_le_bytes();
        buf[blk * Q4_0_BLOCK_BYTES]     = sb[0];
        buf[blk * Q4_0_BLOCK_BYTES + 1] = sb[1];

        for i in 0..16 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[blk * Q4_0_BLOCK_BYTES + 2 + i] = (state >> 33) as u8;
        }
    }
    buf
}

/// Transpose a row-major `[rows × cols]` FP16 tensor on the host.
/// Input is interpreted as `rows * cols` elements of `f16` laid out row-
/// major; output has shape `[cols × rows]` row-major.
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

    // 1. Generate inputs
    let a_f32 = seeded_floats(m * k, 0xAABB ^ (m * 11 + k * 7) as u64);
    let w_q4_host = gen_q4_0_weights(n, k, 0xCCDD ^ (n * 11 + k * 7) as u64);

    // 2. Upload activations and Q4_0 weights
    let d_a_f32 = GpuBuffer::alloc(m * k * std::mem::size_of::<f32>()).unwrap();
    let d_w_q4  = GpuBuffer::alloc(w_q4_host.len()).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a_f32.as_ptr(), a_f32.as_ptr() as *const u8, m * k * 4).unwrap();
        hip_memcpy_h2d(d_w_q4.as_ptr(),  w_q4_host.as_ptr() as *const u8, w_q4_host.len()).unwrap();
    }

    // 3. Run Phase 2b (inline Q4_0 dequant WMMA) — the code under test
    let d_d_q4 = GpuBuffer::alloc(m * n * std::mem::size_of::<f32>()).unwrap();
    launch_wmma_gemm_q4_0(
        d_a_f32.as_ptr() as *const f32,
        d_w_q4.as_ptr() as *const u8,
        d_d_q4.as_ptr() as *mut f32,
        m, n, k,
        hipStream_t::null(),
    )
    .expect("phase 2b launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut q4_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(q4_out.as_mut_ptr() as *mut u8, d_d_q4.as_ptr(), m * n * 4).unwrap();
    }

    // 4. Build the Phase 2a reference:
    //    4a. Convert A_f32 → A_f16 on-device (same __float2half as inside Phase 2b)
    let d_a_f16 = GpuBuffer::alloc(m * k * std::mem::size_of::<u16>()).unwrap();
    convert_f32_to_f16_on_stream(
        d_a_f32.as_ptr() as *const f32,
        d_a_f16.as_ptr(),
        m * k,
        hipStream_t::null(),
    )
    .unwrap();

    //    4b. Dequantise the Q4_0 weights to FP16 on-device (row-major [N × K])
    let d_w_f16_nk = GpuBuffer::alloc(n * k * std::mem::size_of::<u16>()).unwrap();
    dequantize_q4_0_to_f16_on_stream(d_w_q4.as_ptr(), d_w_f16_nk.as_ptr(), n * k, hipStream_t::null()).unwrap();

    //    4c. Transpose [N × K] → [K × N] on host so Phase 2a sees the right layout.
    //         (A small host transpose is fine — this is a test, not a benchmark.)
    let mut w_nk_host = vec![f16::from_f32(0.0); n * k];
    hip_stream_synchronize(hipStream_t::null()).unwrap();
    unsafe {
        hip_memcpy_d2h(
            w_nk_host.as_mut_ptr() as *mut u8,
            d_w_f16_nk.as_ptr(),
            n * k * 2,
        )
        .unwrap();
    }
    let b_kn_host = transpose_f16_host(&w_nk_host, n, k);
    let d_b_kn = GpuBuffer::alloc(k * n * std::mem::size_of::<u16>()).unwrap();
    unsafe {
        hip_memcpy_h2d(d_b_kn.as_ptr(), b_kn_host.as_ptr() as *const u8, k * n * 2).unwrap();
    }

    //    4d. Run Phase 2a WMMA with (A_f16, B_f16) → D_ref
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

    // 5. Compare. Both paths use the same __float2half conversion of A,
    //    the same Q4_0 → FP16 dequant, the same 16-wide WMMA order, and
    //    the same FP32 accumulation order → bit-identical is expected.
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

#[test]
#[serial]
fn wmma_q4_0_matches_phase2a_64x64x64() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "64x64x64", m: 64, n: 64, k: 64 });
}

#[test]
#[serial]
fn wmma_q4_0_matches_phase2a_256x256x256() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "256x256x256", m: 256, n: 256, k: 256 });
}

#[test]
#[serial]
fn wmma_q4_0_matches_phase2a_qkv_o() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "256x3584x3584 (QKV/O)", m: 256, n: 3584, k: 3584 });
}

#[test]
#[serial]
fn wmma_q4_0_matches_phase2a_gate_up() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "256x18944x3584 (Gate/Up)", m: 256, n: 18944, k: 3584 });
}

#[test]
#[serial]
fn wmma_q4_0_matches_phase2a_down() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape(&ShapeCase { label: "256x3584x18944 (Down)", m: 256, n: 3584, k: 18944 });
}
