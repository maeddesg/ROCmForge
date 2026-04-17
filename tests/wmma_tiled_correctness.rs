#![cfg(feature = "gpu")]

//! Phase 2a Step 2 correctness test for the tiled 64×64 WMMA GEMM.
//!
//! Uses `hipblasHgemm` (validated in Phase 1) as the reference oracle so
//! we can exercise realistic prefill shapes (3584×18944) without waiting
//! for a CPU reference to finish. FP16 inputs, FP32 output, absolute
//! tolerance set against the expected FP16 accumulation noise floor at
//! the given K dimension.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::hipblas_ffi::{f16_bits_from_f32, hgemm, hipblasOperation_t, HipBlasHandle};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_tiled;
use rocmforge::gpu::prefill_gemm::convert_f16_to_f32_on_stream;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn seeded_halfs(n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = (state >> 33) as u32;
        let normalised = ((raw & 0xFFFF) as f32 / 65535.0 - 0.5) * 0.5; // [-0.25, 0.25]
        out.push(f16::from_f32(normalised));
    }
    out
}

/// Run hipBLAS `Hgemm` as an FP16 reference and return an FP32 output,
/// using the same row-major → column-major trick as the production code.
fn hipblas_reference(
    a: &GpuBuffer,
    b: &GpuBuffer,
    m: usize,
    n: usize,
    k: usize,
    stream: hipStream_t,
) -> Vec<f32> {
    // hipBLAS Hgemm writes FP16 output; we follow up with FP16 → FP32
    // elementwise conversion so we can compare against the WMMA FP32 output.
    let ref_f16 = GpuBuffer::alloc(m * n * std::mem::size_of::<u16>()).expect("alloc ref f16");
    let ref_f32 = GpuBuffer::alloc(m * n * std::mem::size_of::<f32>()).expect("alloc ref f32");

    let handle = HipBlasHandle::create().expect("hipblasCreate");
    handle.set_stream(stream).expect("hipblasSetStream");

    let alpha = f16_bits_from_f32(1.0);
    let beta = f16_bits_from_f32(0.0);

    // D_row[M, N] = A_row[M, K] × B_row[K, N], no transposes on either
    // operand. Re-interpreted column-major, D becomes (N×M) = B_col × A_col
    // with both operands in their natural "N" form:
    //   hipblas_A = B, lda = N   (B_col has leading dim N)
    //   hipblas_B = A, ldb = K   (A_col has leading dim K)
    //   ldc       = N            (D_col has leading dim N)
    unsafe {
        hgemm(
            &handle,
            hipblasOperation_t::HIPBLAS_OP_N,
            hipblasOperation_t::HIPBLAS_OP_N,
            n as i32, // m_hipblas = N
            m as i32, // n_hipblas = M
            k as i32, // K
            alpha,
            b.as_ptr() as *const u16,
            n as i32, // lda = N
            a.as_ptr() as *const u16,
            k as i32, // ldb = K
            beta,
            ref_f16.as_ptr() as *mut u16,
            n as i32, // ldc = N
        )
    }
    .expect("hipblasHgemm");

    convert_f16_to_f32_on_stream(ref_f16.as_ptr(), ref_f32.as_ptr() as *mut f32, m * n, stream)
        .expect("f16 -> f32 conversion");

    hip_stream_synchronize(stream).expect("sync");

    let mut out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(
            out.as_mut_ptr() as *mut u8,
            ref_f32.as_ptr(),
            m * n * std::mem::size_of::<f32>(),
        )
        .expect("d2h ref");
    }
    out
}

/// One end-to-end test: upload random FP16 A,B; run WMMA and hipBLAS;
/// check absolute differences are within tolerance.
fn check_shape(label: &str, m: usize, n: usize, k: usize) {
    println!("== {label}: M={m} N={n} K={k} ==");

    let a_host = seeded_halfs(m * k, 0xA1A2A3A4 ^ (m * 131 + k * 17) as u64);
    let b_host = seeded_halfs(k * n, 0xB1B2B3B4 ^ (k * 131 + n * 17) as u64);

    let d_a = GpuBuffer::alloc(m * k * 2).expect("alloc A");
    let d_b = GpuBuffer::alloc(k * n * 2).expect("alloc B");
    let d_d = GpuBuffer::alloc(m * n * 4).expect("alloc D");

    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a_host.as_ptr() as *const u8, m * k * 2)
            .expect("h2d A");
        hip_memcpy_h2d(d_b.as_ptr(), b_host.as_ptr() as *const u8, k * n * 2)
            .expect("h2d B");
    }

    // WMMA under test
    launch_wmma_gemm_tiled(
        d_a.as_ptr() as *const u16,
        d_b.as_ptr() as *const u16,
        d_d.as_ptr() as *mut f32,
        m,
        n,
        k,
        hipStream_t::null(),
    )
    .expect("wmma launch");
    hip_stream_synchronize(hipStream_t::null()).expect("sync");

    let mut wmma_out = vec![0.0f32; m * n];
    unsafe {
        hip_memcpy_d2h(
            wmma_out.as_mut_ptr() as *mut u8,
            d_d.as_ptr(),
            m * n * 4,
        )
        .expect("d2h wmma");
    }

    // hipBLAS reference
    let ref_out = hipblas_reference(&d_a, &d_b, m, n, k, hipStream_t::null());

    // Tolerance envelope: WMMA accumulates in FP32, hipBLAS Hgemm does not
    // (kernel trace carries no `f32acc` marker), so the two paths diverge
    // proportionally to K because the reference discards bits we keep. The
    // observed per-element error therefore scales roughly as K * eps(FP16)
    // on worst-case elements. Tolerances below allow for that drift without
    // being loose enough to mask a real bug — the 64×64×64 case, where
    // both sides accumulate only 64 products, still holds to ~1e-3.
    let tolerance = if k >= 8192 {
        1.2e0
    } else if k >= 1024 {
        3.0e-1
    } else if k >= 128 {
        1.5e-2
    } else {
        5.0e-3
    };

    // Also sanity-check the mean: if my kernel is systematically wrong
    // (transposed, wrong lanes, etc.) the MEAN diff rockets because every
    // element misses, not just worst-case ones. A correct kernel with
    // FP16 accumulation drift has mean ≪ max.
    let mean_tolerance = tolerance * 0.15;

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut worst_idx = 0usize;
    for i in 0..(m * n) {
        let d = (wmma_out[i] - ref_out[i]).abs();
        sum_diff += d as f64;
        if d > max_diff {
            max_diff = d;
            worst_idx = i;
        }
    }
    let mean_diff = sum_diff / (m * n) as f64;
    println!(
        "  max_abs_diff = {:.3e}  mean = {:.3e}  (tolerance {:.1e})",
        max_diff, mean_diff, tolerance
    );
    if mean_diff as f32 > mean_tolerance {
        panic!(
            "{label}: mean_abs_diff {:.3e} exceeds mean tolerance {:.1e} — likely a logic bug, not accumulation noise",
            mean_diff, mean_tolerance
        );
    }
    if max_diff > tolerance {
        // Diagnostic: does WMMA output equal ref^T?
        let mut transposed = m == n;
        if transposed {
            'outer: for i in 0..m {
                for j in 0..n {
                    if (wmma_out[i * n + j] - ref_out[j * n + i]).abs() > tolerance {
                        transposed = false;
                        break 'outer;
                    }
                }
            }
        }
        panic!(
            "{label}: max_abs_diff {:.3e} exceeds tolerance {:.1e}. worst idx={} wmma={} ref={}{}",
            max_diff,
            tolerance,
            worst_idx,
            wmma_out[worst_idx],
            ref_out[worst_idx],
            if transposed {
                " (output appears transposed)"
            } else {
                ""
            }
        );
    }
}

#[test]
#[serial]
fn wmma_tiled_matches_hipblas_64x64x64() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape("64x64x64", 64, 64, 64);
}

#[test]
#[serial]
fn wmma_tiled_matches_hipblas_256x256x256() {
    if skip_if_no_gpu() {
        return;
    }
    check_shape("256x256x256", 256, 256, 256);
}

#[test]
#[serial]
fn wmma_tiled_matches_hipblas_qkv_o_shape() {
    if skip_if_no_gpu() {
        return;
    }
    // QKV / O projection at pp256: M=256 (seq), N=3584 (out), K=3584 (in).
    check_shape("256x3584x3584 (QKV/O)", 256, 3584, 3584);
}

#[test]
#[serial]
fn wmma_tiled_matches_hipblas_gate_up_shape() {
    if skip_if_no_gpu() {
        return;
    }
    // Gate / Up at pp256: M=256, N=18944, K=3584.
    check_shape("256x18944x3584 (Gate/Up)", 256, 18944, 3584);
}

#[test]
#[serial]
fn wmma_tiled_matches_hipblas_down_shape() {
    if skip_if_no_gpu() {
        return;
    }
    // Down at pp256: M=256, N=3584, K=18944.
    check_shape("256x3584x18944 (Down)", 256, 3584, 18944);
}
