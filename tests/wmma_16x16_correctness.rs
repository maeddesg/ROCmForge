#![cfg(feature = "gpu")]

//! Phase 2a correctness test for the single-block WMMA kernel.
//!
//! Drives `launch_wmma_gemm_16x16` (wrapping
//! `hip_kernels/wmma/wmma_gemm_16x16.hip`) and compares against a CPU
//! reference with FP32 accumulation. Three sub-tests mirror the
//! standalone HIP harness in `profiling/wmma_poc/`:
//!
//!   1. deterministic inputs — bit-exact vs CPU reference.
//!   2. B = I_16  → expect D = A. Isolates the store pattern.
//!   3. A = 0    → expect D = 0. Catches accumulator-init bugs.
//!
//! These tests run by default (no env gate) because they do not load
//! a model — just allocate three 256-element device buffers.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::wmma::launch_wmma_gemm_16x16;
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const N: usize = 16;
const ELEMENTS: usize = N * N; // 256
const A_BYTES: usize = ELEMENTS * 2; // FP16
const B_BYTES: usize = ELEMENTS * 2;
const D_BYTES: usize = ELEMENTS * 4; // FP32

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

fn cpu_matmul_reference(a: &[f16; ELEMENTS], b: &[f16; ELEMENTS]) -> [f32; ELEMENTS] {
    let mut d = [0.0f32; ELEMENTS];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0f32;
            for k in 0..N {
                acc += a[i * N + k].to_f32() * b[k * N + j].to_f32();
            }
            d[i * N + j] = acc;
        }
    }
    d
}

fn deterministic() -> [f16; ELEMENTS] {
    let mut m = [f16::from_f32(0.0); ELEMENTS];
    for i in 0..ELEMENTS {
        let v = (i % 64) as f32 / 128.0;
        m[i] = f16::from_f32(v);
    }
    m
}

fn identity() -> [f16; ELEMENTS] {
    let mut m = [f16::from_f32(0.0); ELEMENTS];
    for i in 0..N {
        m[i * N + i] = f16::from_f32(1.0);
    }
    m
}

fn zeros() -> [f16; ELEMENTS] {
    [f16::from_f32(0.0); ELEMENTS]
}

fn run_on_gpu(a: &[f16; ELEMENTS], b: &[f16; ELEMENTS]) -> [f32; ELEMENTS] {
    let d_a = GpuBuffer::alloc(A_BYTES).expect("alloc A");
    let d_b = GpuBuffer::alloc(B_BYTES).expect("alloc B");
    let d_d = GpuBuffer::alloc(D_BYTES).expect("alloc D");

    // Safety: slices backing the arrays live for the duration of this call.
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, A_BYTES)
            .expect("h2d A");
        hip_memcpy_h2d(d_b.as_ptr(), b.as_ptr() as *const u8, B_BYTES)
            .expect("h2d B");
    }

    launch_wmma_gemm_16x16(
        d_a.as_ptr() as *const u16,
        d_b.as_ptr() as *const u16,
        d_d.as_ptr() as *mut f32,
        hipStream_t::null(),
    )
    .expect("wmma launch");

    hip_stream_synchronize(hipStream_t::null()).expect("sync");

    let mut out = [0.0f32; ELEMENTS];
    unsafe {
        hip_memcpy_d2h(
            out.as_mut_ptr() as *mut u8,
            d_d.as_ptr(),
            D_BYTES,
        )
        .expect("d2h D");
    }
    out
}

fn assert_bit_exact(label: &str, gpu: &[f32; ELEMENTS], cpu: &[f32; ELEMENTS]) {
    let mut mismatches = 0;
    let mut max_diff = 0.0f32;
    for i in 0..ELEMENTS {
        let diff = (gpu[i] - cpu[i]).abs();
        if gpu[i] != cpu[i] {
            mismatches += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    if mismatches != 0 {
        // Diagnostic: does the GPU output match CPU^T?
        let mut transposed = true;
        for i in 0..N {
            for j in 0..N {
                if gpu[i * N + j] != cpu[j * N + i] {
                    transposed = false;
                    break;
                }
            }
            if !transposed {
                break;
            }
        }
        panic!(
            "{label}: {mismatches}/256 mismatches, max_abs_diff = {max_diff}{}",
            if transposed {
                " (GPU output matches CPU^T — store pattern is transposed)"
            } else {
                ""
            }
        );
    }
}

#[test]
#[serial]
fn wmma_16x16_bit_exact_against_cpu_reference() {
    if skip_if_no_gpu() {
        eprintln!("SKIP: no HIP device available");
        return;
    }
    let a = deterministic();
    let b = deterministic();
    let gpu = run_on_gpu(&a, &b);
    let cpu = cpu_matmul_reference(&a, &b);
    assert_bit_exact("test 1 (deterministic)", &gpu, &cpu);
}

#[test]
#[serial]
fn wmma_16x16_identity_b_yields_a() {
    if skip_if_no_gpu() {
        eprintln!("SKIP: no HIP device available");
        return;
    }
    let a = deterministic();
    let b = identity();
    let gpu = run_on_gpu(&a, &b);

    // Expected: D = A, element-by-element (after f16 → f32 cast).
    let mut expected = [0.0f32; ELEMENTS];
    for i in 0..ELEMENTS {
        expected[i] = a[i].to_f32();
    }
    assert_bit_exact("test 2 (B = I)", &gpu, &expected);
}

#[test]
#[serial]
fn wmma_16x16_zero_a_yields_zero() {
    if skip_if_no_gpu() {
        eprintln!("SKIP: no HIP device available");
        return;
    }
    let a = zeros();
    let b = deterministic();
    let gpu = run_on_gpu(&a, &b);
    let zero = [0.0f32; ELEMENTS];
    assert_bit_exact("test 3 (A = 0)", &gpu, &zero);
}
