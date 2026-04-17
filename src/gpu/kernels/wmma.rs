//! Phase 2a WMMA proof-of-concept kernel wrapper.
//!
//! Wraps the single-block 16×16 FP16 × FP16 → FP32 kernel defined in
//! `hip_kernels/wmma/wmma_gemm_16x16.hip`. This is a correctness-only
//! exposure for the Rust test harness — no dispatch integration.
//!
//! The kernel launches one workgroup with one wave (32 threads). Inputs
//! must be device pointers to 256-element FP16 buffers (row-major 16×16),
//! the output is a 256-element FP32 buffer (row-major 16×16).

use super::super::error::{GpuError, GpuResult};
use super::super::ffi::{hipError_t, hipStream_t};
use std::os::raw::c_void;

unsafe extern "C" {
    fn wmma_gemm_16x16_launch(
        a: *const c_void,
        b: *const c_void,
        d: *mut f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn wmma_gemm_tiled_launch(
        a: *const c_void,
        b: *const c_void,
        d: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn wmma_gemm_q4_0_launch(
        input: *const f32,
        weights_q4_0: *const c_void,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

/// Launch the 16×16 WMMA kernel on the device's default stream.
///
/// Safety: `a`, `b`, `d` must be valid device pointers with the correct
/// sizes (256 × 2 B for `a` / `b`, 256 × 4 B for `d`).
pub fn launch_wmma_gemm_16x16(
    a: *const u16,
    b: *const u16,
    d: *mut f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let code = unsafe {
        wmma_gemm_16x16_launch(
            a as *const c_void,
            b as *const c_void,
            d,
            stream,
        )
    };
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: code as i32,
            description: format!("wmma_gemm_16x16_launch failed: {:?}", code),
        })
    }
}

/// Launch the Phase 2b WMMA GEMM with inline Q4_0 weight dequant.
///
/// Computes `D[M×N] = A[M×K] × B[K×N]` where A is FP32 activations and
/// B is a row-major Q4_0 weight tensor laid out as `[N × K/32]` 18-byte
/// blocks (the standard GGUF layout for a Linear layer's `W[out, in]`
/// tensor, reinterpreted so the GEMM formula `sum_k A[m,k]·B[k,n]`
/// reads `W[n,k]` from the Q4_0 memory at `(n, k)`).
///
/// `M` and `N` must be multiples of 64, `K` must be a multiple of 32.
pub fn launch_wmma_gemm_q4_0(
    input: *const f32,
    weights_q4_0: *const u8,
    output: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if m % 64 != 0 || n % 64 != 0 || k % 32 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "wmma_gemm_q4_0: M={} N={} K={} must be multiples of 64/64/32",
                m, n, k
            ),
        });
    }
    let code = unsafe {
        wmma_gemm_q4_0_launch(
            input,
            weights_q4_0 as *const c_void,
            output,
            m as i32,
            n as i32,
            k as i32,
            stream,
        )
    };
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: code as i32,
            description: format!("wmma_gemm_q4_0_launch failed: {:?}", code),
        })
    }
}

/// Launch the tiled 64×64 WMMA GEMM kernel.
///
/// Computes `D[M×N] = A[M×K] × B[K×N]` in FP16 × FP16 → FP32. All three
/// tensors are row-major. `M` and `N` must be multiples of 64, `K` must
/// be a multiple of 16 — padding for arbitrary shapes lands in Phase 2c.
///
/// Safety: `a`, `b`, `d` must be valid device pointers with at least
/// `M*K*2`, `K*N*2`, `M*N*4` bytes respectively.
pub fn launch_wmma_gemm_tiled(
    a: *const u16,
    b: *const u16,
    d: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if m % 64 != 0 || n % 64 != 0 || k % 16 != 0 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "wmma_gemm_tiled: M={} N={} K={} must be multiples of 64/64/16",
                m, n, k
            ),
        });
    }
    let code = unsafe {
        wmma_gemm_tiled_launch(
            a as *const c_void,
            b as *const c_void,
            d,
            m as i32,
            n as i32,
            k as i32,
            stream,
        )
    };
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: code as i32,
            description: format!("wmma_gemm_tiled_launch failed: {:?}", code),
        })
    }
}
