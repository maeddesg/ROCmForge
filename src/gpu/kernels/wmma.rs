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
