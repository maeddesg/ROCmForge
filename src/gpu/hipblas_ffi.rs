//! Safe-ish FFI bindings to hipBLAS (subset needed for the prefill GEMM path).
//!
//! We only wrap what the prefill path actually calls:
//! - `hipblasCreate` / `hipblasDestroy` for handle lifetime.
//! - `hipblasSetStream` to bind the handle to our device stream.
//! - `hipblasHgemm` for FP16 × FP16 → FP16 GEMM against dequantised weights.
//!
//! All other hipBLAS entry points can be added later on demand.

use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use std::os::raw::{c_int, c_void};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types, dead_code)]
pub enum hipblasStatus_t {
    HIPBLAS_STATUS_SUCCESS = 0,
    HIPBLAS_STATUS_NOT_INITIALIZED = 1,
    HIPBLAS_STATUS_ALLOC_FAILED = 2,
    HIPBLAS_STATUS_INVALID_VALUE = 3,
    HIPBLAS_STATUS_MAPPING_ERROR = 4,
    HIPBLAS_STATUS_EXECUTION_FAILED = 5,
    HIPBLAS_STATUS_INTERNAL_ERROR = 6,
    HIPBLAS_STATUS_NOT_SUPPORTED = 7,
    HIPBLAS_STATUS_ARCH_MISMATCH = 8,
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9,
    HIPBLAS_STATUS_INVALID_ENUM = 10,
    HIPBLAS_STATUS_UNKNOWN = 11,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum hipblasOperation_t {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    #[allow(dead_code)]
    HIPBLAS_OP_C = 113,
}

/// Opaque hipBLAS handle. hipBLAS uses `typedef void* hipblasHandle_t`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub struct hipblasHandle_t(pub *mut c_void);

impl hipblasHandle_t {
    pub const fn null() -> Self {
        Self(std::ptr::null_mut())
    }
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

unsafe impl Send for hipblasHandle_t {}
unsafe impl Sync for hipblasHandle_t {}

#[link(name = "hipblas")]
extern "C" {
    fn hipblasCreate(handle: *mut hipblasHandle_t) -> hipblasStatus_t;
    fn hipblasDestroy(handle: hipblasHandle_t) -> hipblasStatus_t;
    fn hipblasSetStream(handle: hipblasHandle_t, stream: hipStream_t) -> hipblasStatus_t;

    fn hipblasHgemm(
        handle: hipblasHandle_t,
        transa: hipblasOperation_t,
        transb: hipblasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const u16,
        a: *const u16,
        lda: c_int,
        b: *const u16,
        ldb: c_int,
        beta: *const u16,
        c: *mut u16,
        ldc: c_int,
    ) -> hipblasStatus_t;
}

fn status_check(status: hipblasStatus_t) -> GpuResult<()> {
    if status == hipblasStatus_t::HIPBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: status as i32,
            description: format!("hipBLAS status {:?}", status),
        })
    }
}

/// RAII wrapper around a `hipblasHandle_t`.
pub struct HipBlasHandle {
    inner: hipblasHandle_t,
}

impl HipBlasHandle {
    pub fn create() -> GpuResult<Self> {
        let mut raw = hipblasHandle_t::null();
        let status = unsafe { hipblasCreate(&mut raw) };
        status_check(status)?;
        if raw.is_null() {
            return Err(GpuError::HipApiError {
                code: -1,
                description: "hipblasCreate returned null handle".into(),
            });
        }
        Ok(Self { inner: raw })
    }

    pub fn set_stream(&self, stream: hipStream_t) -> GpuResult<()> {
        status_check(unsafe { hipblasSetStream(self.inner, stream) })
    }

    pub fn raw(&self) -> hipblasHandle_t {
        self.inner
    }
}

impl Drop for HipBlasHandle {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            let _ = unsafe { hipblasDestroy(self.inner) };
            self.inner = hipblasHandle_t::null();
        }
    }
}

/// FP16 GEMM. Pointers must already live on the device.
///
/// Computes `C[m × n] = alpha · op(A) · op(B) + beta · C` in column-major
/// terms. Callers building this for row-major tensors should use the
/// standard "compute C^T via swapped operands" trick.
///
/// Safety: all pointers must be valid device pointers with sufficient
/// storage, and `handle` must be bound to the stream that owns them.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm(
    handle: &HipBlasHandle,
    transa: hipblasOperation_t,
    transb: hipblasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    alpha: u16,
    a: *const u16,
    lda: i32,
    b: *const u16,
    ldb: i32,
    beta: u16,
    c: *mut u16,
    ldc: i32,
) -> GpuResult<()> {
    let status = hipblasHgemm(
        handle.raw(),
        transa,
        transb,
        m,
        n,
        k,
        &alpha,
        a,
        lda,
        b,
        ldb,
        &beta,
        c,
        ldc,
    );
    status_check(status)
}

/// Convert an `f32` scalar to the `u16` bit pattern of the equivalent
/// FP16 value. Used for `alpha` / `beta` which hipBLAS passes as `__half*`.
pub fn f16_bits_from_f32(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}
