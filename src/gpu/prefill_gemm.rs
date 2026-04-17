//! hipBLAS-backed prefill GEMM path.
//!
//! Replaces the sequential GEMV stack used for `seq_len > 1` projections with
//! a dequant-to-FP16 + `hipblasHgemm` + FP16-to-FP32 sequence. See
//! `docs/prefill_gemm_plan.md` for the design.
//!
//! Layout convention: all tensors are row-major in ROCmForge. hipBLAS reads
//! column-major, so we compute `C^T = W^T * X^T` by swapping operands:
//!   - view row-major `W[out × in]` as column-major `[in × out]`
//!   - view row-major `X[seq × in]` as column-major `[in × seq]`
//!   - want row-major `Y[seq × out]` = column-major `[out × seq]`
//!   - call hipblasHgemm with transa=T (W), transb=N (X), m=out, n=seq, k=in.

use super::error::GpuResult;
use super::ffi::{hipError_t, hipStream_t};
use super::hipblas_ffi::{f16_bits_from_f32, hgemm, hipblasOperation_t, HipBlasHandle};
use std::os::raw::{c_int, c_void};

const QK4_0: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;

unsafe extern "C" {
    fn dequant_q4_0_to_f16(
        input: *const c_void,
        output: *mut c_void,
        n_elements: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn convert_f32_to_f16(
        input: *const f32,
        output: *mut c_void,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn convert_f16_to_f32(
        input: *const c_void,
        output: *mut f32,
        n: c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

fn check(code: hipError_t, ctx: &str) -> GpuResult<()> {
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(super::error::GpuError::HipApiError {
            code: code as i32,
            description: format!("{}: {:?}", ctx, code),
        })
    }
}

/// Dequantise a Q4_0 weight tensor into an FP16 scratch buffer on the stream.
///
/// `n_elements` must be a multiple of 32 (the Q4_0 block size). `input` must
/// point to at least `n_elements / 32 * 18` bytes; `output` must point to at
/// least `n_elements * 2` bytes.
pub fn dequantize_q4_0_to_f16_on_stream(
    input: *const u8,
    output: *mut u8,
    n_elements: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    assert!(
        n_elements % QK4_0 == 0,
        "dequant_q4_0_to_f16: n_elements ({}) must be a multiple of 32",
        n_elements
    );
    let code = unsafe {
        dequant_q4_0_to_f16(
            input as *const c_void,
            output as *mut c_void,
            n_elements as c_int,
            stream,
        )
    };
    check(code, "dequant_q4_0_to_f16")
}

pub fn convert_f32_to_f16_on_stream(
    input: *const f32,
    output: *mut u8,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let code = unsafe { convert_f32_to_f16(input, output as *mut c_void, n as c_int, stream) };
    check(code, "convert_f32_to_f16")
}

pub fn convert_f16_to_f32_on_stream(
    input: *const u8,
    output: *mut f32,
    n: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let code = unsafe { convert_f16_to_f32(input as *const c_void, output, n as c_int, stream) };
    check(code, "convert_f16_to_f32")
}

/// FP16 × FP16 prefill GEMM, wrapped with the row-major → column-major
/// convention described at the top of the file. All pointers are raw u16
/// (FP16 bit pattern) pointers on the device.
#[allow(clippy::too_many_arguments)]
pub fn hgemm_row_major(
    handle: &HipBlasHandle,
    weight_f16: *const u16,
    input_f16: *const u16,
    output_f16: *mut u16,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> GpuResult<()> {
    let alpha = f16_bits_from_f32(1.0);
    let beta = f16_bits_from_f32(0.0);

    unsafe {
        hgemm(
            handle,
            hipblasOperation_t::HIPBLAS_OP_T, // weight transposed
            hipblasOperation_t::HIPBLAS_OP_N, // input not transposed
            out_dim as i32, // m
            seq_len as i32, // n
            in_dim as i32,  // k
            alpha,
            weight_f16,
            in_dim as i32, // lda = rows of the non-transposed A = in_dim
            input_f16,
            in_dim as i32, // ldb = in_dim (column-major view of seq × in row-major)
            beta,
            output_f16,
            out_dim as i32, // ldc = out_dim (column-major view of seq × out row-major)
        )
    }
}

/// Minimum `seq_len` at which the hipBLAS GEMM path takes over from the
/// custom GEMV stack. Below this, the GEMV paths are faster because the
/// setup cost (dequant + FP16 conversion + hipBLAS dispatch) dominates
/// the modest per-token compute. Empirically tuned against the
/// Qwen2.5-7B Q4_0 prefill sweep: at pp=19 hipBLAS is ~12% slower, at
/// pp=64 hipBLAS is ~25% faster — crossover sits around seq_len 32.
pub const PREFILL_GEMM_THRESHOLD: usize = 32;

/// Largest single projection we may need to dequantise into the FP16
/// scratch buffer. Tied to Qwen2.5-7B Gate/Up/Down = 3584 × 18944 = 68M f16
/// values (~136 MB). We round up a bit for safety; the scratch lives in
/// `GpuPrefillScratch` and is allocated once.
pub const F16_WEIGHT_SCRATCH_ELEMENTS: usize = 3584 * 18944;
