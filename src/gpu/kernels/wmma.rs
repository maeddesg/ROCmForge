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

    fn wmma_attention_prefill_64_launch(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut f32,
        scale: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    fn wmma_attention_prefill_multihead_launch(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut f32,
        seq_len: i32,
        num_heads: i32,
        row_stride: i32,
        scale: f32,
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

/// Launch the Phase 3a PoC WMMA prefill-attention kernel.
///
/// Computes `O = softmax(Q · K^T / sqrt(head_dim)) · V` for a single
/// head with fixed `seq_len = 64` and `head_dim = 128`. No causal
/// mask, no GQA, no online softmax — those are Phases 3b / 3c.
///
/// Inputs Q, K, V are FP16 row-major `[64 × 128]`. Output O is FP32
/// row-major `[64 × 128]`. All pointers must live on the device.
///
/// `scale` is typically `1.0 / sqrt(head_dim)`.
pub fn launch_wmma_attention_prefill_64(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    o: *mut f32,
    scale: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    let code = unsafe {
        wmma_attention_prefill_64_launch(
            q as *const c_void,
            k as *const c_void,
            v as *const c_void,
            o,
            scale,
            stream,
        )
    };
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: code as i32,
            description: format!("wmma_attention_prefill_64_launch failed: {:?}", code),
        })
    }
}

/// Launch the Phase 3a Step 2 multi-head WMMA prefill-attention kernel.
///
/// Computes `O = softmax(Q · K^T / sqrt(head_dim)) · V` for every head
/// in one dispatch. Q, K, V, O are row-major `[seq × row_stride]` FP16
/// (O is FP32), where `row_stride = num_heads * head_dim` is the
/// per-token stride of the no-GQA layout. `head_dim = 128` is baked in;
/// `seq_len ∈ {64, 128}` is supported (larger sequences need online
/// softmax, Phase 3b).
pub fn launch_wmma_attention_prefill_multihead(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    o: *mut f32,
    seq_len: usize,
    num_heads: usize,
    row_stride: usize,
    scale: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if seq_len != 64 && seq_len != 128 {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "wmma_attention_prefill_multihead: seq_len {} unsupported (must be 64 or 128)",
                seq_len
            ),
        });
    }
    let code = unsafe {
        wmma_attention_prefill_multihead_launch(
            q as *const c_void,
            k as *const c_void,
            v as *const c_void,
            o,
            seq_len as i32,
            num_heads as i32,
            row_stride as i32,
            scale,
            stream,
        )
    };
    if code == hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(GpuError::HipApiError {
            code: code as i32,
            description: format!(
                "wmma_attention_prefill_multihead_launch failed: {:?}",
                code
            ),
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
