//! FFI bindings for Phase-2 Q8_1 activation pre-quantizer (Step 1/3 of
//! the llama.cpp MMVQ kernel port).
//!
//! The kernel lives at `hip_kernels_v1/quantize/quantize_q8_1.hip`.
//! Ported from `llama.cpp/ggml/src/ggml-cuda/quantize.cu:4-48` with
//! bit-identical output layout: each 32-element block is 36 bytes
//! (`half2 ds` at offset 0, `int8_t qs[32]` at offset 4).
//!
//! Design rationale (see `results/phase2_llamacpp_kernel_analysis.md`):
//! llama.cpp pre-quantizes activations once per mat-mul into a reusable
//! Q8_1 buffer, whereas the ROCmForge `q8_inline` family re-quantizes
//! inside every GEMV call. Moving the quant out of the hot kernel is
//! root-cause hypothesis H3 (+2 to +5 pp BW standalone) and also
//! unblocks the full MMVQ port.

use std::ffi::c_void;

use super::hip_ffi::{hipError_t, hipStream_t};

/// Bit-identical mirror of llama.cpp's `block_q8_1`.
///
/// `ds[0]` is the per-block scale `d`, `ds[1]` is `d * sum(qs[i])`
/// (stored as the warp-reduced float sum of the original `xi` values,
/// which equals `d * Σ qs` up to rounding — this is what the ggml
/// struct comment documents and what downstream Q4_0/Q4_1/Q5_1 MMVQ
/// kernels expect). Both fields are IEEE-754 binary16.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ81 {
    pub ds: [u16; 2],
    pub qs: [i8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockQ81>() == 36);
const _: () = assert!(std::mem::align_of::<BlockQ81>() >= 2);

/// Number of elements per Q8_1 block. Must match `QK8_1` in the kernel
/// and in `ggml-common.h`.
pub const QK8_1: usize = 32;

#[link(name = "v1_quantize_q8_1", kind = "static")]
extern "C" {
    /// Quantize `n_elements` FP32 activations into `n_elements / 32`
    /// Q8_1 blocks. `input` and `q8_1_output` must be device pointers;
    /// the call is asynchronous on `stream`.
    ///
    /// Constraints:
    ///   * `n_elements > 0`
    ///   * `n_elements % 32 == 0`
    ///
    /// Block layout (per 32-element slice):
    ///   * bytes 0–3:  `half2 ds = (d, d*Σqs)` where `d = max(|xi|)/127`
    ///   * bytes 4–35: `int8_t qs[32] = round(xi / d)`, 0 when `d == 0`
    pub fn rocmforge_launch_quantize_q8_1(
        input: *const f32,
        q8_1_output: *mut c_void,
        n_elements: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
