//! Parity-Validation als GA-Hard-Gate (`ga_tuning_spec §2.8`,
//! `dequant_ir_spec §7.4` + `§8.6`).
//!
//! Säule 6's VALU reference path: every GA candidate runs against a
//! scalar FP32 CPU reference, and `max_abs_err` must stay below a
//! precision-level-dependent tolerance. Failures flip `fitness = 0`
//! before the 20-run benchmark — the GA must not spend its budget
//! timing numerically-wrong kernels.
//!
//! The CPU reference is built on top of `crate::v1::ir::interpreter`
//! (the dequant ground truth from step 1.6 Block 2). That keeps a
//! single source of truth for dequantisation semantics and follows the
//! `ga_tuning_spec §2.8` recipe ("Level-3-VALU-Kernel mit demselben
//! Input").
//!
//! For step 2.1.2 the "candidate kernel" side is the set of
//! Phase-1 GPU GEMV kernels we already have — `KnownKernel` picks one
//! and hands off to the matching `rocmforge_launch_*`. When GA codegen
//! lands in step 2.1.3 the enum gets one more variant
//! (`KnownKernel::Candidate(Arc<CompiledKernel>)`).

#[cfg(feature = "gpu")]
use std::ffi::c_void;

use half::f16;

use crate::v1::ir::interpreter::dequant_block;
use crate::v1::ir::types::QuantFormat;

use super::rng::SeededRng;
use super::types::{KernelShape, PrecisionLevel};

/// How aggressive a parity check is.
///
/// `tolerance` is the **per-dequant-element** ceiling from
/// `ga_tuning_spec §2.8`. For a full GEMV the effective output-level
/// tolerance must grow with `sqrt(K) × max_magnitude` (FP32
/// accumulation noise) plus an additional factor for kernels that
/// quantise the activation inline (Q8-inline trades speed for
/// precision — ~2⁻⁷ input error per element). The
/// `scaled_output_tolerance()` helper produces that value from the
/// CPU reference magnitudes; the pass/fail logic uses it, not the
/// raw per-element number.
#[derive(Debug, Clone, Copy)]
pub struct ParityConfig {
    pub n_blocks: usize,
    pub tolerance: f32,
    /// Multiplier on the per-element tolerance for kernels that
    /// quantise the activation (Q8-inline). v0.x gemv_test uses
    /// `(max_mag + 1e-2) × sqrt(K) × 1e-2` ≈ 10 × per-element-FP16.
    pub q8_inline_factor: f32,
}

impl ParityConfig {
    /// GA-inline parity: 10 blocks, precision-level tolerance from
    /// `ga_tuning_spec §2.8`.
    pub fn for_ga(level: PrecisionLevel) -> Self {
        Self {
            n_blocks: 10,
            tolerance: match level {
                PrecisionLevel::Fp8 => 0.0078,  // 2⁻⁷
                PrecisionLevel::Fp16 => 0.001,  // 2⁻¹⁰
                PrecisionLevel::Bf16 => 0.0078, // 2⁻⁷
                PrecisionLevel::Fp32 => 0.0001, // near-exact — the tolerance
                                                // still needs some slack
                                                // because the candidate's
                                                // summation order differs.
            },
            q8_inline_factor: 10.0,
        }
    }

    /// Stability parity: 1000 blocks, tolerance slightly stricter than
    /// GA-inline (`§2.9`) because we can afford the gate to be tight
    /// when only Top-5 winners run it.
    pub fn for_stability() -> Self {
        Self {
            n_blocks: 1000,
            tolerance: 0.001,
            q8_inline_factor: 10.0,
        }
    }

    /// Compute the effective output-level tolerance for a GEMV of
    /// contraction size `k` against a reference vector whose absolute
    /// max is `max_mag`. `is_q8_inline` adds the INT8-activation
    /// quantisation headroom.
    ///
    /// Rationale matches `tests_v1/gemv_test.rs::q8_inline_tolerance`:
    /// FP32 accumulation has RMS-growth proportional to `sqrt(K)`, and
    /// inline INT8 quant on top adds ~2⁻⁷ error per activation
    /// multiplication. Combined:
    ///
    ///   tol = (max_mag + headroom) × sqrt(K) × per_element_tolerance
    ///         × (q8_inline_factor if q8-inline else 1)
    pub fn scaled_output_tolerance(&self, max_mag: f32, k: usize, is_q8_inline: bool) -> f32 {
        let base = (max_mag + 1e-2) * (k as f32).sqrt() * self.tolerance;
        if is_q8_inline {
            base * self.q8_inline_factor
        } else {
            // Non-inline kernels only see dequant error — still scale
            // with sqrt(K) for FP32 accumulation.
            base
        }
    }
}

/// One per-element disagreement between candidate and reference output.
#[derive(Debug, Clone, Copy)]
pub struct ParityViolation {
    pub block_idx: usize,
    pub element_idx: usize,
    pub candidate_value: f32,
    pub reference_value: f32,
    pub abs_err: f32,
}

/// Aggregate outcome of a parity check.
#[derive(Debug, Clone)]
pub struct ParityResult {
    pub passed: bool,
    pub max_abs_err: f32,
    pub mean_abs_err: f32,
    pub n_blocks_tested: usize,
    pub violations: Vec<ParityViolation>,
    /// Effective output-level tolerance used for the pass/fail
    /// decision. Always ≥ `ParityConfig::tolerance` because the
    /// per-element spec number has to be scaled up for GEMV
    /// accumulation noise (see `ParityConfig::scaled_output_tolerance`).
    pub effective_tolerance: f32,
}

/// Deterministic test block consumed by both CPU reference and GPU
/// candidate. Seeded PRNG = reproducible comparisons across runs.
///
/// `weights` is `N × (K / elements_per_block)` Q-blocks laid out
/// row-major, matching the Phase-1 GEMV kernel convention (output row
/// `n` contiguous along K). `input` is `K` FP32 activations in
/// `[-1.0, 1.0]` — the post-RMSNorm range (extreme values would trip
/// a different failure mode, FP8 saturation, which is `2.2.3` scope).
#[derive(Debug, Clone)]
pub struct TestBlock {
    pub weights: Vec<u8>,
    pub input: Vec<f32>,
}

impl TestBlock {
    /// Build one test block sized for `(shape.n × shape.k)` in `fmt`.
    /// Weight bytes are drawn raw — the parity check just needs a
    /// deterministic bit pattern, not semantically "good" quantisation.
    pub fn random(shape: &KernelShape, fmt: &QuantFormat, rng: &mut SeededRng) -> Self {
        let blocks_per_row = shape.k / fmt.elements_per_block;
        let weight_bytes = shape.n * blocks_per_row * fmt.block_bytes;
        let mut weights = Vec::with_capacity(weight_bytes);
        for _ in 0..weight_bytes {
            weights.push(rng.next_u32() as u8);
        }
        // Q4_K scales need to sit in a sane FP16 range, otherwise the
        // dequantised values explode and the FP32 accumulation in both
        // CPU and GPU overflows deterministically but differently
        // (the GPU uses reduce-across-lanes, the CPU is scalar). We
        // patch the `d` and `dmin` fields of every Q4_K block to small
        // values in [-0.25, 0.25]; that's enough to keep outputs
        // within ~±8 and well below FP16 saturation.
        if fmt.name == "Q4_K" {
            patch_q4k_scales(&mut weights, fmt, shape.n * blocks_per_row, rng);
        } else if fmt.name == "Q6_K" {
            patch_q6k_scales(&mut weights, fmt, shape.n * blocks_per_row, rng);
        } else if fmt.name == "Q4_0" || fmt.name == "Q8_0" {
            patch_single_scale_format(&mut weights, fmt, shape.n * blocks_per_row, rng);
        } else if fmt.name == "Q4_1" {
            patch_q4_1_scales(&mut weights, fmt, shape.n * blocks_per_row, rng);
        }

        let input: Vec<f32> = (0..shape.k)
            .map(|_| {
                let u = rng.gen_f64();
                (u as f32) * 2.0 - 1.0
            })
            .collect();
        Self { weights, input }
    }
}

/// Generate `n` deterministic test blocks; fixed seed so reruns are
/// comparable.
pub fn generate_deterministic_test_blocks(
    n: usize,
    shape: &KernelShape,
    fmt: &QuantFormat,
    seed: u64,
) -> Vec<TestBlock> {
    let mut rng = SeededRng::new(seed);
    (0..n).map(|_| TestBlock::random(shape, fmt, &mut rng)).collect()
}

fn write_fp16(bytes: &mut [u8], offset: usize, value: f32) {
    let raw = f16::from_f32(value).to_bits().to_le_bytes();
    bytes[offset..offset + 2].copy_from_slice(&raw);
}

/// Q4_K block = 144 bytes: `d` @0 (fp16), `dmin` @2 (fp16),
/// `scales[12]` @4, `qs[128]` @16. We pick `d, dmin ∈ [-0.25, 0.25]`
/// so that `d * sub_scale * (nibble - dmin * sub_min)` stays bounded
/// and FP16 accumulation doesn't overflow.
fn patch_q4k_scales(bytes: &mut [u8], fmt: &QuantFormat, n_blocks: usize, rng: &mut SeededRng) {
    for b in 0..n_blocks {
        let base = b * fmt.block_bytes;
        let d = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        let dmin = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        write_fp16(bytes, base, d);
        write_fp16(bytes, base + 2, dmin);
    }
}

/// Q6_K block = 210 bytes: `ql[128]` @0, `qh[64]` @128, `scales[16]`
/// @192 (int8), `d` @208 (fp16). The d scale is what we constrain.
fn patch_q6k_scales(bytes: &mut [u8], fmt: &QuantFormat, n_blocks: usize, rng: &mut SeededRng) {
    for b in 0..n_blocks {
        let base = b * fmt.block_bytes;
        let d = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        write_fp16(bytes, base + 208, d);
        // Scales are int8; clamp them into a moderate range so the
        // per-element multiplier `d * scale` stays bounded.
        for i in 0..16 {
            bytes[base + 192 + i] = ((rng.next_u32() as i8) % 16) as u8;
        }
    }
}

/// Q4_0 / Q8_0: single FP16 `d` at offset 0.
fn patch_single_scale_format(
    bytes: &mut [u8],
    fmt: &QuantFormat,
    n_blocks: usize,
    rng: &mut SeededRng,
) {
    for b in 0..n_blocks {
        let base = b * fmt.block_bytes;
        let d = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        write_fp16(bytes, base, d);
    }
}

/// Q4_1: `d` @0, `m` @2 (both fp16).
fn patch_q4_1_scales(bytes: &mut [u8], fmt: &QuantFormat, n_blocks: usize, rng: &mut SeededRng) {
    for b in 0..n_blocks {
        let base = b * fmt.block_bytes;
        let d = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        let m = ((rng.gen_f64() as f32) * 2.0 - 1.0) * 0.25;
        write_fp16(bytes, base, d);
        write_fp16(bytes, base + 2, m);
    }
}

/// VALU reference for the Block-C gate_up_swiglu shape:
///   `out[n] = silu(gate[n]) * up[n]`
/// with `silu(x) = x / (1 + exp(-x))` and both `gate` and `up`
/// computed via [`valu_reference_gemv`] against their own weight
/// matrix. Same dequant interpreter as the plain GEMV — the SwiGLU
/// epilogue matches the hot path in the emitted kernel
/// (`rf_v1_ga_silu_f32(g) * u`).
pub fn valu_reference_gate_up_swiglu(
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
    format: &QuantFormat,
    shape: &KernelShape,
) -> Vec<f32> {
    let gate = valu_reference_gemv(weights_gate, input, format, shape);
    let up = valu_reference_gemv(weights_up, input, format, shape);
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect()
}

/// VALU reference: FP32 scalar GEMV, `output[n] = Σ_k W[n,k] · in[k]`.
/// Dequantisation goes through the shared interpreter from
/// `src_v1/ir/interpreter.rs` so parity against the GPU is exactly a
/// parity against the same dequant semantics — no hidden second
/// implementation.
pub fn valu_reference_gemv(
    weights: &[u8],
    input: &[f32],
    format: &QuantFormat,
    shape: &KernelShape,
) -> Vec<f32> {
    let epb = format.elements_per_block;
    let bb = format.block_bytes;
    let blocks_per_row = shape.k / epb;

    let mut out = vec![0.0f32; shape.n];
    for row in 0..shape.n {
        let mut acc = 0.0f32;
        for blk in 0..blocks_per_row {
            let byte_offset = (row * blocks_per_row + blk) * bb;
            let block_bytes = &weights[byte_offset..byte_offset + bb];
            let elems = dequant_block(format, block_bytes)
                .expect("dequant_block: format mismatch or truncated block");
            let k_base = blk * epb;
            for (e, &w) in elems.iter().enumerate() {
                acc += w * input[k_base + e];
            }
        }
        out[row] = acc;
    }
    out
}

/// Unit-level parity: compare a candidate output vector to a reference
/// vector. `ParityResult` records every per-element violation so a
/// failed check lets callers pinpoint exactly which column of which
/// test block is bad. Used both by the full `check_parity` flow and by
/// tests that want to confirm the detection logic directly.
pub fn check_parity_output_pair(
    candidate: &[f32],
    reference: &[f32],
    tolerance: f32,
    block_idx: usize,
) -> (f32, f32, Vec<ParityViolation>) {
    assert_eq!(candidate.len(), reference.len(), "length mismatch");
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f32;
    let mut violations = Vec::new();
    for (i, (&cand, &r)) in candidate.iter().zip(reference.iter()).enumerate() {
        let err = (cand - r).abs();
        sum_err += err;
        if err > max_err {
            max_err = err;
        }
        if err > tolerance {
            violations.push(ParityViolation {
                block_idx,
                element_idx: i,
                candidate_value: cand,
                reference_value: r,
                abs_err: err,
            });
        }
    }
    let mean = if candidate.is_empty() {
        0.0
    } else {
        sum_err / candidate.len() as f32
    };
    (max_err, mean, violations)
}

/// Enumerates the Phase-1 GEMV kernels we know how to dispatch for a
/// parity check. The GA codegen in step 2.1.3 will extend this by
/// carrying an `Arc<CompiledKernel>` — the enum stays backward
/// compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownKernel {
    Q4KStandard,
    Q4KQ8Inline,
    Q4KQ8InlineResidual,
    Q6KStandard,
}

impl KnownKernel {
    pub fn quant_format(&self) -> QuantFormat {
        use crate::v1::ir::formats;
        match self {
            Self::Q4KStandard | Self::Q4KQ8Inline | Self::Q4KQ8InlineResidual => formats::q4_k(),
            Self::Q6KStandard => formats::q6_k(),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q4KStandard => "q4_k_standard",
            Self::Q4KQ8Inline => "q4_k_q8_inline",
            Self::Q4KQ8InlineResidual => "q4_k_q8_inline_residual",
            Self::Q6KStandard => "q6_k_standard",
        }
    }
}

/// End-to-end parity on a Phase-1 kernel. Iterates `cfg.n_blocks`
/// deterministic test blocks, runs each on the GPU, compares against
/// the CPU reference, and aggregates violations. The effective
/// tolerance is computed per-block from the reference magnitude via
/// `ParityConfig::scaled_output_tolerance`.
#[cfg(feature = "gpu")]
pub fn check_parity_known_kernel(
    kind: KnownKernel,
    shape: &KernelShape,
    cfg: &ParityConfig,
    seed: u64,
) -> Result<ParityResult, String> {
    let fmt = kind.quant_format();
    let blocks = generate_deterministic_test_blocks(cfg.n_blocks, shape, &fmt, seed);
    let is_q8_inline = matches!(
        kind,
        KnownKernel::Q4KQ8Inline | KnownKernel::Q4KQ8InlineResidual
    );

    let mut max_err = 0.0f32;
    let mut max_tol = 0.0f32;
    let mut sum_err = 0.0f64;
    let mut total_elements = 0usize;
    let mut all_violations = Vec::new();

    for (block_idx, tb) in blocks.iter().enumerate() {
        let reference = valu_reference_gemv(&tb.weights, &tb.input, &fmt, shape);
        let candidate = run_known_kernel_gpu(kind, &tb.weights, &tb.input, shape)?;
        let max_mag = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tol_here = cfg.scaled_output_tolerance(max_mag, shape.k, is_q8_inline);
        if tol_here > max_tol {
            max_tol = tol_here;
        }
        let (blk_max, blk_mean, blk_violations) =
            check_parity_output_pair(&candidate, &reference, tol_here, block_idx);
        if blk_max > max_err {
            max_err = blk_max;
        }
        sum_err += (blk_mean as f64) * (reference.len() as f64);
        total_elements += reference.len();
        all_violations.extend(blk_violations);
    }

    let mean_abs_err = if total_elements > 0 {
        (sum_err / total_elements as f64) as f32
    } else {
        0.0
    };

    Ok(ParityResult {
        passed: all_violations.is_empty(),
        max_abs_err: max_err,
        mean_abs_err,
        n_blocks_tested: cfg.n_blocks,
        violations: all_violations,
        effective_tolerance: max_tol,
    })
}

/// Dispatch a single GEMV launch on the GPU and return host-side
/// output. Shape follows the Phase-1 convention from
/// `src_v1/backend/gpu/gemv.rs` — weights are `[N × (K / epb)]`
/// Q-blocks row-major, input is `[K]` FP32, output is `[N]` FP32.
/// Each call allocates its own stream + buffers; for timing-sensitive
/// stability runs use `run_known_kernel_pooled` instead.
#[cfg(feature = "gpu")]
pub fn run_known_kernel_gpu(
    kind: KnownKernel,
    weights: &[u8],
    input: &[f32],
    shape: &KernelShape,
) -> Result<Vec<f32>, String> {
    use crate::v1::backend::gpu::error::check;
    use crate::v1::backend::gpu::gemv::{
        rocmforge_launch_gemv_q4_k_q8_inline, rocmforge_launch_gemv_q4_k_q8_inline_residual,
        rocmforge_launch_gemv_q4_k_standard, rocmforge_launch_gemv_q6_k_standard,
    };
    use crate::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
    use crate::v1::backend::gpu::wrappers::{HipBuffer, HipStream};

    let stream = HipStream::new().map_err(|e| format!("stream create: {e:?}"))?;

    let mut d_w = HipBuffer::new(weights.len()).map_err(|e| format!("d_w alloc: {e:?}"))?;
    d_w.copy_from_host(weights)
        .map_err(|e| format!("d_w upload: {e:?}"))?;

    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len()).map_err(|e| format!("d_in alloc: {e:?}"))?;
    d_in.copy_from_host(in_bytes)
        .map_err(|e| format!("d_in upload: {e:?}"))?;

    let out_bytes = shape.n * 4;
    let mut d_out = HipBuffer::new(out_bytes).map_err(|e| format!("d_out alloc: {e:?}"))?;

    let rc = unsafe {
        match kind {
            KnownKernel::Q4KStandard => rocmforge_launch_gemv_q4_k_standard(
                d_w.as_ptr() as *const u8,
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream.raw(),
            ),
            KnownKernel::Q4KQ8Inline => rocmforge_launch_gemv_q4_k_q8_inline(
                d_w.as_ptr() as *const u8,
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream.raw(),
            ),
            KnownKernel::Q4KQ8InlineResidual => {
                // Residual-fused wants a `residual` buffer that gets
                // added into the output. For a pure parity check (no
                // residual path), zero the residual buffer and the
                // fused kernel degenerates to plain GEMV.
                let zero_residual = vec![0.0f32; shape.n];
                let res_bytes = unsafe {
                    std::slice::from_raw_parts(
                        zero_residual.as_ptr() as *const u8,
                        zero_residual.len() * 4,
                    )
                };
                let mut d_res =
                    HipBuffer::new(res_bytes.len()).map_err(|e| format!("d_res alloc: {e:?}"))?;
                d_res
                    .copy_from_host(res_bytes)
                    .map_err(|e| format!("d_res upload: {e:?}"))?;
                rocmforge_launch_gemv_q4_k_q8_inline_residual(
                    d_w.as_ptr() as *const u8,
                    d_in.as_ptr() as *const f32,
                    d_res.as_ptr() as *const f32,
                    d_out.as_mut_ptr() as *mut f32,
                    shape.k as i32,
                    shape.n as i32,
                    stream.raw(),
                )
            }
            KnownKernel::Q6KStandard => rocmforge_launch_gemv_q6_k_standard(
                d_w.as_ptr() as *const u8,
                d_in.as_ptr() as *const f32,
                d_out.as_mut_ptr() as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream.raw(),
            ),
        }
    };
    check(rc, "parity kernel launch").map_err(|e| format!("launch: {e:?}"))?;
    stream.synchronize().map_err(|e| format!("sync: {e:?}"))?;

    let mut host = vec![0u8; out_bytes];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut c_void,
            d_out.as_ptr(),
            out_bytes,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "parity D2H").map_err(|e| format!("readback: {e:?}"))?;

    let mut out = Vec::with_capacity(shape.n);
    for chunk in host.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Launch a known Phase-1 kernel using pre-allocated device buffers
/// + a pre-created stream. Reusing buffers across runs is what makes
/// stability timing meaningful — otherwise the 3 ms of alloc + copy
/// per call swamps the ~30 µs kernel.
#[cfg(feature = "gpu")]
pub fn run_known_kernel_pooled(
    kind: KnownKernel,
    d_w: *const core::ffi::c_void,
    d_in: *const core::ffi::c_void,
    d_res: *const core::ffi::c_void,
    d_out: *mut core::ffi::c_void,
    shape: &KernelShape,
    stream: crate::v1::backend::gpu::hip_ffi::hipStream_t,
) -> Result<(), String> {
    use crate::v1::backend::gpu::error::check;
    use crate::v1::backend::gpu::gemv::{
        rocmforge_launch_gemv_q4_k_q8_inline, rocmforge_launch_gemv_q4_k_q8_inline_residual,
        rocmforge_launch_gemv_q4_k_standard, rocmforge_launch_gemv_q6_k_standard,
    };

    let rc = unsafe {
        match kind {
            KnownKernel::Q4KStandard => rocmforge_launch_gemv_q4_k_standard(
                d_w as *const u8,
                d_in as *const f32,
                d_out as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream,
            ),
            KnownKernel::Q4KQ8Inline => rocmforge_launch_gemv_q4_k_q8_inline(
                d_w as *const u8,
                d_in as *const f32,
                d_out as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream,
            ),
            KnownKernel::Q4KQ8InlineResidual => rocmforge_launch_gemv_q4_k_q8_inline_residual(
                d_w as *const u8,
                d_in as *const f32,
                d_res as *const f32,
                d_out as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream,
            ),
            KnownKernel::Q6KStandard => rocmforge_launch_gemv_q6_k_standard(
                d_w as *const u8,
                d_in as *const f32,
                d_out as *mut f32,
                shape.k as i32,
                shape.n as i32,
                stream,
            ),
        }
    };
    check(rc, "parity kernel pooled launch").map_err(|e| format!("launch: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::ir::formats;

    #[test]
    fn parity_config_tolerances_match_spec() {
        assert!((ParityConfig::for_ga(PrecisionLevel::Fp8).tolerance - 0.0078).abs() < 1e-6);
        assert!((ParityConfig::for_ga(PrecisionLevel::Fp16).tolerance - 0.001).abs() < 1e-6);
        assert!((ParityConfig::for_ga(PrecisionLevel::Bf16).tolerance - 0.0078).abs() < 1e-6);
        assert_eq!(ParityConfig::for_ga(PrecisionLevel::Fp16).n_blocks, 10);
        assert_eq!(ParityConfig::for_stability().n_blocks, 1000);
    }

    #[test]
    fn valu_reference_deterministic() {
        let fmt = formats::q4_k();
        let shape = KernelShape::new(1, 256, 256);
        let blocks = generate_deterministic_test_blocks(1, &shape, &fmt, 12345);
        let tb = &blocks[0];
        let a = valu_reference_gemv(&tb.weights, &tb.input, &fmt, &shape);
        let b = valu_reference_gemv(&tb.weights, &tb.input, &fmt, &shape);
        assert_eq!(a, b, "VALU reference must be deterministic");
    }

    #[test]
    fn output_pair_passes_identical_arrays() {
        let cand = vec![0.0, 1.0, -0.5];
        let refr = cand.clone();
        let (max, mean, viols) = check_parity_output_pair(&cand, &refr, 0.001, 0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert!(viols.is_empty());
    }

    #[test]
    fn output_pair_detects_one_bad_element() {
        let cand = vec![0.0, 1.005, 0.0];
        let refr = vec![0.0, 1.0, 0.0];
        let (max, mean, viols) = check_parity_output_pair(&cand, &refr, 0.001, 7);
        assert!((max - 0.005).abs() < 1e-6);
        assert!(mean > 0.0);
        assert_eq!(viols.len(), 1);
        assert_eq!(viols[0].block_idx, 7);
        assert_eq!(viols[0].element_idx, 1);
        assert!((viols[0].abs_err - 0.005).abs() < 1e-6);
    }
}
