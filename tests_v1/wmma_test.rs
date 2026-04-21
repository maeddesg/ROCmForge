//! Phase 1 / Schritt 1.7 Blocks A + B — WMMA GEMM (FP16, Level 1).
//!
//! Covers the four Phase-1 WMMA kernels produced by `codegen_gpu`:
//!   - Q4_0 FP16 (Block A, ported 1:1 from v0.x)
//!   - Q4_K FP16 (Block B, ported 1:1 from v0.x)
//!   - Q6_K FP16 (Block B, ported 1:1 from v0.x)
//!   - Q8_0 FP16 (Block B, new — v0.x has no Q8_0 weight WMMA)
//!
//! Verification per format:
//!   (1) Shape vs CPU-FP32 reference GEMM built on the Dequant-IR
//!       interpreter. Tolerance `5e-2 * sqrt(K/64)` — FP16 accumulation
//!       noise scales as `O(sqrt(K))`.
//!   (2) v0.x handwritten launcher at same precision (Q4_0 / Q4_K /
//!       Q6_K only; Q8_0 has no v0.x counterpart). Tolerance `1e-3`
//!       — same tile geometry and same FP16 path, only symbol names
//!       and the IEEE-denormal prolog differ; in practice the result
//!       is bit-exact.
//!
//! Q8_0 correctness rests on CPU-FP32 alone plus the bit-exact
//! Dequant-IR parity check (schritt 1.6) — the Q8_0 dequant itself has
//! 81 green parity-tests behind it.
//!
//! All GPU tests run `#[serial]`; multi-GB allocations are not
//! parallel-safe on a consumer GPU.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_wmma_gemm_q4_0_fp16, rocmforge_launch_wmma_gemm_q4_0_fp8,
    rocmforge_launch_wmma_gemm_q4_k_fp16, rocmforge_launch_wmma_gemm_q4_k_fp8,
    rocmforge_launch_wmma_gemm_q6_k_fp16, rocmforge_launch_wmma_gemm_q6_k_fp8,
    rocmforge_launch_wmma_gemm_q8_0_fp16, rocmforge_launch_wmma_gemm_q8_0_fp8,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::{q4_0, q4_k, q6_k, q8_0};
use rocmforge::v1::ir::interpreter::dequant_block;
use rocmforge::v1::ir::types::QuantFormat;
use serial_test::serial;

// ─── Random-block generation ───────────────────────────────────────────────
//
// Each format's block-bytes layout is documented in
// `src_v1/ir/formats.rs`. We keep FP16 block scales inside a small range
// (±0.5 for K-formats, ±1.0 for Q4_0/Q8_0) so accumulated products over
// K=4096 stay inside FP16 representable range and don't saturate.

fn gen_q4_0_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 18];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d_f32 = rng.f32() * 2.0 - 1.0;
        buf[b * 18..b * 18 + 2].copy_from_slice(&f16::from_f32(d_f32).to_bits().to_le_bytes());
        for i in 2..18 {
            buf[b * 18 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q8_0_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 34];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d_f32 = rng.f32() * 2.0 - 1.0;
        buf[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d_f32).to_bits().to_le_bytes());
        for i in 2..34 {
            buf[b * 34 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q4_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        // Clamp d / dmin magnitudes so the FP16 `d_scale * nibble - d_mn`
        // stays inside ±10 — well within FP16 normal range.
        let d = (rng.f32() * 2.0 - 1.0) * 0.5;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.5;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4]
            .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            buf[b * 144 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q6_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 210];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let base = b * 210;
        // Fill ql, qh, scales with random bytes.
        for i in 0..208 {
            buf[base + i] = rng.u8(..);
        }
        // d at offset 208 (end of block, signed int8 scales already included above).
        let d = (rng.f32() * 2.0 - 1.0) * 0.5;
        buf[base + 208..base + 210]
            .copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    buf
}

fn gen_input(m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

// ─── CPU reference GEMM (generic over format) ──────────────────────────────

fn cpu_reference_gemm(
    format: &QuantFormat,
    input: &[f32],
    weights: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let epb = format.elements_per_block;
    let bb = format.block_bytes;
    let blocks_per_row = k / epb;

    // Dequant W [N × K] to FP32 row-major.
    let mut w_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for blk in 0..blocks_per_row {
            let ofs = (row * blocks_per_row + blk) * bb;
            let block_bytes = &weights[ofs..ofs + bb];
            let elems = dequant_block(format, block_bytes).expect("dequant");
            let out_base = row * k + blk * epb;
            w_f32[out_base..out_base + epb].copy_from_slice(&elems);
        }
    }

    // Naive GEMM: D[m][n] = Σ_k A[m][k] * W[n][k].
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += input[row * k + kk] * w_f32[col * k + kk];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

// ─── GPU launcher type + generic runner ────────────────────────────────────

type GpuWmmaLauncher = unsafe extern "C" fn(
    *const f32,
    *const u8,
    *mut f32,
    i32,
    i32,
    i32,
    rocmforge::v1::backend::gpu::hip_ffi::hipStream_t,
) -> i32;

fn run_gpu_wmma(
    launcher: GpuWmmaLauncher,
    input: &[f32],
    weights: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let out_bytes = m * n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        launcher(
            d_in.as_ptr() as *const f32,
            d_w.as_ptr() as *const u8,
            d_out.as_mut_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
            stream.raw(),
        )
    };
    check(rc, "WMMA kernel launch")?;
    stream.synchronize()?;

    let mut host_bytes = vec![0u8; out_bytes];
    let rc = unsafe {
        hipMemcpy(
            host_bytes.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            out_bytes,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H readback")?;

    let mut out = Vec::with_capacity(m * n);
    for chunk in host_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// v0.x kernels — directly FFI-bound to the `*_launch` C symbols to
// avoid mixing v0.x and v1 `hipStream_t` type aliases.
extern "C" {
    fn wmma_gemm_q4_0_launch(
        input: *const f32,
        weights: *const std::ffi::c_void,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    fn wmma_gemm_q4_k_launch(
        input: *const f32,
        weights: *const std::ffi::c_void,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    fn wmma_gemm_q6_k_launch(
        input: *const f32,
        weights: *const std::ffi::c_void,
        output: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

type V0xLauncher = unsafe extern "C" fn(
    *const f32,
    *const std::ffi::c_void,
    *mut f32,
    i32,
    i32,
    i32,
    *mut std::ffi::c_void,
) -> i32;

fn run_gpu_wmma_v0x(
    launcher: V0xLauncher,
    input: &[f32],
    weights: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let out_bytes = m * n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        launcher(
            d_in.as_ptr() as *const f32,
            d_w.as_ptr() as *const std::ffi::c_void,
            d_out.as_mut_ptr() as *mut f32,
            m as i32,
            n as i32,
            k as i32,
            stream.raw() as *mut std::ffi::c_void,
        )
    };
    check(rc, "v0.x WMMA launch")?;
    stream.synchronize()?;

    let mut host_bytes = vec![0u8; out_bytes];
    let rc = unsafe {
        hipMemcpy(
            host_bytes.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            out_bytes,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H readback (v0.x)")?;

    let mut out = Vec::with_capacity(m * n);
    for chunk in host_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// ─── Diff helpers ──────────────────────────────────────────────────────────

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Magnitude-aware FP16 tolerance. Absolute FP16 error per multiply is
/// `~2^-10 * |product|`; over `K` accumulations with random-sign
/// products it grows like `sqrt(K) * max_magnitude * 1e-3`. Block A's
/// flat 5e-2 figure was valid for Q4_0 where weights peak near ±7, but
/// Q4_K / Q6_K / Q8_0 have weight magnitudes one to two orders larger,
/// so we scale by the observed CPU output magnitude. A 1e-3 floor
/// handles all-near-zero outputs.
fn fp16_tolerance(cpu: &[f32], k: usize) -> f32 {
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_mag + 1e-3) * (k as f32).sqrt() * 5e-3
}

// ─── Q4_0 (Block A) ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_wmma_q4_0_minimal_64x64x64() {
    let m = 64;
    let n = 64;
    let k = 64;
    let input = gen_input(m, k, 0x11);
    let weights = gen_q4_0_weights(n, k, 0x22);

    let cpu = cpu_reference_gemm(&q4_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_0_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_0 64x64x64 max_abs_err = {err} (tol {tol})");
    println!("Q4_0 64x64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_q4_0_64x4096x4096() {
    let m = 64;
    let n = 4096;
    let k = 4096;
    let input = gen_input(m, k, 0x33);
    let weights = gen_q4_0_weights(n, k, 0x44);

    let cpu = cpu_reference_gemm(&q4_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_0_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_0 64x4096x4096 max_abs_err = {err} (tol {tol})");
    println!("Q4_0 64x4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_q4_0_vs_v0x_fp16() {
    let m = 64;
    let n = 64;
    let k = 64;
    let input = gen_input(m, k, 0x55);
    let weights = gen_q4_0_weights(n, k, 0x66);

    let v1 = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_0_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();
    let v0 = run_gpu_wmma_v0x(wmma_gemm_q4_0_launch, &input, &weights, m, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-3, "Q4_0 v1 vs v0.x max_abs_err = {err} (tol 1e-3)");
    println!("Q4_0 v1 vs v0.x (64x64x64): max_abs_err = {err:.4e}");
}

// ─── Q4_K (Block B) ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_wmma_q4_k_minimal_64x64x256() {
    // Smallest valid Q4_K shape: K multiple of 256 (one super-block).
    let m = 64;
    let n = 64;
    let k = 256;
    let input = gen_input(m, k, 0x77);
    let weights = gen_q4_k_weights(n, k, 0x88);

    let cpu = cpu_reference_gemm(&q4_k(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_k_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_K 64x64x256 max_abs_err = {err} (tol {tol})");
    println!("Q4_K 64x64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_q4_k_vs_v0x_fp16() {
    // Qwen3 QKV shape (Q4_K needs K multiple of 256; 4096 = 16 super-blocks).
    let m = 64;
    let n = 64;
    let k = 4096;
    let input = gen_input(m, k, 0x99);
    let weights = gen_q4_k_weights(n, k, 0xAA);

    let v1 = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_k_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();
    let v0 = run_gpu_wmma_v0x(wmma_gemm_q4_k_launch, &input, &weights, m, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-3, "Q4_K v1 vs v0.x max_abs_err = {err} (tol 1e-3)");
    println!("Q4_K v1 vs v0.x (64x64x4096): max_abs_err = {err:.4e}");
}

// ─── Q6_K (Block B) ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_wmma_q6_k_minimal_64x64x256() {
    let m = 64;
    let n = 64;
    let k = 256;
    let input = gen_input(m, k, 0xBB);
    let weights = gen_q6_k_weights(n, k, 0xCC);

    let cpu = cpu_reference_gemm(&q6_k(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q6_k_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q6_K 64x64x256 max_abs_err = {err} (tol {tol})");
    println!("Q6_K 64x64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_q6_k_vs_v0x_fp16() {
    let m = 64;
    let n = 64;
    let k = 4096;
    let input = gen_input(m, k, 0xDD);
    let weights = gen_q6_k_weights(n, k, 0xEE);

    let v1 = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q6_k_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();
    let v0 = run_gpu_wmma_v0x(wmma_gemm_q6_k_launch, &input, &weights, m, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-3, "Q6_K v1 vs v0.x max_abs_err = {err} (tol 1e-3)");
    println!("Q6_K v1 vs v0.x (64x64x4096): max_abs_err = {err:.4e}");
}

// ─── Q8_0 (Block B, no v0.x partner) ───────────────────────────────────────

#[test]
#[serial]
fn test_wmma_q8_0_minimal_64x64x64() {
    let m = 64;
    let n = 64;
    let k = 64;
    let input = gen_input(m, k, 0xFF);
    let weights = gen_q8_0_weights(n, k, 0x100);

    let cpu = cpu_reference_gemm(&q8_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q8_0_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q8_0 64x64x64 max_abs_err = {err} (tol {tol})");
    println!("Q8_0 64x64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_q8_0_64x4096x4096() {
    let m = 64;
    let n = 4096;
    let k = 4096;
    let input = gen_input(m, k, 0x111);
    let weights = gen_q8_0_weights(n, k, 0x222);

    let cpu = cpu_reference_gemm(&q8_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q8_0_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp16_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q8_0 64x4096x4096 max_abs_err = {err} (tol {tol})");
    println!("Q8_0 64x4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

// ─── Block C: FP8 (Level 0) ─────────────────────────────────────────────────
//
// FP8 E4M3 representable range is ±448. For random-block tests we must
// keep dequantised weights inside that range, else SATFINITE saturation
// dominates the error and the test tells us nothing about WMMA
// correctness. The generators below scale the FP16 `d`/`dmin` so typical
// weight magnitudes stay comfortably inside ±100.

fn gen_q4_0_weights_fp8(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    // Q4_0 weights peak at |d| * 8. Choose |d| ≤ 0.5 → |w| ≤ 4.
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 18];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d = (rng.f32() * 2.0 - 1.0) * 0.5;
        buf[b * 18..b * 18 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 2..18 {
            buf[b * 18 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q8_0_weights_fp8(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    // Q8_0 weights peak at |d| * 127. Choose |d| ≤ 0.02 → |w| ≤ 2.5.
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 34];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d = (rng.f32() * 2.0 - 1.0) * 0.02;
        buf[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 2..34 {
            buf[b * 34 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q4_k_weights_fp8(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    // Q4_K weights peak at |d| * 63 * 15 − |dmin| * 63. Keep d, dmin ≤ 0.05
    // → peak ≤ 47 + 3 = ~50.
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d = (rng.f32() * 2.0 - 1.0) * 0.05;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.05;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4]
            .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            buf[b * 144 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_q6_k_weights_fp8(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    // Q6_K weights peak at |d| * 127 * 31. Keep |d| ≤ 0.02 → peak ≤ 79.
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 210];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let base = b * 210;
        for i in 0..208 {
            buf[base + i] = rng.u8(..);
        }
        let d = (rng.f32() * 2.0 - 1.0) * 0.02;
        buf[base + 208..base + 210]
            .copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    buf
}

/// FP8-specific tolerance. E4M3 has 3-bit mantissa + 1 sign → ~4 significant
/// bits → ~6% relative precision per multiply. Over K accumulations
/// with random-sign products the noise grows as `sqrt(K)`. Roughly 10–20×
/// looser than the FP16 bound.
fn fp8_tolerance(cpu: &[f32], k: usize) -> f32 {
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_mag + 1e-2) * (k as f32).sqrt() * 6e-2
}

// ── Correctness: CPU-FP32 reference vs GPU-FP8 ──────────────────────────────

#[test]
#[serial]
fn test_wmma_fp8_q4_0_minimal_64x64x64() {
    let m = 64;
    let n = 64;
    let k = 64;
    let input = gen_input(m, k, 0x301);
    let weights = gen_q4_0_weights_fp8(n, k, 0x302);

    let cpu = cpu_reference_gemm(&q4_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_0_fp8,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp8_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "FP8 Q4_0 64x64x64 err={err} tol={tol}");
    println!("FP8 Q4_0 64x64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_fp8_q4_k_minimal_64x64x256() {
    let m = 64;
    let n = 64;
    let k = 256;
    let input = gen_input(m, k, 0x401);
    let weights = gen_q4_k_weights_fp8(n, k, 0x402);

    let cpu = cpu_reference_gemm(&q4_k(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_k_fp8,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp8_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "FP8 Q4_K 64x64x256 err={err} tol={tol}");
    println!("FP8 Q4_K 64x64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_fp8_q6_k_minimal_64x64x256() {
    let m = 64;
    let n = 64;
    let k = 256;
    let input = gen_input(m, k, 0x501);
    let weights = gen_q6_k_weights_fp8(n, k, 0x502);

    let cpu = cpu_reference_gemm(&q6_k(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q6_k_fp8,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp8_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "FP8 Q6_K 64x64x256 err={err} tol={tol}");
    println!("FP8 Q6_K 64x64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_wmma_fp8_q8_0_minimal_64x64x64() {
    let m = 64;
    let n = 64;
    let k = 64;
    let input = gen_input(m, k, 0x601);
    let weights = gen_q8_0_weights_fp8(n, k, 0x602);

    let cpu = cpu_reference_gemm(&q8_0(), &input, &weights, m, n, k);
    let gpu = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q8_0_fp8,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let tol = fp8_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "FP8 Q8_0 64x64x64 err={err} tol={tol}");
    println!("FP8 Q8_0 64x64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

// ── Quality comparison: FP8 vs FP16 at same inputs ──────────────────────────

#[test]
#[serial]
fn test_wmma_q4_k_fp8_vs_fp16_quality() {
    // Same weights & input through FP8 and FP16 kernels; confirm FP8
    // error vs CPU-FP32 is within an acceptable multiple of FP16 error.
    let m = 64;
    let n = 64;
    let k = 256;
    let input = gen_input(m, k, 0x701);
    let weights = gen_q4_k_weights_fp8(n, k, 0x702);

    let cpu = cpu_reference_gemm(&q4_k(), &input, &weights, m, n, k);
    let fp16 = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_k_fp16,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();
    let fp8 = run_gpu_wmma(
        rocmforge_launch_wmma_gemm_q4_k_fp8,
        &input,
        &weights,
        m,
        n,
        k,
    )
    .unwrap();

    let err_fp16 = max_abs_err(&cpu, &fp16);
    let err_fp8 = max_abs_err(&cpu, &fp8);
    let ratio = err_fp8 / err_fp16.max(1e-6);
    println!(
        "Q4_K FP8 vs FP16 quality: fp16_err={err_fp16:.4e}, fp8_err={err_fp8:.4e}, ratio={ratio:.1}×"
    );
    // FP8 should be no more than ~50× noisier than FP16 given 3-bit vs
    // 10-bit mantissa. That's a very loose correctness bound; the
    // absolute tolerance below is the real gate.
    assert!(
        err_fp8 < fp8_tolerance(&cpu, k),
        "FP8 err {err_fp8} exceeds its own tolerance — bug"
    );
}

// ── Performance (informational, not a gate) ─────────────────────────────────

#[test]
#[serial]
fn test_wmma_q4_k_fp8_vs_fp16_performance() {
    use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
    use std::time::Instant;

    let m = 64;
    let n = 4096;
    let k = 4096;
    let input = gen_input(m, k, 0x801);
    let weights = gen_q4_k_weights_fp8(n, k, 0x802);

    // Warm-up + 10 timed runs, median.
    let measure = |launcher: GpuWmmaLauncher| -> f64 {
        let stream = HipStream::new().unwrap();

        let mut d_in = HipBuffer::new(input.len() * 4).unwrap();
        let in_bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
        };
        d_in.copy_from_host(in_bytes).unwrap();

        let mut d_w = HipBuffer::new(weights.len()).unwrap();
        d_w.copy_from_host(&weights).unwrap();

        let out_bytes = m * n * 4;
        let mut d_out = HipBuffer::new(out_bytes).unwrap();

        // Warm-up.
        for _ in 0..3 {
            unsafe {
                launcher(
                    d_in.as_ptr() as *const f32,
                    d_w.as_ptr() as *const u8,
                    d_out.as_mut_ptr() as *mut f32,
                    m as i32,
                    n as i32,
                    k as i32,
                    stream.raw(),
                );
            }
        }
        stream.synchronize().unwrap();

        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let t0 = Instant::now();
            unsafe {
                launcher(
                    d_in.as_ptr() as *const f32,
                    d_w.as_ptr() as *const u8,
                    d_out.as_mut_ptr() as *mut f32,
                    m as i32,
                    n as i32,
                    k as i32,
                    stream.raw(),
                );
            }
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_micros() as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };

    let fp16_us = measure(rocmforge_launch_wmma_gemm_q4_k_fp16);
    let fp8_us = measure(rocmforge_launch_wmma_gemm_q4_k_fp8);
    let ratio = fp8_us / fp16_us;

    println!(
        "Q4_K WMMA GEMM 64×4096×4096: FP16 {fp16_us:.0} µs, FP8 {fp8_us:.0} µs → FP8/FP16 = {ratio:.2}×"
    );
    // Informational only — no hard assertion. Phase-2 GA will pick the
    // winner per shape.
}
