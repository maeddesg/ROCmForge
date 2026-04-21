//! Phase 1 / Schritt 1.8 Block A — GEMV Decode PoC: Q4_0 standard.
//!
//! Verifies the v1 GEMV kernel produced by `codegen_gpu` against:
//!   (1) CPU-FP32 reference GEMV built on the Dequant-IR interpreter.
//!   (2) v0.x handwritten `gemv_q4_0_f32_launch` at same precision —
//!       both FP32 dot product, so the comparison should be bit-exact
//!       (modulo warp-reduce order, which is identical by construction).
//!
//! Acceptance:
//!   * CPU-FP32 (dense): `max_abs_err < 1e-3 * sqrt(K)` — FP32 dot
//!     accumulation noise, nothing else.
//!   * v0.x vs v1 (both FP32 standard): `max_abs_err < 1e-5` — the
//!     kernel body is the same warp-parallel reduction, only symbol
//!     names and the IEEE-denormal prolog differ.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_0_standard, rocmforge_launch_gemv_q4_k_standard,
    rocmforge_launch_gemv_q6_k_standard, rocmforge_launch_gemv_q8_0_standard,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::{q4_0, q4_k, q6_k, q8_0};
use rocmforge::v1::ir::interpreter::dequant_block;
use rocmforge::v1::ir::types::QuantFormat;
use serial_test::serial;

// ─── Random Q4_0 weight / input generation ──────────────────────────────────

fn gen_q4_0_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 18];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        // Small d-range keeps products representable and avoids FP32
        // accumulation domination by outliers.
        let d = (rng.f32() * 2.0 - 1.0) * 0.1;
        buf[b * 18..b * 18 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 2..18 {
            buf[b * 18 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_input(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

// ─── CPU reference GEMV ────────────────────────────────────────────────────
//
// `output[n] = Σ_k input[k] * weight[n][k]`. Weights come dequantised
// from the IR interpreter.

fn cpu_reference_gemv(
    format: &QuantFormat,
    input: &[f32],
    weights: &[u8],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let epb = format.elements_per_block;
    let bb = format.block_bytes;
    let blocks_per_row = k / epb;

    // Dequant all weight blocks to FP32.
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

    // Naive dot per output row.
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += input[kk] * w_f32[row * k + kk];
        }
        out[row] = acc;
    }
    out
}

// ─── GPU launch helpers ─────────────────────────────────────────────────────

fn run_gpu_gemv_q4_0_standard(
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let out_bytes = n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        rocmforge_launch_gemv_q4_0_standard(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw(),
        )
    };
    check(rc, "rocmforge_launch_gemv_q4_0_standard")?;
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

    let mut out = Vec::with_capacity(n);
    for chunk in host_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// v0.x launcher — direct FFI to the C symbol avoids cross-alias type
// headaches between v0.x and v1 `hipStream_t`.
extern "C" {
    fn gemv_q4_0_f32_launch(
        weights_q4_0: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

fn run_gpu_gemv_q4_0_v0x(
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let out_bytes = n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        gemv_q4_0_f32_launch(
            d_w.as_ptr() as *const std::ffi::c_void,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw() as *mut std::ffi::c_void,
        )
    };
    check(rc, "gemv_q4_0_f32_launch (v0.x)")?;
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

    let mut out = Vec::with_capacity(n);
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

/// Dense FP32 dot accumulates rounding with `O(sqrt(K))` RMS growth;
/// 1e-3 per multiplication is a comfortable absolute-ULP bound for
/// values below ~1.
fn fp32_dot_tolerance(cpu: &[f32], k: usize) -> f32 {
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_mag + 1e-3) * (k as f32).sqrt() * 1e-5
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_gemv_q4_0_minimal_64x64() {
    // Smallest valid shape: N=64 (8 waves per block), K=64 (= 2 Q4_0 blocks).
    let n = 64;
    let k = 64;
    let input = gen_input(k, 0x11);
    let weights = gen_q4_0_weights(n, k, 0x22);

    let cpu = cpu_reference_gemv(&q4_0(), &input, &weights, n, k);
    let gpu = run_gpu_gemv_q4_0_standard(&weights, &input, n, k).unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_0 GEMV 64x64 err={err} tol={tol}");
    println!("Q4_0 GEMV 64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q4_0_qkv_4096x4096() {
    // Qwen2.5 QKV shape: N=4096, K=4096. 16 KiB shared input — fits.
    let n = 4096;
    let k = 4096;
    let input = gen_input(k, 0x33);
    let weights = gen_q4_0_weights(n, k, 0x44);

    let cpu = cpu_reference_gemv(&q4_0(), &input, &weights, n, k);
    let gpu = run_gpu_gemv_q4_0_standard(&weights, &input, n, k).unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_0 GEMV 4096x4096 err={err} tol={tol}");
    println!("Q4_0 GEMV 4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q4_0_vs_v0x() {
    // Same inputs through v1 and v0.x; expect bit-exact (or within
    // 1e-5 if __shfl_down ordering varies).
    let n = 64;
    let k = 4096;
    let input = gen_input(k, 0x55);
    let weights = gen_q4_0_weights(n, k, 0x66);

    let v1 = run_gpu_gemv_q4_0_standard(&weights, &input, n, k).unwrap();
    let v0 = run_gpu_gemv_q4_0_v0x(&weights, &input, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(
        err < 1e-5,
        "Q4_0 GEMV v1 vs v0.x err = {err} (tol 1e-5)"
    );
    println!("Q4_0 GEMV v1 vs v0.x (N=64, K=4096): max_abs_err = {err:.4e}");
}

// ─── Block B: Q4_K, Q6_K, Q8_0 standard GEMV ────────────────────────────────

fn gen_q4_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        // Small d / dmin keeps accumulated magnitude reasonable.
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

fn gen_q6_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
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

fn gen_q8_0_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 32, 0);
    let blocks_per_row = k / 32;
    let total_blocks = n_rows * blocks_per_row;
    let mut buf = vec![0u8; total_blocks * 34];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[b * 34..b * 34 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 2..34 {
            buf[b * 34 + i] = rng.u8(..);
        }
    }
    buf
}

// Generic GPU runner — parameterised by the launcher function pointer.
type GpuGemvLauncher = unsafe extern "C" fn(
    *const u8,
    *const f32,
    *mut f32,
    i32,
    i32,
    rocmforge::v1::backend::gpu::hip_ffi::hipStream_t,
) -> i32;

fn run_gpu_gemv(
    launcher: GpuGemvLauncher,
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let out_bytes = n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        launcher(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw(),
        )
    };
    check(rc, "GEMV launcher")?;
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

    let mut out = Vec::with_capacity(n);
    for chunk in host_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// v0.x launchers for Q4_K, Q6_K (Q8_0 GEMV has a different signature — skip
// bit-exact v0.x comparison for Q8_0, rely on CPU-FP32 reference alone).
extern "C" {
    fn gemv_q4_k_f32_launch(
        weights_q4_k: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    fn gemv_q6_k_f32_launch(
        weights_q6_k: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

type V0xGemvLauncher = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const f32,
    *mut f32,
    i32,
    i32,
    *mut std::ffi::c_void,
) -> i32;

fn run_gpu_gemv_v0x(
    launcher: V0xGemvLauncher,
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let out_bytes = n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        launcher(
            d_w.as_ptr() as *const std::ffi::c_void,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw() as *mut std::ffi::c_void,
        )
    };
    check(rc, "v0.x GEMV launcher")?;
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

    let mut out = Vec::with_capacity(n);
    for chunk in host_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// ── Q4_K ────────────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_gemv_q4_k_minimal_64x256() {
    // Smallest valid Q4_K shape: K = 256 (one super-block).
    let n = 64;
    let k = 256;
    let input = gen_input(k, 0x101);
    let weights = gen_q4_k_weights(n, k, 0x102);

    let cpu = cpu_reference_gemv(&q4_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_K GEMV 64x256 err={err} tol={tol}");
    println!("Q4_K GEMV 64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q4_k_qkv_4096x4096() {
    let n = 4096;
    let k = 4096;
    let input = gen_input(k, 0x103);
    let weights = gen_q4_k_weights(n, k, 0x104);

    let cpu = cpu_reference_gemv(&q4_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_K GEMV 4096x4096 err={err} tol={tol}");
    println!("Q4_K GEMV 4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q4_k_vs_v0x() {
    let n = 64;
    let k = 4096;
    let input = gen_input(k, 0x105);
    let weights = gen_q4_k_weights(n, k, 0x106);

    let v1 = run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();
    let v0 = run_gpu_gemv_v0x(gemv_q4_k_f32_launch, &weights, &input, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-5, "Q4_K v1 vs v0.x err = {err} (tol 1e-5)");
    println!("Q4_K GEMV v1 vs v0.x (N=64, K=4096): max_abs_err = {err:.4e}");
}

// ── Q6_K ────────────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_gemv_q6_k_minimal_64x256() {
    let n = 64;
    let k = 256;
    let input = gen_input(k, 0x201);
    let weights = gen_q6_k_weights(n, k, 0x202);

    let cpu = cpu_reference_gemv(&q6_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q6_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q6_K GEMV 64x256 err={err} tol={tol}");
    println!("Q6_K GEMV 64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q6_k_attnv_1024x4096() {
    // Qwen3 attn_v GEMV shape.
    let n = 1024;
    let k = 4096;
    let input = gen_input(k, 0x203);
    let weights = gen_q6_k_weights(n, k, 0x204);

    let cpu = cpu_reference_gemv(&q6_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q6_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q6_K GEMV 1024x4096 err={err} tol={tol}");
    println!("Q6_K GEMV 1024x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q6_k_vs_v0x() {
    let n = 64;
    let k = 4096;
    let input = gen_input(k, 0x205);
    let weights = gen_q6_k_weights(n, k, 0x206);

    let v1 = run_gpu_gemv(
        rocmforge_launch_gemv_q6_k_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();
    let v0 = run_gpu_gemv_v0x(gemv_q6_k_f32_launch, &weights, &input, n, k).unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(err < 1e-5, "Q6_K v1 vs v0.x err = {err} (tol 1e-5)");
    println!("Q6_K GEMV v1 vs v0.x (N=64, K=4096): max_abs_err = {err:.4e}");
}

// ── Q8_0 (no v0.x multi-row partner — CPU-FP32 reference only) ─────────────

#[test]
#[serial]
fn test_gemv_q8_0_minimal_64x64() {
    let n = 64;
    let k = 64;
    let input = gen_input(k, 0x301);
    let weights = gen_q8_0_weights(n, k, 0x302);

    let cpu = cpu_reference_gemv(&q8_0(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q8_0_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q8_0 GEMV 64x64 err={err} tol={tol}");
    println!("Q8_0 GEMV 64x64: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q8_0_4096x4096() {
    let n = 4096;
    let k = 4096;
    let input = gen_input(k, 0x303);
    let weights = gen_q8_0_weights(n, k, 0x304);

    let cpu = cpu_reference_gemv(&q8_0(), &input, &weights, n, k);
    let gpu = run_gpu_gemv(
        rocmforge_launch_gemv_q8_0_standard,
        &weights,
        &input,
        n,
        k,
    )
    .unwrap();

    let tol = fp32_dot_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q8_0 GEMV 4096x4096 err={err} tol={tol}");
    println!("Q8_0 GEMV 4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}
