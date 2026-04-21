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

// ─── Block C: Q4_K Q8-Inline + Gate+Up+SwiGLU ──────────────────────────────

use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_gate_up_swiglu, rocmforge_launch_gemv_q4_k_q8_inline,
};

fn run_gpu_gemv_q4_k_q8_inline(
    weights: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_q8_inline,
        weights,
        input,
        n,
        k,
    )
}

// v0.x Q8-inline launcher for bit-exact comparison.
extern "C" {
    fn gemv_q4_k_f32_q8_inline_launch(
        weights_q4_k: *const std::ffi::c_void,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    fn gemv_gate_up_swiglu_q4_k_f32_launch(
        weights_gate_q4_k: *const std::ffi::c_void,
        weights_up_q4_k: *const std::ffi::c_void,
        input: *const f32,
        swiglu_out: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

// ── Q4_K Q8-Inline correctness ─────────────────────────────────────────────

/// Q8-quantised activation introduces up to ~0.8% relative error per
/// element. Over K multiplications the error accumulates with a
/// sqrt(K) RMS pattern. Absolute tolerance scales with output
/// magnitude.
fn q8_inline_tolerance(cpu: &[f32], k: usize) -> f32 {
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_mag + 1e-2) * (k as f32).sqrt() * 1e-2
}

#[test]
#[serial]
fn test_gemv_q4_k_q8_inline_minimal_64x256() {
    let n = 64;
    let k = 256;
    let input = gen_input(k, 0x401);
    let weights = gen_q4_k_weights(n, k, 0x402);

    let cpu = cpu_reference_gemv(&q4_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv_q4_k_q8_inline(&weights, &input, n, k).unwrap();

    let tol = q8_inline_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_K Q8-inline 64x256 err={err} tol={tol}");
    println!("Q4_K Q8-inline 64x256: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

#[test]
#[serial]
fn test_gemv_q4_k_q8_inline_qkv_4096x4096() {
    // Qwen3 QKV shape. LDS: K/32 × 34 = 128 × 34 = 4.4 KB, easy.
    let n = 4096;
    let k = 4096;
    let input = gen_input(k, 0x403);
    let weights = gen_q4_k_weights(n, k, 0x404);

    let cpu = cpu_reference_gemv(&q4_k(), &input, &weights, n, k);
    let gpu = run_gpu_gemv_q4_k_q8_inline(&weights, &input, n, k).unwrap();

    let tol = q8_inline_tolerance(&cpu, k);
    let err = max_abs_err(&cpu, &gpu);
    assert!(err < tol, "Q4_K Q8-inline 4096x4096 err={err} tol={tol}");
    println!("Q4_K Q8-inline 4096x4096: max_abs_err = {err:.4e} (tol {tol:.4e})");
}

/// Dual-accumulator regression guard (spec §5.3 dmin-offset formula).
/// Construct weights where all nibbles are zero but dmin > 0 and
/// min_j > 0, so:
///
///   value[i] = d · scale_j · 0 − dmin · min_j = −dmin · min_j
///
/// With Q8-inline activations:
///
///   Σ value[i] · x[i] = −dmin · min_j · Σ x[i] = −dmin · min_j · q8_sum
///
/// If the kernel accidentally used only `int_dot` it would produce 0.
/// Only the dual-accumulator formula (`d·scale·int_dot − dmin·min·q8_sum`)
/// yields the correct non-zero answer.
#[test]
#[serial]
fn test_gemv_q4_k_q8_inline_dual_accumulator() {
    let n = 64;
    let k = 256;
    // Craft a Q4_K super-block: d=0, dmin=1, scale_j=0 for all j,
    // min_j=1 for all j, all nibbles=0. The scales-byte packing
    // (get_scale_min_k4) requires:
    //   j∈0..4: scale = scales[j] & 0x3F = 0   → scales[0..4] = 0
    //           min   = scales[j+4] & 0x3F = 1 → scales[4..8] = 1
    //   j∈4..8: scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
    //                 = (1 & 0x0F) | ((0 >> 6) << 4) = 1           ← we want 0!
    //
    // Getting scale_j = 0 everywhere while min_j = 1 for j<4 is not
    // expressible given the packing. Instead use a simpler well-defined
    // configuration: scales[0..4] = 0, scales[4..8] = 1. Then:
    //   j∈0..4: scale=0, min=1   ← values = −dmin, independent of nibbles
    //   j∈4..8: scale = (scales[j+4]=1 & 0x0F)|…  = 1
    //           min   = (scales[j+4]=1 >> 4)|…    = 0
    //           → values = d·1·nibble − dmin·0 = d·nibble (zero here since d=0)
    //
    // So for j∈0..4 we have the pure dual-accumulator scenario
    // (nibbles don't matter; the output must be −dmin × q8_sum).
    let mut weights = vec![0u8; n * (k / 256) * 144];
    for row in 0..n {
        let base = row * 144; // k/256 == 1 block
        let d = 0.0f32;
        let dmin = 1.0f32;
        weights[base..base + 2]
            .copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        weights[base + 2..base + 4]
            .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        // scales[0..4] = 0, scales[4..8] = 1, scales[8..12] = 0.
        for i in 0..4 {
            weights[base + 4 + i] = 0;
        }
        for i in 4..8 {
            weights[base + 4 + i] = 1;
        }
        for i in 8..12 {
            weights[base + 4 + i] = 0;
        }
        // qs stays zero.
    }

    let input = gen_input(k, 0x405);
    let gpu = run_gpu_gemv_q4_k_q8_inline(&weights, &input, n, k).unwrap();

    // Each output[row] = −dmin × Σ_{j=0..4} q8_sum(sub_block j).
    // After Q8-quantisation, input in [-1, 1) maps to [-127, 127]. The
    // first 128 elements of input contribute. With dmin=1 and min_j=1,
    // every output row equals:
    //   −1 × sum(q8[0..128] * their per-block x_scale)
    // Instead of computing the exact reference (depends on Q8 scales),
    // just verify: (a) outputs are NOT zero (dual-accu is active), and
    // (b) all rows produce the same value (since weights are identical
    // and only input/q8_sum determines the result).
    let nonzero_count = gpu.iter().filter(|&&v| v.abs() > 1e-4).count();
    assert!(
        nonzero_count > 0,
        "dual-accumulator inactive: all {} outputs are zero",
        gpu.len()
    );
    // All rows share the same weights, so all outputs must match.
    let first = gpu[0];
    for (i, &v) in gpu.iter().enumerate() {
        assert!(
            (v - first).abs() < 1e-3,
            "rows differ: gpu[0]={first} gpu[{i}]={v}"
        );
    }
    println!(
        "Q4_K Q8-inline dual-accumulator: output = {first:.4e} (non-zero → dual-accu active)"
    );
}

#[test]
#[serial]
fn test_gemv_q4_k_q8_inline_vs_v0x() {
    // v1 vs v0.x with identical Q8-inline logic.
    let n = 64;
    let k = 4096;
    let input = gen_input(k, 0x406);
    let weights = gen_q4_k_weights(n, k, 0x407);

    let v1 = run_gpu_gemv_q4_k_q8_inline(&weights, &input, n, k).unwrap();
    let v0 = run_gpu_gemv_v0x(gemv_q4_k_f32_q8_inline_launch, &weights, &input, n, k)
        .unwrap();

    let err = max_abs_err(&v1, &v0);
    assert!(
        err < 1e-3,
        "Q4_K Q8-inline v1 vs v0.x err = {err} (tol 1e-3)"
    );
    println!("Q4_K Q8-inline v1 vs v0.x (N=64, K=4096): max_abs_err = {err:.4e}");
}

// ── Q4_K Gate+Up+SwiGLU ────────────────────────────────────────────────────

fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn run_gpu_gate_up_swiglu(
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
    n: usize,
    k: usize,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    let mut d_wg = HipBuffer::new(weights_gate.len())?;
    d_wg.copy_from_host(weights_gate)?;
    let mut d_wu = HipBuffer::new(weights_up.len())?;
    d_wu.copy_from_host(weights_up)?;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes)?;

    let out_bytes = n * 4;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        rocmforge_launch_gemv_q4_k_gate_up_swiglu(
            d_wg.as_ptr() as *const u8,
            d_wu.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw(),
        )
    };
    check(rc, "launch_gemv_q4_k_gate_up_swiglu")?;
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

#[test]
#[serial]
fn test_gemv_q4_k_gate_up_swiglu_correctness() {
    // Compute gate = GEMV(W_gate), up = GEMV(W_up) with standard Q4_K,
    // then silu(gate) * up on the CPU; compare to the fused kernel.
    let n = 64;
    let k = 256;
    let input = gen_input(k, 0x501);
    let w_gate = gen_q4_k_weights(n, k, 0x502);
    let w_up = gen_q4_k_weights(n, k, 0x503);

    let gate = run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_standard,
        &w_gate,
        &input,
        n,
        k,
    )
    .unwrap();
    let up = run_gpu_gemv(
        rocmforge_launch_gemv_q4_k_standard,
        &w_up,
        &input,
        n,
        k,
    )
    .unwrap();
    let cpu_fused: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu_f32(g) * u)
        .collect();

    let gpu_fused =
        run_gpu_gate_up_swiglu(&w_gate, &w_up, &input, n, k).unwrap();

    let err = max_abs_err(&cpu_fused, &gpu_fused);
    let max_mag = cpu_fused.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let tol = (max_mag + 1e-3) * 1e-4; // SiLU + one extra mul, tight tol
    assert!(
        err < tol,
        "Gate+Up+SwiGLU err={err} tol={tol} (max_mag={max_mag})"
    );
    println!(
        "Q4_K Gate+Up+SwiGLU correctness: max_abs_err = {err:.4e} (tol {tol:.4e})"
    );
}

#[test]
#[serial]
fn test_gemv_q4_k_gate_up_swiglu_vs_v0x() {
    let n = 64;
    let k = 4096;
    let input = gen_input(k, 0x504);
    let w_gate = gen_q4_k_weights(n, k, 0x505);
    let w_up = gen_q4_k_weights(n, k, 0x506);

    let v1 = run_gpu_gate_up_swiglu(&w_gate, &w_up, &input, n, k).unwrap();

    // v0.x fused-gate-up launch.
    let stream = HipStream::new().unwrap();
    let mut d_wg = HipBuffer::new(w_gate.len()).unwrap();
    d_wg.copy_from_host(&w_gate).unwrap();
    let mut d_wu = HipBuffer::new(w_up.len()).unwrap();
    d_wu.copy_from_host(&w_up).unwrap();
    let mut d_in = HipBuffer::new(input.len() * 4).unwrap();
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    d_in.copy_from_host(in_bytes).unwrap();
    let mut d_out = HipBuffer::new(n * 4).unwrap();
    let rc = unsafe {
        gemv_gate_up_swiglu_q4_k_f32_launch(
            d_wg.as_ptr() as *const std::ffi::c_void,
            d_wu.as_ptr() as *const std::ffi::c_void,
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr() as *mut f32,
            k as i32,
            n as i32,
            stream.raw() as *mut std::ffi::c_void,
        )
    };
    check(rc, "v0.x gate_up_swiglu").unwrap();
    stream.synchronize().unwrap();
    let mut host_bytes = vec![0u8; n * 4];
    let rc = unsafe {
        hipMemcpy(
            host_bytes.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            n * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H v0.x").unwrap();
    let v0: Vec<f32> = host_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let err = max_abs_err(&v1, &v0);
    assert!(
        err < 1e-4,
        "Gate+Up+SwiGLU v1 vs v0.x err = {err} (tol 1e-4)"
    );
    println!(
        "Q4_K Gate+Up+SwiGLU v1 vs v0.x (N=64, K=4096): max_abs_err = {err:.4e}"
    );
}

// ── Performance (informational) ────────────────────────────────────────────

#[test]
#[serial]
fn test_gemv_q4_k_performance_standard_vs_q8_inline() {
    use std::time::Instant;

    let n = 4096;
    let k = 4096;
    let input = gen_input(k, 0x601);
    let weights = gen_q4_k_weights(n, k, 0x602);

    let measure_std = || -> f64 {
        let stream = HipStream::new().unwrap();
        let mut d_w = HipBuffer::new(weights.len()).unwrap();
        d_w.copy_from_host(&weights).unwrap();
        let mut d_in = HipBuffer::new(input.len() * 4).unwrap();
        let in_bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
        };
        d_in.copy_from_host(in_bytes).unwrap();
        let mut d_out = HipBuffer::new(n * 4).unwrap();

        // warm-up
        for _ in 0..3 {
            unsafe {
                rocmforge_launch_gemv_q4_k_standard(
                    d_w.as_ptr() as *const u8,
                    d_in.as_ptr() as *const f32,
                    d_out.as_mut_ptr() as *mut f32,
                    k as i32,
                    n as i32,
                    stream.raw(),
                );
            }
        }
        stream.synchronize().unwrap();

        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let t0 = Instant::now();
            unsafe {
                rocmforge_launch_gemv_q4_k_standard(
                    d_w.as_ptr() as *const u8,
                    d_in.as_ptr() as *const f32,
                    d_out.as_mut_ptr() as *mut f32,
                    k as i32,
                    n as i32,
                    stream.raw(),
                );
            }
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_micros() as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };

    let measure_q8 = || -> f64 {
        let stream = HipStream::new().unwrap();
        let mut d_w = HipBuffer::new(weights.len()).unwrap();
        d_w.copy_from_host(&weights).unwrap();
        let mut d_in = HipBuffer::new(input.len() * 4).unwrap();
        let in_bytes = unsafe {
            std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
        };
        d_in.copy_from_host(in_bytes).unwrap();
        let mut d_out = HipBuffer::new(n * 4).unwrap();

        for _ in 0..3 {
            unsafe {
                rocmforge_launch_gemv_q4_k_q8_inline(
                    d_w.as_ptr() as *const u8,
                    d_in.as_ptr() as *const f32,
                    d_out.as_mut_ptr() as *mut f32,
                    k as i32,
                    n as i32,
                    stream.raw(),
                );
            }
        }
        stream.synchronize().unwrap();

        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let t0 = Instant::now();
            unsafe {
                rocmforge_launch_gemv_q4_k_q8_inline(
                    d_w.as_ptr() as *const u8,
                    d_in.as_ptr() as *const f32,
                    d_out.as_mut_ptr() as *mut f32,
                    k as i32,
                    n as i32,
                    stream.raw(),
                );
            }
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_micros() as f64);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };

    let std_us = measure_std();
    let q8_us = measure_q8();
    let speedup = std_us / q8_us;
    println!(
        "Q4_K GEMV N=4096, K=4096: Standard {std_us:.0} µs, Q8-inline {q8_us:.0} µs → speedup = {speedup:.2}×"
    );
    // No hard assertion — informational. v0.x sees 2-3× for this shape.
}
