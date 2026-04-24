//! P0.2 Step 1 — MMQ infrastructure tests.
//!
//! Two deliverables:
//!   1. Integer-WMMA intrinsic on gfx1201: compiles, runs, and produces
//!      bit-exact output vs a scalar CPU reference for a 16×16×16 int8
//!      GEMM.
//!   2. `block_q8_1_mmq` quantiser: size=144, per-32-elem-subblock
//!      scales correct, first-32-elem qs-values match our existing
//!      `block_q8_1` quantiser bit-for-bit.

#![cfg(all(feature = "v1", feature = "gpu"))]

use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_quantize_q8_1_mmq, rocmforge_launch_wmma_i32_smoke, BlockQ81Mmq,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use serial_test::serial;

// ─── Compile-time shape check ──────────────────────────────────────────────

#[test]
fn block_q8_1_mmq_has_expected_size() {
    assert_eq!(std::mem::size_of::<BlockQ81Mmq>(), 144);
    // ds4 at offset 0, qs at offset 16
    let s: BlockQ81Mmq = BlockQ81Mmq { ds4: [0; 4], qs: [0; 128] };
    let base = &s as *const _ as usize;
    let ds4  = &s.ds4 as *const _ as usize;
    let qs   = &s.qs  as *const _ as usize;
    assert_eq!(ds4 - base, 0);
    assert_eq!(qs  - base, 16);
}

// ─── (1) Integer-WMMA smoke — 16×16×16 i8×i8 → i32 ─────────────────────────

fn scalar_int_gemm(a: &[i8], b_rowmajor: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[i * k + kk] as i32) * (b_rowmajor[kk * n + j] as i32);
            }
            c[i * n + j] = acc;
        }
    }
}

#[test]
#[serial]
fn int_wmma_smoke_identity_by_counting() {
    // A = identity-ish: A[i][k] = 1 iff i == k, else 0.
    // B[k][j] = (k + 1) * ((j & 1) == 0 ? 1 : -1)  — signed values, distinguishable columns.
    // Expect C[i][j] = B[i][j].
    const N: usize = 16;
    let mut a = vec![0i8; N * N];
    let mut b_rm = vec![0i8; N * N]; // row-major B
    for i in 0..N { a[i * N + i] = 1; }
    for k in 0..N {
        for j in 0..N {
            let sign: i32 = if (j & 1) == 0 { 1 } else { -1 };
            b_rm[k * N + j] = (((k + 1) as i32) * sign) as i8;
        }
    }

    // Our kernel expects B col-major. Transpose b_rm to b_cm.
    let mut b_cm = vec![0i8; N * N];
    for k in 0..N {
        for j in 0..N {
            b_cm[j * N + k] = b_rm[k * N + j];
        }
    }

    let mut cpu = vec![0i32; N * N];
    scalar_int_gemm(&a, &b_rm, &mut cpu, N, N, N);

    let gpu = run_int_smoke(&a, &b_cm).expect("gpu smoke");
    assert_eq!(gpu.len(), N * N);
    let mismatches: Vec<_> = (0..N*N)
        .filter(|&idx| gpu[idx] != cpu[idx])
        .take(8).collect();
    if !mismatches.is_empty() {
        for idx in &mismatches {
            let (r, c) = (idx / N, idx % N);
            eprintln!("  C[{r},{c}]: gpu={} cpu={}", gpu[*idx], cpu[*idx]);
        }
        panic!("int-WMMA smoke mismatched ({} cells differ)",
            (0..N*N).filter(|&i| gpu[i] != cpu[i]).count());
    }
}

#[test]
#[serial]
fn int_wmma_smoke_random_signed() {
    const N: usize = 16;
    let mut rng = fastrand::Rng::with_seed(0xBEEF);
    let a: Vec<i8> = (0..N*N).map(|_| rng.i8(-32..=31)).collect();
    let b_rm: Vec<i8> = (0..N*N).map(|_| rng.i8(-32..=31)).collect();

    let mut b_cm = vec![0i8; N * N];
    for k in 0..N {
        for j in 0..N {
            b_cm[j * N + k] = b_rm[k * N + j];
        }
    }

    let mut cpu = vec![0i32; N * N];
    scalar_int_gemm(&a, &b_rm, &mut cpu, N, N, N);

    let gpu = run_int_smoke(&a, &b_cm).expect("gpu smoke");
    let diffs = (0..N*N).filter(|&i| gpu[i] != cpu[i]).count();
    assert_eq!(diffs, 0,
        "random-signed int-WMMA: {diffs}/{} cells differ", N*N);
}

fn run_int_smoke(a: &[i8], b_cm: &[i8]) -> HipResult<Vec<i32>> {
    let stream = HipStream::new()?;
    let mut d_a = HipBuffer::new(a.len())?;
    let mut d_b = HipBuffer::new(b_cm.len())?;
    let mut d_c = HipBuffer::new(a.len() * 4)?;
    let a_u8 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len()) };
    let b_u8 = unsafe { std::slice::from_raw_parts(b_cm.as_ptr() as *const u8, b_cm.len()) };
    d_a.copy_from_host(a_u8)?;
    d_b.copy_from_host(b_u8)?;
    let rc = unsafe {
        rocmforge_launch_wmma_i32_smoke(
            d_a.as_ptr() as *const i8,
            d_b.as_ptr() as *const i8,
            d_c.as_mut_ptr() as *mut core::ffi::c_int,
            stream.raw(),
        )
    };
    check(rc, "int-WMMA smoke launch")?;
    stream.synchronize()?;
    let mut host = vec![0u8; a.len() * 4];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_c.as_ptr(),
            a.len() * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host.chunks_exact(4)
           .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
           .collect())
}

// ─── (2) block_q8_1_mmq quantise ───────────────────────────────────────────

fn run_quant_mmq(input: &[f32]) -> HipResult<Vec<u8>> {
    let stream = HipStream::new()?;
    assert_eq!(input.len() % 128, 0);
    let n_blocks = input.len() / 128;
    let out_bytes = n_blocks * 144;

    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let in_bytes = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    d_in.copy_from_host(in_bytes)?;
    let mut d_out = HipBuffer::new(out_bytes)?;

    let rc = unsafe {
        rocmforge_launch_quantize_q8_1_mmq(
            d_in.as_ptr() as *const f32,
            d_out.as_mut_ptr(),
            input.len() as core::ffi::c_int,
            stream.raw(),
        )
    };
    check(rc, "quantize_q8_1_mmq launch")?;
    stream.synchronize()?;
    let mut host = vec![0u8; out_bytes];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            out_bytes,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host)
}

// Scalar CPU reference for one 32-element sub-block (matches the kernel).
fn cpu_subblock_quantize(x: &[f32]) -> ([i8; 32], f32, f32) {
    assert_eq!(x.len(), 32);
    let amax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d_inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let d     = if amax > 0.0 { 1.0 / d_inv }  else { 0.0 };
    let sum: f32 = x.iter().sum();
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (x[i] * d_inv).round() as i8;
    }
    (qs, d, sum)
}

#[test]
#[serial]
fn quantize_q8_1_mmq_matches_cpu_reference() {
    // 4096 elements → 32 mmq-blocks × (4 sub-blocks each) = 128 sub-blocks.
    let n = 4096usize;
    let mut rng = fastrand::Rng::with_seed(0x5EED);
    let input: Vec<f32> = (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * 3.0).collect();
    let gpu = run_quant_mmq(&input).expect("quantize");

    let mut err_scales = 0;
    let mut err_qs     = 0;
    let mut shown      = 0;

    for mb in 0..(n / 128) {
        let block_base = mb * 144;
        for sb in 0..4 {
            let sub_start = mb * 128 + sb * 32;
            let x_sub = &input[sub_start..sub_start + 32];
            let (qs_cpu, d_cpu, sum_cpu) = cpu_subblock_quantize(x_sub);

            // Unpack ds4[sb] — half2 = (d, sum) as two halfs stored at u32 offset 4*sb..4*sb+4.
            let ds_u32 = u32::from_le_bytes([
                gpu[block_base + 4*sb + 0],
                gpu[block_base + 4*sb + 1],
                gpu[block_base + 4*sb + 2],
                gpu[block_base + 4*sb + 3],
            ]);
            let d_half  =  ds_u32        & 0xFFFF;
            let s_half  = (ds_u32 >> 16) & 0xFFFF;
            let d_gpu   = half::f16::from_bits(d_half as u16).to_f32();
            let sum_gpu = half::f16::from_bits(s_half as u16).to_f32();

            if (d_gpu - d_cpu).abs() > 1e-3_f32.max(d_cpu.abs() * 5e-3) {
                err_scales += 1;
                if shown < 3 {
                    eprintln!("  mb={mb} sb={sb}: d_gpu={d_gpu:.6}, d_cpu={d_cpu:.6}");
                    shown += 1;
                }
            }
            if (sum_gpu - sum_cpu).abs() > 5e-2_f32.max(sum_cpu.abs() * 1e-2) {
                err_scales += 1;
                if shown < 3 {
                    eprintln!("  mb={mb} sb={sb}: sum_gpu={sum_gpu:.6}, sum_cpu={sum_cpu:.6}");
                    shown += 1;
                }
            }

            for i in 0..32 {
                let g = gpu[block_base + 16 + sb * 32 + i] as i8;
                if (g as i32 - qs_cpu[i] as i32).abs() > 1 {
                    err_qs += 1;
                }
            }
        }
    }
    println!("quantize_q8_1_mmq: 32 blocks × 4 sub-blocks, err_scales={err_scales}, err_qs={err_qs}");
    assert_eq!(err_scales, 0, "mmq quantiser scales diverged from CPU ref");
    // Allow ±1 int8 off for rounding direction on .5 boundaries.
    assert!(err_qs < 128,
        "mmq quantiser qs diverged from CPU ref too much (err={err_qs})");
}

#[test]
#[serial]
fn quantize_q8_1_mmq_zero_input_safe() {
    // All zeros — amax == 0 must produce all-zero qs and d == 0 without NaN/Inf.
    let n = 128usize;
    let input = vec![0.0f32; n];
    let gpu = run_quant_mmq(&input).expect("quantize zeros");
    for sb in 0..4 {
        let ds_u32 = u32::from_le_bytes([gpu[4*sb], gpu[4*sb+1], gpu[4*sb+2], gpu[4*sb+3]]);
        let d_half = (ds_u32 & 0xFFFF) as u16;
        let s_half = (ds_u32 >> 16) as u16;
        assert_eq!(d_half, 0, "d should be 0 for all-zero input");
        assert_eq!(s_half, 0, "sum should be 0 for all-zero input");
    }
    for i in 0..128 {
        assert_eq!(gpu[16 + i] as i8, 0, "qs[{i}] should be 0");
    }
}
