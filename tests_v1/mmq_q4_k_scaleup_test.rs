//! P0.2 Schritt 3 — scaled MMQ-Q4_K parity.
//!
//! Tests the variable-(M,N,K) kernel on progressively larger shapes,
//! comparing against a FP32 CPU reference (dequant-then-matmul).

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_mmq_q4_k, rocmforge_launch_quantize_q8_1_mmq,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::q4_k;
use rocmforge::v1::ir::interpreter::dequant_block;
use serial_test::serial;

fn gen_q4_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let n_super = k / 256;
    let total = n_rows * n_super;
    let mut buf = vec![0u8; total * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
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

fn cpu_reference(weights: &[u8], acts_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let fmt = q4_k();
    let n_super = k / 256;
    let mut w_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for s in 0..n_super {
            let offset = (row * n_super + s) * 144;
            let elems = dequant_block(&fmt, &weights[offset..offset + 144]).expect("dequant");
            let dst = row * k + s * 256;
            w_f32[dst..dst + 256].copy_from_slice(&elems);
        }
    }
    // [N × M] row-major: out[n * M + m] = Σ_k w[n,k] * act[m,k].
    let mut out = vec![0.0f32; n * m];
    for row_n in 0..n {
        for col_m in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += w_f32[row_n * k + kk] * acts_f32[col_m * k + kk];
            }
            out[row_n * m + col_m] = acc;
        }
    }
    out
}

fn run_scaled(weights: &[u8], acts_f32: &[f32], m: usize, n: usize, k: usize) -> HipResult<Vec<f32>> {
    assert_eq!(m % 16, 0);
    assert_eq!(n % 16, 0);
    assert_eq!(k % 256, 0);
    let stream = HipStream::new()?;

    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    let mut d_act_f32 = HipBuffer::new(acts_f32.len() * 4)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(acts_f32.as_ptr() as *const u8, acts_f32.len() * 4)
    };
    d_act_f32.copy_from_host(bytes)?;
    let n_mmq_blocks = (m * k) / 128;
    let mut d_act_mmq = HipBuffer::new(n_mmq_blocks * 144)?;
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1_mmq(
            d_act_f32.as_ptr() as *const f32,
            d_act_mmq.as_mut_ptr(),
            (m * k) as core::ffi::c_int,
            stream.raw(),
        )
    };
    check(rc, "quantize_q8_1_mmq")?;

    let mut d_out = HipBuffer::new(n * m * 4)?;
    let rc = unsafe {
        rocmforge_launch_mmq_q4_k(
            d_w.as_ptr(),
            d_act_mmq.as_ptr(),
            d_out.as_mut_ptr() as *mut f32,
            m as core::ffi::c_int,
            n as core::ffi::c_int,
            k as core::ffi::c_int,
            stream.raw(),
        )
    };
    check(rc, "mmq_q4_k scaled")?;
    stream.synchronize()?;

    let mut host = vec![0u8; n * m * 4];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            n * m * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn check_parity(label: &str, m: usize, n: usize, k: usize, seed_w: u64, seed_a: u64) {
    let weights = gen_q4_k_weights(n, k, seed_w);
    let mut rng = fastrand::Rng::with_seed(seed_a);
    let acts: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let gpu = run_scaled(&weights, &acts, m, n, k).expect("gpu run");
    let cpu = cpu_reference(&weights, &acts, m, n, k);
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let d = (g - c).abs();
        if d > max_abs { max_abs = d; }
        let mag = c.abs().max(g.abs());
        if mag > max_mag { max_mag = mag; }
    }
    let tol = (max_mag + 1e-3) * (k as f32).sqrt() * 5e-3;
    println!("{label:20} M={m:5} N={n:5} K={k:5}: max_abs={max_abs:.4e}, max_mag={max_mag:.4e}, tol={tol:.4e}");
    assert!(max_abs < tol,
        "{label} parity failed (max_abs={max_abs:.4e}, tol={tol:.4e})");
}

#[test]
#[serial]
fn scaled_16x16_k256_matches_minimal_shape() {
    // Same shape as the minimal-kernel test to verify scale-up didn't regress.
    check_parity("16×16×256", 16, 16, 256, 0xBEEF, 0xCAFE);
}

#[test]
#[serial]
fn scaled_k_variable_single_tile() {
    // Variable K via outer loop over super-blocks.
    check_parity("16×16×512", 16, 16, 512, 0xC0DE, 0xF00D);
    check_parity("16×16×1024", 16, 16, 1024, 0x5EED, 0xF1FA);
    check_parity("16×16×4096", 16, 16, 4096, 0x1234, 0xABCD);
}

#[test]
#[serial]
fn scaled_multi_tile_small() {
    // Grid parallelism: multiple output tiles.
    check_parity("32×32×256", 32, 32, 256, 0x11, 0x22);
    check_parity("32×64×512", 32, 64, 512, 0x33, 0x44);
    check_parity("48×16×256", 48, 16, 256, 0x55, 0x66);
}

#[test]
#[serial]
fn scaled_qwen3_qkv_shape() {
    // M=64, N=4096, K=4096 — realistic Qwen3 QKV dimensions but small M
    // to keep the CPU reference tractable.
    check_parity("64×4096×4096", 64, 4096, 4096, 0xAAAA, 0xBBBB);
}
