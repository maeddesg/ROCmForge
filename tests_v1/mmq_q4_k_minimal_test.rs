//! P0.2 Schritt 2 — minimal MMQ-Q4_K kernel parity test.
//!
//! Only shape: M=N=16, K=256 (one Q4_K super-block per A-row, one
//! Q8_1_mmq block per B-row). If this passes bit-exact (mod FP16/int
//! rounding) vs a scalar CPU reference doing the same dequantised
//! GEMM, the integer-WMMA + scale-fixup core path is proven correct
//! and scaling to multi-tile MMQ becomes mechanical.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_mmq_q4_k_minimal, rocmforge_launch_quantize_q8_1_mmq,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::q4_k;
use rocmforge::v1::ir::interpreter::dequant_block;
use serial_test::serial;

// Random Q4_K weights — identical generator to wmma_test.rs.
fn gen_q4_k_weights(n_rows: usize, seed: u64) -> Vec<u8> {
    // 1 super-block per row (K=256).
    let total_blocks = n_rows;
    let mut buf = vec![0u8; total_blocks * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
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

// CPU reference: dequantise Q4_K rows, then do 16×16×256 GEMM with FP32 inputs.
fn cpu_reference(weights: &[u8], acts_f32: &[f32]) -> Vec<f32> {
    let fmt = q4_k();
    let mut w_f32 = vec![0.0f32; 16 * 256];
    for n in 0..16 {
        let elems = dequant_block(&fmt, &weights[n * 144..n * 144 + 144]).expect("dequant");
        w_f32[n * 256..n * 256 + 256].copy_from_slice(&elems);
    }
    // C[n,m] = Σ_k w[n,k] * act[m,k]
    let mut out = vec![0.0f32; 16 * 16];
    for n in 0..16 {
        for m in 0..16 {
            let mut acc = 0.0f32;
            for k in 0..256 {
                acc += w_f32[n * 256 + k] * acts_f32[m * 256 + k];
            }
            out[n * 16 + m] = acc;
        }
    }
    out
}

fn run_kernel(weights: &[u8], acts_f32: &[f32]) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;

    // Upload weights.
    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;

    // Quantise activations on GPU via the existing mmq quantiser (Schritt 1).
    let mut d_act_f32 = HipBuffer::new(16 * 256 * 4)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(acts_f32.as_ptr() as *const u8, acts_f32.len() * 4)
    };
    d_act_f32.copy_from_host(bytes)?;
    let mut d_act_mmq = HipBuffer::new(16 * 144)?;
    let rc = unsafe {
        rocmforge_launch_quantize_q8_1_mmq(
            d_act_f32.as_ptr() as *const f32,
            d_act_mmq.as_mut_ptr(),
            16 * 256,
            stream.raw(),
        )
    };
    check(rc, "quantize_q8_1_mmq")?;

    let mut d_out = HipBuffer::new(16 * 16 * 4)?;
    let rc = unsafe {
        rocmforge_launch_mmq_q4_k_minimal(
            d_w.as_ptr(),
            d_act_mmq.as_ptr(),
            d_out.as_mut_ptr() as *mut f32,
            stream.raw(),
        )
    };
    check(rc, "mmq_q4_k_minimal")?;
    stream.synchronize()?;

    let mut host = vec![0u8; 16 * 16 * 4];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            16 * 16 * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
#[serial]
fn mmq_q4_k_minimal_matches_cpu_reference() {
    let weights = gen_q4_k_weights(16, 0xBEEF);
    let mut rng = fastrand::Rng::with_seed(0xCAFE);
    let acts: Vec<f32> = (0..16 * 256).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let gpu = run_kernel(&weights, &acts).expect("gpu run");
    let cpu = cpu_reference(&weights, &acts);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_mag = 0.0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let d = (g - c).abs();
        if d > max_abs { max_abs = d; }
        let mag = c.abs().max(g.abs());
        if mag > max_mag { max_mag = mag; }
        if mag > 1e-6 {
            let r = d / mag;
            if r > max_rel { max_rel = r; }
        }
    }
    println!("MMQ-Q4_K minimal @ 16×16×256: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}, max_mag={max_mag:.4e}");

    // Activations ∈ [-1, 1], weights dequantise to ~[-7.5, 7.5]. Output
    // magnitudes can reach ~60-120. Tolerance: integer quantisation of
    // activations (±1/127 ≈ 0.8%) combined with FP16 scales contributes
    // a relative error of a few percent. Allow up to 5%.
    assert!(max_rel < 0.05,
        "MMQ-Q4_K minimal parity failed: max_rel={max_rel:.4e} (tol 5%). \
         First 8 GPU/CPU pairs: {:?}",
        gpu.iter().zip(cpu.iter()).take(8).collect::<Vec<_>>());
}
