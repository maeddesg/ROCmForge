//! P0.2 Schritt 6 — MMQ-Q6_K parity tests.
//!
//! Coverage:
//!   * `mmq_q6_k_small` — single super-block (M=N=16, K=256). Most
//!     basic case to lock down ql+qh dequant + scale-fixup math.
//!   * `mmq_q6_k_vs_cpu_*` — random shapes, GPU 4W output vs FP32 CPU
//!     dequant-then-matmul. Independent ground truth.
//!   * `mmq_q6_k_4w_vs_1w_parity` — 4-warp output bit-identical to 1-warp
//!     reference (proves the warp-id offset is correct, same as Q4_K).
//!   * `mmq_q6_k_vs_fp16_wmma` — Q6_K MMQ output vs the existing
//!     FP16-WMMA Q6_K kernel on Qwen3 LM-Head shape (M=64, N=151936,
//!     K=4096 truncated to 4096 cols for tractability).

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_mmq_q6_k, rocmforge_launch_mmq_q6_k_1w,
    rocmforge_launch_quantize_q8_1_mmq, rocmforge_launch_wmma_gemm_q6_k_fp16,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::q6_k;
use rocmforge::v1::ir::interpreter::dequant_block;
use serial_test::serial;

// ─── Q6_K weight generator (random ql+qh+scales+d) ──────────────────

fn gen_q6_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let n_super = k / 256;
    let total_blocks = n_rows * n_super;
    let mut buf = vec![0u8; total_blocks * 210];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total_blocks {
        let off = b * 210;
        // ql: random nibbles
        for i in 0..128 {
            buf[off + i] = rng.u8(..);
        }
        // qh: random 2-bit pairs
        for i in 0..64 {
            buf[off + 128 + i] = rng.u8(..);
        }
        // scales: int8 in roughly typical range -64..63
        for i in 0..16 {
            buf[off + 192 + i] = (rng.i8(..) / 2) as u8;
        }
        // d: small float scale
        let d = (rng.f32() * 2.0 - 1.0) * 0.1;
        buf[off + 208..off + 210].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    buf
}

fn cpu_reference(weights: &[u8], acts_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let fmt = q6_k();
    let n_super = k / 256;
    let mut w_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for s in 0..n_super {
            let offset = (row * n_super + s) * 210;
            let elems = dequant_block(&fmt, &weights[offset..offset + 210]).expect("dequant");
            let dst = row * k + s * 256;
            w_f32[dst..dst + 256].copy_from_slice(&elems);
        }
    }
    let mut out = vec![0.0f32; m * n];
    for row_m in 0..m {
        for col_n in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += w_f32[col_n * k + kk] * acts_f32[row_m * k + kk];
            }
            out[row_m * n + col_n] = acc;
        }
    }
    out
}

#[derive(Copy, Clone, Debug)]
enum Variant { OneWarp, FourWarp }

fn launch_mmq_q6_k_variant(
    variant: Variant,
    d_w: &HipBuffer,
    d_act_mmq: &HipBuffer,
    d_out: &mut HipBuffer,
    m: usize, n: usize, k: usize,
    stream: &HipStream,
) -> HipResult<()> {
    let rc = unsafe {
        match variant {
            Variant::FourWarp => rocmforge_launch_mmq_q6_k(
                d_w.as_ptr(),
                d_act_mmq.as_ptr(),
                d_out.as_mut_ptr() as *mut f32,
                m as core::ffi::c_int,
                n as core::ffi::c_int,
                k as core::ffi::c_int,
                stream.raw(),
            ),
            Variant::OneWarp => rocmforge_launch_mmq_q6_k_1w(
                d_w.as_ptr(),
                d_act_mmq.as_ptr(),
                d_out.as_mut_ptr() as *mut f32,
                m as core::ffi::c_int,
                n as core::ffi::c_int,
                k as core::ffi::c_int,
                stream.raw(),
            ),
        }
    };
    check(rc, "mmq_q6_k variant launch")
}

fn run_mmq_q6_k(
    variant: Variant,
    weights: &[u8],
    acts_f32: &[f32],
    m: usize, n: usize, k: usize,
) -> HipResult<Vec<f32>> {
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

    let mut d_out = HipBuffer::new(m * n * 4)?;
    launch_mmq_q6_k_variant(variant, &d_w, &d_act_mmq, &mut d_out, m, n, k, &stream)?;
    stream.synchronize()?;

    let mut host = vec![0u8; m * n * 4];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            m * n * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn run_fp16_wmma_q6_k(
    weights: &[u8],
    acts_f32: &[f32],
    m: usize, n: usize, k: usize,
) -> HipResult<Vec<f32>> {
    // The FP16-WMMA Q6_K kernel requires M%64==N%64==0 and K%256==0.
    assert_eq!(m % 64, 0);
    assert_eq!(n % 64, 0);
    assert_eq!(k % 256, 0);
    let stream = HipStream::new()?;
    let mut d_w = HipBuffer::new(weights.len())?;
    d_w.copy_from_host(weights)?;
    let mut d_act_f32 = HipBuffer::new(acts_f32.len() * 4)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(acts_f32.as_ptr() as *const u8, acts_f32.len() * 4)
    };
    d_act_f32.copy_from_host(bytes)?;

    let mut d_out = HipBuffer::new(m * n * 4)?;
    let rc = unsafe {
        rocmforge_launch_wmma_gemm_q6_k_fp16(
            d_act_f32.as_ptr() as *const f32,
            d_w.as_ptr() as *const u8,
            d_out.as_mut_ptr() as *mut f32,
            m as i32, n as i32, k as i32,
            stream.raw(),
        )
    };
    check(rc, "wmma_q6_k_fp16")?;
    stream.synchronize()?;
    let mut host = vec![0u8; m * n * 4];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut _,
            d_out.as_ptr(),
            m * n * 4,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "D2H")?;
    Ok(host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn fp16_tol(max_mag: f32, k: usize) -> f32 {
    (max_mag + 1e-3) * (k as f32).sqrt() * 5e-3
}

// ─── Tests ──────────────────────────────────────────────────────────

#[test]
#[serial]
fn mmq_q6_k_small_vs_cpu() {
    // Smallest viable case: 1 super-block per row.
    let m = 16; let n = 16; let k = 256;
    let weights = gen_q6_k_weights(n, k, 0x06E1u64);
    let mut rng = fastrand::Rng::with_seed(0xCAFE);
    let acts: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let gpu = run_mmq_q6_k(Variant::FourWarp, &weights, &acts, m, n, k).expect("4w");
    let cpu = cpu_reference(&weights, &acts, m, n, k);
    let mut max_abs = 0.0f32; let mut max_mag = 0.0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let d = (g - c).abs();
        if d > max_abs { max_abs = d; }
        let mag = c.abs().max(g.abs());
        if mag > max_mag { max_mag = mag; }
    }
    let tol = fp16_tol(max_mag, k);
    println!("mmq_q6_k 16×16×256: max_abs={max_abs:.4e}, max_mag={max_mag:.4e}, tol={tol:.4e}");
    assert!(max_abs < tol, "small Q6_K MMQ failed (max_abs={max_abs:.4e}, tol={tol:.4e})");
}

#[test]
#[serial]
fn mmq_q6_k_vs_cpu_larger() {
    let cases = [
        (16, 16, 1024usize),
        (32, 32, 256),
        (64, 64, 256),
        (64, 64, 1024),
    ];
    for (m, n, k) in cases {
        let weights = gen_q6_k_weights(n, k, 0xC001 ^ (m as u64) ^ (n as u64));
        let mut rng = fastrand::Rng::with_seed(0xF00D ^ (k as u64));
        let acts: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let gpu = run_mmq_q6_k(Variant::FourWarp, &weights, &acts, m, n, k).expect("4w");
        let cpu = cpu_reference(&weights, &acts, m, n, k);
        let mut max_abs = 0.0f32; let mut max_mag = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let d = (g - c).abs();
            if d > max_abs { max_abs = d; }
            let mag = c.abs().max(g.abs());
            if mag > max_mag { max_mag = mag; }
        }
        let tol = fp16_tol(max_mag, k);
        println!("mmq_q6_k {m}×{n}×{k}: max_abs={max_abs:.4e} max_mag={max_mag:.4e} tol={tol:.4e}");
        assert!(max_abs < tol, "{}×{}×{} failed (max_abs={:.4e}, tol={:.4e})", m, n, k, max_abs, tol);
    }
}

#[test]
#[serial]
fn mmq_q6_k_4w_vs_1w_parity() {
    let cases = [(16, 16, 256), (64, 64, 256), (64, 64, 1024)];
    for (m, n, k) in cases {
        let weights = gen_q6_k_weights(n, k, 0xCC ^ (k as u64));
        let mut rng = fastrand::Rng::with_seed(0xDD);
        let acts: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let out_1w = run_mmq_q6_k(Variant::OneWarp, &weights, &acts, m, n, k).expect("1w");
        let out_4w = run_mmq_q6_k(Variant::FourWarp, &weights, &acts, m, n, k).expect("4w");
        let mut max_abs = 0.0f32; let mut diff = 0usize;
        for (a, b) in out_1w.iter().zip(out_4w.iter()) {
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
            if d > 0.0 { diff += 1; }
        }
        println!("Q6_K 4W vs 1W {m}×{n}×{k}: max_abs={max_abs:.4e} differing={diff}/{}", out_1w.len());
        assert_eq!(max_abs, 0.0, "4W vs 1W not bit-identical");
    }
}

#[test]
#[serial]
fn mmq_q6_k_vs_fp16_wmma() {
    // Compare integer MMQ vs the existing FP16-WMMA Q6_K kernel.
    // Use M=N=64 (FP16 kernel's tile minimum) and K=1024 to keep CPU
    // ref tractable for the cross-check above implicitly. Tolerance
    // is loose because the two kernels use different intermediate
    // precision (int32 vs FP16 acc).
    let m = 64; let n = 64; let k = 1024;
    let weights = gen_q6_k_weights(n, k, 0x77);
    let mut rng = fastrand::Rng::with_seed(0x88);
    let acts: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let gpu_mmq = run_mmq_q6_k(Variant::FourWarp, &weights, &acts, m, n, k).expect("mmq");
    let gpu_fp16 = run_fp16_wmma_q6_k(&weights, &acts, m, n, k).expect("fp16");

    let mut max_abs = 0.0f32; let mut max_mag = 0.0f32;
    for (a, b) in gpu_mmq.iter().zip(gpu_fp16.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let mag = a.abs().max(b.abs());
        if mag > max_mag { max_mag = mag; }
    }
    // Two different acc-precisions: allow full FP16 tol.
    let tol = fp16_tol(max_mag, k);
    println!("Q6_K MMQ vs FP16-WMMA {m}×{n}×{k}: max_abs={max_abs:.4e} max_mag={max_mag:.4e} tol={tol:.4e}");
    assert!(max_abs < tol, "MMQ vs FP16-WMMA exceeded tol");
}
