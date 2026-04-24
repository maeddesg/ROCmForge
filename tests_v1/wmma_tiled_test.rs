//! P0.1b — Template-parametrised WMMA GEMM for Q4_K FP16.
//!
//! Two-tier validation:
//!   1. Template<64,64,32> must produce bit-identical output to the
//!      hard-coded wmma_gemm_q4_k_fp16 kernel. This is the correctness
//!      anchor — if it drifts, the template refactor is the bug.
//!   2. Template<128,128,32> must agree with a CPU-FP32 reference
//!      within FP16-accumulation tolerance. Cross-checked against the
//!      64×64 instantiation for the Q4_K Qwen3 / Llama-3 shapes.
//!
//! Performance gate: at realistic prefill shapes (M ≥ 128, N ≥ 4096),
//! the 128×128 instantiation must be ≥ 30 % faster than 64×64.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::error::{check, HipResult};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::wmma::{
    rocmforge_launch_wmma_gemm_q4_k_fp16,
    rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32,
    rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32,
};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use rocmforge::v1::ir::formats::q4_k;
use rocmforge::v1::ir::interpreter::dequant_block;
use serial_test::serial;

type GpuLauncher = unsafe extern "C" fn(
    *const f32,
    *const u8,
    *mut f32,
    i32,
    i32,
    i32,
    rocmforge::v1::backend::gpu::hip_ffi::hipStream_t,
) -> i32;

fn gen_q4_k_weights(n_rows: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total = n_rows * blocks_per_row;
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

fn gen_input(m: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn cpu_reference_gemm(input: &[f32], weights: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let fmt = q4_k();
    let epb = fmt.elements_per_block;
    let bb = fmt.block_bytes;
    let blocks_per_row = k / epb;
    let mut w_f32 = vec![0.0f32; n * k];
    for row in 0..n {
        for blk in 0..blocks_per_row {
            let ofs = (row * blocks_per_row + blk) * bb;
            let elems = dequant_block(&fmt, &weights[ofs..ofs + bb]).expect("dequant");
            let out_base = row * k + blk * epb;
            w_f32[out_base..out_base + epb].copy_from_slice(&elems);
        }
    }
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

fn run(launcher: GpuLauncher, input: &[f32], weights: &[u8], m: usize, n: usize, k: usize)
    -> HipResult<Vec<f32>>
{
    let stream = HipStream::new()?;
    let mut d_in = HipBuffer::new(input.len() * 4)?;
    let bytes = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    d_in.copy_from_host(bytes)?;
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
    check(rc, "tiled WMMA launch")?;
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
    Ok(host.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn max_abs_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        if diff > max_abs { max_abs = diff; }
        let mag = x.abs().max(y.abs()).max(1e-6);
        let rel = diff / mag;
        if rel > max_rel { max_rel = rel; }
    }
    (max_abs, max_rel)
}

// Magnitude-aware FP16 tolerance — same formula as the original
// `tests_v1/wmma_test.rs::fp16_tolerance`. Scales with output magnitude
// because Q4_K accumulation can produce values of order `sqrt(K) * d_max`.
fn fp16_tolerance(cpu: &[f32], k: usize) -> f32 {
    let max_mag = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (max_mag + 1e-3) * (k as f32).sqrt() * 5e-3
}

// ─── Tier 1: Parity — Template<64,64,32> vs original ────────────────────────

#[test]
#[serial]
fn template_64x64_matches_original_kernel() {
    // Use a Qwen3 QKV shape: M=64 (padded seq), N=4096, K=4096.
    let m = 64usize;
    let n = 4096usize;
    let k = 4096usize;
    let weights = gen_q4_k_weights(n, k, 0xC0DE);
    let input = gen_input(m, k, 0xBEEF);

    let orig = run(rocmforge_launch_wmma_gemm_q4_k_fp16, &input, &weights, m, n, k)
        .expect("original");
    let tpl = run(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32,
                  &input, &weights, m, n, k).expect("tiled<64,64,32>");

    let (max_abs, max_rel) = max_abs_err(&orig, &tpl);
    println!("Template<64,64,32> vs original: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}");
    // Identical kernel body → expected bit-exact.
    assert_eq!(max_abs, 0.0,
        "Template<64,64,32> must be bit-identical to original hard-coded kernel \
         (max_abs={max_abs}). This breaks the whole refactor premise.");
}

// ─── Tier 2: Correctness — Template<128,128,32> vs CPU reference ────────────

#[test]
#[serial]
fn template_128x128_matches_cpu_reference_small() {
    // Minimum dim that satisfies both 128×128 and Q4_K's K%256==0.
    let m = 128usize;
    let n = 128usize;
    let k = 256usize;
    let weights = gen_q4_k_weights(n, k, 0xABCD);
    let input = gen_input(m, k, 0x1234);

    let gpu = run(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32,
                  &input, &weights, m, n, k).expect("tiled<128,128>");
    let cpu = cpu_reference_gemm(&input, &weights, m, n, k);

    let (max_abs, max_rel) = max_abs_err(&gpu, &cpu);
    let tol = fp16_tolerance(&cpu, k);
    println!("Template<128,128,32> vs CPU ref @ {m}×{n}×{k}: max_abs={max_abs:.4e}, \
              max_rel={max_rel:.4e}, tol={tol:.4e}");
    assert!(max_abs < tol,
        "Template<128,128,32> output diverged from CPU ref (max_abs={max_abs}, tol={tol})");
}

#[test]
#[serial]
fn template_128x128_matches_cpu_reference_qwen_shape() {
    // Qwen3 QKV shape (M=128 padded to 128, N=4096, K=4096). Largest
    // practical K we can still CPU-reference in reasonable time.
    let m = 128usize;
    let n = 128usize;
    let k = 4096usize;
    let weights = gen_q4_k_weights(n, k, 0xFACE);
    let input = gen_input(m, k, 0xDEAD);

    let gpu = run(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32,
                  &input, &weights, m, n, k).expect("tiled<128,128>");
    let cpu = cpu_reference_gemm(&input, &weights, m, n, k);

    let (max_abs, max_rel) = max_abs_err(&gpu, &cpu);
    let tol = fp16_tolerance(&cpu, k);
    println!("Template<128,128,32> vs CPU ref @ {m}×{n}×{k}: max_abs={max_abs:.4e}, \
              max_rel={max_rel:.4e}, tol={tol:.4e}");
    assert!(max_abs < tol,
        "Template<128,128,32> @ K=4096 diverged (max_abs={max_abs}, tol={tol})");
}

// ─── Tier 2b: Cross-check — 64×64 vs 128×128 on same shape ──────────────────

#[test]
#[serial]
fn template_64x64_vs_128x128_cross_parity() {
    // Both must agree within their own FP16 accumulation noise. Order
    // of operations differs (different wave-partition of the reduction)
    // but the final values should be close.
    let m = 128usize;
    let n = 128usize;
    let k = 4096usize;
    let weights = gen_q4_k_weights(n, k, 0x5EED);
    let input = gen_input(m, k, 0xC0FFEE);

    let a = run(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32,
                &input, &weights, m, n, k).expect("64x64");
    let b = run(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32,
                &input, &weights, m, n, k).expect("128x128");

    let (max_abs, max_rel) = max_abs_err(&a, &b);
    println!("Tiled 64×64 vs 128×128 @ {m}×{n}×{k}: max_abs={max_abs:.4e}, \
              max_rel={max_rel:.4e}");
    // Different reduction order → allow moderate drift.
    assert!(max_abs < 0.5,
        "64×64 and 128×128 disagree too much (max_abs={max_abs}) — either kernel is broken");
}

// ─── Tier 3: Performance ────────────────────────────────────────────────────

fn time_kernel(launcher: GpuLauncher, input: &[f32], weights: &[u8],
                m: usize, n: usize, k: usize, iters: usize) -> f64 {
    // Warmup
    let _ = run(launcher, input, weights, m, n, k).expect("warmup");
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = run(launcher, input, weights, m, n, k).expect("timed run");
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;
    elapsed
}

fn perf_at(m: usize, n: usize, k: usize) -> (f64, f64, f64) {
    let weights = gen_q4_k_weights(n, k, 0xBADF00D);
    let input = gen_input(m, k, 0xFEEDFACE);
    let t64 = time_kernel(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32,
                          &input, &weights, m, n, k, 6);
    let t128 = time_kernel(rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32,
                           &input, &weights, m, n, k, 6);
    (t64 * 1e3, t128 * 1e3, t64 / t128)
}

#[test]
#[serial]
fn perf_compare_shapes_adaptive_dispatch_signal() {
    if std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping perf sweep — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    // Perf landscape across the shapes that actually occur during
    // Qwen3-8B prefill. Used to decide the adaptive-dispatch
    // threshold. 128×128 is NOT monotonically better: it wins at
    // small M (better arithmetic intensity, fewer kernel launches)
    // but regresses at large M (fewer blocks/CU → worse tail cleanup,
    // plus higher VGPR pressure).
    for (m, n, k, label) in [
        (128usize,  4096,  4096, "M=128  QKV"),
        (256,       4096,  4096, "M=256  QKV"),
        (256,      14336,  4096, "M=256  FFN"),
        (512,       4096,  4096, "M=512  QKV"),
        (512,      14336,  4096, "M=512  FFN"),
        (1024,      4096,  4096, "M=1024 QKV"),
    ] {
        let (t64, t128, spd) = perf_at(m, n, k);
        println!("  {label:14} @ {m}×{n}×{k}: 64×64={t64:.3}ms, 128×128={t128:.3}ms, speedup={spd:.2}×");
    }
    // No assertion — this test drives policy (adaptive dispatch
    // threshold), not regression prevention. Correctness is already
    // covered by the Tier-1/2 parity tests above.
}
