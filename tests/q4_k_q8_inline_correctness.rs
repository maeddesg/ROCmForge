#![cfg(feature = "gpu")]

//! Phase 8b Step 3 correctness tests for the Q4_K × Q8-inline decode
//! kernels. Each test generates random Q4_K weights at a real Qwen3 /
//! Llama-3.1 shape, runs the GPU kernel, and compares against a CPU
//! reference that dequantises Q4_K to FP32 and does a plain GEMV.
//!
//! Q8-inline adds one extra quantisation step (FP32 activation → Q8_0
//! with shared scale per 32 elements), so tolerances are looser than
//! a pure FP32 path but the output must still match the CPU reference
//! to within ~0.5 % of the dominant magnitude.

use half::f16;
use rocmforge::gpu::ffi::{
    hip_get_device_count, hip_memcpy_d2h, hip_memcpy_h2d, hip_stream_synchronize, hipStream_t,
};
use rocmforge::gpu::kernels::{
    gemv_gate_up_swiglu_q4_k_f32_q8_inline_on_stream, gemv_q4_k_f32_q8_inline_on_stream,
    gemv_q4_k_f32_q8_inline_residual_on_stream,
};
use rocmforge::gpu::weights::GpuBuffer;
use serial_test::serial;

const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_ELEMS_PER_BLOCK: usize = 256;

fn skip_if_no_gpu() -> bool {
    hip_get_device_count().map(|n| n <= 0).unwrap_or(true)
}

// ─── CPU reference dequant (matches hip_kernels/quant/q4_k_gemv.hip) ────

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

fn dequant_q4_k_block(block: &[u8]) -> [f32; 256] {
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = [0.0f32; 256];
    for j in 0..8 {
        let (sc, mn) = get_scale_min_k4(j, scales);
        let d_scale = d * sc as f32;
        let d_min = dmin * mn as f32;
        let pair_base = (j >> 1) * 32;
        let is_upper = (j & 1) != 0;
        for i in 0..32 {
            let byte = qs[pair_base + i];
            let nib = if is_upper { byte >> 4 } else { byte & 0x0F };
            out[j * 32 + i] = d_scale * nib as f32 - d_min;
        }
    }
    out
}

fn gen_q4_k_weights(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % Q4_K_ELEMS_PER_BLOCK == 0);
    let blocks_per_row = k / Q4_K_ELEMS_PER_BLOCK;
    let mut buf = vec![0u8; n * blocks_per_row * Q4_K_BLOCK_BYTES];
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for block_idx in 0..(n * blocks_per_row) {
        let base = block_idx * Q4_K_BLOCK_BYTES;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let d = f16::from_f32(0.001 + 0.010 * ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dmin =
            f16::from_f32(0.001 + 0.010 * ((state >> 33) as u32 & 0xFFFF) as f32 / 65535.0);
        let db = d.to_bits().to_le_bytes();
        let mb = dmin.to_bits().to_le_bytes();
        buf[base] = db[0];
        buf[base + 1] = db[1];
        buf[base + 2] = mb[0];
        buf[base + 3] = mb[1];
        for i in 0..12 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 4 + i] = (state >> 33) as u8;
        }
        for i in 0..128 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf[base + 16 + i] = (state >> 33) as u8;
        }
    }
    buf
}

fn dequant_q4_k_matrix(buf: &[u8], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / Q4_K_ELEMS_PER_BLOCK;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for b in 0..blocks_per_row {
            let base = (row * blocks_per_row + b) * Q4_K_BLOCK_BYTES;
            let block = &buf[base..base + Q4_K_BLOCK_BYTES];
            let decoded = dequant_q4_k_block(block);
            for i in 0..Q4_K_ELEMS_PER_BLOCK {
                out[row * k + b * Q4_K_ELEMS_PER_BLOCK + i] = decoded[i];
            }
        }
    }
    out
}

fn seeded_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 0.5
        })
        .collect()
}

fn cpu_gemv(a: &[f32], w_nk: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|col| {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[i] * w_nk[col * k + i];
            }
            acc
        })
        .collect()
}

fn rel_max_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    let mut max_mag = 0.0f32;
    for i in 0..gpu.len() {
        max_diff = max_diff.max((gpu[i] - cpu[i]).abs());
        max_mag = max_mag.max(cpu[i].abs());
    }
    (max_diff, max_mag)
}

// ─── Tests ──────────────────────────────────────────────────────────────

fn check_gemv_shape(label: &str, n: usize, k: usize, rel_tol: f32) {
    println!("== {}: N={} K={} ==", label, n, k);
    let a = seeded_floats(k, 0xAA ^ (n * 11 + k * 7) as u64);
    let w = gen_q4_k_weights(n, k, 0xCC ^ (n * 11 + k * 7) as u64);
    let w_deq = dequant_q4_k_matrix(&w, n, k);

    let d_a = GpuBuffer::alloc(k * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_o = GpuBuffer::alloc(n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, k * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }
    gemv_q4_k_f32_q8_inline_on_stream(
        d_w.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_o.as_ptr() as *mut f32,
        k,
        n,
        hipStream_t::null(),
    )
    .expect("q4_k q8_inline launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; n];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_o.as_ptr(), n * 4).unwrap();
    }
    let cpu_out = cpu_gemv(&a, &w_deq, n, k);
    let (d, mag) = rel_max_diff(&gpu_out, &cpu_out);
    let rel = d / mag.max(1e-6);
    println!("  max_abs_diff = {:.3e}, max_mag = {:.3e}, rel = {:.3e} (tol {:.0e})",
        d, mag, rel, rel_tol);
    assert!(rel < rel_tol, "{}: rel diff {:.3e} exceeds {:.0e}", label, rel, rel_tol);
}

// Q8 quantisation introduces ~0.8 % per-element relative error by design
// (max_abs(x) / 127 bucket). Accumulated over K dot-products the rel error
// against FP32 ground truth sits at ~1 % — we use 1.5e-2 as a loose but
// meaningful tolerance. A correctness regression (e.g. a sign flip or a
// dropped `dmin` term) would push the error several orders of magnitude
// higher.

#[test]
#[serial]
fn q4_k_q8_inline_gemv_qwen3_qkv_shape() {
    if skip_if_no_gpu() { return; }
    check_gemv_shape("Qwen3 QKV 4096x4096", 4096, 4096, 1.5e-2);
}

#[test]
#[serial]
fn q4_k_q8_inline_gemv_qwen3_gate_up_shape() {
    if skip_if_no_gpu() { return; }
    check_gemv_shape("Qwen3 gate/up 12288x4096", 12288, 4096, 1.5e-2);
}

#[test]
#[serial]
fn q4_k_q8_inline_gemv_qwen3_ffn_down_shape() {
    if skip_if_no_gpu() { return; }
    check_gemv_shape("Qwen3 ffn_down 4096x12288", 4096, 12288, 1.5e-2);
}

// ─── Residual variant ───────────────────────────────────────────────────

#[test]
#[serial]
fn q4_k_q8_inline_residual_qwen3_shape() {
    if skip_if_no_gpu() { return; }
    let n = 4096;
    let k = 4096;
    let a = seeded_floats(k, 0xBB);
    let r = seeded_floats(n, 0xDD);
    let w = gen_q4_k_weights(n, k, 0xEE);
    let w_deq = dequant_q4_k_matrix(&w, n, k);

    let d_a = GpuBuffer::alloc(k * 4).unwrap();
    let d_r = GpuBuffer::alloc(n * 4).unwrap();
    let d_w = GpuBuffer::alloc(w.len()).unwrap();
    let d_o = GpuBuffer::alloc(n * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, k * 4).unwrap();
        hip_memcpy_h2d(d_r.as_ptr(), r.as_ptr() as *const u8, n * 4).unwrap();
        hip_memcpy_h2d(d_w.as_ptr(), w.as_ptr() as *const u8, w.len()).unwrap();
    }
    gemv_q4_k_f32_q8_inline_residual_on_stream(
        d_w.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_r.as_ptr() as *const f32,
        d_o.as_ptr() as *mut f32,
        k,
        n,
        hipStream_t::null(),
    )
    .expect("q4_k q8_inline residual");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; n];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_o.as_ptr(), n * 4).unwrap();
    }
    let cpu_raw = cpu_gemv(&a, &w_deq, n, k);
    let cpu_out: Vec<f32> = cpu_raw.iter().zip(r.iter()).map(|(g, r)| g + r).collect();
    let (d, mag) = rel_max_diff(&gpu_out, &cpu_out);
    let rel = d / mag.max(1e-6);
    println!("residual max_abs={:.3e} mag={:.3e} rel={:.3e}", d, mag, rel);
    assert!(rel < 1.5e-2, "residual rel diff {} too high", rel);
}

// ─── Fused gate+up+SwiGLU ──────────────────────────────────────────────

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
#[serial]
fn q4_k_q8_inline_gate_up_swiglu_qwen3_shape() {
    if skip_if_no_gpu() { return; }
    let ff = 12288;
    let h = 4096;
    let a = seeded_floats(h, 0x12);
    let w_gate = gen_q4_k_weights(ff, h, 0x34);
    let w_up = gen_q4_k_weights(ff, h, 0x56);
    let wg_deq = dequant_q4_k_matrix(&w_gate, ff, h);
    let wu_deq = dequant_q4_k_matrix(&w_up, ff, h);

    let d_a = GpuBuffer::alloc(h * 4).unwrap();
    let d_g = GpuBuffer::alloc(w_gate.len()).unwrap();
    let d_u = GpuBuffer::alloc(w_up.len()).unwrap();
    let d_o = GpuBuffer::alloc(ff * 4).unwrap();
    unsafe {
        hip_memcpy_h2d(d_a.as_ptr(), a.as_ptr() as *const u8, h * 4).unwrap();
        hip_memcpy_h2d(d_g.as_ptr(), w_gate.as_ptr() as *const u8, w_gate.len()).unwrap();
        hip_memcpy_h2d(d_u.as_ptr(), w_up.as_ptr() as *const u8, w_up.len()).unwrap();
    }
    gemv_gate_up_swiglu_q4_k_f32_q8_inline_on_stream(
        d_g.as_ptr() as *const u8,
        d_u.as_ptr() as *const u8,
        d_a.as_ptr() as *const f32,
        d_o.as_ptr() as *mut f32,
        h,
        ff,
        hipStream_t::null(),
    )
    .expect("q4_k q8_inline gate_up launch");
    hip_stream_synchronize(hipStream_t::null()).unwrap();

    let mut gpu_out = vec![0.0f32; ff];
    unsafe {
        hip_memcpy_d2h(gpu_out.as_mut_ptr() as *mut u8, d_o.as_ptr(), ff * 4).unwrap();
    }
    let gate = cpu_gemv(&a, &wg_deq, ff, h);
    let up = cpu_gemv(&a, &wu_deq, ff, h);
    let cpu_out: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * u)
        .collect();
    let (d, mag) = rel_max_diff(&gpu_out, &cpu_out);
    let rel = d / mag.max(1e-6);
    println!("gate_up max_abs={:.3e} mag={:.3e} rel={:.3e}", d, mag, rel);
    assert!(rel < 1e-2, "gate_up rel diff {} too high", rel);
}
