//! Minimal debug test — compare MMVQ vs q8_inline vs CPU reference
//! on one row × one super-block, fixed weights, fixed input.
#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::backend::gpu::gemv::{
    rocmforge_launch_gemv_q4_k_mmvq, rocmforge_launch_gemv_q4_k_q8_inline,
};
use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use rocmforge::v1::backend::gpu::quantize::{rocmforge_launch_quantize_q8_1, BlockQ81};
use rocmforge::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use std::ffi::c_void;

fn gen_q4k(n: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks_per_row = k / 256;
    let total = n * blocks_per_row;
    let mut buf = vec![0u8; total * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..16 { buf[b * 144 + i] = rng.u8(..) & 0x3F; }
        for i in 16..144 { buf[b * 144 + i] = rng.u8(..); }
    }
    buf
}

fn run_both(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let w = gen_q4k(n, k, seed);
    let mut rng = fastrand::Rng::with_seed(seed.wrapping_add(1));
    let input: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let stream = HipStream::new().unwrap();
    let mut d_w = HipBuffer::new(w.len()).unwrap();
    d_w.copy_from_host(&w).unwrap();
    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    let mut d_in = HipBuffer::new(in_bytes.len()).unwrap();
    d_in.copy_from_host(in_bytes).unwrap();
    let mut d_q81 = HipBuffer::new((k / 32) * 36).unwrap();
    let mut d_out_inline = HipBuffer::new(n * 4).unwrap();
    let mut d_out_mmvq = HipBuffer::new(n * 4).unwrap();

    unsafe {
        rocmforge_launch_gemv_q4_k_q8_inline(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_inline.as_mut_ptr() as *mut f32,
            k as i32, n as i32, stream.raw());
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32, d_q81.as_mut_ptr(),
            k as i32, stream.raw());
        rocmforge_launch_gemv_q4_k_mmvq(
            d_w.as_ptr(), d_q81.as_ptr(), d_out_mmvq.as_mut_ptr(),
            n as i32, k as i32, stream.raw());
    }
    stream.synchronize().unwrap();

    let mut r_inline = vec![0f32; n];
    let mut r_mmvq = vec![0f32; n];
    unsafe {
        hipMemcpy(r_inline.as_mut_ptr() as *mut _, d_out_inline.as_ptr(), n*4, hipMemcpyDeviceToHost);
        hipMemcpy(r_mmvq.as_mut_ptr() as *mut _, d_out_mmvq.as_ptr(), n*4, hipMemcpyDeviceToHost);
    }
    (r_inline, r_mmvq)
}

fn report(label: &str, ref_: &[f32], mmvq: &[f32]) {
    let mut rels: Vec<f32> = Vec::new();
    let mut max_abs = 0.0f32;
    let mag_max = ref_.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for (a, b) in ref_.iter().zip(mmvq.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let denom = a.abs().max(b.abs()).max(mag_max * 0.001);
        rels.push(d / denom);
    }
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = rels[rels.len() / 2];
    let p99 = rels[(rels.len() * 99) / 100];
    let max = rels[rels.len() - 1];
    println!(
        "  {}: N={} |out|max={:.2}  max_abs={:.4}  rel(median={:.5} p99={:.4} max={:.4})",
        label, ref_.len(), mag_max, max_abs, median, p99, max
    );
}

#[test]
fn dbg_n1_k256()   { let (r, m) = run_both(1, 256, 0x1);    report("N=1  K=256",   &r, &m); }
#[test]
fn dbg_n1_k512()   { let (r, m) = run_both(1, 512, 0x2);    report("N=1  K=512",   &r, &m); }
#[test]
fn dbg_n1_k4096()  { let (r, m) = run_both(1, 4096, 0x3);   report("N=1  K=4096",  &r, &m); }
#[test]
fn dbg_n2_k256()   { let (r, m) = run_both(2, 256, 0x4);    report("N=2  K=256",   &r, &m); }
#[test]
fn dbg_n4_k4096()  { let (r, m) = run_both(4, 4096, 0x5);   report("N=4  K=4096",  &r, &m); }
#[test]
fn dbg_n8_k256()   { let (r, m) = run_both(8, 256, 0x6);    report("N=8  K=256",   &r, &m); }
#[test]
fn dbg_n1024_k4096() { let (r, m) = run_both(1024, 4096, 0x7); report("N=1024 K=4096", &r, &m); }

// ─── CPU FP32 reference ────────────────────────────────────────────────────

/// Dequantize one Q4_K super-block (256 floats) using the official formula.
fn dequant_q4k_block(bytes: &[u8]) -> [f32; 256] {
    assert_eq!(bytes.len(), 144);
    let d = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([bytes[2], bytes[3]])).to_f32();
    let scales = &bytes[4..16];
    let qs = &bytes[16..144];
    let mut out = [0.0f32; 256];
    for j in 0..8 {
        let (sc, mn): (u8, u8) = if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            (
                (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
            )
        };
        let pair_base = (j >> 1) * 32;
        let is_upper = (j & 1) != 0;
        for i in 0..32 {
            let byte = qs[pair_base + i];
            let nib = if is_upper { byte >> 4 } else { byte & 0x0F };
            out[j * 32 + i] = d * (sc as f32) * (nib as f32) - dmin * (mn as f32);
        }
    }
    out
}

fn cpu_gemv(weights: &[u8], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    let blocks_per_row = k / 256;
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let row_offset = row * blocks_per_row * 144;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let block_bytes = &weights[row_offset + b * 144..row_offset + (b + 1) * 144];
            let dequant = dequant_q4k_block(block_bytes);
            for i in 0..256 {
                acc += dequant[i] * input[b * 256 + i];
            }
        }
        out[row] = acc;
    }
    out
}

#[test]
fn dbg_vs_cpu_n1_k4096() {
    let w = gen_q4k(1, 4096, 0x101);
    let mut rng = fastrand::Rng::with_seed(0x102);
    let input: Vec<f32> = (0..4096).map(|_| rng.f32() * 2.0 - 1.0).collect();

    let cpu = cpu_gemv(&w, &input, 1, 4096);

    let stream = HipStream::new().unwrap();
    let mut d_w = HipBuffer::new(w.len()).unwrap();
    d_w.copy_from_host(&w).unwrap();
    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    let mut d_in = HipBuffer::new(in_bytes.len()).unwrap();
    d_in.copy_from_host(in_bytes).unwrap();
    let mut d_q81 = HipBuffer::new((4096 / 32) * 36).unwrap();
    let mut d_out_inline = HipBuffer::new(1 * 4).unwrap();
    let mut d_out_mmvq = HipBuffer::new(1 * 4).unwrap();

    unsafe {
        rocmforge_launch_gemv_q4_k_q8_inline(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_inline.as_mut_ptr() as *mut f32,
            4096, 1, stream.raw());
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32, d_q81.as_mut_ptr(),
            4096, stream.raw());
        rocmforge_launch_gemv_q4_k_mmvq(
            d_w.as_ptr(), d_q81.as_ptr(), d_out_mmvq.as_mut_ptr(),
            1, 4096, stream.raw());
    }
    stream.synchronize().unwrap();

    let mut r_inline = vec![0f32; 1];
    let mut r_mmvq = vec![0f32; 1];
    unsafe {
        hipMemcpy(r_inline.as_mut_ptr() as *mut _, d_out_inline.as_ptr(), 4, hipMemcpyDeviceToHost);
        hipMemcpy(r_mmvq.as_mut_ptr() as *mut _, d_out_mmvq.as_ptr(), 4, hipMemcpyDeviceToHost);
    }

    println!("  CPU:       {}", cpu[0]);
    println!("  q8_inline: {}  (diff from CPU: {:.5}, rel: {:.4})",
             r_inline[0], r_inline[0] - cpu[0], (r_inline[0] - cpu[0]) / cpu[0].abs());
    println!("  mmvq:      {}  (diff from CPU: {:.5}, rel: {:.4})",
             r_mmvq[0], r_mmvq[0] - cpu[0], (r_mmvq[0] - cpu[0]) / cpu[0].abs());
    println!("  inline vs mmvq diff: {:.5}", (r_inline[0] - r_mmvq[0]).abs());
}

#[test]
fn dbg_mmvq_minimal() {
    // 1 row, K=256 (exactly 1 super-block). Known weights & input.
    const N: usize = 1;
    const K: usize = 256;

    // Build one super-block: d=0.01, dmin=0.005, scales all 0x10 (=16), 
    // mins all 0x05 (=5), nibbles: first 32 bytes are pattern 0x54 ...
    let mut w = vec![0u8; 144];
    w[0..2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
    w[2..4].copy_from_slice(&f16::from_f32(0.005).to_bits().to_le_bytes());
    // Scales: use simple 6-bit values
    for i in 4..12 { w[i] = 0x10; }  // scales[0..3] = 0x10, mins[0..3] = 0x10
    for i in 8..12 { w[i] = 0x05; }  // scales[4..7] lowers / mins 4..7 upper packed — keep simple
    // Actually, leave all 12 bytes same pattern for simplicity
    for i in 4..16 { w[i] = 0x10; }
    // Nibbles: byte pattern 0x21 = low nib 1, high nib 2
    for i in 16..144 { w[i] = 0x21; }

    // Input: 256 floats all = 1.0
    let input: Vec<f32> = vec![1.0f32; K];

    // Upload and run both kernels
    let stream = HipStream::new().unwrap();
    let mut d_w = HipBuffer::new(w.len()).unwrap();
    d_w.copy_from_host(&w).unwrap();
    let in_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4)
    };
    let mut d_in = HipBuffer::new(in_bytes.len()).unwrap();
    d_in.copy_from_host(in_bytes).unwrap();
    let mut d_q81 = HipBuffer::new((K/32) * 36).unwrap();
    let mut d_out_inline = HipBuffer::new(N * 4).unwrap();
    let mut d_out_mmvq = HipBuffer::new(N * 4).unwrap();

    unsafe {
        let rc = rocmforge_launch_gemv_q4_k_q8_inline(
            d_w.as_ptr() as *const u8,
            d_in.as_ptr() as *const f32,
            d_out_inline.as_mut_ptr() as *mut f32,
            K as i32, N as i32, stream.raw());
        assert_eq!(rc, 0);
        rocmforge_launch_quantize_q8_1(
            d_in.as_ptr() as *const f32,
            d_q81.as_mut_ptr(),
            K as i32, stream.raw());
        let rc = rocmforge_launch_gemv_q4_k_mmvq(
            d_w.as_ptr(),
            d_q81.as_ptr(),
            d_out_mmvq.as_mut_ptr(),
            N as i32, K as i32, stream.raw());
        assert_eq!(rc, 0);
    }
    stream.synchronize().unwrap();

    let mut r_inline = vec![0f32; N];
    let mut r_mmvq = vec![0f32; N];
    unsafe {
        hipMemcpy(r_inline.as_mut_ptr() as *mut _, d_out_inline.as_ptr(), N*4, hipMemcpyDeviceToHost);
        hipMemcpy(r_mmvq.as_mut_ptr() as *mut _, d_out_mmvq.as_ptr(), N*4, hipMemcpyDeviceToHost);
    }
    println!("q8_inline: {:?}", r_inline);
    println!("mmvq:      {:?}", r_mmvq);
    println!("diff:      {}", (r_inline[0] - r_mmvq[0]).abs());

    // Also dump first Q8_1 block
    let mut q81 = vec![0u8; 36];
    unsafe { hipMemcpy(q81.as_mut_ptr() as *mut _, d_q81.as_ptr(), 36, hipMemcpyDeviceToHost); }
    let q81_d = f16::from_bits(u16::from_le_bytes([q81[0], q81[1]])).to_f32();
    let q81_s = f16::from_bits(u16::from_le_bytes([q81[2], q81[3]])).to_f32();
    println!("Q8_1 block 0: d={}, s={}, qs[0..8]={:?}", q81_d, q81_s, &q81[4..12]);
}
