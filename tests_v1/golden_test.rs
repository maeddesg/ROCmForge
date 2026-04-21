//! Phase 1 / Schritt 1.6 — Block 2: golden-reference tests for the
//! Dequant IR CPU interpreter.
//!
//! Two sources of ground truth:
//!   Option A (handcrafted): hand-built blocks with known `d`/nibble/
//!     scale values. Expected FP32 output computed from the spec formula.
//!   Option B (from GGUF):   pull real Q4_0/Q4_K/Q6_K blocks out of the
//!     shipped GGUF models and compare against the v0.x reference
//!     dequant in `src/cpu/quant.rs` (llama.cpp-compatible).
//!
//! Acceptance gate: `max_abs_err < 1e-6` vs. the reference. Single-
//! rounding FMA keeps the interpreter bit-identical to the v0.x `d *
//! nibble * sc` reference for the quantisation formulas that are plain
//! multiply-only; the Q4_1 FMA form may differ by up to 1 ULP (the
//! spec allows this in §3.5).

#![cfg(feature = "v1")]

use half::f16;
use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::ir::formats::{q4_0, q4_1, q4_k, q6_k, q8_0};
use rocmforge::v1::ir::interpreter::dequant_block;

// ── Handcrafted blocks (Option A) ────────────────────────────────────────────

fn f16_bytes(v: f32) -> [u8; 2] {
    f16::from_f32(v).to_bits().to_le_bytes()
}

#[test]
fn test_golden_q4_0_zero_scale() {
    // All zeros → all outputs zero (d = 0 zeroes out the -8 offset too).
    let block = [0u8; 18];
    let out = dequant_block(&q4_0(), &block).unwrap();
    assert_eq!(out.len(), 32);
    for v in &out {
        assert_eq!(*v, 0.0);
    }
}

#[test]
fn test_golden_q4_0_d_half_nibble_10() {
    // d = 0.5, every nibble = 0xA (=10). value = 0.5 * (10 - 8) = 1.0.
    let mut block = [0u8; 18];
    block[0..2].copy_from_slice(&f16_bytes(0.5));
    for i in 2..18 {
        block[i] = 0xAA; // low nibble = 10, high nibble = 10
    }
    let out = dequant_block(&q4_0(), &block).unwrap();
    for (i, v) in out.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "element {i}: expected 1.0, got {v}"
        );
    }
}

#[test]
fn test_golden_q4_0_negative_values() {
    // d = 1.0, nibble = 0 → value = -8.0 for every element.
    let mut block = [0u8; 18];
    block[0..2].copy_from_slice(&f16_bytes(1.0));
    // qs stays zero.
    let out = dequant_block(&q4_0(), &block).unwrap();
    for v in &out {
        assert_eq!(*v, -8.0);
    }
}

#[test]
fn test_golden_q4_k_scales_no_div64() {
    // Historical-bug guard (spec §3.10 Bug 1): the 6-bit scale 63 must
    // contribute as 63, not 63/64 ≈ 0.984.
    // Build a block where sub_block 0 has scale = 63 and min = 0:
    //   scales[0] = 0x3F (scale_0 = 63)
    //   scales[4] = 0x00 (min_0   = 0)
    // d = 1.0, dmin = 0.0 so the min term vanishes; nibble = 1 so
    // value = (1 * 63) * 1 - 0 = 63.
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&f16_bytes(1.0)); // d
    block[2..4].copy_from_slice(&f16_bytes(0.0)); // dmin
    block[4] = 0x3F; // scales[0] → scale_0 = 63
                      // scales[4] stays 0 → min_0 = 0
                      // Set qs[0] low nibble = 1 → element 0 gets nibble 1.
    block[16] = 0x01;

    let out = dequant_block(&q4_k(), &block).unwrap();
    // element 0 is in sub_block 0 (even), low nibble of qs[0].
    // Expected: 63.0 (= 63 * 1 * 1 - 0).
    assert!(
        (out[0] - 63.0).abs() < 1e-6,
        "expected element 0 = 63.0 (bug-guard for /64), got {}",
        out[0]
    );
    // If the /64 bug were active, we'd see 63/64 ≈ 0.984 instead.
    assert!(
        out[0] > 1.0,
        "element 0 = {} suggests /64 bug is present",
        out[0]
    );
}

#[test]
fn test_golden_q4_k_handcrafted_simple() {
    // d = 2.0, dmin = 1.0, all scales = 1, all mins = 0, every qs byte
    // has low = 4, high = 4. Sub-block 0 (even) gets low nibbles →
    // d_eff * nibble - dmin_eff = (2*1)*4 - (1*0) = 8.
    // Sub-block 1 (odd) gets high nibbles → same value = 8.
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&f16_bytes(2.0));
    block[2..4].copy_from_slice(&f16_bytes(1.0));
    // scales[0..4] = 0x01 → scales_{0..4} = 1
    for i in 4..8 {
        block[i] = 0x01;
    }
    // scales[4..8] = 0 → mins_{0..4} = 0; j≥4 scales combine bits from
    // scales[0..4]>>6 (which is 0) and scales[4..8]&0x0F (also 0).
    // So sc[4..8] = 0 and mi[4..8] = 0. Only sub_blocks 0..4 have non-
    // zero contribution.
    for i in 16..144 {
        block[i] = 0x44; // low = 4, high = 4
    }

    let out = dequant_block(&q4_k(), &block).unwrap();

    // The first 64 elements (sub_j=0,1 → pair 0 using qs[0..32]):
    //   sub_j=0 (even, scale=1, min=0) gets low nibble = 4 → 2*1*4 = 8
    //   sub_j=1 (odd,  scale=1, min=0) gets high nibble = 4 → 2*1*4 = 8
    for i in 0..64 {
        assert!((out[i] - 8.0).abs() < 1e-6, "element {i}: got {}", out[i]);
    }
    // Elements 128..256 belong to sub_blocks 4..8 with scale=0, min=0
    // so value is exactly 0.
    for i in 128..256 {
        assert_eq!(out[i], 0.0, "element {i} should be 0 with scale=0");
    }
}

#[test]
fn test_golden_q6_k_d_at_offset_208() {
    // Historical-bug guard (spec §3.10 Bug 2): d is at offset 208.
    // If we put d at offset 208 but nothing at offset 0, and the
    // implementation correctly reads from 208, we should see non-zero
    // output (ql[0..2] = 0 here). If the implementation mistakenly
    // reads offset 0 → d = 0 → every output is zero.
    let mut block = [0u8; 210];
    block[0..2].fill(0); // ql[0..2] stays zero — acts as wrong-offset d
    block[208..210].copy_from_slice(&f16_bytes(1.0)); // real d

    // Scale[0] = 1 (signed i8), q6 = 0 (ql=0, qh=0) → q6 - 32 = -32 → -32.0
    block[192] = 0x01;

    let out = dequant_block(&q6_k(), &block).unwrap();
    assert_eq!(out.len(), 256);
    assert!(
        (out[0] - (-32.0)).abs() < 1e-6,
        "expected -32.0 (d at 208 correctly read), got {}",
        out[0]
    );
    // If d were read at offset 0, d would be 0 → out[0] = 0.0.
    assert_ne!(
        out[0], 0.0,
        "out[0] == 0 suggests d is being read at the wrong offset"
    );
}

#[test]
fn test_golden_q6_k_handcrafted() {
    // d = 1.0, scale[0] = 1, ql[0] = 0x21 (low=1, high=2), qh[0] = 0x00
    // q1 (element 0) = (ql[0] & 0xF) | (((qh[0] >> 0) & 3) << 4) - 32
    //               = 1 | 0 - 32 = -31
    // out[0] = d * sc * q1 = 1 * 1 * -31 = -31
    let mut block = [0u8; 210];
    block[208..210].copy_from_slice(&f16_bytes(1.0));
    block[192] = 0x01;
    block[0] = 0x21;

    let out = dequant_block(&q6_k(), &block).unwrap();
    assert!(
        (out[0] - (-31.0)).abs() < 1e-6,
        "out[0] = {}, expected -31.0",
        out[0]
    );
}

#[test]
fn test_golden_q8_0_signed() {
    // Q8_0 bytes are signed int8. d = 2.0, qs[0] = 0x80 (-128), qs[1] = 0x7F (127).
    let mut block = [0u8; 34];
    block[0..2].copy_from_slice(&f16_bytes(2.0));
    block[2] = 0x80; // -128
    block[3] = 0x7F; // 127
    block[4] = 0xFF; // -1
    block[5] = 0x00; // 0
    // The rest stay zero.

    let out = dequant_block(&q8_0(), &block).unwrap();
    assert!((out[0] - (-256.0)).abs() < 1e-6, "out[0] = {}", out[0]);
    assert!((out[1] - 254.0).abs() < 1e-6, "out[1] = {}", out[1]);
    assert!((out[2] - (-2.0)).abs() < 1e-6, "out[2] = {}", out[2]);
    assert_eq!(out[3], 0.0);
}

#[test]
fn test_golden_q4_1_handcrafted() {
    // d = 1.0, m = 0.25, nibble = 2 → 1*2 + 0.25 = 2.25.
    let mut block = [0u8; 20];
    block[0..2].copy_from_slice(&f16_bytes(1.0));
    block[2..4].copy_from_slice(&f16_bytes(0.25));
    // qs[0] = 0x22 → low = 2, high = 2. Element 0 = low nibble.
    block[4] = 0x22;

    let out = dequant_block(&q4_1(), &block).unwrap();
    assert!((out[0] - 2.25).abs() < 1e-6, "out[0] = {}", out[0]);
}

// ── Blocks from real GGUF files (Option B) ───────────────────────────────────

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .expect("HOME must be set")
        .join("models")
        .join(name)
}

fn open_first_tensor<'a>(
    gguf: &'a GGUFFile,
    predicate: impl Fn(&str) -> bool,
) -> Option<&'a rocmforge::v1::core::tensor_info::TensorInfo> {
    gguf.tensors().iter().find(|t| predicate(&t.name))
}

fn get_block<'a>(
    gguf: &'a GGUFFile,
    tensor: &rocmforge::v1::core::tensor_info::TensorInfo,
    block_offset: usize,
    block_bytes: usize,
) -> &'a [u8] {
    let start = (gguf.data_start() + tensor.file_offset) as usize + block_offset;
    &gguf.mmap_bytes()[start..start + block_bytes]
}

/// Reference dequant for Q4_0 mirroring v0.x `dequant_q4_0_block`
/// (`src/cpu/quant.rs:597`). Straight `value = d * (nibble - 8)` —
/// bit-identical to our interpreter.
fn v0_reference_q4_0(block: &[u8]) -> Vec<f32> {
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let qs = &block[2..18];
    let mut out = vec![0.0f32; 32];
    for i in 0..16 {
        let lo = ((qs[i] & 0x0F) as i32) - 8;
        let hi = ((qs[i] >> 4) as i32) - 8;
        out[i] = lo as f32 * d;
        out[i + 16] = hi as f32 * d;
    }
    out
}

/// Reference dequant for Q4_K mirroring v0.x `embed_q4_k` (llama.cpp
/// `dequantize_row_q4_K`).
fn v0_reference_q4_k(block: &[u8]) -> Vec<f32> {
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let get_scale_min = |j: usize| -> (u8, u8) {
        if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, mn)
        }
    };

    let mut out = vec![0.0f32; 256];
    let mut out_idx = 0;
    let mut qs_off = 0;
    for k in 0..4 {
        let (sc_e, m_e) = get_scale_min(2 * k);
        let (sc_o, m_o) = get_scale_min(2 * k + 1);
        let d_e = d * sc_e as f32;
        let m_offset_e = dmin * m_e as f32;
        let d_o = d * sc_o as f32;
        let m_offset_o = dmin * m_o as f32;
        for l in 0..32 {
            let q_lo = (qs[qs_off + l] & 0x0F) as f32;
            let q_hi = (qs[qs_off + l] >> 4) as f32;
            // v0.x uses separate mul-sub (no FMA), matches interpreter
            // mul_add form when dmin * m is subtracted at single rounding.
            out[out_idx + l] = d_e * q_lo - m_offset_e;
            out[out_idx + 32 + l] = d_o * q_hi - m_offset_o;
        }
        out_idx += 64;
        qs_off += 32;
    }
    out
}

/// Reference dequant for Q6_K mirroring v0.x `embed_q6_k` (llama.cpp
/// `dequantize_row_q6_K`).
fn v0_reference_q6_k(block: &[u8]) -> Vec<f32> {
    let mut ql = &block[0..128];
    let mut qh = &block[128..192];
    let mut sc: &[u8] = &block[192..208];
    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();

    let mut out = vec![0.0f32; 256];
    let mut base = 0usize;
    for _ in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
            let q3 = (((ql[l] >> 4) & 0xF) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
            let q4 = (((ql[l + 32] >> 4) & 0xF) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
            let s1 = d * (sc[is] as i8) as f32;
            let s2 = d * (sc[is + 2] as i8) as f32;
            let s3 = d * (sc[is + 4] as i8) as f32;
            let s4 = d * (sc[is + 6] as i8) as f32;
            out[base + l] = s1 * q1 as f32;
            out[base + l + 32] = s2 * q2 as f32;
            out[base + l + 64] = s3 * q3 as f32;
            out[base + l + 96] = s4 * q4 as f32;
        }
        base += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc = &sc[8..];
    }
    out
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn test_golden_q4_0_from_gguf_qwen25() {
    let gguf = GGUFFile::open(model_path("Qwen2.5-7B-Instruct-Q4_0.gguf"))
        .expect("Qwen2.5-Q4_0 open");
    // Any Q4_0 weight tensor; pick `blk.0.attn_q.weight`.
    let tensor = open_first_tensor(&gguf, |n| {
        n == "blk.0.attn_q.weight"
            && matches!(
                gguf.tensors()
                    .iter()
                    .find(|t| t.name == n)
                    .map(|t| t.ggml_type),
                Some(rocmforge::v1::core::tensor_info::GgmlType::Q4_0)
            )
    })
    .expect("blk.0.attn_q.weight Q4_0");
    let fmt = q4_0();

    for block_idx in [0usize, 37, 123] {
        let block = get_block(&gguf, tensor, block_idx * fmt.block_bytes, fmt.block_bytes);
        let ours = dequant_block(&fmt, block).unwrap();
        let theirs = v0_reference_q4_0(block);
        let err = max_abs_err(&ours, &theirs);
        assert!(
            err < 1e-6,
            "Q4_0 block {block_idx}: max_abs_err = {err}"
        );
    }
}

#[test]
fn test_golden_q4_k_from_gguf_qwen3() {
    let gguf = GGUFFile::open(model_path("Qwen3-8B-Q4_K_M.gguf")).expect("Qwen3-8B open");
    // Find a Q4_K tensor (e.g. token_embd.weight or most block weights).
    let tensor = gguf
        .tensors()
        .iter()
        .find(|t| matches!(t.ggml_type, rocmforge::v1::core::tensor_info::GgmlType::Q4_K))
        .expect("at least one Q4_K tensor");
    let fmt = q4_k();
    for block_idx in [0usize, 17, 41] {
        let block = get_block(&gguf, tensor, block_idx * fmt.block_bytes, fmt.block_bytes);
        let ours = dequant_block(&fmt, block).unwrap();
        let theirs = v0_reference_q4_k(block);
        let err = max_abs_err(&ours, &theirs);
        assert!(
            err < 1e-6,
            "Q4_K block {block_idx}: max_abs_err = {err}"
        );
    }
}

#[test]
fn test_golden_q6_k_from_gguf_qwen3() {
    let gguf = GGUFFile::open(model_path("Qwen3-8B-Q4_K_M.gguf")).expect("Qwen3-8B open");
    let tensor = gguf
        .tensors()
        .iter()
        .find(|t| matches!(t.ggml_type, rocmforge::v1::core::tensor_info::GgmlType::Q6_K))
        .expect("at least one Q6_K tensor");
    let fmt = q6_k();
    for block_idx in [0usize, 9, 55] {
        let block = get_block(&gguf, tensor, block_idx * fmt.block_bytes, fmt.block_bytes);
        let ours = dequant_block(&fmt, block).unwrap();
        let theirs = v0_reference_q6_k(block);
        let err = max_abs_err(&ours, &theirs);
        assert!(
            err < 1e-6,
            "Q6_K block {block_idx}: max_abs_err = {err}"
        );
    }
}
