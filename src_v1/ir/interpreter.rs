//! CPU reference interpreter for Dequant IR programs.
//!
//! Ground-truth FP32 dequant against which the GPU codegen (Block 3) is
//! parity-checked. Correctness first — no performance goals.
//!
//! Scope. The public entry point `dequant_block` maps a `QuantFormat`
//! + block bytes to `Vec<f32>` using the spec-defined dequant formulas
//! (§3.4–3.8). Arithmetic follows the op semantics of `types.rs` /
//! spec §2 — in particular, `FmaF32` is single-rounding (via Rust's
//! `f32::mul_add`) to stay bit-identical with GPU `v_fma_f32` (§7.4
//! Rule 1).
//!
//! Q6_K note (spec discrepancy, deliberate). Spec §3.7 and §6.3 describe
//! a `qh[i/4] shift (i%4)*2` layout. Real GGUF files (produced by
//! llama.cpp `dequantize_row_q6_K`) use a more involved mapping where
//! each `qh[l]` byte carries the upper 2 bits for four elements spread
//! across positions `l, l+32, l+64, l+96` within a 128-element chunk.
//! This interpreter follows the llama.cpp pattern (which v0.x
//! `embed_q6_k` implements correctly) so that golden-vector tests from
//! real GGUF blocks pass.

use half::f16;

use super::types::QuantFormat;

/// Convert FP16 raw bits to FP32.
fn f16_to_f32(raw: u16) -> f32 {
    f16::from_bits(raw).to_f32()
}

fn read_fp16(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

/// Dispatch by format id — matches llama.cpp `ggml_type` codes (§3.11).
pub fn dequant_block(format: &QuantFormat, block_data: &[u8]) -> Result<Vec<f32>, String> {
    if block_data.len() < format.block_bytes {
        return Err(format!(
            "block_data len {} < block_bytes {} for {}",
            block_data.len(),
            format.block_bytes,
            format.name
        ));
    }
    Ok(match format.id {
        2 => dequant_q4_0(block_data),
        3 => dequant_q4_1(block_data),
        8 => dequant_q8_0(block_data),
        12 => dequant_q4_k(block_data),
        14 => dequant_q6_k(block_data),
        other => return Err(format!("no interpreter for format id {other}")),
    })
}

// --- Q4_0 (§3.4) -----------------------------------------------------
//
// Layout: d[fp16] @0, qs[16] @2
// Formula: value = d * (nibble - 8)
fn dequant_q4_0(block: &[u8]) -> Vec<f32> {
    let d = read_fp16(block, 0);
    let qs = &block[2..18];
    let mut out = Vec::with_capacity(32);
    for i in 0..32 {
        let byte_idx = i % 16;
        let nibble = if i >= 16 {
            (qs[byte_idx] >> 4) & 0x0F
        } else {
            qs[byte_idx] & 0x0F
        };
        out.push(d * (nibble as f32 - 8.0));
    }
    out
}

// --- Q4_1 (§3.5) -----------------------------------------------------
//
// Layout: d[fp16] @0, m[fp16] @2, qs[16] @4
// Formula: value = d * nibble + m  (spec §3.5 → FmaF32 form)
fn dequant_q4_1(block: &[u8]) -> Vec<f32> {
    let d = read_fp16(block, 0);
    let m = read_fp16(block, 2);
    let qs = &block[4..20];
    let mut out = Vec::with_capacity(32);
    for i in 0..32 {
        let byte_idx = i % 16;
        let nibble = if i >= 16 {
            (qs[byte_idx] >> 4) & 0x0F
        } else {
            qs[byte_idx] & 0x0F
        };
        // Single-rounding FMA to stay bit-identical with v_fma_f32 (§7.4).
        out.push((nibble as f32).mul_add(d, m));
    }
    out
}

// --- Q4_K (§3.6) -----------------------------------------------------
//
// Layout: d[fp16] @0, dmin[fp16] @2, scales[12] @4, qs[128] @16
// Scales: 6-bit packed via get_scale_min_k4. Raw integers 0..63, **no /64**.
// Formula: value = (d * sc_j) * nibble − (dmin * m_j), via Fma(d_eff, nibble, −dmin_eff).
// Nibble layout: pair-interleaved — even sub_j = low nibble, odd sub_j = high nibble
// of the same 32 qs bytes.
fn dequant_q4_k(block: &[u8]) -> Vec<f32> {
    let d = read_fp16(block, 0);
    let dmin = read_fp16(block, 2);
    let scales = &block[4..16];
    let qs = &block[16..144];

    // WARNING (v0.x Phase-7 regression): values are raw 0..63 integers.
    // Do NOT divide by 64.
    let mut sc = [0u8; 8];
    let mut mi = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        mi[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        mi[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    let mut out = vec![0.0f32; 256];

    // Iterate in pair-of-sub-blocks (matches llama.cpp dequantize_row_q4_K).
    // Each pair (even_j, odd_j) covers 64 elements and consumes 32 qs bytes.
    let mut qs_off = 0usize;
    let mut out_idx = 0usize;
    for k in 0..4 {
        let even_j = 2 * k;
        let odd_j = 2 * k + 1;
        let d_e = d * sc[even_j] as f32;
        let dmin_e = dmin * mi[even_j] as f32;
        let d_o = d * sc[odd_j] as f32;
        let dmin_o = dmin * mi[odd_j] as f32;
        for l in 0..32 {
            let q_lo = (qs[qs_off + l] & 0x0F) as f32;
            let q_hi = ((qs[qs_off + l] >> 4) & 0x0F) as f32;
            out[out_idx + l] = q_lo.mul_add(d_e, -dmin_e);
            out[out_idx + 32 + l] = q_hi.mul_add(d_o, -dmin_o);
        }
        qs_off += 32;
        out_idx += 64;
    }
    out
}

// --- Q6_K (§3.7, llama.cpp pattern) ----------------------------------
//
// Layout: ql[128] @0, qh[64] @128, scales[16, i8] @192, d[fp16] @208 (end!)
// Formula: value = d * sc_j * (q6 − 32), q6 = ql_nibble | (qh_2bits << 4).
//
// Element order follows llama.cpp dequantize_row_q6_K:
//   for n in [0, 128]:   // outer chunk of 128
//     for l in 0..32:    // inner; writes positions l+0, l+32, l+64, l+96
//       q1 = (ql[l +  0] & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
//       q2 = (ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
//       q3 = (ql[l +  0]  >> 4) | ((qh[l] >> 4) & 3) << 4) - 32
//       q4 = (ql[l + 32]  >> 4) | ((qh[l] >> 6) & 3) << 4) - 32
//       y[l +  0] = d * sc[is + 0] * q1  with is = l/16
//       y[l + 32] = d * sc[is + 2] * q2
//       y[l + 64] = d * sc[is + 4] * q3
//       y[l + 96] = d * sc[is + 6] * q4
//     then: y += 128, ql += 64, qh += 32, sc += 8
fn dequant_q6_k(block: &[u8]) -> Vec<f32> {
    let mut ql = &block[0..128];
    let mut qh = &block[128..192];
    // Signed int8 scales (cast via byte-to-i8).
    let scales_raw = &block[192..208];
    let mut sc_slice: &[u8] = scales_raw;
    let d = read_fp16(block, 208);

    let mut out = vec![0.0f32; 256];
    let mut y_base = 0usize;

    for _chunk in 0..2 {
        for l in 0..32 {
            let is = l / 16;

            let q1_combined = (ql[l] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4);
            let q1 = (q1_combined as i8 as i32) - 32;

            let q2_combined = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4);
            let q2 = (q2_combined as i8 as i32) - 32;

            let q3_combined = ((ql[l] >> 4) & 0x0F) | (((qh[l] >> 4) & 0x03) << 4);
            let q3 = (q3_combined as i8 as i32) - 32;

            let q4_combined = ((ql[l + 32] >> 4) & 0x0F) | (((qh[l] >> 6) & 0x03) << 4);
            let q4 = (q4_combined as i8 as i32) - 32;

            // Signed int8 scale.
            let sc1 = d * (sc_slice[is] as i8) as f32;
            let sc2 = d * (sc_slice[is + 2] as i8) as f32;
            let sc3 = d * (sc_slice[is + 4] as i8) as f32;
            let sc4 = d * (sc_slice[is + 6] as i8) as f32;

            out[y_base + l] = sc1 * q1 as f32;
            out[y_base + l + 32] = sc2 * q2 as f32;
            out[y_base + l + 64] = sc3 * q3 as f32;
            out[y_base + l + 96] = sc4 * q4 as f32;
        }
        y_base += 128;
        ql = &ql[64..];
        qh = &qh[32..];
        sc_slice = &sc_slice[8..];
    }
    out
}

// --- Q8_0 (§3.8) -----------------------------------------------------
//
// Layout: d[fp16] @0, qs[32, i8] @2
// Formula: value = d * q8_signed  (signed int8, -128..127)
fn dequant_q8_0(block: &[u8]) -> Vec<f32> {
    let d = read_fp16(block, 0);
    let mut out = Vec::with_capacity(32);
    for i in 0..32 {
        // Critical: signed int8.
        let q8_signed = block[2 + i] as i8 as f32;
        out.push(d * q8_signed);
    }
    out
}
