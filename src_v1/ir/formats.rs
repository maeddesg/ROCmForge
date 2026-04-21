//! Phase-1 mandatory QuantFormats: Q4_0, Q4_1, Q4_K, Q6_K, Q8_0.
//!
//! Programs are lifted verbatim from `dequant_ir_spec.md §3.4–§3.8`.
//! No local invention — format spec is the single source of truth.
//!
//! Register-ID scheme. Each program uses a small, local namespace of
//! `RegId`s. The numeric values only need to be unique *within* a
//! single program (the validator checks that). Kept as module-level
//! `const`s so the program definitions read like the spec listings.
//!
//! Historical bug warnings (from v0.x, §3.10) are called out at the
//! relevant ops.

use super::types::{
    DequantOp, Fp8Variant, QuantFormat, RegId, ScalarType, SubScalesLayout,
};

// --- Convention for kernel-prolog-supplied registers -----------------
//
// The kernel prolog loads block-header fields into a fixed set of
// RegIds before the dequant program runs (§3.2). They appear here as
// *sources* without ever being written by the program.
//
// For the element-loop-bound values (`R_QS_ELEM`, `R_HIGH_FLAG`, etc.)
// the codegen synthesises them from the loop index (§6.3); they are
// read by the program but never assigned inside it.

// Shared low IDs used across programs (no cross-program collisions:
// each program has its own validation scope).
const R_D: RegId = 0; // block scale d, F32, set by prolog
const R_DMIN: RegId = 1; // block min dmin / m, F32, set by prolog
const R_SCALE_J: RegId = 2; // active sub-block scale, F32, set by prolog / unpack
const R_MIN_J: RegId = 3; // active sub-block min, F32, set by prolog / unpack

// --- Q4_0 ------------------------------------------------------------

pub fn q4_0() -> QuantFormat {
    const R_QS: RegId = 10; // qs[16] bytes
    const R_QS_ELEM: RegId = 11; // element-loop-bound: current qs byte
    const R_NIBBLE: RegId = 12;
    const R_CONST_8: RegId = 13;
    const R_Q_FP: RegId = 14;
    const R_Q_CENTERED: RegId = 15;
    const R_VAL: RegId = 16;
    const R_OUT: RegId = 17;

    QuantFormat {
        id: 2,
        name: "Q4_0",
        block_bytes: 18,
        elements_per_block: 32,
        sub_blocks_per_block: 1,
        sub_block_size: 32,

        block_scale_offset: 0,
        block_scale_type: ScalarType::Fp16,

        block_min_offset: None,
        block_min_type: None,

        sub_scales_layout: SubScalesLayout::None,

        dequant_program: vec![
            // Phase A — load qs bytes
            DequantOp::LoadBytes {
                offset: 2,
                count: 16,
                reg: R_QS,
            },

            // Phase B — element loop:
            //   for i < 16:  byte = qs[i],       high = false
            //   for i >= 16: byte = qs[i - 16],  high = true
            // Codegen synthesises R_QS_ELEM / R_HIGH_FLAG from the index.
            DequantOp::ExtractNibble {
                src: R_QS_ELEM,
                high: false,
                dst: R_NIBBLE,
            },

            // value = d * (nibble - 8)
            DequantOp::Const {
                value: 8.0,
                dst: R_CONST_8,
            },
            DequantOp::IntToFloat {
                src: R_NIBBLE,
                offset: 0,
                dst: R_Q_FP,
            },
            DequantOp::SubF32 {
                a: R_Q_FP,
                b: R_CONST_8,
                dst: R_Q_CENTERED,
            },
            DequantOp::MulF32 {
                a: R_D,
                b: R_Q_CENTERED,
                dst: R_VAL,
            },

            // Downcast + store
            DequantOp::DowncastToFP8 {
                src: R_VAL,
                dst: R_OUT,
                variant: Fp8Variant::E4M3,
                saturate: true,
            },
            DequantOp::StoreFP8 {
                src: R_OUT,
                variant: Fp8Variant::E4M3,
                lds_offset_expr: "elem_idx * TILE_N + col".to_string(),
            },
        ],
    }
}

// --- Q4_1 ------------------------------------------------------------

pub fn q4_1() -> QuantFormat {
    const R_QS: RegId = 10;
    const R_QS_ELEM: RegId = 11;
    const R_NIBBLE: RegId = 12;
    const R_Q_FP: RegId = 13;
    const R_VAL: RegId = 14;
    const R_OUT: RegId = 15;

    QuantFormat {
        id: 3,
        name: "Q4_1",
        block_bytes: 20,
        elements_per_block: 32,
        sub_blocks_per_block: 1,
        sub_block_size: 32,

        block_scale_offset: 0,
        block_scale_type: ScalarType::Fp16,

        block_min_offset: Some(2),
        block_min_type: Some(ScalarType::Fp16),

        sub_scales_layout: SubScalesLayout::None,

        dequant_program: vec![
            DequantOp::LoadBytes {
                offset: 4,
                count: 16,
                reg: R_QS,
            },

            DequantOp::ExtractNibble {
                src: R_QS_ELEM,
                high: false,
                dst: R_NIBBLE,
            },
            DequantOp::IntToFloat {
                src: R_NIBBLE,
                offset: 0,
                dst: R_Q_FP,
            },

            // value = d * nibble + m (R_DMIN carries m in Q4_1)
            DequantOp::FmaF32 {
                a: R_D,
                b: R_Q_FP,
                c: R_DMIN,
                dst: R_VAL,
            },

            DequantOp::DowncastToFP8 {
                src: R_VAL,
                dst: R_OUT,
                variant: Fp8Variant::E4M3,
                saturate: true,
            },
            DequantOp::StoreFP8 {
                src: R_OUT,
                variant: Fp8Variant::E4M3,
                lds_offset_expr: "elem_idx * TILE_N + col".to_string(),
            },
        ],
    }
}

// --- Q4_K ------------------------------------------------------------
//
// WARNING (v0.x Phase-7 regression): the 6-bit scales and mins are raw
// integers in 0..63. Do NOT divide by 64. The unpack program keeps them
// as integers until the final `IntToFloat` — `d * scale * nibble`, not
// `d * (scale/64) * nibble`.
//
// Pair-interleaved nibble layout: for sub-block pair (j, j+1), the 32
// qs bytes (pair_base..pair_base+32) hold 64 elements. The low nibble
// feeds sub-block j, the high nibble feeds j+1. See §3.6 / §6.4.

/// Unpack program for Q4_K 6-bit packed scales (`get_scale_min_k4`).
///
/// Writes 8 (scale[j], min[j]) FP32 pairs into the registers the main
/// program expects via `ScaleBlockStart`. `R_SCALES_J*` are codegen-
/// synthesised addresses (§6.3); this listing captures the op sequence
/// the codegen will emit for each (j < 4) / (j ≥ 4) path.
pub fn q4_k_unpack_scales() -> Vec<DequantOp> {
    const R_SCALES: RegId = 20;
    const R_SCALES_J: RegId = 21;
    const R_SCALES_J_PLUS_4: RegId = 22;
    const R_SCALES_J_MINUS_4: RegId = 23;
    const R_SCALE_J_INT: RegId = 24;
    const R_MIN_J_INT: RegId = 25;
    const R_LO_S: RegId = 26;
    const R_HI_S: RegId = 27;
    const R_LO_M: RegId = 28;
    const R_HI_M: RegId = 29;

    vec![
        // Phase A — load the 12 scale bytes.
        DequantOp::LoadBytes {
            offset: 0,
            count: 12,
            reg: R_SCALES,
        },

        // j ∈ 0..4: direct 6-bit extract.
        //   scale[j] = scales[j]     & 0x3F
        //   min[j]   = scales[j + 4] & 0x3F
        DequantOp::ExtractBits {
            src: R_SCALES_J,
            shift: 0,
            mask: 0x3F,
            dst: R_SCALE_J_INT,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 0,
            mask: 0x3F,
            dst: R_MIN_J_INT,
        },

        // j ∈ 4..8: CombineBits formula.
        //   scale[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 0,
            mask: 0x0F,
            dst: R_LO_S,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J_MINUS_4,
            shift: 6,
            mask: 0x03,
            dst: R_HI_S,
        },
        //   min[j] = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 4,
            mask: 0x0F,
            dst: R_LO_M,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J,
            shift: 6,
            mask: 0x03,
            dst: R_HI_M,
        },
        DequantOp::CombineBits {
            lo: R_LO_S,
            hi: R_HI_S,
            hi_shift: 4,
            dst: R_SCALE_J_INT,
        },
        DequantOp::CombineBits {
            lo: R_LO_M,
            hi: R_HI_M,
            hi_shift: 4,
            dst: R_MIN_J_INT,
        },

        // Finally promote to F32 so `d * scale` / `dmin * min` can run.
        DequantOp::IntToFloat {
            src: R_SCALE_J_INT,
            offset: 0,
            dst: R_SCALE_J,
        },
        DequantOp::IntToFloat {
            src: R_MIN_J_INT,
            offset: 0,
            dst: R_MIN_J,
        },
    ]
}

pub fn q4_k() -> QuantFormat {
    const R_QS: RegId = 30;
    const R_QS_ELEM: RegId = 31;
    const R_NIBBLE: RegId = 32;
    const R_Q_FP: RegId = 33;
    const R_D_EFF: RegId = 34;
    const R_DMIN_EFF: RegId = 35;
    const R_NEG_DMIN_EFF: RegId = 36;
    const R_VAL: RegId = 37;
    const R_OUT: RegId = 38;

    QuantFormat {
        id: 12,
        name: "Q4_K",
        block_bytes: 144,
        elements_per_block: 256,
        sub_blocks_per_block: 8,
        sub_block_size: 32,

        block_scale_offset: 0,
        block_scale_type: ScalarType::Fp16,

        block_min_offset: Some(2),
        block_min_type: Some(ScalarType::Fp16),

        sub_scales_layout: SubScalesLayout::Packed6Bit {
            offset: 4,
            count: 12,
            unpack_program: q4_k_unpack_scales(),
        },

        dequant_program: vec![
            // Phase A — load the 128 qs bytes
            DequantOp::LoadBytes {
                offset: 16,
                count: 128,
                reg: R_QS,
            },

            // Phase B — sub-block marker tells the codegen to hoist
            // r_scale[j] / r_min[j] loads outside the element loop.
            DequantOp::ScaleBlockStart { sub_block_idx: 0 },

            // Nibble extract — pair-interleaving resolved by the codegen.
            DequantOp::ExtractNibble {
                src: R_QS_ELEM,
                high: false,
                dst: R_NIBBLE,
            },
            DequantOp::IntToFloat {
                src: R_NIBBLE,
                offset: 0,
                dst: R_Q_FP,
            },

            // d_eff = d * scale[j] (once per sub-block; codegen hoists).
            DequantOp::MulF32 {
                a: R_D,
                b: R_SCALE_J,
                dst: R_D_EFF,
            },
            // dmin_eff = dmin * min[j] (once per sub-block).
            DequantOp::MulF32 {
                a: R_DMIN,
                b: R_MIN_J,
                dst: R_DMIN_EFF,
            },
            // Codegen peephole lifts this into the FMA `c`-operand modifier.
            DequantOp::NegF32 {
                src: R_DMIN_EFF,
                dst: R_NEG_DMIN_EFF,
            },

            // value = d_eff * nibble_fp + (-dmin_eff)
            DequantOp::FmaF32 {
                a: R_D_EFF,
                b: R_Q_FP,
                c: R_NEG_DMIN_EFF,
                dst: R_VAL,
            },

            DequantOp::DowncastToFP8 {
                src: R_VAL,
                dst: R_OUT,
                variant: Fp8Variant::E4M3,
                saturate: true,
            },
            DequantOp::StoreFP8 {
                src: R_OUT,
                variant: Fp8Variant::E4M3,
                lds_offset_expr: "sub_j * 32 + e * TILE_N + col".to_string(),
            },
        ],
    }
}

// --- Q6_K ------------------------------------------------------------
//
// WARNING (v0.x Phase-5 regression): the FP16 block scale `d` sits at
// offset 208 — the END of the block, not the start. The first 128 bytes
// are `ql`, the next 64 are `qh`, then 16 int8 sub-scales, then `d`.

pub fn q6_k() -> QuantFormat {
    const R_QL: RegId = 40;
    const R_QH: RegId = 41;
    const R_QL_ELEM: RegId = 42;
    const R_QH_ELEM: RegId = 43;
    const R_QL_N: RegId = 44;
    const R_QH_2: RegId = 45;
    const R_Q6_INT: RegId = 46;
    const R_CONST_32: RegId = 47;
    const R_Q_FP: RegId = 48;
    const R_Q_CENTERED: RegId = 49;
    const R_D_EFF: RegId = 50;
    const R_VAL: RegId = 51;
    const R_OUT: RegId = 52;

    QuantFormat {
        id: 14,
        name: "Q6_K",
        block_bytes: 210,
        elements_per_block: 256,
        sub_blocks_per_block: 16,
        sub_block_size: 16,

        // ACHTUNG: block scale at the *end* of the block.
        block_scale_offset: 208,
        block_scale_type: ScalarType::Fp16,

        block_min_offset: None,
        block_min_type: None,

        sub_scales_layout: SubScalesLayout::Int8Array {
            offset: 192,
            count: 16,
        },

        dequant_program: vec![
            // Phase A — ql, qh
            DequantOp::LoadBytes {
                offset: 0,
                count: 128,
                reg: R_QL,
            },
            DequantOp::LoadBytes {
                offset: 128,
                count: 64,
                reg: R_QH,
            },

            // Phase B
            DequantOp::ScaleBlockStart { sub_block_idx: 0 },

            // ql nibble (high flag is element-index dependent).
            DequantOp::ExtractNibble {
                src: R_QL_ELEM,
                high: false,
                dst: R_QL_N,
            },

            // qh 2-bit slice — the shift parameter is compile-time
            // synthesised by the codegen from (e % 4) * 2.
            DequantOp::ExtractBits {
                src: R_QH_ELEM,
                shift: 0,
                mask: 0x03,
                dst: R_QH_2,
            },

            // q6 = ql_nibble | (qh_2bits << 4) — declarative form.
            DequantOp::Combine6Bit {
                ql: R_QL_N,
                qh: R_QH_ELEM,
                shift: 0,
                dst: R_Q6_INT,
            },

            // Centre at 32: value = d_eff * (q6 - 32).
            DequantOp::Const {
                value: 32.0,
                dst: R_CONST_32,
            },
            DequantOp::IntToFloat {
                src: R_Q6_INT,
                offset: 0,
                dst: R_Q_FP,
            },
            DequantOp::SubF32 {
                a: R_Q_FP,
                b: R_CONST_32,
                dst: R_Q_CENTERED,
            },

            // d_eff = d * scale[j] (int8 scale → F32 at prolog).
            DequantOp::MulF32 {
                a: R_D,
                b: R_SCALE_J,
                dst: R_D_EFF,
            },
            DequantOp::MulF32 {
                a: R_D_EFF,
                b: R_Q_CENTERED,
                dst: R_VAL,
            },

            DequantOp::DowncastToFP8 {
                src: R_VAL,
                dst: R_OUT,
                variant: Fp8Variant::E4M3,
                saturate: true,
            },
            DequantOp::StoreFP8 {
                src: R_OUT,
                variant: Fp8Variant::E4M3,
                lds_offset_expr: "sub_j * 16 + e * TILE_N + col".to_string(),
            },
        ],
    }
}

// --- Q8_0 ------------------------------------------------------------
//
// Bytes are signed int8 (-128..127). The codegen emits
// `v_cvt_f32_i32` rather than `v_cvt_f32_ubyte0` based on the
// QuantFormat id (§3.8).

pub fn q8_0() -> QuantFormat {
    const R_QS: RegId = 60;
    const R_QS_ELEM: RegId = 61;
    const R_Q_FP: RegId = 62;
    const R_VAL: RegId = 63;
    const R_OUT: RegId = 64;

    QuantFormat {
        id: 8,
        name: "Q8_0",
        block_bytes: 34,
        elements_per_block: 32,
        sub_blocks_per_block: 1,
        sub_block_size: 32,

        block_scale_offset: 0,
        block_scale_type: ScalarType::Fp16,

        block_min_offset: None,
        block_min_type: None,

        sub_scales_layout: SubScalesLayout::None,

        dequant_program: vec![
            DequantOp::LoadBytes {
                offset: 2,
                count: 32,
                reg: R_QS,
            },

            DequantOp::IntToFloat {
                src: R_QS_ELEM,
                offset: 0,
                dst: R_Q_FP,
            },

            DequantOp::MulF32 {
                a: R_D,
                b: R_Q_FP,
                dst: R_VAL,
            },

            DequantOp::DowncastToFP8 {
                src: R_VAL,
                dst: R_OUT,
                variant: Fp8Variant::E4M3,
                saturate: true,
            },
            DequantOp::StoreFP8 {
                src: R_OUT,
                variant: Fp8Variant::E4M3,
                lds_offset_expr: "elem_idx * TILE_N + col".to_string(),
            },
        ],
    }
}

// --- Convenience ------------------------------------------------------

/// All five Phase-1 mandatory formats in stable order.
pub fn all_formats() -> Vec<QuantFormat> {
    vec![q4_0(), q4_1(), q4_k(), q6_k(), q8_0()]
}
