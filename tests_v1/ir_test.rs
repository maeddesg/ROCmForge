//! Phase 1 / Schritt 1.6 — Block 1: IR core-types, QuantFormats, validator,
//! printer.
//!
//! Tests cover the spec-mandated `dequant_ir_spec.md §2.10` (17-op enum),
//! §3.4–3.8 (five Phase-1 formats), §2.11 invariants (validator), and
//! the debug printer output.

#![cfg(feature = "v1")]

use rocmforge::v1::ir::formats::{all_formats, q4_0, q4_1, q4_k, q6_k, q8_0};
use rocmforge::v1::ir::printer::format_program;
use rocmforge::v1::ir::types::{DequantOp, Fp8Variant, HalfType, QuantFormat, SubScalesLayout};
use rocmforge::v1::ir::validator::{
    validate_format, validate_program, ValidationError,
};

// ── Core-type coverage ──────────────────────────────────────────────────────

#[test]
fn test_dequant_op_enum_complete() {
    // Build one of each of the 17 DequantOp variants and confirm they
    // all have a `.name()` and a canonical string shape.
    let all_ops: Vec<DequantOp> = vec![
        DequantOp::LoadBytes { offset: 0, count: 2, reg: 0 },
        DequantOp::LoadFP8 {
            offset: 0,
            count: 4,
            variant: Fp8Variant::E4M3,
            reg: 1,
        },
        DequantOp::ExtractNibble { src: 1, high: false, dst: 2 },
        DequantOp::Combine6Bit { ql: 2, qh: 3, shift: 0, dst: 4 },
        DequantOp::ExtractBits { src: 4, shift: 0, mask: 0x3F, dst: 5 },
        DequantOp::CombineBits { lo: 5, hi: 6, hi_shift: 4, dst: 7 },
        DequantOp::IntToFloat { src: 7, offset: 0, dst: 8 },
        DequantOp::MulF32 { a: 8, b: 9, dst: 10 },
        DequantOp::FmaF32 { a: 10, b: 11, c: 12, dst: 13 },
        DequantOp::SubF32 { a: 13, b: 14, dst: 15 },
        DequantOp::AddF32 { a: 15, b: 16, dst: 17 },
        DequantOp::NegF32 { src: 17, dst: 18 },
        DequantOp::DowncastToHalf {
            src: 18,
            dst: 19,
            target: HalfType::Fp16,
        },
        DequantOp::DowncastToFP8 {
            src: 19,
            dst: 20,
            variant: Fp8Variant::E4M3,
            saturate: true,
        },
        DequantOp::StoreHalf {
            src: 19,
            lds_offset_expr: "i".into(),
        },
        DequantOp::StoreFP8 {
            src: 20,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "i".into(),
        },
        DequantOp::ScaleBlockStart { sub_block_idx: 0 },
    ];
    // Const is the final variant — kept separate because it uses a
    // different arm and tests self-documentation.
    //
    // NB: `dequant_ir_spec.md §2.10` states "Gesamt: 17 DequantOp-Varianten"
    // but the enum body directly above lists 18 (arch-doc contributes 11,
    // not 10 as the prose claims). The implementation follows the listing.
    let const_op = DequantOp::Const { value: 8.0, dst: 99 };
    assert_eq!(all_ops.len() + 1, 18, "full enum has 18 variants");
    assert_eq!(const_op.name(), "Const");
    for op in &all_ops {
        // All ops have a non-empty name.
        assert!(!op.name().is_empty());
    }
}

// ── QuantFormat validation ──────────────────────────────────────────────────

#[test]
fn test_quant_format_q4_0_valid() {
    let fmt = q4_0();
    assert_eq!(fmt.block_bytes, 18);
    assert_eq!(fmt.elements_per_block, 32);
    assert_eq!(fmt.sub_blocks_per_block, 1);
    assert_eq!(fmt.block_scale_offset, 0);
    assert!(matches!(fmt.sub_scales_layout, SubScalesLayout::None));
    validate_format(&fmt).expect("Q4_0 must validate");
}

#[test]
fn test_quant_format_q4_1_valid() {
    let fmt = q4_1();
    assert_eq!(fmt.block_bytes, 20);
    assert_eq!(fmt.elements_per_block, 32);
    assert_eq!(fmt.block_min_offset, Some(2));
    validate_format(&fmt).expect("Q4_1 must validate");
}

#[test]
fn test_quant_format_q4_k_valid() {
    let fmt = q4_k();
    assert_eq!(fmt.block_bytes, 144);
    assert_eq!(fmt.elements_per_block, 256);
    assert_eq!(fmt.sub_blocks_per_block, 8);
    assert!(matches!(
        fmt.sub_scales_layout,
        SubScalesLayout::Packed6Bit { offset: 4, count: 12, .. }
    ));
    validate_format(&fmt).expect("Q4_K must validate");
}

#[test]
fn test_quant_format_q6_k_valid() {
    let fmt = q6_k();
    assert_eq!(fmt.block_bytes, 210);
    assert_eq!(fmt.elements_per_block, 256);
    assert_eq!(fmt.sub_blocks_per_block, 16);
    // v0.x Phase-5 regression guard: d is at the END of the block.
    assert_eq!(fmt.block_scale_offset, 208);
    assert!(matches!(
        fmt.sub_scales_layout,
        SubScalesLayout::Int8Array { offset: 192, count: 16 }
    ));
    validate_format(&fmt).expect("Q6_K must validate");
}

#[test]
fn test_quant_format_q8_0_valid() {
    let fmt = q8_0();
    assert_eq!(fmt.block_bytes, 34);
    assert_eq!(fmt.elements_per_block, 32);
    validate_format(&fmt).expect("Q8_0 must validate");
}

#[test]
fn test_all_formats_validate() {
    for fmt in all_formats() {
        let res = validate_format(&fmt);
        assert!(res.is_ok(), "format {} failed: {:?}", fmt.name, res);
    }
}

// ── Validator failure modes ─────────────────────────────────────────────────

/// Minimal body: one LoadBytes, one terminal Store — so individual
/// tests can prepend a single offending op and still hit the specific
/// rule they're exercising.
fn minimal_terminated(extra: Vec<DequantOp>) -> Vec<DequantOp> {
    let mut ops = vec![DequantOp::LoadBytes {
        offset: 0,
        count: 2,
        reg: 100,
    }];
    ops.extend(extra);
    ops.push(DequantOp::StoreFP8 {
        src: 200,
        variant: Fp8Variant::E4M3,
        lds_offset_expr: "i".into(),
    });
    ops
}

#[test]
fn test_validator_ssa_invariant() {
    // Two LoadBytes writing the same RegId — must be rejected.
    let program = vec![
        DequantOp::LoadBytes { offset: 0, count: 2, reg: 5 },
        DequantOp::LoadBytes { offset: 2, count: 2, reg: 5 },
        DequantOp::StoreHalf {
            src: 5,
            lds_offset_expr: "i".into(),
        },
    ];
    let errs = validate_program(&program, 144).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::DuplicateAssignment { reg: 5, .. }
        )),
        "expected DuplicateAssignment, got {:?}",
        errs
    );
}

#[test]
fn test_validator_use_before_def() {
    // Reads r200 as F32 source BEFORE writing it. r100 is external (never
    // written), r200 IS written later — so this is a real use-before-def.
    let program = vec![
        DequantOp::LoadBytes { offset: 0, count: 2, reg: 100 },
        // src = 200 (will be written later — use-before-def).
        DequantOp::MulF32 { a: 200, b: 100, dst: 201 },
        // The later write of 200.
        DequantOp::Const { value: 1.0, dst: 200 },
        DequantOp::StoreHalf {
            src: 201,
            lds_offset_expr: "i".into(),
        },
    ];
    let errs = validate_program(&program, 144).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::UseBeforeDef { reg: 200, .. }
        )),
        "expected UseBeforeDef, got {:?}",
        errs
    );
}

#[test]
fn test_validator_type_mismatch() {
    // LoadBytes(count=1) → U8. MulF32 requires F32 — type mismatch.
    let program = vec![
        DequantOp::LoadBytes { offset: 0, count: 1, reg: 10 },
        DequantOp::Const { value: 1.0, dst: 11 },
        DequantOp::MulF32 { a: 10, b: 11, dst: 12 },
        DequantOp::StoreHalf {
            src: 12,
            lds_offset_expr: "i".into(),
        },
    ];
    let errs = validate_program(&program, 144).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::TypeMismatch { op_name: "MulF32", .. }
        )),
        "expected TypeMismatch on MulF32, got {:?}",
        errs
    );
}

#[test]
fn test_validator_offset_out_of_bounds() {
    // block_bytes = 18 (Q4_0). Load at offset=16, count=16 overruns.
    let program = minimal_terminated(vec![DequantOp::LoadBytes {
        offset: 16,
        count: 16,
        reg: 101,
    }]);
    let errs = validate_program(&program, 18).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::OffsetOutOfBounds { .. }
        )),
        "expected OffsetOutOfBounds, got {:?}",
        errs
    );
}

#[test]
fn test_validator_no_terminal_store() {
    // Program ends on MulF32 — no Store.
    let program = vec![
        DequantOp::LoadBytes { offset: 0, count: 2, reg: 10 },
        DequantOp::Const { value: 1.0, dst: 11 },
        DequantOp::Const { value: 2.0, dst: 12 },
        DequantOp::MulF32 { a: 11, b: 12, dst: 13 },
    ];
    let errs = validate_program(&program, 144).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(e, ValidationError::NoTerminalStore)),
        "expected NoTerminalStore, got {:?}",
        errs
    );
}

#[test]
fn test_validator_sub_block_order() {
    let program = vec![
        DequantOp::LoadBytes { offset: 0, count: 2, reg: 10 },
        DequantOp::ScaleBlockStart { sub_block_idx: 3 },
        DequantOp::ScaleBlockStart { sub_block_idx: 2 }, // decreasing → error
        DequantOp::StoreHalf {
            src: 10,
            lds_offset_expr: "i".into(),
        },
    ];
    let errs = validate_program(&program, 144).unwrap_err();
    assert!(
        errs.iter().any(|e| matches!(
            e,
            ValidationError::InvalidSubBlockOrder { .. }
        )),
        "expected InvalidSubBlockOrder, got {:?}",
        errs
    );
}

// ── Printer output ──────────────────────────────────────────────────────────

#[test]
fn test_printer_q4_0() {
    let fmt = q4_0();
    let output = format_program(&fmt);
    assert!(output.contains("Q4_0"));
    assert!(output.contains("LoadBytes"));
    assert!(output.contains("Store"));
    println!("{output}");
}

#[test]
fn test_printer_q4_k() {
    let fmt = q4_k();
    let output = format_program(&fmt);
    // Q4_K has both a main program and an unpack-program template.
    assert!(output.contains("Q4_K"));
    assert!(output.contains("Unpack Program"));
    assert!(output.contains("CombineBits"));
    assert!(output.contains("ScaleBlockStart"));
    assert!(output.lines().count() > 10, "Q4_K should be >10 lines");
    println!("{output}");
}

#[test]
fn test_printer_all_formats() {
    // Smoke: none of the formats panic when rendered.
    for fmt in all_formats() {
        let out = format_program(&fmt);
        assert!(!out.is_empty(), "{} rendered empty", fmt.name);
    }
}
