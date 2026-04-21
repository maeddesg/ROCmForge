//! IR validator — well-formedness checks that must pass BEFORE codegen.
//!
//! Rules derived from `dequant_ir_spec.md §2.11`:
//!   1. SSA — each RegId is written at most once by the program.
//!   2. Use-before-def — every read reference must be either external
//!      (kernel-prolog / loop-bound, §6.3) or written by a preceding op.
//!   3. Type consistency — e.g. arithmetic ops require FP32 operands;
//!      ExtractNibble/ExtractBits require U8/U16.
//!   4. Offset bounds — `LoadBytes`/`LoadFP8` offsets + count must fit
//!      inside `block_bytes`.
//!   5. Monotonic `ScaleBlockStart.sub_block_idx` when multiple are
//!      present in the same program.
//!   6. Terminal Store — the last op must be `StoreHalf` or `StoreFP8`.
//!
//! "External RegIds" convention. The dequant programs reference prolog-
//! provided values (`r_d`, `r_dmin`, `r_scale_j`) and loop-bound values
//! synthesised by the codegen from the element index (`r_qs_elem`,
//! `r_qh_elem`, `r_ql_elem`, …) — §6.3. Any RegId that is read but
//! never written inside the program is treated as external.
//!
//! Pattern-template caveat. `SubScalesLayout::Packed6Bit::unpack_program`
//! is inherently a *template* for the j<4 and j≥4 paths; the spec's own
//! Q4_K listing (§3.6) writes `R_SCALE_J_INT` and `R_MIN_J_INT` twice.
//! The codegen unrolls it per-j into two SSA-valid emissions. We
//! therefore validate `dequant_program` strictly, and only type-check
//! `unpack_program` (see `validate_pattern_program`).

use std::collections::HashSet;

use super::types::{DequantOp, QuantFormat, RegType, SubScalesLayout};

/// Validator diagnostics. Each variant carries enough context to point
/// a user at the offending op index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Same RegId is written by two ops in the program.
    DuplicateAssignment {
        reg: u32,
        first_op: usize,
        second_op: usize,
    },
    /// Op at `op_index` reads a RegId that was never defined (neither
    /// external nor by a preceding op in the program).
    UseBeforeDef { reg: u32, op_index: usize },
    /// Op operand has the wrong type — e.g. MulF32 on a U8 register.
    TypeMismatch {
        op_index: usize,
        op_name: &'static str,
        expected: &'static str,
        got: &'static str,
    },
    /// `LoadBytes`/`LoadFP8` reaches past the end of the block.
    OffsetOutOfBounds {
        op_index: usize,
        offset: usize,
        count: usize,
        block_bytes: usize,
    },
    /// Consecutive `ScaleBlockStart` markers are not monotonically
    /// non-decreasing.
    InvalidSubBlockOrder {
        op_index: usize,
        previous: u8,
        current: u8,
    },
    /// The program does not end with a Store op.
    NoTerminalStore,
    /// The program is empty.
    EmptyProgram,
}

/// Validate a `QuantFormat`'s main `dequant_program` against all six
/// rules. `block_bytes` comes from the format itself — callers usually
/// use [`validate_format`].
pub fn validate_program(
    program: &[DequantOp],
    block_bytes: usize,
) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    if program.is_empty() {
        errors.push(ValidationError::EmptyProgram);
        return Err(errors);
    }

    // Rule 6 — terminal store.
    if !program.last().unwrap().is_terminal_store() {
        errors.push(ValidationError::NoTerminalStore);
    }

    // Precompute the external-RegId set: any RegId read by some op and
    // never assigned by any op in the program is assumed to come from
    // the kernel prolog or be loop-synthesised (§6.3).
    let mut assigned: HashSet<u32> = HashSet::new();
    for op in program {
        if let Some(d) = op.dst() {
            assigned.insert(d);
        }
    }
    let mut read: HashSet<u32> = HashSet::new();
    for op in program {
        for s in op.srcs() {
            read.insert(s);
        }
    }
    let externals: HashSet<u32> = read.difference(&assigned).copied().collect();

    // Running state for rules 1–5.
    let mut first_def: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut reg_types: std::collections::HashMap<u32, RegType> = Default::default();
    // Externals default to F32 (prolog-provided r_d/r_dmin/r_scale_j are
    // all F32 after prolog promotion; loop-bound byte-register externals
    // are U8 — tracked case-by-case below).
    for e in &externals {
        reg_types.insert(*e, RegType::F32);
    }
    let mut last_sub_idx: Option<u8> = None;

    for (i, op) in program.iter().enumerate() {
        // Rule 2 — use-before-def.
        for src in op.srcs() {
            if !first_def.contains_key(&src) && !externals.contains(&src) {
                errors.push(ValidationError::UseBeforeDef {
                    reg: src,
                    op_index: i,
                });
            }
        }

        // Rule 1 — SSA.
        if let Some(dst) = op.dst() {
            if let Some(prev) = first_def.get(&dst) {
                errors.push(ValidationError::DuplicateAssignment {
                    reg: dst,
                    first_op: *prev,
                    second_op: i,
                });
            } else {
                first_def.insert(dst, i);
            }
        }

        // Loop-bound byte externals read by ExtractNibble / Combine6Bit
        // / ExtractBits need to be typed U8 (not F32) so downstream type
        // checks pass. Update their type as soon as we see them used.
        coerce_external_types(op, &externals, &mut reg_types);

        // Rule 3 — type checks (best-effort based on current map).
        type_check_op(op, i, &reg_types, &mut errors);

        // Record the dst type for downstream checks.
        if let Some(d) = op.dst() {
            if let Some(t) = op.result_type(&[]) {
                reg_types.insert(d, t);
            }
        }

        // Rule 4 — offset bounds.
        match *op {
            DequantOp::LoadBytes { offset, count, .. }
            | DequantOp::LoadFP8 { offset, count, .. } => {
                if offset + count > block_bytes {
                    errors.push(ValidationError::OffsetOutOfBounds {
                        op_index: i,
                        offset,
                        count,
                        block_bytes,
                    });
                }
            }
            _ => {}
        }

        // Rule 5 — monotonic sub-block markers.
        if let DequantOp::ScaleBlockStart { sub_block_idx } = *op {
            if let Some(prev) = last_sub_idx {
                if sub_block_idx < prev {
                    errors.push(ValidationError::InvalidSubBlockOrder {
                        op_index: i,
                        previous: prev,
                        current: sub_block_idx,
                    });
                }
            }
            last_sub_idx = Some(sub_block_idx);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Coerce external RegIds that are *read* as byte-register inputs
/// (ExtractNibble / ExtractBits src, IntToFloat src, Combine6Bit ql/qh)
/// into a U8 type so the type checker accepts them. The kernel prolog
/// sets these from block memory.
fn coerce_external_types(
    op: &DequantOp,
    externals: &HashSet<u32>,
    reg_types: &mut std::collections::HashMap<u32, RegType>,
) {
    let as_u8 = |rt: &mut std::collections::HashMap<u32, RegType>, reg: u32| {
        if externals.contains(&reg) && matches!(rt.get(&reg), Some(RegType::F32)) {
            rt.insert(reg, RegType::U8);
        }
    };
    match *op {
        DequantOp::ExtractNibble { src, .. }
        | DequantOp::ExtractBits { src, .. }
        | DequantOp::IntToFloat { src, .. } => as_u8(reg_types, src),
        DequantOp::Combine6Bit { ql, qh, .. } => {
            as_u8(reg_types, ql);
            as_u8(reg_types, qh);
        }
        DequantOp::CombineBits { lo, hi, .. } => {
            as_u8(reg_types, lo);
            as_u8(reg_types, hi);
        }
        _ => {}
    }
}

fn type_check_op(
    op: &DequantOp,
    op_index: usize,
    reg_types: &std::collections::HashMap<u32, RegType>,
    errors: &mut Vec<ValidationError>,
) {
    let require = |reg: u32,
                   expected: &'static str,
                   ok: fn(RegType) -> bool,
                   errors: &mut Vec<ValidationError>| {
        if let Some(t) = reg_types.get(&reg) {
            if !ok(*t) {
                errors.push(ValidationError::TypeMismatch {
                    op_index,
                    op_name: op.name(),
                    expected,
                    got: reg_type_name(*t),
                });
            }
        }
    };

    match *op {
        DequantOp::ExtractNibble { src, .. } | DequantOp::ExtractBits { src, .. } => {
            require(
                src,
                "U8/U16",
                |t| matches!(t, RegType::U8 | RegType::U16),
                errors,
            );
        }
        DequantOp::Combine6Bit { ql, qh, .. } => {
            require(ql, "U8", |t| matches!(t, RegType::U8), errors);
            require(qh, "U8", |t| matches!(t, RegType::U8), errors);
        }
        DequantOp::CombineBits { lo, hi, .. } => {
            require(lo, "U8", |t| matches!(t, RegType::U8), errors);
            require(hi, "U8", |t| matches!(t, RegType::U8), errors);
        }
        DequantOp::IntToFloat { src, .. } => {
            require(
                src,
                "U8/I8",
                |t| matches!(t, RegType::U8 | RegType::I8),
                errors,
            );
        }
        DequantOp::MulF32 { a, b, .. }
        | DequantOp::SubF32 { a, b, .. }
        | DequantOp::AddF32 { a, b, .. } => {
            require(a, "F32", |t| matches!(t, RegType::F32), errors);
            require(b, "F32", |t| matches!(t, RegType::F32), errors);
        }
        DequantOp::FmaF32 { a, b, c, .. } => {
            require(a, "F32", |t| matches!(t, RegType::F32), errors);
            require(b, "F32", |t| matches!(t, RegType::F32), errors);
            require(c, "F32", |t| matches!(t, RegType::F32), errors);
        }
        DequantOp::NegF32 { src, .. } => {
            require(src, "F32", |t| matches!(t, RegType::F32), errors);
        }
        DequantOp::DowncastToHalf { src, .. } | DequantOp::DowncastToFP8 { src, .. } => {
            require(src, "F32", |t| matches!(t, RegType::F32), errors);
        }
        DequantOp::StoreHalf { src, .. } => {
            require(
                src,
                "F16/BF16",
                |t| matches!(t, RegType::F16 | RegType::BF16),
                errors,
            );
        }
        DequantOp::StoreFP8 { src, variant, .. } => match variant {
            crate::v1::ir::types::Fp8Variant::E4M3 => {
                require(src, "Fp8E4M3", |t| matches!(t, RegType::Fp8E4M3), errors);
            }
            crate::v1::ir::types::Fp8Variant::E5M2 => {
                require(src, "Fp8E5M2", |t| matches!(t, RegType::Fp8E5M2), errors);
            }
        },
        _ => {}
    }
}

fn reg_type_name(t: RegType) -> &'static str {
    match t {
        RegType::U8 => "U8",
        RegType::U16 => "U16",
        RegType::I8 => "I8",
        RegType::F16 => "F16",
        RegType::BF16 => "BF16",
        RegType::Fp8E4M3 => "Fp8E4M3",
        RegType::Fp8E5M2 => "Fp8E5M2",
        RegType::F32 => "F32",
    }
}

/// Validate a complete format: runs [`validate_program`] on the main
/// dequant program (strict). The optional `unpack_program` in
/// `Packed6Bit` is a pattern template (see module docs) and is
/// NOT SSA-validated here.
pub fn validate_format(fmt: &QuantFormat) -> Result<(), Vec<ValidationError>> {
    validate_program(&fmt.dequant_program, fmt.block_bytes)?;

    // Offset bounds for the sub-scales layout itself (the layout's
    // offset/count is read by the prolog, not by dequant_program ops).
    match fmt.sub_scales_layout {
        SubScalesLayout::Int8Array { offset, count } => {
            if offset + count > fmt.block_bytes {
                return Err(vec![ValidationError::OffsetOutOfBounds {
                    op_index: usize::MAX,
                    offset,
                    count,
                    block_bytes: fmt.block_bytes,
                }]);
            }
        }
        SubScalesLayout::Packed6Bit { offset, count, .. } => {
            if offset + count > fmt.block_bytes {
                return Err(vec![ValidationError::OffsetOutOfBounds {
                    op_index: usize::MAX,
                    offset,
                    count,
                    block_bytes: fmt.block_bytes,
                }]);
            }
        }
        SubScalesLayout::None => {}
    }

    Ok(())
}
