//! Human-readable IR printer for DequantOp programs.
//!
//! Output format (matches the sketch in the Phase-1.6 prompt):
//!
//! ```text
//! Q4_K Dequant Program (144 bytes, 256 elements, 8 sub-blocks):
//! ──────────────────────────────────────────────────────────
//! [0] r30 = LoadBytes(offset=16, count=128)
//! [1]     ScaleBlockStart(sub_block_idx=0)
//! [2] r32 = ExtractNibble(r31, high=false)
//! [3] r33 = IntToFloat(r32, offset=0)
//! [4] r34 = MulF32(r0, r2)                            // r_d * r_scale_j
//! …
//! [N]     StoreFP8(rX, E4M3, "sub_j * 32 + e * TILE_N + col")
//! ```

use super::types::{DequantOp, Fp8Variant, HalfType, QuantFormat, SubScalesLayout};

/// Render a full format: header + main program + optional unpack program.
pub fn format_program(fmt: &QuantFormat) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} Dequant Program ({} bytes, {} elements, {} sub-blocks):\n",
        fmt.name, fmt.block_bytes, fmt.elements_per_block, fmt.sub_blocks_per_block
    ));
    out.push_str(&"─".repeat(64));
    out.push('\n');
    out.push_str(&render_ops(&fmt.dequant_program));

    if let SubScalesLayout::Packed6Bit {
        unpack_program, ..
    } = &fmt.sub_scales_layout
    {
        out.push_str("\n-- Unpack Program (pattern template, both j-paths):\n");
        out.push_str(&render_ops(unpack_program));
    }
    out
}

/// Convenience — print [`format_program`] to stdout.
pub fn print_program(fmt: &QuantFormat) {
    println!("{}", format_program(fmt));
}

fn render_ops(program: &[DequantOp]) -> String {
    let mut s = String::new();
    for (i, op) in program.iter().enumerate() {
        s.push_str(&format!("[{i:>3}] {}\n", render_op(op)));
    }
    s
}

fn render_op(op: &DequantOp) -> String {
    match op {
        DequantOp::LoadBytes {
            offset,
            count,
            reg,
        } => format!("r{reg} = LoadBytes(offset={offset}, count={count})"),
        DequantOp::LoadFP8 {
            offset,
            count,
            variant,
            reg,
        } => format!(
            "r{reg} = LoadFP8(offset={offset}, count={count}, {})",
            fp8_name(*variant)
        ),
        DequantOp::ExtractNibble { src, high, dst } => {
            format!("r{dst} = ExtractNibble(r{src}, high={high})")
        }
        DequantOp::Combine6Bit {
            ql,
            qh,
            shift,
            dst,
        } => format!("r{dst} = Combine6Bit(ql=r{ql}, qh=r{qh}, shift={shift})"),
        DequantOp::ExtractBits {
            src,
            shift,
            mask,
            dst,
        } => format!("r{dst} = ExtractBits(r{src}, shift={shift}, mask=0x{mask:02X})"),
        DequantOp::CombineBits {
            lo,
            hi,
            hi_shift,
            dst,
        } => format!("r{dst} = CombineBits(lo=r{lo}, hi=r{hi}, hi_shift={hi_shift})"),
        DequantOp::IntToFloat { src, offset, dst } => {
            format!("r{dst} = IntToFloat(r{src}, offset={offset})")
        }
        DequantOp::MulF32 { a, b, dst } => format!("r{dst} = MulF32(r{a}, r{b})"),
        DequantOp::FmaF32 { a, b, c, dst } => format!("r{dst} = FmaF32(r{a}, r{b}, r{c})"),
        DequantOp::SubF32 { a, b, dst } => format!("r{dst} = SubF32(r{a}, r{b})"),
        DequantOp::AddF32 { a, b, dst } => format!("r{dst} = AddF32(r{a}, r{b})"),
        DequantOp::NegF32 { src, dst } => format!("r{dst} = NegF32(r{src})"),
        DequantOp::DowncastToHalf { src, dst, target } => format!(
            "r{dst} = DowncastToHalf(r{src}, {})",
            half_name(*target)
        ),
        DequantOp::DowncastToFP8 {
            src,
            dst,
            variant,
            saturate,
        } => format!(
            "r{dst} = DowncastToFP8(r{src}, {}, saturate={saturate})",
            fp8_name(*variant)
        ),
        DequantOp::StoreHalf {
            src,
            lds_offset_expr,
        } => format!("    StoreHalf(r{src}, \"{lds_offset_expr}\")"),
        DequantOp::StoreFP8 {
            src,
            variant,
            lds_offset_expr,
        } => format!(
            "    StoreFP8(r{src}, {}, \"{lds_offset_expr}\")",
            fp8_name(*variant)
        ),
        DequantOp::ScaleBlockStart { sub_block_idx } => {
            format!("    ScaleBlockStart(sub_block_idx={sub_block_idx})")
        }
        DequantOp::Const { value, dst } => format!("r{dst} = Const({value})"),
    }
}

fn fp8_name(v: Fp8Variant) -> &'static str {
    match v {
        Fp8Variant::E4M3 => "E4M3",
        Fp8Variant::E5M2 => "E5M2",
    }
}

fn half_name(h: HalfType) -> &'static str {
    match h {
        HalfType::Fp16 => "Fp16",
        HalfType::Bf16 => "Bf16",
    }
}
