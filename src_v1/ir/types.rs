//! Dequant IR — core types (§2 of `dequant_ir_spec.md`).
//!
//! All 17 DequantOps, typed RegIds (SSA), scalar/half-type discriminants,
//! and the `QuantFormat` struct live here. The names match
//! `architecture_v1.2.0-draft.md` §2.4 where both docs agree; every
//! `[SPEC-ERWEITERUNG]` op from the Dequant IR spec is included.
//!
//! Nomenclature (arch-doc wins on conflicts, per project rule):
//!   - `Combine6Bit` (NOT `CombineBits` for the 6-bit path; `CombineBits`
//!     is a separate op for generic 2-field combines).
//!   - `DowncastToHalf` / `StoreHalf` (NOT `ToFP16` / `StoreFp16`).
//!   - `LoadBytes` (NOT `LoadU8`; byte count is explicit via `count`).
//!
//! Invariants enforced by `super::validator`:
//!   1. SSA — every `dst` / `reg` RegId is written exactly once.
//!   2. Use-before-def — every source RegId is defined beforehand.
//!   3. Type consistency — arithmetic ops operate on F32 registers.
//!   4. Offset bounds — `LoadBytes`/`LoadFP8` offsets stay inside the block.
//!   5. Monotonic `ScaleBlockStart` — sub_block_idx is non-decreasing.
//!   6. Terminal Store — each program ends with `StoreHalf` or `StoreFP8`.

/// Small expression sub-language used by `StoreHalf`/`StoreFP8` to describe
/// the LDS write offset — matches the kanonical definition in
/// `architecture_v1.2.0-draft.md §2.4` (`pub type Expr = String;`).
///
/// Parsed by the code-generator, not at IR level. Typical form:
/// `"elem_idx * TILE_N + col"` or `"sub_j * 32 + e * TILE_N + col"`.
pub type Expr = String;

/// Register identifier — mapped to physical VGPR / ZMM by the codegen's
/// linear-scan allocator (§6.7). Every RegId is written exactly once
/// (SSA, §2.2), so its type is fixed at definition time.
pub type RegId = u32;

/// Typed register contents. Arithmetic is always FP32 (§1.6 principle 3);
/// sub-FP32 types only appear at Load/Store boundaries and the final
/// downcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegType {
    /// 8-bit unsigned — one byte out of block memory.
    U8,
    /// 16-bit unsigned — byte pair (e.g. FP16 raw bits before `IntToFloat`).
    U16,
    /// 8-bit signed — Q6_K int8 scales, Q8_0 signed quants.
    I8,
    /// IEEE-754 binary16 (ggml `ggml_fp16_t`).
    F16,
    /// Brain-float: 8-bit exponent, 7-bit mantissa.
    BF16,
    /// OCP E4M3 — ±448 range, default WMMA input.
    Fp8E4M3,
    /// OCP E5M2 — ±57 344 range, KV-cache default.
    Fp8E5M2,
    /// IEEE-754 binary32 — the IR arithmetic format.
    F32,
}

/// FP8 sub-variant discriminant — matches `architecture_v1.2.0-draft §2.4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Variant {
    /// 4 exponent bits, 3 mantissa bits; ±448. Default for weights.
    E4M3,
    /// 5 exponent bits, 2 mantissa bits; ±57 344. Default for KV cache.
    E5M2,
}

/// Target format for `DowncastToHalf`. Reduced to `{Fp16, Bf16}` per
/// STOP-1 resolution point 5 — FP8 downcast lives in its own op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HalfType {
    Fp16,
    Bf16,
}

/// Scalar format for block-header fields (`d`, `dmin`, …).
/// Taken from `architecture_v1.2.0-draft §2.4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Fp16,
    Bf16,
    Fp32,
    Int8,
    Fp8E4M3,
    Fp8E5M2,
}

/// The 17 DequantOps covering all five Phase-1 formats (§2.10).
///
/// Layout groups follow §2 — storage, extraction/bit-ops, arithmetic,
/// int→float, downcast, store, specials. Each variant's semantics match
/// the spec paragraph of the same name in §2.3 – §2.9.
#[derive(Debug, Clone, PartialEq)]
pub enum DequantOp {
    // --- Category 1: Memory ops (§2.3) -------------------------------
    /// Load `count` bytes at `block_ptr + offset` into `reg`.
    /// Count → reg type: 1 → `U8`, 2 → `U16`, ≥4 → vector-U8.
    LoadBytes { offset: usize, count: usize, reg: RegId },
    /// Load `count` FP8 values (weights/KV native). Not used by the five
    /// Phase-1 formats; reserved for future native-FP8 models.
    LoadFP8 {
        offset: usize,
        count: usize,
        variant: Fp8Variant,
        reg: RegId,
    },

    // --- Category 2: Extraction / bit manipulation (§2.4) ------------
    /// Single-byte nibble extract.
    /// `high == false`: `dst = src & 0x0F`. `high == true`: `dst = src >> 4`.
    ExtractNibble { src: RegId, high: bool, dst: RegId },
    /// Q6_K 6-bit reconstruction: `dst = (ql & 0x0F) | (((qh >> shift) & 0x03) << 4)`.
    Combine6Bit {
        ql: RegId,
        qh: RegId,
        shift: u8,
        dst: RegId,
    },
    /// Generic bitfield extract — `dst = (src >> shift) & mask`.
    /// [SPEC-ERWEITERUNG] §2.4 for Q4_K scale unpack.
    ExtractBits {
        src: RegId,
        shift: u8,
        mask: u32,
        dst: RegId,
    },
    /// Combine two bit-fields — `dst = lo | (hi << hi_shift)`.
    /// [SPEC-ERWEITERUNG] §2.4 for Q4_K scale unpack (j ≥ 4 path).
    CombineBits {
        lo: RegId,
        hi: RegId,
        hi_shift: u8,
        dst: RegId,
    },

    // --- Category 3: Int → FP32 (§2.6) -------------------------------
    /// `dst = (src as fp32) + offset`. `src` may be `U8` or `I8` — sign
    /// handling follows the register type.
    IntToFloat { src: RegId, offset: i32, dst: RegId },

    // --- Category 4: FP32 arithmetic (§2.5) --------------------------
    /// `dst = a * b`, RNE.
    MulF32 { a: RegId, b: RegId, dst: RegId },
    /// `dst = a * b + c`, single-rounding RNE (IEEE-754 `fma`).
    FmaF32 { a: RegId, b: RegId, c: RegId, dst: RegId },
    /// `dst = a - b`, RNE. [SPEC-ERWEITERUNG]
    SubF32 { a: RegId, b: RegId, dst: RegId },
    /// `dst = a + b`, RNE. [SPEC-ERWEITERUNG]
    AddF32 { a: RegId, b: RegId, dst: RegId },
    /// `dst = -src`, bit-flip on the sign bit (exact). [SPEC-ERWEITERUNG]
    NegF32 { src: RegId, dst: RegId },

    // --- Category 5: Downcast (terminal before store, §2.7) ----------
    /// FP32 → FP16 / BF16.
    DowncastToHalf {
        src: RegId,
        dst: RegId,
        target: HalfType,
    },
    /// FP32 → FP8 (E4M3 or E5M2); `saturate == true` clamps to ±max.
    DowncastToFP8 {
        src: RegId,
        dst: RegId,
        variant: Fp8Variant,
        saturate: bool,
    },

    // --- Category 6: Store (terminal, §2.8) --------------------------
    /// Write `src` (F16 or BF16) into LDS at `lds_offset_expr`.
    StoreHalf { src: RegId, lds_offset_expr: Expr },
    /// Write `src` (FP8) into LDS at `lds_offset_expr`; 1 byte per elem.
    StoreFP8 {
        src: RegId,
        variant: Fp8Variant,
        lds_offset_expr: Expr,
    },

    // --- Category 7: Specials (§2.9) ---------------------------------
    /// Non-emitting marker — tells the codegen a sub-block boundary was
    /// reached, so `r_scale[j]` / `r_min[j]` can be hoisted out of the
    /// element loop. [SPEC-ERWEITERUNG]
    ScaleBlockStart { sub_block_idx: u8 },
    /// `dst = value` — emitted as an immediate by the codegen.
    /// [SPEC-ERWEITERUNG] — lets Q4_0 / Q6_K programs read as
    /// "centre at 8" / "centre at 32" instead of hiding the offset
    /// inside `IntToFloat`.
    Const { value: f32, dst: RegId },
}

impl DequantOp {
    /// Op-kind name (for error messages and the IR printer).
    pub fn name(&self) -> &'static str {
        match self {
            Self::LoadBytes { .. } => "LoadBytes",
            Self::LoadFP8 { .. } => "LoadFP8",
            Self::ExtractNibble { .. } => "ExtractNibble",
            Self::Combine6Bit { .. } => "Combine6Bit",
            Self::ExtractBits { .. } => "ExtractBits",
            Self::CombineBits { .. } => "CombineBits",
            Self::IntToFloat { .. } => "IntToFloat",
            Self::MulF32 { .. } => "MulF32",
            Self::FmaF32 { .. } => "FmaF32",
            Self::SubF32 { .. } => "SubF32",
            Self::AddF32 { .. } => "AddF32",
            Self::NegF32 { .. } => "NegF32",
            Self::DowncastToHalf { .. } => "DowncastToHalf",
            Self::DowncastToFP8 { .. } => "DowncastToFP8",
            Self::StoreHalf { .. } => "StoreHalf",
            Self::StoreFP8 { .. } => "StoreFP8",
            Self::ScaleBlockStart { .. } => "ScaleBlockStart",
            Self::Const { .. } => "Const",
        }
    }

    /// `true` iff this op terminates a dequant program.
    pub fn is_terminal_store(&self) -> bool {
        matches!(self, Self::StoreHalf { .. } | Self::StoreFP8 { .. })
    }

    /// RegId that this op writes (if any). `LoadBytes`/`LoadFP8` write
    /// into `reg`; arithmetic/bit ops write into `dst`; stores and
    /// `ScaleBlockStart` do not write any register.
    pub fn dst(&self) -> Option<RegId> {
        match *self {
            Self::LoadBytes { reg, .. } | Self::LoadFP8 { reg, .. } => Some(reg),
            Self::ExtractNibble { dst, .. }
            | Self::Combine6Bit { dst, .. }
            | Self::ExtractBits { dst, .. }
            | Self::CombineBits { dst, .. }
            | Self::IntToFloat { dst, .. }
            | Self::MulF32 { dst, .. }
            | Self::FmaF32 { dst, .. }
            | Self::SubF32 { dst, .. }
            | Self::AddF32 { dst, .. }
            | Self::NegF32 { dst, .. }
            | Self::DowncastToHalf { dst, .. }
            | Self::DowncastToFP8 { dst, .. }
            | Self::Const { dst, .. } => Some(dst),
            Self::StoreHalf { .. } | Self::StoreFP8 { .. } | Self::ScaleBlockStart { .. } => None,
        }
    }

    /// All source RegIds read by this op.
    pub fn srcs(&self) -> Vec<RegId> {
        match *self {
            Self::LoadBytes { .. } | Self::LoadFP8 { .. } | Self::ScaleBlockStart { .. } => vec![],
            Self::ExtractNibble { src, .. }
            | Self::NegF32 { src, .. }
            | Self::IntToFloat { src, .. }
            | Self::ExtractBits { src, .. }
            | Self::DowncastToHalf { src, .. }
            | Self::DowncastToFP8 { src, .. } => vec![src],
            Self::StoreHalf { src, .. } | Self::StoreFP8 { src, .. } => vec![src],
            Self::Combine6Bit { ql, qh, .. } => vec![ql, qh],
            Self::CombineBits { lo, hi, .. } => vec![lo, hi],
            Self::MulF32 { a, b, .. } | Self::SubF32 { a, b, .. } | Self::AddF32 { a, b, .. } => {
                vec![a, b]
            }
            Self::FmaF32 { a, b, c, .. } => vec![a, b, c],
            Self::Const { .. } => vec![],
        }
    }

    /// Register type produced by this op, given the source types. Used
    /// by the validator's type inference. Returns `None` for stores and
    /// `ScaleBlockStart`.
    pub fn result_type(&self, src_types: &[RegType]) -> Option<RegType> {
        match *self {
            Self::LoadBytes { count, .. } => match count {
                1 => Some(RegType::U8),
                2 => Some(RegType::U16),
                _ => Some(RegType::U8), // vector-U8 — treated as U8 for scalar type-check
            },
            Self::LoadFP8 { variant, .. } => Some(match variant {
                Fp8Variant::E4M3 => RegType::Fp8E4M3,
                Fp8Variant::E5M2 => RegType::Fp8E5M2,
            }),
            Self::ExtractNibble { .. } | Self::ExtractBits { .. } | Self::CombineBits { .. } => {
                Some(RegType::U8)
            }
            Self::Combine6Bit { .. } => Some(RegType::U8),
            Self::IntToFloat { .. } => Some(RegType::F32),
            Self::MulF32 { .. }
            | Self::FmaF32 { .. }
            | Self::SubF32 { .. }
            | Self::AddF32 { .. }
            | Self::NegF32 { .. } => Some(RegType::F32),
            Self::DowncastToHalf { target, .. } => Some(match target {
                HalfType::Fp16 => RegType::F16,
                HalfType::Bf16 => RegType::BF16,
            }),
            Self::DowncastToFP8 { variant, .. } => Some(match variant {
                Fp8Variant::E4M3 => RegType::Fp8E4M3,
                Fp8Variant::E5M2 => RegType::Fp8E5M2,
            }),
            Self::Const { .. } => Some(RegType::F32),
            Self::StoreHalf { .. } | Self::StoreFP8 { .. } | Self::ScaleBlockStart { .. } => None,
        }
    }
    // Parameter hook used by the validator/type inference — some ops
    // (e.g. `ExtractNibble`) only accept integer-typed sources.
    #[inline]
    fn _keep_srctypes_unused(_t: &[RegType]) {}
}

/// `QuantFormatId` — matches llama.cpp `ggml_type` numeric codes so the
/// loader can map 1:1 from the GGUF header (§3).
pub type QuantFormatId = u32;

/// Sub-block scale layout, from `architecture_v1.2.0-draft §2.4`.
/// `None` means the format has no sub-blocks (Q4_0, Q4_1, Q8_0).
#[derive(Debug, Clone)]
pub enum SubScalesLayout {
    /// One int8 per sub-block, packed contiguously. Q6_K.
    Int8Array { offset: usize, count: usize },
    /// Sub-scales packed into 6-bit fields; `unpack_program` extracts
    /// them. Q4_K.
    Packed6Bit {
        offset: usize,
        count: usize,
        unpack_program: Vec<DequantOp>,
    },
    /// No sub-block scales.
    None,
}

/// One quantisation format — full dequant program + block layout.
#[derive(Debug, Clone)]
pub struct QuantFormat {
    pub id: QuantFormatId,
    pub name: &'static str,
    pub block_bytes: usize,
    pub elements_per_block: usize,
    pub sub_blocks_per_block: usize,
    pub sub_block_size: usize,

    pub dequant_program: Vec<DequantOp>,

    pub block_scale_offset: usize,
    pub block_scale_type: ScalarType,

    pub block_min_offset: Option<usize>,
    pub block_min_type: Option<ScalarType>,

    pub sub_scales_layout: SubScalesLayout,
}
