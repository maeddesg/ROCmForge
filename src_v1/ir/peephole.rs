//! Peephole transforms — **shared between GPU and CPU codegens**.
//!
//! This is the single point at which `MulF32 + AddF32` can become
//! `FmaF32`. `dequant_ir_spec.md §7.4 Rule 1` makes it a Parity
//! invariant: GPU and CPU must either both fuse (single-rounding
//! result) or both not fuse (double-rounding result). Putting the
//! transform in a shared module prevents the two emitters from
//! drifting.
//!
//! Passes (§6.8 / §7.2):
//!   1. **Mul + Add → Fma.** `MulF32(a,b,T) + AddF32(T,z,W)` where
//!      `T` is consumed only by the immediately-following `AddF32` is
//!      rewritten to `FmaF32(a,b,z,W)`. The guard on single-use is
//!      important — if `T` is reused later, we have to keep the
//!      separate `Mul`.
//!   2. **Mul + Sub → Fma with negated c.** `MulF32(a,b,T) +
//!      SubF32(T,z,W)` where `T` is a single-use intermediate is
//!      rewritten to `FmaF32(a,b,-z_neg,W)` — but we need a way to
//!      name `-z`. We leave this to the emitter (operand-modifier on
//!      the FmaF32's `c` operand) and do **not** fuse at IR level,
//!      because that would require synthesising a `NegF32` op just to
//!      satisfy the data-flow. Phase-1 formats don't hit this pattern.
//!   3. **NegF32 → operand modifier.** We leave `NegF32` in the IR —
//!      the emitter recognises the `NegF32(X)` → `FmaF32 c: X`
//!      pattern and lifts it into the hardware's built-in operand
//!      modifier (§6.8). No IR-level rewrite.
//!   4. **Const-inline.** `Const { value, dst }` keeps its identity
//!      at IR level; the emitter materialises it as an immediate at
//!      the consumer. No IR-level rewrite.
//!
//! Phase 1 concretely implements pass 1 only. It is enough to cover
//! the "user accidentally wrote Mul+Add instead of Fma" mistake that
//! would otherwise break Parity.

use super::types::DequantOp;

/// Apply all shared peephole passes. Deterministic — same input always
/// yields the same output — so GPU and CPU emitters see the same IR.
pub fn apply_peephole(program: &[DequantOp]) -> Vec<DequantOp> {
    fuse_mul_add_to_fma(program)
}

/// Fuse `MulF32(a, b, T) + AddF32(T, z, W)` (or `AddF32(z, T, W)`) into
/// `FmaF32(a, b, z, W)` when `T` is used nowhere else in the rest of
/// the program.
fn fuse_mul_add_to_fma(program: &[DequantOp]) -> Vec<DequantOp> {
    let mut out: Vec<DequantOp> = Vec::with_capacity(program.len());
    let mut i = 0usize;
    while i < program.len() {
        // Look one op ahead for the Mul→Add pattern.
        if i + 1 < program.len() {
            if let (
                DequantOp::MulF32 { a, b, dst: t1 },
                DequantOp::AddF32 {
                    a: add_a,
                    b: add_b,
                    dst: w,
                },
            ) = (&program[i], &program[i + 1])
            {
                // AddF32 uses `T` as one of its operands. Use the other
                // as `c` in the FMA.
                let (c_reg, uses_t) = if *add_a == *t1 {
                    (*add_b, true)
                } else if *add_b == *t1 {
                    (*add_a, true)
                } else {
                    (*add_a, false)
                };
                if uses_t && !is_used_after(program, i + 2, *t1) {
                    out.push(DequantOp::FmaF32 {
                        a: *a,
                        b: *b,
                        c: c_reg,
                        dst: *w,
                    });
                    i += 2;
                    continue;
                }
            }
        }
        out.push(program[i].clone());
        i += 1;
    }
    out
}

/// `true` iff RegId `r` is referenced as a source by any op in
/// `program[from..]`.
fn is_used_after(program: &[DequantOp], from: usize, r: u32) -> bool {
    program[from..].iter().any(|op| op.srcs().contains(&r))
}
