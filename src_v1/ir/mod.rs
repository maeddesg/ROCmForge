//! Säule 3 — Dequant IR.
//!
//! Declarative quantisation-format descriptions (`QuantFormat`) plus the
//! `DequantOp` IR and rule-based codegen for gfx1201 and Zen4. Full
//! specification: `docs/v1.0/dequant_ir_spec.md`.

pub mod codegen_gpu;
pub mod diff;
pub mod formats;
pub mod interpreter;
pub mod peephole;
pub mod printer;
pub mod types;
pub mod validator;
