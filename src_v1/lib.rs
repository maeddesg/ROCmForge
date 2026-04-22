//! ROCmForge v1.0 — rebuild root.
//!
//! Six-pillar architecture targeting gfx1201 (RDNA 4) + Zen4 AVX-512.
//! Pillars live in their own modules; shared types are in [`core`].
//! Blueprint: `docs/v1.0/architecture_v1.2.0-draft.md`.
//! Dequant IR: `docs/v1.0/dequant_ir_spec.md`.
//!
//! This file is included from `src/lib.rs` under `#[cfg(feature = "v1")]`
//! via a `#[path = "../src_v1/lib.rs"]` attribute — v1.0 code lives in a
//! sibling directory but ships as part of the single `rocmforge` package.

pub mod core;
pub mod ga;
pub mod graph;
pub mod ir;
pub mod runtime;
pub mod monitor;
pub mod introspection;
pub mod backend;
pub mod cli;
