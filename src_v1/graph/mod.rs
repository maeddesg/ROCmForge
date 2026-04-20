//! Säule 2 — Computation Graph + Fusion-Passes.
//!
//! Builds a typed operator DAG from GGUF metadata, applies rule-based
//! fusion patterns, and feeds the fused graph into the kernel-codegen
//! pipeline. See architecture_v1.2.0-draft §2.3.
