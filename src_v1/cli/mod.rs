//! CLI entry points — `rocmforge` (inference only) and hooks for
//! `rf-forge` (offline GA tuning, separate binary under `tools/rf-forge`).
//!
//! See architecture_v1.2.0-draft §4.7.

#[cfg(feature = "gpu")]
pub mod inference_test;
pub mod list_tensors;
