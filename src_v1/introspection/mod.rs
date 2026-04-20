//! Säule 1 — Model Introspection.
//!
//! Scans GGUF weights at load time, produces a `ModelProfile` with
//! magnitude statistics, SNR risk scores, and per-layer precision
//! recommendations. See architecture_v1.2.0-draft §2.2.
