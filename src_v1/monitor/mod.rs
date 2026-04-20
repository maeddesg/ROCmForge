//! Säule 5 — Quality Monitor.
//!
//! Fused-epilog hidden-state checks (mean-abs, max-abs, NaN, FP8
//! saturation) write a dirty flag into pinned host memory. The CPU polls
//! the flag at the token-end sync (no extra sync). See
//! architecture_v1.2.0-draft §2.6.
