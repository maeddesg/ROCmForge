//! Säule 4 — Self-Tuning Runtime.
//!
//! UCB1 multi-armed bandit selects between GA-optimised kernel variants
//! per shape. Timing via HIP events batched at the token-end sync point
//! (zero extra sync in the hot path). See architecture_v1.2.0-draft §2.5.
