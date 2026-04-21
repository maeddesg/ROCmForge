//! Säule 5 — Quality Monitor.
//!
//! Phase 1: calibration pass at pipeline init, periodic
//! hidden-state drift checks (NaN/Inf/mean-abs z-score/max-abs),
//! token-level repetition heuristic. Phase 1 only **logs** —
//! Phase 2 reuses the same `PrecisionRevisionSignal` to escalate
//! FP8 → FP16 → BF16 → FP32 and re-run the affected layer.
//!
//! The doc's target implementation is a fused epilog in the last
//! kernel of each layer-block that writes a dirty flag into
//! pinned host memory. Phase 1 uses the cheaper CPU-side path:
//! hipMemcpy the post-output-norm hidden state every
//! `sample_rate` tokens and run the check on the host. At
//! sample_rate=32 and ~30 tok/s that's ~1 readback/second —
//! negligible relative to the rest of decode. The fused-epilog
//! path is a Phase-2 optimisation.
//!
//! See architecture_v1.2.0-draft §2.6 and §5.4.

pub mod calibration;
pub mod check;
pub mod repetition;
pub mod types;

pub use calibration::{CALIBRATION_PROMPT, MIN_CALIBRATION_STEPS, OUTPUT_HIDDEN};
pub use types::{
    DriftReason, ExpectedRange, NodeId, PrecisionRevisionSignal, QualityMonitor, RevisionEvent,
};
