//! Types for the Quality Monitor — exact field names from
//! architecture_v1.2.0-draft §2.6. Phase-1 extensions over the doc
//! core set (added `InfDetected`, `AttentionCollapse`,
//! `RepetitionDetected`) are additive and covered by the same
//! `DriftReason` enum; downstream code matches on the new variants
//! explicitly, so existing doc-variant consumers stay source-compatible.

use std::collections::HashMap;

use crate::v1::introspection::PrecisionHint;

/// Identifies a graph node within the executor's node list. Phase-1
/// uses this as an opaque handle — the monitor stores one
/// `ExpectedRange` per watch-point and the executor maps those to
/// the buffers it reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Per-watch-point calibration band. Populated by
/// `QualityMonitor::calibrate` and read by `check_hidden_state`.
#[derive(Debug, Clone)]
pub struct ExpectedRange {
    pub mean_abs_expected: f32,
    pub mean_abs_stddev: f32,
    pub max_abs_expected: f32,
    /// Number of NaN values that is still tolerated (normally 0 —
    /// any NaN in a hidden state means the forward pass broke).
    pub nan_tolerance: u32,
}

/// Precision-revision signal — Phase 1 only logs these; Phase 2
/// consumes them to escalate FP8 → FP16 → BF16 → FP32 on the
/// affected layer.
#[derive(Debug, Clone)]
pub struct PrecisionRevisionSignal {
    pub affected_node: NodeId,
    pub current_precision: PrecisionHint,
    pub recommended_precision: PrecisionHint,
    pub reason: DriftReason,
}

/// Reasons a monitor check can fire. `Fp8SaturationExceeded`,
/// `AttentionCollapse` and `RepetitionDetected` are Phase-1+
/// extensions but included in the enum for forward compatibility
/// with Phase 2.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftReason {
    /// `|z_score| > tolerance_factor` against the calibrated band.
    MeanAbsExceedsStdTolerance {
        observed: f32,
        expected: f32,
        z_score: f32,
    },
    /// Max-Abs exceeds the FP16-overflow threshold (default 60000).
    MaxAbsExceedsThreshold { observed: f32, threshold: f32 },
    /// One or more NaN values in the hidden state.
    NaNDetected { count: u32 },
    /// One or more Inf values in the hidden state.
    InfDetected { count: u32 },
    /// FP8 saturation count beyond tolerance (Phase 2).
    Fp8SaturationExceeded {
        count: u32,
        total: u32,
        fraction: f32,
    },
    /// Attention-entropy collapse (Phase 2 candidate).
    AttentionCollapse { entropy: f32, threshold: f32 },
    /// Same token repeated N times in a row — Phase 1 heuristic.
    RepetitionDetected { token_id: u32, count: u32 },
}

/// One logged drift event — Phase 1 writes the struct and moves on.
/// Phase 2 flips `resolved=true` once the re-run with higher
/// precision passes the same check.
#[derive(Debug, Clone)]
pub struct RevisionEvent {
    pub token_index: u64,
    pub node_id: NodeId,
    pub signal: PrecisionRevisionSignal,
    pub resolved: bool,
}

pub struct QualityMonitor {
    /// Calibrated watch-points. Populated by `calibrate`.
    pub expected_ranges: HashMap<NodeId, ExpectedRange>,
    /// Drift band: `|z_score| > tolerance_factor` triggers a signal.
    /// Default 3.0 — conservative, favours false negatives over
    /// false positives during Phase-1 validation.
    pub tolerance_factor: f32,
    /// Run the hidden-state check every `sample_rate` decode tokens.
    /// Phase 1 default 32 — at 30 tok/s that's ~1 check/second, so
    /// the GPU→CPU copy overhead is negligible.
    pub sample_rate: u32,
    /// All drift events seen during the current session. Phase 1
    /// just logs to stderr; this vector gives callers a
    /// machine-readable copy for reports.
    pub revision_log: Vec<RevisionEvent>,
    /// Token counter since the last sampled hidden-state check.
    pub(crate) tokens_since_check: u64,
    /// Most recent decoded token ids, used by `check_repetition`.
    pub(crate) recent_tokens: Vec<u32>,
}

impl QualityMonitor {
    pub fn new(sample_rate: u32, tolerance_factor: f32) -> Self {
        Self {
            expected_ranges: HashMap::new(),
            tolerance_factor,
            sample_rate,
            revision_log: Vec::new(),
            tokens_since_check: 0,
            recent_tokens: Vec::new(),
        }
    }
}
