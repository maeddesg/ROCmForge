//! Calibration pass for the Quality Monitor.
//!
//! Arch-Doc §2.6: "Die Kalibrierungsdaten werden beim ersten
//! Inference-Durchlauf mit einem kurzen Prompt aufgenommen — eine
//! Sekunde Warmup, einmalig pro Modell." We run the classic
//! "The quick brown fox…" prompt through the executor, snapshot
//! the final hidden state on each decode step, and compute
//! per-step magnitude stats. The mean and stddev across steps
//! become the calibrated band for the `NodeId::OUTPUT_HIDDEN`
//! watch-point.
//!
//! Phase 1 only calibrates one watch-point (the final
//! post-output-norm hidden state) — that's what the executor
//! can expose cheaply and what drives all observable drift
//! symptoms. Phase 2 can sprinkle more points (post-attention,
//! post-FFN per layer) without changing this interface.

use super::types::{ExpectedRange, NodeId, QualityMonitor};

/// Watch-point id used by Phase 1. The executor reads the
/// final post-output-norm hidden state under this key.
pub const OUTPUT_HIDDEN: NodeId = NodeId(0);

/// Phase-1 calibration prompt. English sentence with varied
/// letter frequencies so the hidden state exercises typical
/// token paths without overly model-specific biases.
pub const CALIBRATION_PROMPT: &str = "The quick brown fox jumps over the lazy dog";

/// Minimum decode steps to collect before the band is considered
/// stable. More steps = tighter stddev but longer warm-up.
pub const MIN_CALIBRATION_STEPS: usize = 10;

impl QualityMonitor {
    /// Feed in the collected per-step (mean_abs, max_abs) samples
    /// and build the ExpectedRange for the `OUTPUT_HIDDEN` node.
    /// The caller (pipeline init) runs the executor and harvests
    /// the samples; the monitor stays pure-CPU and testable.
    pub fn install_calibration(
        &mut self,
        node: NodeId,
        samples: &[(f32, f32)],
    ) {
        if samples.is_empty() {
            return;
        }
        let n = samples.len() as f32;
        let mean_abs_mean: f32 = samples.iter().map(|(m, _)| *m).sum::<f32>() / n;
        let mean_abs_var: f32 = samples
            .iter()
            .map(|(m, _)| {
                let d = m - mean_abs_mean;
                d * d
            })
            .sum::<f32>()
            / n;
        let mean_abs_stddev = mean_abs_var.sqrt().max(1e-4);
        let max_abs_mean: f32 = samples.iter().map(|(_, mx)| *mx).sum::<f32>() / n;
        self.expected_ranges.insert(
            node,
            ExpectedRange {
                mean_abs_expected: mean_abs_mean,
                mean_abs_stddev,
                max_abs_expected: max_abs_mean,
                nan_tolerance: 0,
            },
        );
    }
}
