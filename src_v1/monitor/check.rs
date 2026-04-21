//! Hidden-state drift check. Pure CPU — the caller pulls the
//! hidden state GPU→CPU before invoking `check_hidden_state`.

use crate::v1::introspection::PrecisionHint;

use super::types::{DriftReason, NodeId, PrecisionRevisionSignal, QualityMonitor};

/// FP16 upper bound is 65 504. The monitor raises an overflow
/// guard once we get within 10 % of that so the forward pass has
/// headroom before a real overflow takes the logits out.
const MAX_ABS_THRESHOLD: f32 = 60_000.0;

impl QualityMonitor {
    /// Inspect one hidden state vs. the calibrated band for `node`.
    /// Order matters — NaN / Inf checks short-circuit before any
    /// arithmetic, so a corrupted state can't produce misleading
    /// z-scores downstream.
    pub fn check_hidden_state(
        &self,
        node: NodeId,
        hidden_state: &[f32],
    ) -> Option<PrecisionRevisionSignal> {
        let range = self.expected_ranges.get(&node)?;

        // 1. NaN / Inf short-circuit.
        let mut nan_count = 0u32;
        let mut inf_count = 0u32;
        for v in hidden_state {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            }
        }
        if nan_count > range.nan_tolerance {
            return Some(PrecisionRevisionSignal {
                affected_node: node,
                current_precision: PrecisionHint::Fp16Scales,
                recommended_precision: PrecisionHint::Fp32Scales,
                reason: DriftReason::NaNDetected { count: nan_count },
            });
        }
        if inf_count > 0 {
            return Some(PrecisionRevisionSignal {
                affected_node: node,
                current_precision: PrecisionHint::Fp16Scales,
                recommended_precision: PrecisionHint::Fp32Scales,
                reason: DriftReason::InfDetected { count: inf_count },
            });
        }

        // 2. Max-abs threshold — checked before the z-score so a
        // single extreme outlier (FP16-overflow class failure)
        // isn't mis-reported as "mean-abs drifted" just because
        // the outlier dominates the sum.
        let n = hidden_state.len().max(1) as f32;
        let mut sum_abs = 0.0f32;
        let mut max_abs = 0.0f32;
        for x in hidden_state {
            let a = x.abs();
            sum_abs += a;
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs > MAX_ABS_THRESHOLD {
            return Some(PrecisionRevisionSignal {
                affected_node: node,
                current_precision: PrecisionHint::Fp16Scales,
                recommended_precision: PrecisionHint::Fp32Scales,
                reason: DriftReason::MaxAbsExceedsThreshold {
                    observed: max_abs,
                    threshold: MAX_ABS_THRESHOLD,
                },
            });
        }

        // 3. Mean-abs z-score.
        let mean_abs = sum_abs / n;
        let sigma = range.mean_abs_stddev.max(1e-6);
        let z_score = (mean_abs - range.mean_abs_expected) / sigma;
        if z_score.abs() > self.tolerance_factor {
            return Some(PrecisionRevisionSignal {
                affected_node: node,
                current_precision: PrecisionHint::Fp16Scales,
                recommended_precision: PrecisionHint::Bf16Scales,
                reason: DriftReason::MeanAbsExceedsStdTolerance {
                    observed: mean_abs,
                    expected: range.mean_abs_expected,
                    z_score,
                },
            });
        }

        None
    }
}
