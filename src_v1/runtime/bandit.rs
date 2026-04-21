//! UCB1 multi-armed bandit for runtime kernel-variant selection.
//!
//! Auer, Cesa-Bianchi, Fischer (2002), "Finite-time Analysis of the
//! Multiarmed Bandit Problem". We *minimise* execution time, so for
//! each call we pick the arm that maximises
//!
//!     score(arm) = -mean_time_us(arm) + c * sqrt(ln(N_total) / n_arm)
//!
//! matching architecture_v1.2.0-draft §2.5. Using `-mean_time` (rather
//! than `1/time`) keeps the reward and the exploration term on the
//! same order of magnitude: kernel times are ~100 µs, so a c≈1.4
//! exploration bonus of O(0.5) decays quickly and UCB1 converges to
//! the fastest arm within a few dozen pulls — exactly the Phase-1
//! target.
//!
//! The first `n_arms` pulls hand out every variant at least once;
//! only after that does the UCB1 formula kick in.

use super::variants::VariantId;

#[derive(Debug, Clone)]
pub struct BanditArm {
    pub variant_id: VariantId,
    pub n_pulls: u64,
    pub mean_time_us: f64,
    pub best_time_us: f64,
}

impl BanditArm {
    fn new(id: VariantId) -> Self {
        Self {
            variant_id: id,
            n_pulls: 0,
            mean_time_us: 0.0,
            best_time_us: f64::INFINITY,
        }
    }
}

#[derive(Debug)]
pub struct ShapeBandit {
    pub arms: Vec<BanditArm>,
    pub total_pulls: u64,
    pub exploration_constant: f64,
}

impl ShapeBandit {
    pub fn new(variant_ids: &[VariantId]) -> Self {
        let arms = variant_ids.iter().map(|&id| BanditArm::new(id)).collect();
        Self {
            arms,
            total_pulls: 0,
            exploration_constant: 1.4,
        }
    }

    /// Pick the arm to pull this round. Round-robin until every arm
    /// has at least one sample; UCB1 afterwards.
    pub fn select(&self) -> VariantId {
        for arm in &self.arms {
            if arm.n_pulls == 0 {
                return arm.variant_id;
            }
        }

        let ln_n = (self.total_pulls as f64).ln();
        let c = self.exploration_constant;

        self.arms
            .iter()
            .max_by(|a, b| {
                let ucb_a = -a.mean_time_us + c * (ln_n / a.n_pulls as f64).sqrt();
                let ucb_b = -b.mean_time_us + c * (ln_n / b.n_pulls as f64).sqrt();
                ucb_a
                    .partial_cmp(&ucb_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("bandit has at least one arm")
            .variant_id
    }

    /// Record an execution of `variant` that took `time_us`. The
    /// mean is updated as an exact running average rather than
    /// Welford — we don't need the variance in Phase 1 and the
    /// running-average form is cheaper.
    pub fn record(&mut self, variant_id: VariantId, time_us: f64) {
        let t = time_us.max(1e-3);
        if let Some(arm) = self
            .arms
            .iter_mut()
            .find(|a| a.variant_id == variant_id)
        {
            let old_n = arm.n_pulls as f64;
            arm.mean_time_us = (arm.mean_time_us * old_n + t) / (old_n + 1.0);
            arm.n_pulls += 1;
            if t < arm.best_time_us {
                arm.best_time_us = t;
            }
        }
        self.total_pulls += 1;
    }

    /// The arm with the most pulls so far. Equivalent to "the arm
    /// UCB1 would pick in pure exploitation" once the exploration
    /// term has decayed below the gap between arms.
    pub fn best_variant(&self) -> VariantId {
        self.arms
            .iter()
            .max_by_key(|a| a.n_pulls)
            .expect("bandit has at least one arm")
            .variant_id
    }

    /// Heuristic: true once every arm has ≥5 samples and we've taken
    /// ≥50 pulls total. Used by reports to label the state.
    pub fn is_exploiting(&self) -> bool {
        self.arms.iter().all(|a| a.n_pulls >= 5) && self.total_pulls > 50
    }

    /// Pretty-print per-arm stats to stdout.
    pub fn print_stats(&self) {
        println!(
            "  [bandit] total_pulls={} phase={}",
            self.total_pulls,
            if self.is_exploiting() {
                "exploiting"
            } else {
                "exploring"
            }
        );
        for arm in &self.arms {
            let pct = if self.total_pulls > 0 {
                arm.n_pulls as f64 / self.total_pulls as f64 * 100.0
            } else {
                0.0
            };
            let best = if arm.best_time_us.is_finite() {
                format!("{:.2}µs", arm.best_time_us)
            } else {
                "—".to_string()
            };
            println!(
                "    variant {:>3}: {:>5} pulls ({:>5.1}%)  mean={:>6.2}µs  best={}",
                arm.variant_id.0, arm.n_pulls, pct, arm.mean_time_us, best
            );
        }
    }
}
