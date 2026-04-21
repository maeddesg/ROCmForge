//! Säule 4 — Self-Tuning Runtime.
//!
//! UCB1 multi-armed bandit selects between kernel variants per
//! shape. Timing is collected via `Instant::now()` around the
//! launch + stream sync (Phase 1). Phase 2 moves this to HIP-event
//! timing batched at the token-end sync point so the Bandit adds
//! zero extra sync in the hot path — see architecture_v1.2.0-draft
//! §2.5.
//!
//! Public entry points:
//!   * [`VariantRegistry`] — compile-time list of candidate kernels
//!   * [`ShapeBandit`]     — UCB1 state for one shape
//!   * [`Runtime`]         — holds the registry and a Bandit per
//!                            multi-variant shape

pub mod bandit;
pub mod variants;

use std::collections::HashMap;

pub use bandit::{BanditArm, ShapeBandit};
pub use variants::{KernelId, KernelVariant, OpType, ShapeKey, VariantId, VariantRegistry};

pub struct Runtime {
    pub registry: VariantRegistry,
    pub bandits: HashMap<ShapeKey, ShapeBandit>,
    /// Gate for turning the Bandit off in benchmarks / A-B tests.
    /// When `false`, `select_variant` returns `best_variant` (the
    /// arm with the most pulls) instead of the UCB1 pick.
    pub tuning_enabled: bool,
}

impl Runtime {
    pub fn new(registry: VariantRegistry) -> Self {
        let mut bandits = HashMap::new();
        for (shape, variants) in &registry.variants {
            if variants.len() > 1 {
                let ids: Vec<VariantId> = variants.iter().map(|v| v.id).collect();
                bandits.insert(*shape, ShapeBandit::new(&ids));
            }
        }
        Self {
            registry,
            bandits,
            tuning_enabled: true,
        }
    }

    /// Pick the variant to run. Shapes with a single registered
    /// kernel skip the Bandit entirely — there's nothing to tune.
    pub fn select_variant(&self, shape: &ShapeKey) -> Option<VariantId> {
        if let Some(bandit) = self.bandits.get(shape) {
            if self.tuning_enabled {
                Some(bandit.select())
            } else {
                Some(bandit.best_variant())
            }
        } else {
            self.registry.get_variants(shape).first().map(|v| v.id)
        }
    }

    pub fn kernel_for(&self, shape: &ShapeKey, variant: VariantId) -> Option<KernelId> {
        self.registry.lookup(shape, variant)
    }

    pub fn record(&mut self, shape: &ShapeKey, variant_id: VariantId, time_us: f64) {
        if let Some(bandit) = self.bandits.get_mut(shape) {
            bandit.record(variant_id, time_us);
        }
    }

    /// Force a specific variant — used by A/B tests that want to
    /// pin one arm and compare logits against another. No-op if the
    /// shape has no Bandit (single variant anyway).
    pub fn force_variant(&mut self, shape: &ShapeKey, variant_id: VariantId) {
        self.tuning_enabled = false;
        if let Some(bandit) = self.bandits.get_mut(shape) {
            // Bias the selection by setting every other arm's
            // `best_variant`-proxy lower. Implemented as a pull of
            // the chosen variant; sufficient for test fixtures.
            let forced_pull_count = 1_000_000u64;
            for arm in bandit.arms.iter_mut() {
                if arm.variant_id == variant_id {
                    arm.n_pulls = forced_pull_count;
                } else {
                    arm.n_pulls = 0;
                }
            }
        }
    }

    pub fn print_tuning_report(&self) {
        println!("\n=== Self-Tuning Runtime Report ===");
        println!(
            "registry: {} shapes, {} with >1 variant (Bandit active)",
            self.registry.num_shapes(),
            self.bandits.len()
        );
        let mut shapes: Vec<&ShapeKey> = self.bandits.keys().collect();
        shapes.sort_by_key(|s| (s.op_type as u8, s.format as u8, s.n, s.k));
        for shape in shapes {
            println!(
                "\nshape: {:?} {:?} n={} k={}",
                shape.op_type, shape.format, shape.n, shape.k
            );
            let bandit = &self.bandits[shape];
            bandit.print_stats();
            // Name the variants so the report is readable.
            if let Some(variants) = self.registry.variants.get(shape) {
                let names: Vec<String> = variants
                    .iter()
                    .map(|v| format!("{}={}", v.id.0, v.name))
                    .collect();
                println!("    (variants: {})", names.join(", "));
            }
        }
        println!("=== end report ===\n");
    }
}
