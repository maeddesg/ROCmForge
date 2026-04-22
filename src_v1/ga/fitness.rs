//! GA fitness (`ga_tuning_spec §2.4`).
//!
//! `fitness = baseline_us / median(candidate_times_us)` — values > 1.0
//! beat the baseline, < 1.0 lose. Rejected candidates (pre- or
//! post-compile gate) get fitness = 0.0 so the selection drops them
//! without a special-case branch.
//!
//! For step 2.1.1 the real GPU-benchmark entry is stubbed. The
//! framework is validated via `evaluate_toy_fitness`, which skips
//! compile + benchmark and uses `toy::toy_fitness` as the driver. The
//! 5-phase flow is still exercised end-to-end: Stage-1 validate →
//! "compile" (cache lookup) → Stage-2 validate → (skip warmup+bench) →
//! fitness.

use super::compile::{CompileCache, CompileKey, CompiledKernel};
use super::genome::KernelGenome;
use super::parity::ParityResult;
use super::toy::toy_fitness;
use super::types::{CodeObjectResources, KernelTarget, PrecisionLevel};
use super::validation::{
    validate_post_compile, validate_pre_compile, GaQuantFormat, PostCompileResult,
};

/// Median of a slice of f64; `NaN`-free input assumed. For even length
/// returns the average of the two middle values.
pub fn median(times: &[f64]) -> f64 {
    if times.is_empty() {
        return f64::INFINITY;
    }
    let mut sorted: Vec<f64> = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN in times"));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Outcome of evaluating one genome. `fitness = 0.0` for rejected
/// candidates (cheaper to filter in selection than to use `Option`).
#[derive(Debug, Clone)]
pub struct FitnessResult {
    /// baseline_us / candidate_median_us. 0.0 on reject.
    pub fitness: f64,
    /// Candidate median latency (µs). `None` if rejected before bench.
    pub median_us: Option<f64>,
    /// Reject reason; `None` if the candidate ran through cleanly.
    pub reject_reason: Option<String>,
    /// Post-compile resources (available iff compile ran).
    pub post_compile: Option<PostCompileResult>,
    /// Parity snapshot (available iff the 6-phase real-GPU fitness
    /// path ran this candidate through `check_parity_*`). The toy
    /// path leaves this `None` — it has no GPU kernel to compare.
    pub parity: Option<ParityResult>,
}

impl FitnessResult {
    pub fn rejected(reason: &str) -> Self {
        Self {
            fitness: 0.0,
            median_us: None,
            reject_reason: Some(reason.to_string()),
            post_compile: None,
            parity: None,
        }
    }

    pub fn rejected_post_compile(post: PostCompileResult) -> Self {
        Self {
            fitness: 0.0,
            median_us: None,
            reject_reason: Some(format!(
                "post_compile_vgpr: actual={} waves/CU={}",
                post.actual_vgprs, post.max_waves_per_cu
            )),
            post_compile: Some(post),
            parity: None,
        }
    }

    /// Fitness = 0 because the parity check against the VALU reference
    /// failed. The GA's 6-phase pipeline uses this before any
    /// benchmark runs happen — no budget is wasted timing a kernel
    /// whose output is numerically wrong (`ga_tuning_spec §2.8`).
    pub fn parity_violation(parity: ParityResult) -> Self {
        let reason = format!(
            "parity_violation: max_err={:.6} violations={}",
            parity.max_abs_err,
            parity.violations.len()
        );
        Self {
            fitness: 0.0,
            median_us: None,
            reject_reason: Some(reason),
            post_compile: None,
            parity: Some(parity),
        }
    }

    pub fn measured(fitness: f64, median_us: f64, post: PostCompileResult) -> Self {
        Self {
            fitness,
            median_us: Some(median_us),
            reject_reason: None,
            post_compile: Some(post),
            parity: None,
        }
    }

    pub fn measured_with_parity(
        fitness: f64,
        median_us: f64,
        post: PostCompileResult,
        parity: ParityResult,
    ) -> Self {
        Self {
            fitness,
            median_us: Some(median_us),
            reject_reason: None,
            post_compile: Some(post),
            parity: Some(parity),
        }
    }

    pub fn is_rejected(&self) -> bool {
        self.reject_reason.is_some() || self.fitness == 0.0
    }
}

/// Toy-fitness driver used by the framework validation tests. Runs the
/// Stage-1 + Stage-2 gates (the gates themselves are cheap and
/// GPU-independent) and wraps the deterministic toy landscape as the
/// benchmark result. No hipcc, no GPU.
pub fn evaluate_toy_fitness(
    genome: &KernelGenome,
    fmt: &GaQuantFormat,
    level: PrecisionLevel,
    baseline_us: f64,
    cache: &mut CompileCache,
) -> FitnessResult {
    // Phase 1: pre-compile gate.
    if !validate_pre_compile(genome, fmt, level) {
        return FitnessResult::rejected("pre_compile");
    }

    // Phase 2: "compile" — for the toy path the factory returns
    // synthetic resources sized from the genome. This exercises the
    // cache hit/miss counters.
    let key = CompileKey::new(
        genome.into(),
        12, // Q4_K format id, placeholder for step 2.1.1
        level,
        KernelTarget::Gfx1201,
    );
    let kernel: std::sync::Arc<CompiledKernel> = cache.get_or_insert_with(&key, |k| {
        let vgprs = synthetic_vgprs_for(genome, level);
        CompiledKernel {
            key: k.clone(),
            resources: CodeObjectResources {
                vgpr_count: vgprs,
                sgpr_count: 20,
                lds_bytes: 0,
            },
            co_path: None,
        }
    });

    // Phase 3: post-compile VGPR gate.
    let post = validate_post_compile(kernel.resources);
    if !post.accepted {
        return FitnessResult::rejected_post_compile(post);
    }

    // Phases 4+5: no real benchmark — toy fitness is the oracle.
    let toy = toy_fitness(genome);
    // Convert toy fitness to a pretend-median so the result looks
    // uniform in the log (`median = baseline / fitness`).
    let median_us = if toy > 0.0 {
        baseline_us / toy
    } else {
        f64::INFINITY
    };
    FitnessResult::measured(toy, median_us, post)
}

/// Synthetic VGPR count that maps a genome's structural choices to a
/// plausible compile result. Used only by the toy path so the
/// post-compile gate can still reject outrageously expensive
/// configurations (e.g. `tile_m=128 + k_unroll=8`).
fn synthetic_vgprs_for(g: &KernelGenome, level: PrecisionLevel) -> u16 {
    let base = match level {
        PrecisionLevel::Fp8 => 40,
        PrecisionLevel::Fp16 | PrecisionLevel::Bf16 => 56,
        PrecisionLevel::Fp32 => 72,
    };
    let tile_m_cost = (g.tile_m as u32 / 16) * 10;
    let unroll_cost = (g.k_unroll as u32) * 4;
    let prefetch_cost = (g.prefetch_depth as u32) * 6;
    (base + tile_m_cost + unroll_cost + prefetch_cost) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::ga::genome::{DequantStrategy, KernelGenome};

    #[test]
    fn median_odd_and_even() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn fitness_relative_to_baseline() {
        let mut cache = CompileCache::new();
        let g = KernelGenome {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            tiles_per_wave: 2,
            waves_per_block: 4,
            use_lds_for_a: false,
            use_lds_for_b: true,
            prefetch_depth: 1,
            k_unroll: 4,
            double_buffer: false,
            dequant_strategy: DequantStrategy::Inline,
        };
        let r = evaluate_toy_fitness(
            &g,
            &GaQuantFormat::q4_k(),
            PrecisionLevel::Fp16,
            100.0,
            &mut cache,
        );
        assert!(!r.is_rejected());
        assert!(r.fitness > 1.9, "sweet-spot genome should score ~2.0, got {}", r.fitness);
    }
}
