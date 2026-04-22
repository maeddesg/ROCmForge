//! Toy fitness landscape used to validate the GA framework.
//!
//! The real GPU-benchmark path lands in step 2.1.3 — here we only need
//! to prove that the selection + crossover + mutation + elitism +
//! early-exit machinery converges toward a known optimum. The landscape
//! is deliberately small enough that a correctly-implemented GA finds
//! the sweet-spot in ≤ 20 generations.
//!
//! Sweet-spot (fitness ≈ 2.0):
//!   `tile_m = 64, tile_n = 64, k_unroll = 4`.
//!
//! Second peak (fitness 1.35–1.5):
//!   `tile_m in {32, 64}, tile_n = 32, k_unroll in {2, 4}`.
//!
//! Everything else: ≤ 1.0.

use super::compile::CompileCache;
use super::engine::{log_eval_record, GaConfig, GaResult, KernelGa};
use super::genome::KernelGenome;
use super::logger::GaLogger;

/// Fitness landscape for the framework test. No GPU, no compile.
pub fn toy_fitness(g: &KernelGenome) -> f64 {
    let tile_score = match (g.tile_m, g.tile_n) {
        (64, 64) => 2.0,
        (32, 32) => 1.5,
        (32, 64) | (64, 32) => 1.3,
        (64, _) | (_, 64) => 1.0,
        _ => 0.5 + (g.tile_m as f64 / 128.0) * 0.3,
    };
    let unroll_score = match g.k_unroll {
        4 => 1.0,
        2 => 0.9,
        8 => 0.8,
        _ => 0.7,
    };
    tile_score * unroll_score
}

/// Convenience: run the GA on the toy landscape end-to-end. Used both
/// by the framework tests and by the step report to prove convergence
/// in-process.
pub fn run_toy_ga(config: GaConfig, capture_log: bool) -> (GaResult, Option<Vec<String>>) {
    let run_id = config.run_id.clone();
    let seed = config.seed;
    let mut logger = if capture_log {
        GaLogger::capturing(run_id)
    } else {
        GaLogger::null(run_id)
    };
    let mut cache = CompileCache::new();
    let mut ga = KernelGa::new(config);

    // `run_with` already handles per-generation logging via its own
    // logger borrow. `log_eval_record` is available for callers who
    // want per-individual eval records in the log (step 2.1.3 will use
    // it); the toy path keeps the log compact and skips per-eval spam.
    let _ = (log_eval_record, seed); // tie the re-exports into the build
    let result = ga.run_with("toy", &mut logger, &mut cache, |g, _c| toy_fitness(g));

    // For log-capture mode the caller gets the captured lines back.
    let captured = logger.captured_lines().map(|s| s.to_vec());
    (result, captured)
}

/// Smaller, faster variant used by tests: 20 generations, pop 50.
pub fn toy_ga_defaults(seed: u64, run_id: &str) -> GaConfig {
    GaConfig {
        population_size: 50,
        generations: 20,
        tournament_size: 3,
        crossover_rate: 0.7,
        mutation_rate: 0.15,
        elitism_fraction: 0.1,
        // Early exit after 10 generations plateau — but on a fitness
        // landscape that caps at 2.0, typical runs either converge
        // quickly and bail, or burn the full 20 generations.
        early_exit_generations: 10,
        early_exit_threshold: 0.01,
        seed,
        run_id: run_id.to_string(),
    }
}
