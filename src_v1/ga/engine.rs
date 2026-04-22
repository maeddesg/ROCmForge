//! GA engine — tournament selection, crossover, mutation, elitism,
//! early-exit. 1:1 from `ga_tuning_spec §2.5` + `§2.6`.
//!
//! The engine is generic over the fitness function. Step 2.1.1
//! validates it via `toy::run_toy_ga`; step 2.1.3 will plug in the
//! real GPU benchmark.

use serde_json::json;

use super::compile::CompileCache;
use super::genome::KernelGenome;
use super::logger::GaLogger;
use super::rng::SeededRng;

/// Tunable parameters from the GA spec. `Default` matches
/// `ga_tuning_spec §2.5` + `§2.6`.
#[derive(Debug, Clone)]
pub struct GaConfig {
    pub population_size: usize,
    pub generations: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism_fraction: f64,
    pub early_exit_generations: usize,
    pub early_exit_threshold: f64,
    pub seed: u64,
    pub run_id: String,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 50,
            tournament_size: 3,
            crossover_rate: 0.7,
            mutation_rate: 0.1,
            elitism_fraction: 0.05,
            // Amendment 2 in ga_tuning_spec §2.6: 10 generations, not 5,
            // so a punctuated-equilibrium plateau isn't mistaken for
            // convergence.
            early_exit_generations: 10,
            early_exit_threshold: 0.01,
            // Default seeds are caller-provided in practice; 0xC0FFEE is
            // only used when the caller doesn't set anything (never in
            // production, which always supplies a CLI-level seed).
            seed: 0xC0FFEE,
            run_id: "ga-default-run".to_string(),
        }
    }
}

/// One genome + its fitness result, co-sorted together.
#[derive(Debug, Clone)]
pub struct KernelGenomeScored {
    pub genome: KernelGenome,
    pub fitness: f64,
}

/// Aggregated outputs of evaluating a whole generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationResults {
    pub scored: Vec<KernelGenomeScored>,
    pub compile_hits: u64,
    pub compile_misses: u64,
    pub pre_compile_rejects: usize,
    pub post_compile_vgpr_rejects: usize,
    pub benchmarked_individuals: usize,
}

impl GenerationResults {
    pub fn best_fitness(&self) -> f64 {
        self.scored
            .iter()
            .map(|s| s.fitness)
            .fold(0.0f64, f64::max)
    }

    pub fn median_fitness(&self) -> f64 {
        if self.scored.is_empty() {
            return 0.0;
        }
        let mut fits: Vec<f64> = self.scored.iter().map(|s| s.fitness).collect();
        fits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = fits.len();
        if n % 2 == 1 {
            fits[n / 2]
        } else {
            0.5 * (fits[n / 2 - 1] + fits[n / 2])
        }
    }

    pub fn sorted_by_fitness_desc(&self) -> Vec<KernelGenomeScored> {
        let mut v = self.scored.clone();
        v.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

/// Final GA output for one `(shape, format, level)` run.
#[derive(Debug, Clone)]
pub struct GaResult {
    pub top: Vec<KernelGenomeScored>,
    pub generations_ran: usize,
    pub best_fitness: f64,
    pub early_exited: bool,
    /// Post-convergence stability verdicts for the Top-K (populated
    /// when the caller runs [`KernelGa::validate_top_k_stability`]).
    /// Empty if the caller skipped stability validation (e.g. in the
    /// toy-path where no real GPU kernel exists yet).
    pub stable_top: Vec<StableCandidate>,
}

/// One Top-K candidate that survived the post-GA stability pass.
#[derive(Debug, Clone)]
pub struct StableCandidate {
    pub genome: KernelGenome,
    pub fitness: f64,
    pub stability: super::stability::StabilityResult,
}

/// Main GA engine. Takes a user-supplied `evaluate` closure so the
/// same code drives the toy path today and the real GPU path in 2.1.3.
pub struct KernelGa {
    pub config: GaConfig,
    pub rng: SeededRng,
}

impl KernelGa {
    pub fn new(config: GaConfig) -> Self {
        let seed = config.seed;
        Self {
            config,
            rng: SeededRng::new(seed),
        }
    }

    /// Drive the GA. `evaluate` is called once per genome per generation
    /// and must return a fitness (higher = better).
    pub fn run_with<F>(
        &mut self,
        shape_label: &str,
        logger: &mut GaLogger,
        cache: &mut CompileCache,
        mut evaluate: F,
    ) -> GaResult
    where
        F: FnMut(&KernelGenome, &mut CompileCache) -> f64,
    {
        let _ = logger.log_shape_start(shape_label);

        let mut population: Vec<KernelGenome> = (0..self.config.population_size)
            .map(|_| KernelGenome::random(&mut self.rng))
            .collect();

        let mut best_history: Vec<f64> = Vec::with_capacity(self.config.generations);
        let mut last_results: Option<GenerationResults> = None;
        let mut early_exited = false;
        let mut generations_ran = 0usize;

        for gen in 0..self.config.generations {
            // Snapshot cache counters so per-generation stats are
            // incremental rather than cumulative.
            let hits_start = cache.hits;
            let misses_start = cache.misses;

            let mut scored = Vec::with_capacity(population.len());
            for genome in &population {
                let f = evaluate(genome, cache);
                scored.push(KernelGenomeScored {
                    genome: *genome,
                    fitness: f,
                });
            }

            let results = GenerationResults {
                benchmarked_individuals: scored
                    .iter()
                    .filter(|s| s.fitness > 0.0)
                    .count(),
                pre_compile_rejects: scored
                    .iter()
                    .filter(|s| s.fitness == 0.0)
                    .count(),
                post_compile_vgpr_rejects: 0,
                compile_hits: cache.hits - hits_start,
                compile_misses: cache.misses - misses_start,
                scored,
            };

            let _ = logger.log_generation(
                shape_label,
                gen,
                results.compile_hits,
                results.compile_misses,
                results.pre_compile_rejects,
                results.post_compile_vgpr_rejects,
                results.benchmarked_individuals,
                results.best_fitness(),
                results.median_fitness(),
                0,
                0,
            );

            best_history.push(results.best_fitness());
            generations_ran = gen + 1;

            if self.should_early_exit(&best_history) {
                let _ = logger.log_early_exit(shape_label, gen, results.best_fitness());
                early_exited = true;
                last_results = Some(results);
                break;
            }

            population = self.next_generation(&results);
            last_results = Some(results);
        }

        let last = last_results.unwrap_or_default();
        let top = last.sorted_by_fitness_desc().into_iter().take(5).collect::<Vec<_>>();
        let best = last.best_fitness();
        let _ = logger.log_shape_complete(shape_label, best);

        GaResult {
            top,
            generations_ran,
            best_fitness: best,
            early_exited,
            stable_top: Vec::new(),
        }
    }

    /// Post-convergence Stability-Validation (`ga_tuning_spec §2.9`).
    ///
    /// Sleeps [`super::stability::THERMAL_COOLDOWN`] before the first
    /// run so the GPU has a chance to return to base clock after the
    /// ~8 min GA hammering. Each Top-K candidate then gets 3×10 runs
    /// + 1000-block parity; failures are logged as `stability_fail`
    /// and drop out of the final `stable_top` list.
    ///
    /// The closure `kernel_for(&KernelGenome) -> Option<KnownKernel>`
    /// lets the caller map genomes to Phase-1 kernels. Step 2.1.3 will
    /// replace `KnownKernel` with a GA-compiled-kernel handle; the
    /// engine signature stays the same.
    #[cfg(feature = "gpu")]
    pub fn validate_top_k_stability<F>(
        result: &mut GaResult,
        shape_label: &str,
        shape: &super::types::KernelShape,
        logger: &mut GaLogger,
        cfg: &super::stability::StabilityConfig,
        mut kernel_for: F,
    ) where
        F: FnMut(&KernelGenome) -> Option<super::parity::KnownKernel>,
    {
        std::thread::sleep(super::stability::THERMAL_COOLDOWN);

        let candidates = result.top.clone();
        result.stable_top.clear();
        for cand in candidates {
            let Some(kind) = kernel_for(&cand.genome) else {
                // Caller has no binding for this genome yet (step
                // 2.1.3 territory). Skip rather than fail the whole
                // pass.
                continue;
            };
            match super::stability::check_stability_known_kernel(kind, shape, cfg) {
                Ok(stab) => {
                    if stab.passed {
                        let parity_max_err = stab
                            .parity_result
                            .as_ref()
                            .map(|p| p.max_abs_err)
                            .unwrap_or(0.0);
                        let parity_blocks = stab
                            .parity_result
                            .as_ref()
                            .map(|p| p.n_blocks_tested)
                            .unwrap_or(0);
                        let _ = logger.log_stability_pass(
                            shape_label,
                            cand.fitness,
                            stab.variance_pct,
                            parity_max_err,
                            parity_blocks,
                            &stab.median_times_us,
                        );
                        result.stable_top.push(StableCandidate {
                            genome: cand.genome,
                            fitness: cand.fitness,
                            stability: stab,
                        });
                    } else {
                        let reason = stab
                            .reject_reason
                            .as_deref()
                            .unwrap_or("stability failed for unknown reason");
                        let _ = logger.log_stability_fail(
                            shape_label,
                            cand.fitness,
                            stab.variance_pct,
                            reason,
                        );
                    }
                }
                Err(e) => {
                    let _ = logger.log_stability_fail(
                        shape_label,
                        cand.fitness,
                        0.0,
                        &format!("stability error: {e}"),
                    );
                }
            }
        }
    }

    /// Tournament selection — pick `tournament_size` random individuals
    /// and return the one with the highest fitness.
    fn tournament_select<'a>(
        &mut self,
        scored: &'a [KernelGenomeScored],
    ) -> &'a KernelGenomeScored {
        assert!(!scored.is_empty(), "tournament on empty population");
        let mut best_idx = self.rng.gen_index(scored.len());
        for _ in 1..self.config.tournament_size {
            let cand = self.rng.gen_index(scored.len());
            if scored[cand].fitness > scored[best_idx].fitness {
                best_idx = cand;
            }
        }
        &scored[best_idx]
    }

    /// Produce the next generation from the current scored population.
    /// Elitism: top N are copied verbatim. The rest is crossover
    /// (rate = `crossover_rate`) or mutation-only of a tournament winner.
    pub fn next_generation(&mut self, results: &GenerationResults) -> Vec<KernelGenome> {
        let sorted = results.sorted_by_fitness_desc();
        let pop = self.config.population_size;
        let elite_count =
            ((pop as f64) * self.config.elitism_fraction).ceil() as usize;
        let elite_count = elite_count.min(sorted.len()).min(pop);

        let mut next = Vec::with_capacity(pop);
        for s in sorted.iter().take(elite_count) {
            next.push(s.genome);
        }

        while next.len() < pop {
            if self.rng.gen_f64() < self.config.crossover_rate {
                let a = self.tournament_select(&sorted);
                let b = self.tournament_select(&sorted);
                let mut child = KernelGenome::crossover(&a.genome, &b.genome, &mut self.rng);
                child.mutate(self.config.mutation_rate, &mut self.rng);
                next.push(child);
            } else {
                let parent = self.tournament_select(&sorted);
                let mut child = parent.genome;
                child.mutate(self.config.mutation_rate, &mut self.rng);
                next.push(child);
            }
        }

        next
    }

    /// Early-exit when the best fitness over the last `W` generations
    /// improved by < `early_exit_threshold`. `W` is
    /// `early_exit_generations` — spec says 10 to allow punctuated
    /// equilibria (§2.6).
    pub fn should_early_exit(&self, history: &[f64]) -> bool {
        let w = self.config.early_exit_generations;
        if history.len() < w {
            return false;
        }
        let recent = &history[history.len() - w..];
        let first = recent[0];
        let last = *recent.last().unwrap();
        if first <= 0.0 {
            return false;
        }
        let improvement = (last - first) / first;
        improvement < self.config.early_exit_threshold
    }
}

/// Record one evaluation to the JSONL log. Factored out so both the
/// toy path and the (future) real-GPU path can share the serialization
/// code.
pub fn log_eval_record(
    logger: &mut GaLogger,
    shape_label: &str,
    generation: usize,
    individual: usize,
    genome: &KernelGenome,
    fitness: f64,
    median_us: Option<f64>,
    seed: u64,
) {
    let genome_json = json!({
        "tile_m": genome.tile_m,
        "tile_n": genome.tile_n,
        "tile_k": genome.tile_k,
        "tiles_per_wave": genome.tiles_per_wave,
        "waves_per_block": genome.waves_per_block,
        "use_lds_for_a": genome.use_lds_for_a,
        "use_lds_for_b": genome.use_lds_for_b,
        "prefetch_depth": genome.prefetch_depth,
        "k_unroll": genome.k_unroll,
        "double_buffer": genome.double_buffer,
        "dequant_strategy": match genome.dequant_strategy {
            super::genome::DequantStrategy::Inline => "inline".to_string(),
            super::genome::DequantStrategy::PrePass { lds_bytes } => {
                format!("pre_pass_{lds_bytes}")
            }
            super::genome::DequantStrategy::Batched { batch_size } => {
                format!("batched_{batch_size}")
            }
        },
    });
    let metrics = json!({
        "fitness": fitness,
        "median_latency_us": median_us,
    });
    let _ = logger.log_eval(shape_label, generation, individual, genome_json, metrics, seed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_spec() {
        let c = GaConfig::default();
        assert_eq!(c.population_size, 100);
        assert_eq!(c.generations, 50);
        assert_eq!(c.tournament_size, 3);
        assert_eq!(c.crossover_rate, 0.7);
        assert_eq!(c.mutation_rate, 0.1);
        assert_eq!(c.elitism_fraction, 0.05);
        assert_eq!(c.early_exit_generations, 10);
        assert_eq!(c.early_exit_threshold, 0.01);
    }
}
