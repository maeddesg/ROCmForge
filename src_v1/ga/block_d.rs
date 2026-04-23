//! Phase 2 / Schritt 2.1.3 Block D — 2-D GA (num_waves ×
//! multi_row_cols) on `gate_up_swiglu`, plus the wiring from a GA
//! winner into the graph executor for End-to-End Decode.
//!
//! Block D extends Block C along one extra axis: `multi_row_cols ∈
//! {1, 2, 4, 8}`. With `num_waves ∈ {1, 2, 4, 8}` that's a 4 × 4 =
//! **16**-point search space. The remaining genome fields stay at
//! Phase-1 defaults — `lds_strategy` and `k_unroll` are not
//! meaningful axes in this kernel (the scope analysis before Block
//! D writes up why; see `results/phase2_step_2.1.3_block_d_full_tileconfig.md`).
//!
//! Design differences vs. Block C:
//!
//!   * **`BlockDGenome`** — a tiny `(num_waves, multi_row_cols)`
//!     struct rather than overloading `KernelGenome`. Block C used
//!     `KernelGenome::waves_per_block` as the single-axis vehicle;
//!     Block D needs two axes and a local struct keeps the cache
//!     key and the genome congruent. `KernelGenome::from(…)` is not
//!     used here at all — Block D is self-contained.
//!   * **2-D compile cache** — `HashMap<BlockDGenome,
//!     Arc<DynamicKernel>>`. At most 16 unique entries; with a Pop
//!     12 / Gen 8 GA the cache hit-rate is ≥ 75 % after gen 1.
//!   * **Parity on a scaled-down shape, benchmark on the real
//!     shape.** Block C was forced onto N=512 parity AND benchmark
//!     because the Top-K stability step re-ran parity at 1000
//!     blocks. Block D uses `K=4096, N=512` for parity (matching
//!     Block C) but `K=4096, N=14336` for the benchmark phase —
//!     the shape the GA is actually tuning for.
//!   * **`DynamicGateUpHook`** — a small RAII wrapper handed to
//!     `GraphExecutor::set_gate_up_swiglu_dynamic_kernel` so the
//!     winner plugs straight into decode. Living in the GA module
//!     keeps the executor free of GA-specific types.
//!
//! The parity, benchmark, and stability helpers from `block_c.rs`
//! take `&DynamicKernel` and don't care which cache built it — they
//! are re-used as-is.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::sync::Arc;

use crate::v1::graph::executor::DynamicGateUpHook;
use crate::v1::ir::codegen_gpu::emit_q4_k_gemv_gate_up_swiglu_parametric_2d;
use crate::v1::ir::formats::q4_k;
use crate::v1::ir::types::QuantFormat;

use super::block_c::{
    bench_dynamic_gate_up, parity_dynamic_gate_up, stability_dynamic_gate_up,
    DynamicStabilityResult,
};
use super::compile::{compile_hip_source, parse_amdgpu_metadata, CompileError};
use super::dynamic::{DynamicKernel, GateUpSwigluGeometry};
use super::engine::{GaConfig, KernelGa};
use super::logger::GaLogger;
use super::parity::ParityConfig;
use super::rng::SeededRng;
use super::types::{CodeObjectResources, KernelShape, PrecisionLevel};

/// Legal `num_waves` values — same set as Block C.
pub const BLOCK_D_NUM_WAVES_VALUES: &[u8] = &[1, 2, 4, 8];

/// Legal `multi_row_cols` values. `1` is legal but stresses
/// dispatch overhead; `8` stresses VGPR budget (each thread holds
/// `2 × multi_row_cols` FP32 accumulators). The post-compile VGPR
/// gate catches out-of-budget configs.
pub const BLOCK_D_MULTI_ROW_COLS_VALUES: &[u8] = &[1, 2, 4, 8];

/// Block-D genome — the two axes the GA searches over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockDGenome {
    pub num_waves: u8,
    pub multi_row_cols: u8,
}

impl BlockDGenome {
    /// Phase-1 default: `(num_waves=8, multi_row_cols=4)`.
    pub fn phase1_default() -> Self {
        Self {
            num_waves: 8,
            multi_row_cols: 4,
        }
    }

    pub fn block_c_winner() -> Self {
        Self {
            num_waves: 4,
            multi_row_cols: 4,
        }
    }
}

/// Draw a random 2-D genome.
pub fn random_block_d_genome(rng: &mut SeededRng) -> BlockDGenome {
    BlockDGenome {
        num_waves: *rng.choose(BLOCK_D_NUM_WAVES_VALUES),
        multi_row_cols: *rng.choose(BLOCK_D_MULTI_ROW_COLS_VALUES),
    }
}

/// Compile cache keyed on the full 2-D genome.
pub struct DynamicKernelCache2D {
    entries: HashMap<BlockDGenome, Arc<DynamicKernel>>,
    resources: HashMap<BlockDGenome, CodeObjectResources>,
    pub hits: u64,
    pub misses: u64,
}

impl DynamicKernelCache2D {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            resources: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Compile-or-fetch. Runs hipcc + `clang-offload-bundler`
    /// on miss (~ 1 s per unique kernel).
    pub fn get_or_compile(&mut self, g: BlockDGenome) -> Result<Arc<DynamicKernel>, CompileError> {
        if let Some(k) = self.entries.get(&g) {
            self.hits += 1;
            return Ok(k.clone());
        }
        self.misses += 1;

        let (src, symbol) = emit_q4_k_gemv_gate_up_swiglu_parametric_2d(
            g.num_waves as u32,
            g.multi_row_cols as u32,
        );
        let label = format!("ga_gate_up_w{}_c{}", g.num_waves, g.multi_row_cols);
        let co_bytes = compile_hip_source(&src, &label)?;

        let extracted = std::env::temp_dir()
            .join("rocmforge_ga")
            .join(format!("{label}.gfx1201.co"));
        let res = parse_amdgpu_metadata(&extracted)
            .map_err(|e| CompileError::Io(e))?
            .unwrap_or(CodeObjectResources {
                vgpr_count: 0,
                sgpr_count: 0,
                lds_bytes: 0,
            });

        let geometry =
            GateUpSwigluGeometry::for_config(g.num_waves as u32, g.multi_row_cols as u32);
        let kernel = DynamicKernel::from_code_object(&co_bytes, symbol, geometry).map_err(|e| {
            CompileError::HipccFailed {
                kernel_name: label.clone(),
                stderr: format!("HipModule::load: {e:?}"),
                status: None,
            }
        })?;

        let kernel = Arc::new(kernel);
        self.entries.insert(g, kernel.clone());
        self.resources.insert(g, res);
        Ok(kernel)
    }

    pub fn resources_for(&self, g: BlockDGenome) -> Option<CodeObjectResources> {
        self.resources.get(&g).copied()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl Default for DynamicKernelCache2D {
    fn default() -> Self {
        Self::new()
    }
}

/// One evaluated genome.
#[derive(Debug, Clone)]
pub struct BlockDCandidate {
    pub genome: BlockDGenome,
    pub median_us: f64,
    pub fitness: f64,
    pub vgpr_count: u16,
    pub parity_max_err: f32,
    pub parity_passed: bool,
}

/// Stability-validated winner.
#[derive(Debug, Clone)]
pub struct BlockDWinner {
    pub genome: BlockDGenome,
    pub median_us: f64,
    pub fitness: f64,
    pub vgpr_count: u16,
    pub parity_max_err: f32,
    pub stability_variance_pct: f64,
    pub stability_passed: bool,
    pub parity_passed: bool,
}

/// Aggregate GA result.
#[derive(Debug, Clone)]
pub struct BlockDResult {
    pub all_candidates: Vec<BlockDCandidate>,
    pub stable_winners: Vec<BlockDWinner>,
    pub generations_ran: usize,
    pub early_exited: bool,
    pub compile_hits: u64,
    pub compile_misses: u64,
    /// Shape parity was checked on (N=512 by default).
    pub parity_shape: KernelShape,
    /// Shape the benchmark + final winner was measured on (real
    /// model shape, e.g. N=14336 for Qwen3-8B).
    pub bench_shape: KernelShape,
}

/// Tuned `GaConfig` for the 16-point 2-D search. Larger than
/// Block C's Pop 8 / Gen 5 so the GA actually explores the space;
/// smaller than `§2.5`'s 100/50 because 16 × 8 generations already
/// covers the reachable space 8× over.
pub fn block_d_default_config(seed: u64) -> GaConfig {
    GaConfig {
        population_size: 12,
        generations: 8,
        tournament_size: 2,
        crossover_rate: 0.7,
        mutation_rate: 0.3,
        elitism_fraction: 0.1, // Top 1 of 12 + tournament-filled next gen
        early_exit_generations: 4,
        early_exit_threshold: 0.01,
        seed,
        run_id: format!("block_d_mini_ga_seed{seed}"),
    }
}

/// Build a [`DynamicGateUpHook`] from a compiled kernel + the shape
/// it was tuned for. The hook is what `GraphExecutor::
/// set_gate_up_swiglu_dynamic_kernel` accepts.
pub fn make_gate_up_hook(
    kernel: Arc<DynamicKernel>,
    hidden_dim: usize,
    ffn_dim: usize,
) -> DynamicGateUpHook {
    DynamicGateUpHook {
        kernel,
        hidden_dim,
        ffn_dim,
    }
}

// ── Mini-GA driver ──────────────────────────────────────────────────────────

impl KernelGa {
    /// Block-D mini-GA over `(num_waves, multi_row_cols)`. See the
    /// module-level doc for the design split between parity
    /// (`parity_shape`, typically N=512) and benchmark
    /// (`bench_shape`, typically the real model shape).
    ///
    /// The fitness pipeline is identical to Block C:
    ///
    /// ```text
    ///   Phase 1 — pre-compile (trivially passes here)
    ///   Phase 2 — compile (hit or miss)
    ///   Phase 3 — post-compile VGPR gate
    ///   Phase 4 — parity (on `parity_shape`, N=512)
    ///   Phase 5 — warmup
    ///   Phase 6 — benchmark (on `bench_shape`, N=14336)
    /// ```
    ///
    /// The `bench_weights_gate`, `bench_weights_up`, and
    /// `bench_input` buffers are used for Phase 6 only — they must
    /// be sized for `bench_shape`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_num_waves_and_multi_row_cols(
        &mut self,
        parity_shape: &KernelShape,
        bench_shape: &KernelShape,
        _fmt: &QuantFormat,
        _level: PrecisionLevel,
        baseline_us: f32,
        bench_weights_gate: &[u8],
        bench_weights_up: &[u8],
        bench_input: &[f32],
        parity_weights_gate: &[u8],
        parity_weights_up: &[u8],
        parity_input: &[f32],
        cache: &mut DynamicKernelCache2D,
        logger: &mut GaLogger,
    ) -> BlockDResult {
        let shape_label = "gemv_q4_k_gate_up_swiglu_block_d";
        let _ = logger.log_shape_start(shape_label);

        let mut population: Vec<BlockDGenome> = (0..self.config.population_size)
            .map(|_| random_block_d_genome(&mut self.rng))
            .collect();

        // One evaluation per unique genome — tournament may draw
        // the same combination multiple times within a generation.
        let mut per_genome_cand: HashMap<BlockDGenome, BlockDCandidate> = HashMap::new();

        let mut best_history: Vec<f64> = Vec::with_capacity(self.config.generations);
        let mut generations_ran = 0usize;
        let mut early_exited = false;

        for gen in 0..self.config.generations {
            let mut scored = Vec::with_capacity(population.len());

            for (individual, genome) in population.iter().enumerate() {
                if let Some(existing) = per_genome_cand.get(genome) {
                    scored.push((individual, *genome, existing.fitness));
                    continue;
                }
                let eval = evaluate_one_2d_genome(
                    *genome,
                    parity_shape,
                    bench_shape,
                    baseline_us,
                    bench_weights_gate,
                    bench_weights_up,
                    bench_input,
                    cache,
                    logger,
                    shape_label,
                    gen,
                    individual,
                    self.config.seed,
                );
                let cand = BlockDCandidate {
                    genome: *genome,
                    median_us: if eval.fitness > 0.0 {
                        (baseline_us as f64) / eval.fitness
                    } else {
                        f64::INFINITY
                    },
                    fitness: eval.fitness,
                    vgpr_count: eval.vgpr_count,
                    parity_max_err: eval.parity_max_err,
                    parity_passed: eval.parity_passed,
                };
                per_genome_cand.insert(*genome, cand);
                scored.push((individual, *genome, eval.fitness));
            }

            let best = scored.iter().map(|(_, _, f)| *f).fold(0.0f64, f64::max);
            let median = {
                let mut fits: Vec<f64> = scored.iter().map(|(_, _, f)| *f).collect();
                fits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                fits[fits.len() / 2]
            };
            let _ = logger.log_generation(
                shape_label,
                gen,
                cache.hits,
                cache.misses,
                0,
                0,
                scored.iter().filter(|(_, _, f)| *f > 0.0).count(),
                best,
                median,
                0,
                0,
            );

            best_history.push(best);
            generations_ran = gen + 1;

            if self.should_early_exit(&best_history) {
                let _ = logger.log_early_exit(shape_label, gen, best);
                early_exited = true;
                break;
            }

            // Next generation: tournament + crossover + per-axis
            // mutation + 1 elite.
            let mut sorted = scored.clone();
            sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let mut next = Vec::with_capacity(self.config.population_size);
            next.push(sorted[0].1); // elite

            while next.len() < self.config.population_size {
                let a = tournament_pick_2d(&scored, self.config.tournament_size, &mut self.rng);
                if self.rng.gen_f64() < self.config.crossover_rate {
                    let b = tournament_pick_2d(&scored, self.config.tournament_size, &mut self.rng);
                    // 2-D crossover: for each axis, pick a or b.
                    let mut child = a;
                    if self.rng.gen_bool() {
                        child.num_waves = b.num_waves;
                    }
                    if self.rng.gen_bool() {
                        child.multi_row_cols = b.multi_row_cols;
                    }
                    // Per-axis mutation.
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.num_waves = *self.rng.choose(BLOCK_D_NUM_WAVES_VALUES);
                    }
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.multi_row_cols = *self.rng.choose(BLOCK_D_MULTI_ROW_COLS_VALUES);
                    }
                    next.push(child);
                } else {
                    let mut child = a;
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.num_waves = *self.rng.choose(BLOCK_D_NUM_WAVES_VALUES);
                    }
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.multi_row_cols = *self.rng.choose(BLOCK_D_MULTI_ROW_COLS_VALUES);
                    }
                    next.push(child);
                }
            }
            population = next;
        }

        // Sort & Top-K.
        let mut all_candidates: Vec<BlockDCandidate> = per_genome_cand.into_values().collect();
        all_candidates.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = all_candidates.iter().take(3).cloned().collect::<Vec<_>>();

        std::thread::sleep(std::time::Duration::from_secs(2));

        let mut stable_winners = Vec::new();
        for cand in &top_k {
            if !cand.parity_passed || cand.fitness <= 0.0 {
                continue;
            }
            let kernel = match cache.get_or_compile(cand.genome) {
                Ok(k) => k,
                Err(_) => continue,
            };
            // Stability runs on the parity shape. The 1000-block
            // parity inside `stability_dynamic_gate_up` would OOM
            // on bench shape (Block C SIGKILL lesson — 234 MB FP32
            // dequant per block × 1000 ≈ TB of scratch). Timing
            // variance at parity shape is still a valid jitter
            // metric; it rejects kernels that flake, which is the
            // whole point of the stability gate.
            let st: DynamicStabilityResult = match stability_dynamic_gate_up(
                &kernel,
                parity_shape,
                parity_weights_gate,
                parity_weights_up,
                parity_input,
            ) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if st.passed {
                let _ = logger.log_stability_pass(
                    shape_label,
                    cand.fitness,
                    st.variance_pct,
                    st.parity.max_abs_err,
                    st.parity.n_blocks_tested,
                    &[st.median_us],
                );
                stable_winners.push(BlockDWinner {
                    genome: cand.genome,
                    median_us: cand.median_us,
                    fitness: cand.fitness,
                    vgpr_count: cand.vgpr_count,
                    parity_max_err: cand.parity_max_err,
                    stability_variance_pct: st.variance_pct,
                    stability_passed: true,
                    parity_passed: true,
                });
            } else {
                let reason = st.reject_reason.as_deref().unwrap_or("stability failed");
                let _ =
                    logger.log_stability_fail(shape_label, cand.fitness, st.variance_pct, reason);
            }
        }

        let _ = logger.log_shape_complete(
            shape_label,
            stable_winners
                .first()
                .map(|w| w.fitness)
                .unwrap_or(all_candidates.first().map(|c| c.fitness).unwrap_or(0.0)),
        );

        BlockDResult {
            all_candidates,
            stable_winners,
            generations_ran,
            early_exited,
            compile_hits: cache.hits,
            compile_misses: cache.misses,
            parity_shape: *parity_shape,
            bench_shape: *bench_shape,
        }
    }
}

fn tournament_pick_2d(
    scored: &[(usize, BlockDGenome, f64)],
    size: usize,
    rng: &mut SeededRng,
) -> BlockDGenome {
    let mut best_idx = rng.gen_index(scored.len());
    for _ in 1..size {
        let cand = rng.gen_index(scored.len());
        if scored[cand].2 > scored[best_idx].2 {
            best_idx = cand;
        }
    }
    scored[best_idx].1
}

struct OneEval2D {
    fitness: f64,
    vgpr_count: u16,
    parity_max_err: f32,
    parity_passed: bool,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_one_2d_genome(
    genome: BlockDGenome,
    parity_shape: &KernelShape,
    bench_shape: &KernelShape,
    baseline_us: f32,
    bench_weights_gate: &[u8],
    bench_weights_up: &[u8],
    bench_input: &[f32],
    cache: &mut DynamicKernelCache2D,
    logger: &mut GaLogger,
    shape_label: &str,
    generation: usize,
    individual: usize,
    seed: u64,
) -> OneEval2D {
    // Phase 2: compile.
    let kernel = match cache.get_or_compile(genome) {
        Ok(k) => k,
        Err(_) => {
            return OneEval2D {
                fitness: 0.0,
                vgpr_count: 0,
                parity_max_err: 0.0,
                parity_passed: false,
            };
        }
    };

    // Phase 3: post-compile VGPR gate.
    let res = cache.resources_for(genome).unwrap_or(CodeObjectResources {
        vgpr_count: 0,
        sgpr_count: 0,
        lds_bytes: 0,
    });
    let max_waves = res.max_waves_per_cu();
    if max_waves < 4 {
        return OneEval2D {
            fitness: 0.0,
            vgpr_count: res.vgpr_count,
            parity_max_err: 0.0,
            parity_passed: false,
        };
    }

    // Phase 4: parity on the small shape.
    let parity_cfg = ParityConfig::for_ga(PrecisionLevel::Fp16);
    let parity = match parity_dynamic_gate_up(&kernel, parity_shape, &parity_cfg, 12345) {
        Ok(p) => p,
        Err(_) => {
            return OneEval2D {
                fitness: 0.0,
                vgpr_count: res.vgpr_count,
                parity_max_err: f32::INFINITY,
                parity_passed: false,
            };
        }
    };
    if !parity.passed {
        let _ = logger.log_parity_violation(
            shape_label,
            generation,
            individual,
            parity.max_abs_err,
            parity.effective_tolerance,
            parity.violations.len(),
            parity.violations.first().map(|v| v.block_idx),
            parity.violations.first().map(|v| v.element_idx),
        );
        return OneEval2D {
            fitness: 0.0,
            vgpr_count: res.vgpr_count,
            parity_max_err: parity.max_abs_err,
            parity_passed: false,
        };
    }

    // Phases 5+6: warmup + benchmark on the REAL shape.
    let median_us = match bench_dynamic_gate_up(
        &kernel,
        bench_weights_gate,
        bench_weights_up,
        bench_input,
        bench_shape.k as i32,
        bench_shape.n as i32,
        5,
        20,
    ) {
        Ok(v) => v,
        Err(_) => {
            return OneEval2D {
                fitness: 0.0,
                vgpr_count: res.vgpr_count,
                parity_max_err: parity.max_abs_err,
                parity_passed: true,
            };
        }
    };

    let fitness = if median_us > 0.0 {
        (baseline_us as f64) / median_us
    } else {
        0.0
    };

    let _ = logger.log_eval(
        shape_label,
        generation,
        individual,
        serde_json::json!({
            "num_waves": genome.num_waves,
            "multi_row_cols": genome.multi_row_cols,
        }),
        serde_json::json!({
            "fitness": fitness,
            "median_latency_us": median_us,
            "actual_vgpr_count": res.vgpr_count,
            "actual_sgpr_count": res.sgpr_count,
            "actual_waves_per_cu": max_waves,
            "parity_max_err": parity.max_abs_err,
            "parity_passed": parity.passed,
        }),
        seed,
    );

    OneEval2D {
        fitness,
        vgpr_count: res.vgpr_count,
        parity_max_err: parity.max_abs_err,
        parity_passed: true,
    }
}

#[cfg(test)]
mod inline_tests {
    use super::*;

    #[test]
    fn block_d_genome_defaults() {
        assert_eq!(BlockDGenome::phase1_default().num_waves, 8);
        assert_eq!(BlockDGenome::phase1_default().multi_row_cols, 4);
        assert_eq!(BlockDGenome::block_c_winner().num_waves, 4);
        assert_eq!(BlockDGenome::block_c_winner().multi_row_cols, 4);
    }

    #[test]
    fn random_block_d_genome_is_legal() {
        let mut rng = SeededRng::new(7);
        for _ in 0..100 {
            let g = random_block_d_genome(&mut rng);
            assert!(BLOCK_D_NUM_WAVES_VALUES.contains(&g.num_waves));
            assert!(BLOCK_D_MULTI_ROW_COLS_VALUES.contains(&g.multi_row_cols));
        }
    }

    #[test]
    fn block_d_default_config_spec() {
        let cfg = block_d_default_config(42);
        assert_eq!(cfg.population_size, 12);
        assert_eq!(cfg.generations, 8);
        assert_eq!(cfg.tournament_size, 2);
        assert_eq!(cfg.seed, 42);
        assert!(cfg.run_id.contains("seed42"));
    }
}
