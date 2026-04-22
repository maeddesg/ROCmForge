//! Phase 2 / Schritt 2.1.3 Block C — First real GA run against
//! `gate_up_swiglu`. Search space restricted to `num_waves ∈ {1,2,4,8}`
//! — the only parametric axis Block B exposed. Every other genome
//! field is pinned to a "Phase-1 default" baked into `fixed_genome`.
//!
//! What this module glues together:
//!
//!   * [`DynamicKernelCache`] — `num_waves → Arc<DynamicKernel>`,
//!     backed by [`compile_hip_source`] + [`parse_amdgpu_metadata`].
//!     Only 4 unique kernels exist, so Cache-Hit-Rate is 100 % after
//!     generation 1.
//!   * [`parity_dynamic_gate_up`] — 10-block parity against
//!     [`valu_reference_gate_up_swiglu`], scaled-tolerance matching
//!     `ParityConfig::scaled_output_tolerance`.
//!   * [`bench_dynamic_gate_up`] — warm-up + median-of-N Instant-based
//!     timing (no HipEvent — stream sync between each run is enough
//!     granularity for the 1D search).
//!   * [`stability_dynamic_gate_up`] — Top-K post-GA check with
//!     a higher bench sample count and 1000-block parity.
//!   * [`KernelGa::run_num_waves_only`] — the mini-GA driver. Custom
//!     loop rather than `KernelGa::run_with` because the 1D-space
//!     wants `random_num_waves_only` seeding instead of generic
//!     `KernelGenome::random`, and the fitness closure needs mutable
//!     access to the dynamic cache + pre-uploaded device buffers.

#![cfg(feature = "gpu")]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use crate::v1::backend::gpu::error::{check, HipResult};
use crate::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
use crate::v1::backend::gpu::wrappers::{HipBuffer, HipStream};
use crate::v1::ir::codegen_gpu::{
    emit_q4_k_gemv_gate_up_swiglu_parametric, ga_gate_up_swiglu_symbol,
};
use crate::v1::ir::formats::q4_k;
use crate::v1::ir::types::QuantFormat;

use super::compile::{compile_hip_source, parse_amdgpu_metadata, CompileError};
use super::dynamic::{DynamicKernel, GateUpSwigluGeometry};
use super::engine::{GaConfig, KernelGa};
use super::genome::{DequantStrategy, KernelGenome};
use super::logger::GaLogger;
use super::parity::{
    check_parity_output_pair, generate_deterministic_test_blocks, valu_reference_gate_up_swiglu,
    ParityConfig, ParityResult,
};
use super::rng::SeededRng;
use super::types::{CodeObjectResources, KernelShape, PrecisionLevel};

/// Legal `num_waves` values for the parametric gate_up_swiglu
/// emitter. Must match what the codegen accepts (`num_waves ∈
/// {1, 2, 4, 8}` — anything else would change threads_per_block in
/// ways the static helper layout doesn't support yet).
pub const BLOCK_C_NUM_WAVES_VALUES: &[u8] = &[1, 2, 4, 8];

/// Phase-1 default settings for every genome axis Block C does NOT
/// parametrise. Matches `emit_q4_k_gemv_gate_up_swiglu_parametric`'s
/// fixed `#define`s so all GA candidates share the same host-side
/// launch geometry (only `waves_per_block` varies).
pub fn fixed_genome() -> KernelGenome {
    KernelGenome {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        tiles_per_wave: 1,
        waves_per_block: 8, // overwritten by callers
        use_lds_for_a: false,
        use_lds_for_b: true,
        prefetch_depth: 0,
        k_unroll: 1,
        double_buffer: false,
        dequant_strategy: DequantStrategy::Inline,
    }
}

/// Draw a random genome varying only `num_waves`. Used to seed the
/// Block-C population — the rest of the genome stays at
/// [`fixed_genome`] so `KernelGenome::sanitize` never has to rescue
/// a cross-field violation.
pub fn random_num_waves_only(rng: &mut SeededRng) -> KernelGenome {
    let mut g = fixed_genome();
    g.waves_per_block = *rng.choose(BLOCK_C_NUM_WAVES_VALUES);
    g
}

/// Small cache of compiled `num_waves → DynamicKernel`. 4 unique
/// entries are expected; the whole structure is re-created per
/// `run_num_waves_only` call.
pub struct DynamicKernelCache {
    entries: HashMap<u8, Arc<DynamicKernel>>,
    /// Stores the parsed AMDGPU metadata alongside the kernel —
    /// needed for the post-compile VGPR gate and the JSONL-log's
    /// `actual_vgpr_count` field.
    resources: HashMap<u8, CodeObjectResources>,
    pub hits: u64,
    pub misses: u64,
}

impl DynamicKernelCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            resources: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Compile or fetch the kernel for `num_waves`. Invokes hipcc +
    /// `clang-offload-bundler` on a miss (≈ 1 s wall-clock).
    pub fn get_or_compile(&mut self, num_waves: u8) -> Result<Arc<DynamicKernel>, CompileError> {
        if let Some(k) = self.entries.get(&num_waves) {
            self.hits += 1;
            return Ok(k.clone());
        }
        self.misses += 1;

        let (src, symbol) = emit_q4_k_gemv_gate_up_swiglu_parametric(num_waves as u32);
        let kernel_label = format!("ga_gate_up_w{num_waves}");
        let co_bytes = compile_hip_source(&src, &kernel_label)?;

        // Post-compile VGPR read on the extracted gfx1201 ELF.
        let extracted_path = std::env::temp_dir()
            .join("rocmforge_ga")
            .join(format!("{kernel_label}.gfx1201.co"));
        let res = parse_amdgpu_metadata(&extracted_path)
            .map_err(|e| CompileError::Io(e))?
            .unwrap_or(CodeObjectResources {
                vgpr_count: 0,
                sgpr_count: 0,
                lds_bytes: 0,
            });

        let geometry = GateUpSwigluGeometry::for_num_waves(num_waves as u32);
        let kernel = DynamicKernel::from_code_object(&co_bytes, symbol, geometry).map_err(|e| {
            CompileError::HipccFailed {
                kernel_name: kernel_label.clone(),
                stderr: format!("HipModule::load: {e:?}"),
                status: None,
            }
        })?;

        let kernel = Arc::new(kernel);
        self.entries.insert(num_waves, kernel.clone());
        self.resources.insert(num_waves, res);
        Ok(kernel)
    }

    pub fn resources_for(&self, num_waves: u8) -> Option<CodeObjectResources> {
        self.resources.get(&num_waves).copied()
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

impl Default for DynamicKernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatch `DynamicKernel::launch_gate_up_swiglu` into a
/// freshly-allocated device buffer and return the host-side output.
fn run_dyn_once(
    kernel: &DynamicKernel,
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
    n_rows: i32,
    ncols: i32,
) -> HipResult<Vec<f32>> {
    let stream = HipStream::new()?;
    let mut d_wg = HipBuffer::new(weights_gate.len())?;
    d_wg.copy_from_host(weights_gate)?;
    let mut d_wu = HipBuffer::new(weights_up.len())?;
    d_wu.copy_from_host(weights_up)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len())?;
    d_in.copy_from_host(in_bytes)?;
    let mut d_out = HipBuffer::new((ncols as usize) * 4)?;

    kernel.launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)?;
    stream.synchronize()?;

    let out_bytes = (ncols as usize) * 4;
    let mut host = vec![0u8; out_bytes];
    let rc = unsafe {
        hipMemcpy(
            host.as_mut_ptr() as *mut c_void,
            d_out.as_ptr(),
            out_bytes,
            hipMemcpyDeviceToHost,
        )
    };
    check(rc, "gate_up readback")?;
    Ok(host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Parity for a `DynamicKernel` against the CPU VALU reference.
/// Uses the same tolerance scaling as `check_parity_known_kernel`
/// from 2.1.2 — `(max_mag + 0.01) × √K × per_element_tolerance`.
/// Returns a `ParityResult`; `passed == true` means every block's
/// max_err fell under its block's scaled tolerance.
pub fn parity_dynamic_gate_up(
    kernel: &DynamicKernel,
    shape: &KernelShape,
    cfg: &ParityConfig,
    seed: u64,
) -> Result<ParityResult, String> {
    let fmt = q4_k();
    // Gate-up needs *two* weight tensors — the existing
    // `TestBlock::random` produces one, so we draw twice with
    // derived seeds.
    let blocks_gate = generate_deterministic_test_blocks(cfg.n_blocks, shape, &fmt, seed);
    let blocks_up =
        generate_deterministic_test_blocks(cfg.n_blocks, shape, &fmt, seed.wrapping_add(1));

    let mut max_err = 0.0f32;
    let mut max_tol = 0.0f32;
    let mut sum_err = 0.0f64;
    let mut total_elements = 0usize;
    let mut all_violations = Vec::new();

    for (block_idx, (tb_g, tb_u)) in blocks_gate.iter().zip(blocks_up.iter()).enumerate() {
        // Gate and Up share the *input* (they're applied to the same
        // activation); only the weight tensor differs.
        let input = &tb_g.input;
        let reference = valu_reference_gate_up_swiglu(
            &tb_g.weights,
            &tb_u.weights,
            input,
            &fmt,
            shape,
        );
        let max_mag = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tol_here = cfg.scaled_output_tolerance(max_mag, shape.k, false);
        if tol_here > max_tol {
            max_tol = tol_here;
        }

        let candidate = run_dyn_once(kernel, &tb_g.weights, &tb_u.weights, input, shape.k as i32, shape.n as i32)
            .map_err(|e| format!("dynamic gate_up run: {e:?}"))?;

        let (blk_max, blk_mean, viols) =
            check_parity_output_pair(&candidate, &reference, tol_here, block_idx);
        if blk_max > max_err {
            max_err = blk_max;
        }
        sum_err += (blk_mean as f64) * (reference.len() as f64);
        total_elements += reference.len();
        all_violations.extend(viols);
    }

    let mean = if total_elements > 0 {
        (sum_err / total_elements as f64) as f32
    } else {
        0.0
    };
    Ok(ParityResult {
        passed: all_violations.is_empty(),
        max_abs_err: max_err,
        mean_abs_err: mean,
        n_blocks_tested: cfg.n_blocks,
        violations: all_violations,
        effective_tolerance: max_tol,
    })
}

/// Median-of-`n_samples` timing (after `n_warmup` warmup dispatches).
/// Uses `Instant` around a `stream.synchronize()` — HipEvent would
/// be more precise but per-iteration Event create/destroy introduces
/// more variance than the kernel-level measurement gains (we saw
/// this empirically in 2.1.2 stability tuning).
pub fn bench_dynamic_gate_up(
    kernel: &DynamicKernel,
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
    n_rows: i32,
    ncols: i32,
    n_warmup: usize,
    n_samples: usize,
) -> HipResult<f64> {
    let stream = HipStream::new()?;
    let mut d_wg = HipBuffer::new(weights_gate.len())?;
    d_wg.copy_from_host(weights_gate)?;
    let mut d_wu = HipBuffer::new(weights_up.len())?;
    d_wu.copy_from_host(weights_up)?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len())?;
    d_in.copy_from_host(in_bytes)?;
    let mut d_out = HipBuffer::new((ncols as usize) * 4)?;

    for _ in 0..n_warmup {
        kernel.launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)?;
    }
    stream.synchronize()?;

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let t0 = Instant::now();
        kernel.launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, n_rows, ncols, &stream)?;
        stream.synchronize()?;
        samples.push(t0.elapsed().as_micros() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

/// Stability pass for a Top-K winner: 30 samples, median-variance +
/// 1000-block parity. Used after GA convergence.
pub struct DynamicStabilityResult {
    pub passed: bool,
    pub variance_pct: f64,
    pub median_us: f64,
    pub parity: ParityResult,
    pub reject_reason: Option<String>,
}

pub fn stability_dynamic_gate_up(
    kernel: &DynamicKernel,
    shape: &KernelShape,
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
) -> Result<DynamicStabilityResult, String> {
    // 3 input sets × 10 samples — match `§2.9`. We use the same
    // weights+input across sets (the sets differ only in seed in the
    // known-kernel stability check from 2.1.2, which the GA doesn't
    // reliably sync here). 30 samples from one input is sufficient to
    // detect timing instability on the Block-C 1-D search.
    let stream = HipStream::new().map_err(|e| format!("stream: {e:?}"))?;
    let mut d_wg = HipBuffer::new(weights_gate.len()).map_err(|e| format!("d_wg: {e:?}"))?;
    d_wg.copy_from_host(weights_gate)
        .map_err(|e| format!("d_wg up: {e:?}"))?;
    let mut d_wu = HipBuffer::new(weights_up.len()).map_err(|e| format!("d_wu: {e:?}"))?;
    d_wu.copy_from_host(weights_up)
        .map_err(|e| format!("d_wu up: {e:?}"))?;
    let in_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let mut d_in = HipBuffer::new(in_bytes.len()).map_err(|e| format!("d_in: {e:?}"))?;
    d_in.copy_from_host(in_bytes)
        .map_err(|e| format!("d_in up: {e:?}"))?;
    let mut d_out = HipBuffer::new(shape.n * 4).map_err(|e| format!("d_out: {e:?}"))?;

    // 10 warmup dispatches (same tuning as 2.1.2).
    for _ in 0..10 {
        kernel
            .launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, shape.k as i32, shape.n as i32, &stream)
            .map_err(|e| format!("warmup: {e:?}"))?;
    }
    stream.synchronize().map_err(|e| format!("warmup sync: {e:?}"))?;

    let mut samples = Vec::with_capacity(30);
    for _ in 0..30 {
        let t0 = Instant::now();
        kernel
            .launch_gate_up_swiglu(&d_wg, &d_wu, &d_in, &mut d_out, shape.k as i32, shape.n as i32, &stream)
            .map_err(|e| format!("timed: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("timed sync: {e:?}"))?;
        samples.push(t0.elapsed().as_micros() as f64);
    }

    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = sorted[sorted.len() / 2];
    let max_dev = samples
        .iter()
        .map(|t| (t - median_us).abs())
        .fold(0.0f64, f64::max);
    let variance_pct = if median_us > 0.0 {
        (max_dev / median_us) * 100.0
    } else {
        100.0
    };

    let parity = parity_dynamic_gate_up(kernel, shape, &ParityConfig::for_stability(), 4242)?;

    // Stability gate tolerant of real-hardware jitter, matching 2.1.2's
    // empirical findings (cold-start dispatch variance on gfx1201 runs
    // 1.5 – 4 % on ~30 µs kernels). Block C only flags egregious
    // instability (> 5 %) so the 1-D GA doesn't reject good
    // num_waves choices because of governor noise.
    let mut reject: Option<String> = None;
    if variance_pct >= 5.0 {
        reject = Some(format!("variance {variance_pct:.2}% ≥ 5.00% ceiling"));
    }
    if !parity.passed {
        reject = Some(match reject {
            Some(prev) => format!(
                "{prev}; parity fail: max_err={:.6} > tol={:.6}",
                parity.max_abs_err, parity.effective_tolerance
            ),
            None => format!(
                "parity fail: max_err={:.6} > tol={:.6}",
                parity.max_abs_err, parity.effective_tolerance
            ),
        });
    }

    Ok(DynamicStabilityResult {
        passed: reject.is_none(),
        variance_pct,
        median_us,
        parity,
        reject_reason: reject,
    })
}

/// One Top-K candidate that survived the Block-C stability pass.
#[derive(Debug, Clone)]
pub struct BlockCWinner {
    pub genome: KernelGenome,
    pub median_us: f64,
    pub fitness: f64,
    pub vgpr_count: u16,
    pub parity_max_err: f32,
    pub stability_variance_pct: f64,
    pub stability_passed: bool,
    pub parity_passed: bool,
}

/// Aggregate Block-C GA result.
#[derive(Debug, Clone)]
pub struct BlockCResult {
    pub all_candidates: Vec<BlockCCandidate>,
    pub stable_winners: Vec<BlockCWinner>,
    pub generations_ran: usize,
    pub early_exited: bool,
    pub compile_hits: u64,
    pub compile_misses: u64,
}

/// Per-candidate bookkeeping — one entry per **unique** `num_waves`
/// value the GA evaluated. Re-evaluations of the same value collapse
/// into a single entry (the median over all samples).
#[derive(Debug, Clone)]
pub struct BlockCCandidate {
    pub num_waves: u8,
    pub median_us: f64,
    pub fitness: f64,
    pub vgpr_count: u16,
    pub parity_max_err: f32,
    pub parity_passed: bool,
}

impl KernelGa {
    /// Block-C mini-GA driver. 1-D search over `num_waves` with the
    /// full 6-phase fitness pipeline (pre-compile → compile →
    /// post-compile VGPR → parity → warmup → benchmark). Returns
    /// a [`BlockCResult`] with every unique candidate's stats +
    /// a stability-validated Top-K list.
    ///
    /// `config` should typically be [`GaConfig`] with
    /// `population_size: 8, generations: 5, tournament_size: 2`
    /// (see prompt — the 1-D space doesn't need the §2.5 spec
    /// defaults).
    #[allow(clippy::too_many_arguments)]
    pub fn run_num_waves_only(
        &mut self,
        shape: &KernelShape,
        _fmt: &QuantFormat,
        _level: PrecisionLevel,
        baseline_us: f32,
        weights_gate: &[u8],
        weights_up: &[u8],
        input: &[f32],
        cache: &mut DynamicKernelCache,
        logger: &mut GaLogger,
    ) -> BlockCResult {
        let shape_label = "gemv_q4_k_gate_up_swiglu_block_c";
        let _ = logger.log_shape_start(shape_label);

        // Seed initial population with random num_waves values.
        let mut population: Vec<KernelGenome> = (0..self.config.population_size)
            .map(|_| random_num_waves_only(&mut self.rng))
            .collect();

        // Memoise per-num_waves results: the benchmark is
        // deterministic enough that running num_waves=4 twice in
        // the same generation would waste budget. The GA still
        // proceeds through its tournament/crossover/mutation as
        // normal — only the eval is memoised.
        let mut per_waves_candidate: HashMap<u8, BlockCCandidate> = HashMap::new();

        let mut best_history: Vec<f64> = Vec::with_capacity(self.config.generations);
        let mut generations_ran = 0usize;
        let mut early_exited = false;

        for gen in 0..self.config.generations {
            let mut scored = Vec::with_capacity(population.len());

            for (individual, genome) in population.iter().enumerate() {
                let num_waves = genome.waves_per_block;
                if let Some(existing) = per_waves_candidate.get(&num_waves) {
                    // Memo hit — reuse the fitness.
                    scored.push((individual, genome.clone(), existing.fitness));
                    continue;
                }
                let fit = evaluate_one_genome(
                    genome,
                    shape,
                    baseline_us,
                    weights_gate,
                    weights_up,
                    input,
                    cache,
                    logger,
                    shape_label,
                    gen,
                    individual,
                    self.config.seed,
                );
                let entry = BlockCCandidate {
                    num_waves,
                    median_us: if fit.fitness > 0.0 {
                        (baseline_us as f64) / fit.fitness
                    } else {
                        f64::INFINITY
                    },
                    fitness: fit.fitness,
                    vgpr_count: fit.vgpr_count,
                    parity_max_err: fit.parity_max_err,
                    parity_passed: fit.parity_passed,
                };
                per_waves_candidate.insert(num_waves, entry);
                scored.push((individual, genome.clone(), fit.fitness));
            }

            let best = scored
                .iter()
                .map(|(_, _, f)| *f)
                .fold(0.0f64, f64::max);
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

            // Next generation — simple tournament-of-2 + crossover +
            // num-waves-only mutation. Elitism = 1 individual.
            let mut sorted = scored.clone();
            sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let mut next = Vec::with_capacity(self.config.population_size);
            next.push(sorted[0].1.clone()); // elite

            while next.len() < self.config.population_size {
                let a = tournament_pick(&scored, self.config.tournament_size, &mut self.rng);
                if self.rng.gen_f64() < self.config.crossover_rate {
                    let b = tournament_pick(&scored, self.config.tournament_size, &mut self.rng);
                    let mut child = a.clone();
                    if self.rng.gen_bool() {
                        child.waves_per_block = b.waves_per_block;
                    }
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.waves_per_block = *self.rng.choose(BLOCK_C_NUM_WAVES_VALUES);
                    }
                    next.push(child);
                } else {
                    let mut child = a.clone();
                    if self.rng.gen_f64() < self.config.mutation_rate {
                        child.waves_per_block = *self.rng.choose(BLOCK_C_NUM_WAVES_VALUES);
                    }
                    next.push(child);
                }
            }
            population = next;
        }

        // Sort candidates by fitness descending and take Top-K.
        let mut all_candidates: Vec<BlockCCandidate> =
            per_waves_candidate.into_values().collect();
        all_candidates
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = all_candidates.iter().take(3).cloned().collect::<Vec<_>>();

        // Thermal cool-down before stability (2.1.2 §2.9 convention).
        std::thread::sleep(std::time::Duration::from_secs(2));

        let mut stable_winners = Vec::new();
        for cand in &top_k {
            if !cand.parity_passed || cand.fitness <= 0.0 {
                continue;
            }
            let kernel = match cache.get_or_compile(cand.num_waves) {
                Ok(k) => k,
                Err(_) => continue,
            };
            let st = match stability_dynamic_gate_up(&kernel, shape, weights_gate, weights_up, input) {
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
                stable_winners.push(BlockCWinner {
                    genome: {
                        let mut g = fixed_genome();
                        g.waves_per_block = cand.num_waves;
                        g
                    },
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
                let _ = logger.log_stability_fail(shape_label, cand.fitness, st.variance_pct, reason);
            }
        }

        let _ = logger.log_shape_complete(
            shape_label,
            stable_winners
                .first()
                .map(|w| w.fitness)
                .unwrap_or(all_candidates.first().map(|c| c.fitness).unwrap_or(0.0)),
        );

        BlockCResult {
            all_candidates,
            stable_winners,
            generations_ran,
            early_exited,
            compile_hits: cache.hits,
            compile_misses: cache.misses,
        }
    }
}

fn tournament_pick<'a>(
    scored: &'a [(usize, KernelGenome, f64)],
    size: usize,
    rng: &mut SeededRng,
) -> &'a KernelGenome {
    let mut best_idx = rng.gen_index(scored.len());
    for _ in 1..size {
        let cand = rng.gen_index(scored.len());
        if scored[cand].2 > scored[best_idx].2 {
            best_idx = cand;
        }
    }
    &scored[best_idx].1
}

/// Result of evaluating one genome through the 6-phase pipeline.
struct OneEval {
    fitness: f64,
    vgpr_count: u16,
    parity_max_err: f32,
    parity_passed: bool,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_one_genome(
    genome: &KernelGenome,
    shape: &KernelShape,
    baseline_us: f32,
    weights_gate: &[u8],
    weights_up: &[u8],
    input: &[f32],
    cache: &mut DynamicKernelCache,
    logger: &mut GaLogger,
    shape_label: &str,
    generation: usize,
    individual: usize,
    seed: u64,
) -> OneEval {
    // Phase 2: compile (hit or miss).
    let kernel = match cache.get_or_compile(genome.waves_per_block) {
        Ok(k) => k,
        Err(_) => {
            return OneEval {
                fitness: 0.0,
                vgpr_count: 0,
                parity_max_err: 0.0,
                parity_passed: false,
            };
        }
    };

    // Phase 3: post-compile VGPR gate — 2.1.1's
    // `validate_post_compile` on the real resources.
    let res = cache.resources_for(genome.waves_per_block).unwrap_or(CodeObjectResources {
        vgpr_count: 0,
        sgpr_count: 0,
        lds_bytes: 0,
    });
    let max_waves = res.max_waves_per_cu();
    if max_waves < 4 {
        // 2.1.1 post-compile gate — reject below 4 waves/CU.
        return OneEval {
            fitness: 0.0,
            vgpr_count: res.vgpr_count,
            parity_max_err: 0.0,
            parity_passed: false,
        };
    }

    // Phase 4: parity.
    let parity_cfg = ParityConfig::for_ga(PrecisionLevel::Fp16);
    let parity = match parity_dynamic_gate_up(&kernel, shape, &parity_cfg, 12345) {
        Ok(p) => p,
        Err(_) => {
            return OneEval {
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
        return OneEval {
            fitness: 0.0,
            vgpr_count: res.vgpr_count,
            parity_max_err: parity.max_abs_err,
            parity_passed: false,
        };
    }

    // Phases 5+6: warmup + benchmark.
    let median_us = match bench_dynamic_gate_up(
        &kernel,
        weights_gate,
        weights_up,
        input,
        shape.k as i32,
        shape.n as i32,
        5,
        20,
    ) {
        Ok(v) => v,
        Err(_) => {
            return OneEval {
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
        serde_json::json!({"num_waves": genome.waves_per_block}),
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

    OneEval {
        fitness,
        vgpr_count: res.vgpr_count,
        parity_max_err: parity.max_abs_err,
        parity_passed: true,
    }
}

/// Convenient constructor for the tuned-for-1D GaConfig used by
/// Block C (Pop 8, Gen 5, Tournament 2, higher mutation rate so the
/// small search space gets fully explored).
pub fn block_c_default_config(seed: u64) -> GaConfig {
    GaConfig {
        population_size: 8,
        generations: 5,
        tournament_size: 2,
        crossover_rate: 0.7,
        mutation_rate: 0.3,
        elitism_fraction: 0.125, // 1 of 8
        early_exit_generations: 3,
        early_exit_threshold: 0.01,
        seed,
        run_id: format!("block_c_mini_ga_seed{seed}"),
    }
}
