//! Phase 2 / Schritt 2.1.3 Block C — first real GA run.
//!
//! Validates the wired-up pipeline end-to-end:
//!   genome → emit_parametric → hipcc → HipModule → launch
//!   → parity → benchmark → fitness → GA selection → Top-K → stability.
//!
//! Search space is 1-D (`num_waves ∈ {1,2,4,8}`), so the mini-GA
//! (Pop 8, Gen 5, Tournament 2, Seed 42) converges essentially on
//! the first generation. The tests assert that the GA doesn't drop
//! a valid candidate, produces a seed-reproducible winner, and that
//! the winner beats or matches the Phase-1-default (`num_waves=8`)
//! — i.e. the GA confirms v0.x-lesson #1 (empirical measurement
//! beats guessing).

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::ga::block_c::{
    bench_dynamic_gate_up, block_c_default_config, fixed_genome, parity_dynamic_gate_up,
    random_num_waves_only, BlockCResult, DynamicKernelCache, BLOCK_C_NUM_WAVES_VALUES,
};
use rocmforge::v1::ga::engine::KernelGa;
use rocmforge::v1::ga::logger::GaLogger;
use rocmforge::v1::ga::parity::ParityConfig;
use rocmforge::v1::ga::rng::SeededRng;
use rocmforge::v1::ga::types::{KernelShape, PrecisionLevel};
use rocmforge::v1::ir::formats::q4_k;
use serial_test::serial;

// ── Test fixtures ────────────────────────────────────────────────────────

/// Shape used by the Block-C tests. A scaled-down proxy for the
/// Qwen3-8B gate_up_swiglu (K=4096 real, N=14336 real) — the real
/// N would make the CPU VALU reference spend minutes per parity
/// block (N×K FP32 dequant = 234 MB × 10 blocks × 40 evals).
/// The test shape keeps K=4096 (so per-row Q4_K super-block count
/// matches real weights) and drops N to 512 (divisible by
/// `num_waves × 4` for every legal num_waves). The fitness ratios
/// the GA discovers still transfer because num_waves only changes
/// the launch geometry — not the per-output-column work.
fn block_c_shape() -> KernelShape {
    KernelShape::new(1, 512, 4096)
}

/// Generate one Q4_K weight matrix for the gate_up_swiglu benchmark.
/// Small scales keep dequantised values bounded so FP32 accumulation
/// over K=4096 stays well inside FP16 representable range.
fn gen_q4k_weights(n_rows: usize, ncols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(n_rows % 256, 0);
    let blocks_per_col = n_rows / 256;
    let total = ncols * blocks_per_col;
    let mut buf = vec![0u8; total * 144];
    let mut rng = fastrand::Rng::with_seed(seed);
    for b in 0..total {
        let d = (rng.f32() * 2.0 - 1.0) * 0.01;
        let dmin = (rng.f32() * 2.0 - 1.0) * 0.01;
        buf[b * 144..b * 144 + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        buf[b * 144 + 2..b * 144 + 4]
            .copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..16 {
            buf[b * 144 + i] = rng.u8(..) & 0x3F;
        }
        for i in 16..144 {
            buf[b * 144 + i] = rng.u8(..);
        }
    }
    buf
}

fn gen_input(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

/// Shared GA driver used across the tests — runs the mini-GA once
/// with a supplied seed and returns the full result plus a parallel
/// log capture (so tests can assert on the JSONL event stream).
fn run_mini_ga(seed: u64) -> (BlockCResult, Vec<String>) {
    let shape = block_c_shape();
    let weights_gate = gen_q4k_weights(shape.k, shape.n, 0xA1A1);
    let weights_up = gen_q4k_weights(shape.k, shape.n, 0xA2A2);
    let input = gen_input(shape.k, 0xA3A3);

    let mut cache = DynamicKernelCache::new();
    let mut logger = GaLogger::capturing(format!("block-c-seed-{seed}"));
    let mut ga = KernelGa::new(block_c_default_config(seed));

    let result = ga.run_num_waves_only(
        &shape,
        &q4_k(),
        PrecisionLevel::Fp16,
        432.8, // Post-P0 gate_up_swiglu baseline from 2.0.3
        &weights_gate,
        &weights_up,
        &input,
        &mut cache,
        &mut logger,
    );
    let lines = logger.captured_lines().unwrap().to_vec();
    (result, lines)
}

// ── CompileCache with real compile path ─────────────────────────────────

#[test]
#[serial]
fn test_compile_cache_real_kernel() {
    let mut cache = DynamicKernelCache::new();
    let k1 = cache.get_or_compile(4).expect("first compile");
    let k2 = cache.get_or_compile(4).expect("second lookup");
    assert!(std::sync::Arc::ptr_eq(&k1, &k2), "cache should hit on identical num_waves");
    assert_eq!(cache.hits, 1);
    assert_eq!(cache.misses, 1);
    let res = cache.resources_for(4).expect("resources for num_waves=4");
    assert!(res.vgpr_count > 0 && res.vgpr_count < 256);
    println!(
        "num_waves=4 cache: vgprs={} sgprs={} hit_rate={:.0}%",
        res.vgpr_count,
        res.sgpr_count,
        cache.hit_rate() * 100.0
    );
}

#[test]
#[serial]
fn test_compile_cache_four_unique_entries() {
    let mut cache = DynamicKernelCache::new();
    for &w in BLOCK_C_NUM_WAVES_VALUES {
        cache.get_or_compile(w).expect("compile");
    }
    // Second pass — every call should hit.
    for &w in BLOCK_C_NUM_WAVES_VALUES {
        cache.get_or_compile(w).expect("hit");
    }
    assert_eq!(cache.misses, 4, "exactly 4 unique compiles");
    assert_eq!(cache.hits, 4);
    assert_eq!(cache.hit_rate(), 0.5);
}

// ── Parity on a single dynamic kernel ───────────────────────────────────

#[test]
#[serial]
fn test_dynamic_gate_up_parity() {
    let mut cache = DynamicKernelCache::new();
    let kernel = cache.get_or_compile(8).expect("compile w=8");
    let shape = block_c_shape();
    let cfg = ParityConfig::for_ga(PrecisionLevel::Fp16);
    let parity = parity_dynamic_gate_up(&kernel, &shape, &cfg, 12345).expect("parity");
    println!(
        "dynamic w=8 parity: max_err={:.3} mean={:.3} tol={:.3}",
        parity.max_abs_err, parity.mean_abs_err, parity.effective_tolerance
    );
    assert!(
        parity.passed,
        "dynamic w=8 must pass parity vs VALU reference (max_err={} > tol={})",
        parity.max_abs_err, parity.effective_tolerance
    );
}

// ── Mini-GA convergence + every candidate survived parity ──────────────

#[test]
#[serial]
fn test_mini_ga_finds_winner_and_all_candidates_parity() {
    let (result, _log) = run_mini_ga(42);
    println!(
        "Block C GA result: {} candidates, {} stable winners, generations_ran={}, early_exited={}",
        result.all_candidates.len(),
        result.stable_winners.len(),
        result.generations_ran,
        result.early_exited,
    );
    for cand in &result.all_candidates {
        println!(
            "  num_waves={}  median={:.1} µs  fitness={:.3}  vgprs={}  parity_err={:.3}",
            cand.num_waves, cand.median_us, cand.fitness, cand.vgpr_count, cand.parity_max_err,
        );
    }

    assert!(
        result.all_candidates.len() >= 3,
        "mini-GA must explore at least 3 of the 4 num_waves values; got {}",
        result.all_candidates.len()
    );
    for cand in &result.all_candidates {
        assert!(
            cand.parity_passed,
            "every evaluated candidate must pass parity (num_waves={} failed)",
            cand.num_waves
        );
    }
    let best_fitness = result.all_candidates[0].fitness;
    assert!(
        best_fitness > 0.0,
        "GA must find at least one positive-fitness candidate"
    );
}

// ── Seed reproducibility ────────────────────────────────────────────────

#[test]
#[serial]
fn test_mini_ga_seed_reproducible() {
    let (r1, _) = run_mini_ga(42);
    let (r2, _) = run_mini_ga(42);
    let best1 = r1.all_candidates[0].num_waves;
    let best2 = r2.all_candidates[0].num_waves;
    assert_eq!(
        best1, best2,
        "same seed must pick the same winner (got {best1} vs {best2})"
    );
    // Fitness can differ slightly across runs due to kernel timing
    // jitter — but the winner's num_waves is the GA decision and
    // must be deterministic for a fixed seed.
}

// ── Stability pass ──────────────────────────────────────────────────────

#[test]
#[serial]
fn test_mini_ga_stable_winner_exists() {
    let (result, _) = run_mini_ga(42);
    // At least ONE stable winner should survive the 5 %-variance gate.
    // (If GPU is under load this can flake; we only demand ≥1 because
    // all 4 configs share the same kernel body modulo geometry.)
    assert!(
        !result.stable_winners.is_empty(),
        "expected ≥1 stable winner from Top-3; all rejected"
    );
    for w in &result.stable_winners {
        println!(
            "  stable: num_waves={}  median={:.1} µs  fitness={:.3}  variance={:.2}%",
            w.genome.waves_per_block, w.median_us, w.fitness, w.stability_variance_pct
        );
        assert!(w.parity_passed);
        assert!(w.stability_passed);
    }
}

// ── JSONL log content ───────────────────────────────────────────────────

#[test]
#[serial]
fn test_mini_ga_jsonl_contains_events() {
    let (_result, log_lines) = run_mini_ga(42);
    assert!(log_lines.len() >= 5, "log should have multiple events");
    let joined = log_lines.join("\n");
    assert!(joined.contains("\"event\":\"shape_start\""));
    assert!(joined.contains("\"event\":\"generation_complete\""));
    assert!(joined.contains("\"event\":\"eval\""));
    assert!(joined.contains("\"event\":\"shape_complete\""));
    for line in &log_lines {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("every log line must be valid JSON");
        assert_eq!(v["run_id"].as_str(), Some("block-c-seed-42"));
    }
}

// ── Winner vs baseline (num_waves=8 Phase-1 default) ────────────────────

#[test]
#[serial]
fn test_winner_not_slower_than_phase1_default() {
    // The GA should never pick a kernel strictly slower than the
    // Phase-1 default (num_waves=8). If it does, something's wrong
    // with the fitness evaluation. We allow a 5 % slack for timing
    // jitter — a tied result is fine.
    let (result, _) = run_mini_ga(42);
    let winner = result
        .all_candidates
        .first()
        .expect("at least one candidate");
    let phase1 = result
        .all_candidates
        .iter()
        .find(|c| c.num_waves == 8)
        .expect("num_waves=8 must be evaluated");
    println!(
        "winner num_waves={} @ {:.1} µs  vs  phase-1 default (w=8) @ {:.1} µs",
        winner.num_waves, winner.median_us, phase1.median_us
    );
    assert!(
        winner.median_us <= phase1.median_us * 1.05,
        "GA winner ({} µs) is more than 5 % slower than Phase-1 default ({} µs) — fitness logic is broken",
        winner.median_us,
        phase1.median_us,
    );
}

// ── Sanity: random_num_waves_only respects the legal set ───────────────

#[test]
fn test_random_num_waves_only_is_legal() {
    let mut rng = SeededRng::new(1);
    for _ in 0..100 {
        let g = random_num_waves_only(&mut rng);
        assert!(BLOCK_C_NUM_WAVES_VALUES.contains(&g.waves_per_block));
        // All other fields match the fixed baseline.
        let base = fixed_genome();
        assert_eq!(g.tile_m, base.tile_m);
        assert_eq!(g.tile_k, base.tile_k);
        assert_eq!(g.use_lds_for_a, base.use_lds_for_a);
        assert_eq!(g.use_lds_for_b, base.use_lds_for_b);
    }
}

// ── Isolated winner timing vs Phase-1 default (same shape) ──────────────

#[test]
#[serial]
fn test_winner_isolated_timing_vs_phase1_default() {
    // Honest apples-to-apples: compare the GA winner to the
    // Phase-1 default (num_waves=8) on the same test shape. The
    // 432.8 µs baseline from 2.0.3 applies to real Qwen3 weights at
    // N=14336; this test uses N=512 test-shape weights, so absolute
    // µs aren't comparable to rocprof. What IS comparable: the
    // winner's speed vs the default's speed — both measured under
    // identical conditions in the same process.
    let (result, _) = run_mini_ga(42);
    let winner = &result.all_candidates[0];
    let phase1 = result
        .all_candidates
        .iter()
        .find(|c| c.num_waves == 8)
        .expect("num_waves=8 must be evaluated");
    let delta_pct = (phase1.median_us - winner.median_us) / phase1.median_us * 100.0;
    println!(
        "Block-C on test shape: winner num_waves={} @ {:.1} µs vs Phase-1 default (w=8) @ {:.1} µs — {:.1}% faster",
        winner.num_waves, winner.median_us, phase1.median_us, delta_pct
    );
    // NOTE — projecting this to full-model decode is tempting, but
    // the optimal num_waves at N=14336 may differ from N=512 because
    // grid-X (ncols / (num_waves × 4)) scales differently with N.
    // The only thing this test proves is: at the SAME shape the GA
    // picks a faster configuration than the hard-coded default.
    //
    // The real-model projection lands in the step-2.1.3 report only
    // after a shape-matched benchmark runs end-to-end.
    assert!(
        delta_pct >= 0.0,
        "winner must not be slower than Phase-1 default at the same shape"
    );
}

// ── Standalone bench helper sanity ──────────────────────────────────────

#[test]
#[serial]
fn test_bench_dynamic_gate_up_returns_positive_median() {
    let mut cache = DynamicKernelCache::new();
    let kernel = cache.get_or_compile(4).expect("compile");
    let shape = block_c_shape();
    let weights_gate = gen_q4k_weights(shape.k, shape.n, 0xC1);
    let weights_up = gen_q4k_weights(shape.k, shape.n, 0xC2);
    let input = gen_input(shape.k, 0xC3);
    let us = bench_dynamic_gate_up(
        &kernel,
        &weights_gate,
        &weights_up,
        &input,
        shape.k as i32,
        shape.n as i32,
        5,
        20,
    )
    .expect("bench");
    println!("num_waves=4 @ shape {}×{}×{}: {:.1} µs", shape.m, shape.n, shape.k, us);
    assert!(us > 0.0 && us < 100_000.0);
}
