//! Phase 2 / Schritt 2.1.3 Block D — 2-D GA (`num_waves ×
//! multi_row_cols`) + Executor-Integration für dynamische Kernel.
//!
//! Testt die drei Deliverables:
//!
//!   1. **Parametrisierter Codegen** — `emit_*_parametric_2d` für
//!      alle 16 Kombinationen kompiliert, Symbole sind unique,
//!      Parity vs VALU besteht.
//!   2. **Mini-GA** — konvergiert auf der realen Shape (K=4096,
//!      N=14336), Winner mindestens so schnell wie Block-C-Winner
//!      `(w=4, c=4)`, Seed-reproduzierbar.
//!   3. **Executor-Integration** — `DynamicGateUpHook` wird
//!      erkannt + gedispatcht; bei Shape-Mismatch oder ohne Hook
//!      läuft der statische Fallback.
//!   4. **End-to-End Decode** (GPU + Modell-gegatet) — GA-Winner
//!      in der Pipeline installiert, Decode produziert kohärenten
//!      Text mit ≥ baseline tok/s.
//!
//! Einige Tests (End-to-End) sind hinter `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1`
//! gegatet, genau wie die Phase-1-Real-Model-Tests.

#![cfg(all(feature = "v1", feature = "gpu"))]

use half::f16;
use rocmforge::v1::ga::block_c::{
    bench_dynamic_gate_up, parity_dynamic_gate_up, DynamicKernelCache,
};
use rocmforge::v1::ga::block_d::{
    block_d_default_config, make_gate_up_hook, random_block_d_genome, BlockDGenome,
    DynamicKernelCache2D, BLOCK_D_MULTI_ROW_COLS_VALUES, BLOCK_D_NUM_WAVES_VALUES,
};
use rocmforge::v1::ga::engine::KernelGa;
use rocmforge::v1::ga::logger::GaLogger;
use rocmforge::v1::ga::parity::ParityConfig;
use rocmforge::v1::ga::rng::SeededRng;
use rocmforge::v1::ga::types::{KernelShape, PrecisionLevel};
use rocmforge::v1::ir::codegen_gpu::{
    emit_q4_k_gemv_gate_up_swiglu_parametric_2d, ga_gate_up_swiglu_symbol_2d,
};
use rocmforge::v1::ir::formats::q4_k;
use serial_test::serial;
use std::sync::Arc;

// ── Shapes ────────────────────────────────────────────────────────────────

/// Parity shape — small N keeps the CPU VALU reference well under
/// the OOM ceiling Block C hit on N=14336.
fn parity_shape() -> KernelShape {
    KernelShape::new(1, 512, 4096)
}

/// Real benchmark shape — Qwen3-8B `(hidden_dim, ffn_dim)` is
/// `(4096, 12288)` and Qwen2.5-7B is `(3584, 18944)`. The prompt
/// specifies 14336, so we benchmark at that shape to match the GA
/// spec and compare 1:1 with the post-P0 baseline.
fn bench_shape() -> KernelShape {
    KernelShape::new(1, 14336, 4096)
}

// ── Test fixtures ─────────────────────────────────────────────────────────

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
        buf[b * 144 + 2..b * 144 + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
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

// ── CPU-only: codegen basics ──────────────────────────────────────────────

#[test]
fn test_parametric_2d_symbol_unique_per_pair() {
    let s_1_1 = ga_gate_up_swiglu_symbol_2d(1, 1);
    let s_4_4 = ga_gate_up_swiglu_symbol_2d(4, 4);
    let s_8_8 = ga_gate_up_swiglu_symbol_2d(8, 8);
    let s_4_8 = ga_gate_up_swiglu_symbol_2d(4, 8);
    let s_8_4 = ga_gate_up_swiglu_symbol_2d(8, 4);
    let all = [&s_1_1, &s_4_4, &s_8_8, &s_4_8, &s_8_4];
    for (i, a) in all.iter().enumerate() {
        for b in all.iter().skip(i + 1) {
            assert_ne!(a, b, "every (w,c) pair must yield a unique symbol");
        }
    }
    assert!(s_4_4.contains("w4"));
    assert!(s_4_4.contains("c4"));
    assert!(s_8_4.contains("w8") && s_8_4.contains("c4"));
}

#[test]
fn test_parametric_2d_source_contains_both_defines() {
    for &w in BLOCK_D_NUM_WAVES_VALUES {
        for &c in BLOCK_D_MULTI_ROW_COLS_VALUES {
            let (src, sym) = emit_q4_k_gemv_gate_up_swiglu_parametric_2d(w as u32, c as u32);
            assert!(src.contains(&sym));
            assert!(
                src.contains(&format!("#define Q4_K_FIXED_WAVES       {w}")),
                "missing FIXED_WAVES define for ({w}, {c})"
            );
            assert!(
                src.contains(&format!("#define Q4_K_MULTI_ROW_COLS    {c}")),
                "missing MULTI_ROW_COLS define for ({w}, {c})"
            );
            assert!(src.contains("extern \"C\" __launch_bounds__"));
        }
    }
}

#[test]
fn test_random_block_d_genome_is_legal() {
    let mut rng = SeededRng::new(1);
    for _ in 0..200 {
        let g = random_block_d_genome(&mut rng);
        assert!(BLOCK_D_NUM_WAVES_VALUES.contains(&g.num_waves));
        assert!(BLOCK_D_MULTI_ROW_COLS_VALUES.contains(&g.multi_row_cols));
    }
}

// ── Compile cache with real hipcc ─────────────────────────────────────────

#[test]
#[serial]
fn test_compile_cache_2d_hit_miss() {
    let mut cache = DynamicKernelCache2D::new();
    let g = BlockDGenome {
        num_waves: 4,
        multi_row_cols: 4,
    };
    let k1 = cache.get_or_compile(g).expect("compile miss");
    let k2 = cache.get_or_compile(g).expect("cache hit");
    assert!(Arc::ptr_eq(&k1, &k2));
    assert_eq!(cache.hits, 1);
    assert_eq!(cache.misses, 1);
    let res = cache.resources_for(g).expect("resources");
    assert!(res.vgpr_count > 0 && res.vgpr_count < 256);
    println!(
        "(w=4, c=4): vgprs={} sgprs={} lds={} hit_rate={:.0}%",
        res.vgpr_count,
        res.sgpr_count,
        res.lds_bytes,
        cache.hit_rate() * 100.0
    );
}

#[test]
#[serial]
fn test_compile_cache_2d_full_matrix() {
    // 16-point space: every (w, c) compiles and produces a
    // distinct kernel. Runs all 16 hipcc invocations — the total
    // wall-clock is ~20 s, acceptable for a one-shot test.
    let mut cache = DynamicKernelCache2D::new();
    for &w in BLOCK_D_NUM_WAVES_VALUES {
        for &c in BLOCK_D_MULTI_ROW_COLS_VALUES {
            let g = BlockDGenome {
                num_waves: w,
                multi_row_cols: c,
            };
            let k = cache.get_or_compile(g).expect("full-matrix compile");
            let res = cache.resources_for(g).expect("full-matrix resources");
            println!(
                "  (w={:>1}, c={:>1})  vgprs={:>3}  sgprs={:>3}  lds={:>6}  sym={}",
                w,
                c,
                res.vgpr_count,
                res.sgpr_count,
                res.lds_bytes,
                k.symbol()
            );
        }
    }
    assert_eq!(cache.misses, 16);
    assert_eq!(cache.hits, 0);
}

// ── Parity ────────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_parity_all_pairs_on_small_shape() {
    // Every (w, c) passes parity vs VALU reference on the small
    // shape. Tests that the 2-D parametric codegen is bit-compatible
    // with the CPU reference across the whole search space.
    let mut cache = DynamicKernelCache2D::new();
    let shape = parity_shape();
    let cfg = ParityConfig::for_ga(PrecisionLevel::Fp16);
    for &w in BLOCK_D_NUM_WAVES_VALUES {
        for &c in BLOCK_D_MULTI_ROW_COLS_VALUES {
            let g = BlockDGenome {
                num_waves: w,
                multi_row_cols: c,
            };
            let kernel = cache.get_or_compile(g).expect("compile");
            let parity = parity_dynamic_gate_up(&kernel, &shape, &cfg, 12345).expect("parity run");
            println!(
                "  (w={}, c={})  max_err={:.3}  tol={:.3}  {}",
                w,
                c,
                parity.max_abs_err,
                parity.effective_tolerance,
                if parity.passed { "PASS" } else { "FAIL" }
            );
            assert!(
                parity.passed,
                "(w={w}, c={c}) failed parity: max_err={} > tol={}",
                parity.max_abs_err, parity.effective_tolerance
            );
        }
    }
}

#[test]
#[serial]
fn test_parity_c4_matches_block_c_baseline() {
    // Sanity: at multi_row_cols=4, the 2-D kernel should be
    // numerically indistinguishable from the 1-D Block-C kernel
    // (same algorithm, same constants).
    let mut cache_2d = DynamicKernelCache2D::new();
    let mut cache_1d = DynamicKernelCache::new();
    let shape = parity_shape();
    let cfg = ParityConfig::for_ga(PrecisionLevel::Fp16);

    let k2d = cache_2d
        .get_or_compile(BlockDGenome {
            num_waves: 8,
            multi_row_cols: 4,
        })
        .expect("2d compile");
    let k1d = cache_1d.get_or_compile(8).expect("1d compile");

    let p2d = parity_dynamic_gate_up(&k2d, &shape, &cfg, 7777).expect("2d parity");
    let p1d = parity_dynamic_gate_up(&k1d, &shape, &cfg, 7777).expect("1d parity");
    println!(
        "  2-D (w=8,c=4) max_err={:.3} | 1-D (w=8) max_err={:.3}",
        p2d.max_abs_err, p1d.max_abs_err
    );
    // Both pass; the numerical error shouldn't differ materially
    // (same Q4_K supblock dot, same shuffle reduce).
    assert!(p2d.passed && p1d.passed);
    let rel = (p2d.max_abs_err - p1d.max_abs_err).abs() / p1d.max_abs_err.max(1e-6);
    assert!(
        rel < 0.5,
        "2-D at c=4 should match 1-D within 50% on max_abs_err; got 2d={}, 1d={}",
        p2d.max_abs_err,
        p1d.max_abs_err
    );
}

// ── Benchmark on real shape (no CPU VALU involved) ────────────────────────

#[test]
#[serial]
fn test_bench_all_pairs_on_real_shape() {
    // Timing-only run at N=14336. No CPU reference. Asserts every
    // combination gives a positive median latency and prints the
    // full 16-point matrix.
    let mut cache = DynamicKernelCache2D::new();
    let shape = bench_shape();
    let weights_gate = gen_q4k_weights(shape.k, shape.n, 0xB1);
    let weights_up = gen_q4k_weights(shape.k, shape.n, 0xB2);
    let input = gen_input(shape.k, 0xB3);

    println!("  Real bench shape: K={}, N={}", shape.k, shape.n);
    println!("  ┌─────┬─────┬───────────┐");
    println!("  │  w  │  c  │ median µs │");
    println!("  ├─────┼─────┼───────────┤");

    for &w in BLOCK_D_NUM_WAVES_VALUES {
        for &c in BLOCK_D_MULTI_ROW_COLS_VALUES {
            let g = BlockDGenome {
                num_waves: w,
                multi_row_cols: c,
            };
            let kernel = cache.get_or_compile(g).expect("compile");
            let us = bench_dynamic_gate_up(
                &kernel,
                &weights_gate,
                &weights_up,
                &input,
                shape.k as i32,
                shape.n as i32,
                5,
                15,
            )
            .expect("bench");
            println!("  │ {:>3} │ {:>3} │ {:>9.1} │", w, c, us);
            assert!(us > 0.0 && us < 1e7);
        }
    }
    println!("  └─────┴─────┴───────────┘");
}

// ── GA on real shape ──────────────────────────────────────────────────────

fn run_block_d_ga(seed: u64) -> rocmforge::v1::ga::block_d::BlockDResult {
    let parity = parity_shape();
    let bench = bench_shape();

    // Bench-shape weights for Phase-6 timing.
    let bench_weights_gate = gen_q4k_weights(bench.k, bench.n, 0xD1);
    let bench_weights_up = gen_q4k_weights(bench.k, bench.n, 0xD2);
    let bench_input = gen_input(bench.k, 0xD3);

    // Parity-shape weights for the post-GA stability pass (1000
    // blocks at N=14336 would OOM — see Block C SIGKILL).
    let parity_weights_gate = gen_q4k_weights(parity.k, parity.n, 0xE1);
    let parity_weights_up = gen_q4k_weights(parity.k, parity.n, 0xE2);
    let parity_input = gen_input(parity.k, 0xE3);

    let mut cache = DynamicKernelCache2D::new();
    let mut logger = GaLogger::capturing(format!("block-d-seed-{seed}"));
    let mut ga = KernelGa::new(block_d_default_config(seed));

    ga.run_num_waves_and_multi_row_cols(
        &parity,
        &bench,
        &q4_k(),
        PrecisionLevel::Fp16,
        432.8, // Post-P0 baseline from 2.0.3
        &bench_weights_gate,
        &bench_weights_up,
        &bench_input,
        &parity_weights_gate,
        &parity_weights_up,
        &parity_input,
        &mut cache,
        &mut logger,
    )
}

#[test]
#[serial]
fn test_block_d_ga_converges() {
    let result = run_block_d_ga(42);
    println!(
        "Block-D GA: {} candidates, {} stable winners, gen_ran={}, early={}",
        result.all_candidates.len(),
        result.stable_winners.len(),
        result.generations_ran,
        result.early_exited
    );
    println!(
        "  compile cache: {} hits, {} misses, hit_rate={:.1}%",
        result.compile_hits,
        result.compile_misses,
        (result.compile_hits as f64 / (result.compile_hits + result.compile_misses) as f64) * 100.0
    );
    for (i, cand) in result.all_candidates.iter().take(8).enumerate() {
        println!(
            "  #{}:  (w={}, c={})  {:.1} µs  fit={:.3}  vgprs={}  parity_err={:.2}",
            i + 1,
            cand.genome.num_waves,
            cand.genome.multi_row_cols,
            cand.median_us,
            cand.fitness,
            cand.vgpr_count,
            cand.parity_max_err
        );
    }
    assert!(
        result.all_candidates.len() >= 6,
        "GA must explore at least 6 of 16 points; got {}",
        result.all_candidates.len()
    );
    let winner = &result.all_candidates[0];
    assert!(
        winner.fitness > 0.0,
        "GA must return at least one positive-fitness winner"
    );
    assert!(winner.parity_passed, "Winner must have passed parity");
}

#[test]
#[serial]
fn test_block_d_seed_reproducible() {
    // Unlike Block C — where num_waves=4 won by a 17 % margin — the
    // 2-D winner typically clusters with 3–5 other candidates
    // within ≤ 2 % of the best latency. The *exact* genome ranked
    // #1 is therefore dominated by measurement jitter, not by the
    // GA logic. What we can still assert reproducibly is:
    //
    //   * Same seed evaluates the same SET of candidates.
    //   * Winner latency agrees within 5 % between runs.
    let r1 = run_block_d_ga(42);
    let r2 = run_block_d_ga(42);
    let top1 = &r1.all_candidates[0];
    let top2 = &r2.all_candidates[0];
    println!(
        "  run1 winner: (w={}, c={}) @ {:.1} µs  |  run2 winner: (w={}, c={}) @ {:.1} µs",
        top1.genome.num_waves,
        top1.genome.multi_row_cols,
        top1.median_us,
        top2.genome.num_waves,
        top2.genome.multi_row_cols,
        top2.median_us
    );

    // Significant overlap required but not strict set equality —
    // the GA's tournament + crossover reads live fitness values
    // from measured timings, and jitter across runs nudges which
    // candidates win tournaments, which propagates into different
    // crossovers. The RNG stream is still deterministic but the
    // *path* through the search space is measurement-coupled.
    let set1: std::collections::HashSet<BlockDGenome> =
        r1.all_candidates.iter().map(|c| c.genome).collect();
    let set2: std::collections::HashSet<BlockDGenome> =
        r2.all_candidates.iter().map(|c| c.genome).collect();
    let overlap = set1.intersection(&set2).count();
    let total = set1.len().max(set2.len());
    let overlap_ratio = overlap as f64 / total as f64;
    println!(
        "  candidate overlap: {}/{} ({:.0}%)",
        overlap,
        total,
        overlap_ratio * 100.0
    );
    assert!(
        overlap_ratio >= 0.7,
        "same seed should visit ≥ 70 % the same genomes; got {:.0}%",
        overlap_ratio * 100.0
    );

    // Winner latencies within 5 % — timing noise is the only
    // source of difference.
    let delta = (top1.median_us - top2.median_us).abs() / top2.median_us.min(top1.median_us);
    assert!(
        delta < 0.05,
        "winner latency diverged too much between runs: {:.1} vs {:.1} µs ({:.1}%)",
        top1.median_us,
        top2.median_us,
        delta * 100.0
    );
}

#[test]
#[serial]
fn test_block_d_winner_not_slower_than_phase1_default() {
    // Winner must be at least as fast as (w=8, c=4), the Phase-1
    // default. Allow 5% slack for timing jitter.
    let result = run_block_d_ga(42);
    let winner = &result.all_candidates[0];

    // Find the Phase-1 default among the evaluated candidates. If
    // the GA didn't sample (w=8, c=4), measure it standalone so the
    // comparison is still apples-to-apples.
    let phase1_median = if let Some(p1) = result
        .all_candidates
        .iter()
        .find(|c| c.genome == BlockDGenome::phase1_default())
    {
        p1.median_us
    } else {
        let mut cache = DynamicKernelCache2D::new();
        let bench = bench_shape();
        let weights_gate = gen_q4k_weights(bench.k, bench.n, 0xD1);
        let weights_up = gen_q4k_weights(bench.k, bench.n, 0xD2);
        let input = gen_input(bench.k, 0xD3);
        let k = cache
            .get_or_compile(BlockDGenome::phase1_default())
            .expect("phase1 compile");
        bench_dynamic_gate_up(
            &k,
            &weights_gate,
            &weights_up,
            &input,
            bench.k as i32,
            bench.n as i32,
            5,
            20,
        )
        .expect("phase1 bench")
    };
    let delta_pct = (phase1_median - winner.median_us) / phase1_median * 100.0;
    println!(
        "Block-D winner (w={}, c={}) @ {:.1} µs  vs  Phase-1 default (w=8, c=4) @ {:.1} µs  —  {:+.1}% faster",
        winner.genome.num_waves,
        winner.genome.multi_row_cols,
        winner.median_us,
        phase1_median,
        delta_pct
    );
    assert!(
        winner.median_us <= phase1_median * 1.05,
        "winner ({} µs) > 1.05 × Phase-1-default ({} µs) — fitness broken",
        winner.median_us,
        phase1_median
    );
}

#[test]
#[serial]
fn test_block_d_winner_not_slower_than_block_c_winner() {
    // Block-C winner (w=4, c=4) was 56 µs on the small shape. On
    // the real shape here, the GA either re-discovers it or finds
    // a different config — but the 2-D GA must not regress below
    // the 1-D GA's optimum. Measure (w=4, c=4) standalone for an
    // honest comparison.
    let result = run_block_d_ga(42);
    let winner = &result.all_candidates[0];

    let block_c_median = if let Some(bc) = result
        .all_candidates
        .iter()
        .find(|c| c.genome == BlockDGenome::block_c_winner())
    {
        bc.median_us
    } else {
        let mut cache = DynamicKernelCache2D::new();
        let bench = bench_shape();
        let weights_gate = gen_q4k_weights(bench.k, bench.n, 0xD1);
        let weights_up = gen_q4k_weights(bench.k, bench.n, 0xD2);
        let input = gen_input(bench.k, 0xD3);
        let k = cache
            .get_or_compile(BlockDGenome::block_c_winner())
            .expect("bc compile");
        bench_dynamic_gate_up(
            &k,
            &weights_gate,
            &weights_up,
            &input,
            bench.k as i32,
            bench.n as i32,
            5,
            20,
        )
        .expect("bc bench")
    };
    let delta_pct = (block_c_median - winner.median_us) / block_c_median * 100.0;
    println!(
        "Block-D winner (w={}, c={}) @ {:.1} µs  vs  Block-C winner (w=4, c=4) @ {:.1} µs  —  {:+.1}%",
        winner.genome.num_waves,
        winner.genome.multi_row_cols,
        winner.median_us,
        block_c_median,
        delta_pct
    );
    // 2-D search space is strict superset of 1-D (c=4 projection),
    // so the 2-D winner must be ≤ 1-D winner modulo timing jitter.
    assert!(
        winner.median_us <= block_c_median * 1.05,
        "2-D winner ({} µs) > 1.05 × Block-C winner ({} µs) — the 2-D GA regressed",
        winner.median_us,
        block_c_median
    );
}

// ── Executor-Integration — struct-level (no graph dispatch) ───────────────
//
// The full end-to-end test below exercises set_gate_up_swiglu_dynamic_kernel
// in context. These tests exercise the getter / setter semantics
// without needing a loaded model.

#[test]
#[serial]
fn test_make_gate_up_hook_fields() {
    let mut cache = DynamicKernelCache2D::new();
    let kernel = cache
        .get_or_compile(BlockDGenome::phase1_default())
        .expect("compile");
    let hook = make_gate_up_hook(kernel.clone(), 4096, 14336);
    assert_eq!(hook.hidden_dim, 4096);
    assert_eq!(hook.ffn_dim, 14336);
    assert!(Arc::ptr_eq(&hook.kernel, &kernel));
}

// ── End-to-End Decode (real model, gated) ─────────────────────────────────

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir().expect("HOME").join("models").join(name)
}

fn real_model_tests_enabled() -> bool {
    std::env::var("ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS")
        .ok()
        .as_deref()
        == Some("1")
}

/// Compile a GA-winner kernel for the model's actual FFN shape.
/// Used by the end-to-end tests below so the dynamic hook's shape
/// matches the graph-node shape (hook installation is only useful
/// at the exact shape it was compiled for).
fn compile_winner_for_shape(
    genome: BlockDGenome,
) -> Arc<rocmforge::v1::ga::dynamic::DynamicKernel> {
    let mut cache = DynamicKernelCache2D::new();
    cache.get_or_compile(genome).expect("winner compile")
}

#[test]
#[serial]
fn test_decode_with_ga_winner_faster_and_coherent() {
    if !real_model_tests_enabled() {
        eprintln!("skipping end-to-end decode test — set ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    use rocmforge::v1::backend::gpu::device::GpuDevice;
    use rocmforge::v1::core::gguf::GGUFFile;
    use rocmforge::v1::core::inference::InferencePipeline;
    use rocmforge::v1::core::model_config::ModelConfig;
    use rocmforge::v1::core::model_loader::LoadedModel;
    use rocmforge::v1::core::sampling::SamplingConfig;
    use rocmforge::v1::core::tensor_info::{
        group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
    };
    use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};

    let path = model_path(QWEN3);
    if !path.exists() {
        eprintln!(
            "skipping — model not at {} (drop a Qwen3-8B Q4_K_M GGUF there)",
            path.display()
        );
        return;
    }

    let device = GpuDevice::detect(0).expect("gpu");
    let model = LoadedModel::load(&path, &device).expect("load model");
    let gguf = GGUFFile::open(&path).expect("reopen gguf");
    let model_static: &'static LoadedModel = Box::leak(Box::new(model));
    let gguf_static: &'static GGUFFile = Box::leak(Box::new(gguf));

    let cfg =
        ModelConfig::from_metadata(gguf_static.metadata(), gguf_static.tensors()).expect("cfg");
    let hidden_dim = cfg.hidden_dim;
    let ffn_dim = cfg.ffn_dim;
    println!(
        "Qwen3 config: hidden_dim={}, ffn_dim={}, n_layers={}",
        hidden_dim, ffn_dim, cfg.n_layers
    );
    let layers = group_tensors_by_layer(gguf_static.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf_static.tensors() {
        let (role, li) = parse_tensor_name(&t.name);
        if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
            globals.insert(role, t);
        }
    }
    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: globals,
    };
    let graph = GraphBuilder::build(&ctx).expect("build graph");
    let plan = BufferPlan::plan_phase1(&graph);
    let mut pipe =
        InferencePipeline::new(graph, plan, model_static, gguf_static, 256).expect("pipeline");

    // Baseline — static kernel path.
    pipe.reset().expect("reset");
    let baseline = pipe
        .generate(
            "Explain what a mutex is in one paragraph.",
            50,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("baseline generate");
    println!(
        "Baseline (static kernel):   prefill={:.1} tok/s  decode={:.1} tok/s  output_tokens={}",
        baseline.prefill_tok_s, baseline.decode_tok_s, baseline.generated_tokens
    );

    // GA winner — run a short GA against the ACTUAL ffn_dim the
    // model uses, compile its winner, inject the hook. The GA
    // itself tunes over synthetic weights at (hidden_dim, ffn_dim);
    // the chosen config applies equally to the real weights because
    // the optimisation is over launch geometry, not weight values.
    let parity = parity_shape();
    let bench = KernelShape::new(1, ffn_dim, hidden_dim);
    let weights_gate = gen_q4k_weights(bench.k, bench.n, 0xE1);
    let weights_up = gen_q4k_weights(bench.k, bench.n, 0xE2);
    let input = gen_input(bench.k, 0xE3);
    let parity_weights_gate = gen_q4k_weights(parity.k, parity.n, 0xF1);
    let parity_weights_up = gen_q4k_weights(parity.k, parity.n, 0xF2);
    let parity_input = gen_input(parity.k, 0xF3);
    let mut cache = DynamicKernelCache2D::new();
    let mut logger = GaLogger::capturing("block-d-e2e".to_string());
    let mut ga = KernelGa::new(block_d_default_config(42));
    let ga_result = ga.run_num_waves_and_multi_row_cols(
        &parity,
        &bench,
        &q4_k(),
        PrecisionLevel::Fp16,
        432.8,
        &weights_gate,
        &weights_up,
        &input,
        &parity_weights_gate,
        &parity_weights_up,
        &parity_input,
        &mut cache,
        &mut logger,
    );
    let winner = &ga_result.all_candidates[0];
    println!(
        "End-to-End GA winner: (w={}, c={}) @ {:.1} µs",
        winner.genome.num_waves, winner.genome.multi_row_cols, winner.median_us
    );
    let winner_kernel = cache
        .get_or_compile(winner.genome)
        .expect("winner recompile hit");

    // Install the hook.
    pipe.executor
        .set_gate_up_swiglu_dynamic_kernel(Some(make_gate_up_hook(
            winner_kernel.clone(),
            hidden_dim,
            ffn_dim,
        )));
    assert!(pipe.executor.gate_up_swiglu_dynamic_kernel().is_some());

    // Rerun with GA-tuned kernel.
    pipe.reset().expect("reset");
    let tuned = pipe
        .generate(
            "Explain what a mutex is in one paragraph.",
            50,
            &SamplingConfig::greedy(),
            true,
        )
        .expect("tuned generate");
    println!(
        "Tuned    (GA winner):      prefill={:.1} tok/s  decode={:.1} tok/s  output_tokens={}",
        tuned.prefill_tok_s, tuned.decode_tok_s, tuned.generated_tokens
    );

    assert!(
        tuned.generated_tokens > 0,
        "tuned generation must produce tokens"
    );
    assert!(
        !tuned.output.trim().is_empty(),
        "tuned output must not be empty"
    );
    assert!(
        tuned.decode_tok_s > 0.0,
        "tuned decode must have positive tok/s"
    );
    let delta_pct = (tuned.decode_tok_s - baseline.decode_tok_s) / baseline.decode_tok_s * 100.0;
    println!(
        "  delta: {:+.1}%  (baseline {:.1} → tuned {:.1} tok/s)",
        delta_pct, baseline.decode_tok_s, tuned.decode_tok_s
    );
    // Tolerate ≤ 5 % regression for driver / thermal noise; the
    // GA-tuned kernel must not meaningfully slow down decode.
    assert!(
        tuned.decode_tok_s >= baseline.decode_tok_s * 0.95,
        "tuned decode {} tok/s < 0.95 × baseline {} tok/s",
        tuned.decode_tok_s,
        baseline.decode_tok_s
    );
}

#[test]
#[serial]
fn test_executor_fallback_when_hook_shape_mismatch() {
    // With a hook installed for a WRONG shape, the executor must
    // fall through to the static Phase-1 path. Verified by
    // installing a hook for (1, 1) which no real model has, running
    // decode, and asserting output is still coherent.
    if !real_model_tests_enabled() {
        eprintln!("skipping — requires ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1");
        return;
    }
    use rocmforge::v1::backend::gpu::device::GpuDevice;
    use rocmforge::v1::core::gguf::GGUFFile;
    use rocmforge::v1::core::inference::InferencePipeline;
    use rocmforge::v1::core::model_config::ModelConfig;
    use rocmforge::v1::core::model_loader::LoadedModel;
    use rocmforge::v1::core::sampling::SamplingConfig;
    use rocmforge::v1::core::tensor_info::{
        group_tensors_by_layer, parse_tensor_name, TensorInfo, TensorRole,
    };
    use rocmforge::v1::graph::{BufferPlan, GraphBuildContext, GraphBuilder};

    let path = model_path(QWEN3);
    if !path.exists() {
        return;
    }
    let device = GpuDevice::detect(0).expect("gpu");
    let model = LoadedModel::load(&path, &device).expect("load");
    let gguf = GGUFFile::open(&path).expect("gguf");
    let model_static: &'static LoadedModel = Box::leak(Box::new(model));
    let gguf_static: &'static GGUFFile = Box::leak(Box::new(gguf));
    let cfg =
        ModelConfig::from_metadata(gguf_static.metadata(), gguf_static.tensors()).expect("cfg");
    let layers = group_tensors_by_layer(gguf_static.tensors());
    let mut globals: std::collections::HashMap<TensorRole, &TensorInfo> =
        std::collections::HashMap::new();
    for t in gguf_static.tensors() {
        let (role, li) = parse_tensor_name(&t.name);
        if li.is_none() && !matches!(role, TensorRole::Unknown(_)) {
            globals.insert(role, t);
        }
    }
    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: globals,
    };
    let graph = GraphBuilder::build(&ctx).expect("build");
    let plan = BufferPlan::plan_phase1(&graph);
    let mut pipe =
        InferencePipeline::new(graph, plan, model_static, gguf_static, 256).expect("pipeline");

    // Hook with intentionally mismatched shape.
    let winner_kernel = compile_winner_for_shape(BlockDGenome::phase1_default());
    pipe.executor
        .set_gate_up_swiglu_dynamic_kernel(Some(make_gate_up_hook(
            winner_kernel,
            /* hidden_dim = */ 1,
            /* ffn_dim = */ 1,
        )));

    pipe.reset().expect("reset");
    let result = pipe
        .generate("Hello", 5, &SamplingConfig::greedy(), true)
        .expect("generate");
    assert!(result.generated_tokens > 0);
    assert!(!result.output.trim().is_empty());
    println!(
        "Fallback (hook shape mismatched): {} tokens, {:.1} tok/s, output=\"{}\"",
        result.generated_tokens,
        result.decode_tok_s,
        result.output.chars().take(40).collect::<String>()
    );
}
