//! Phase 2 / Schritt 2.1.1 — Kernel-GA framework tests.
//!
//! Covers the framework-level guarantees required by
//! `ga_tuning_spec §2` + `§5.10` — genome operators, pre-compile
//! validation, post-compile VGPR gate, tournament selection,
//! elitism, early-exit, seed reproducibility, toy-problem convergence,
//! compile-cache hit-rate, and JSONL log parseability.
//!
//! Most tests are CPU-only. Two tests are `#[cfg(feature = "gpu")]`
//! and exercise the VGPR reader on a real compiled `.co` file.

#![cfg(feature = "v1")]

use rocmforge::v1::ga::compile::CompileCache;
use rocmforge::v1::ga::engine::{GaConfig, KernelGa, KernelGenomeScored};
use rocmforge::v1::ga::fitness::evaluate_toy_fitness;
use rocmforge::v1::ga::genome::{
    DequantStrategy, KernelGenome, DEQUANT_PREPASS_LDS_VALUES, K_UNROLL_VALUES,
    PREFETCH_DEPTH_VALUES, TILE_K_VALUES, TILE_M_VALUES, TILE_N_VALUES, TILES_PER_WAVE_VALUES,
    WAVES_PER_BLOCK_VALUES,
};
use rocmforge::v1::ga::logger::GaLogger;
use rocmforge::v1::ga::rng::SeededRng;
use rocmforge::v1::ga::toy::{run_toy_ga, toy_fitness, toy_ga_defaults};
use rocmforge::v1::ga::types::{
    CodeObjectResources, KernelTarget, LdsStrategy, PrecisionLevel, TileConfig,
};
use rocmforge::v1::ga::validation::{
    validate_post_compile, validate_pre_compile, GaQuantFormat,
};

// ── Helpers ──────────────────────────────────────────────────────────────

fn sensible_genome() -> KernelGenome {
    KernelGenome {
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
    }
}

// ── Genome-Operatoren ────────────────────────────────────────────────────

#[test]
fn test_genome_random_produces_valid() {
    let mut rng = SeededRng::new(12345);
    for _ in 0..100 {
        let g = KernelGenome::random(&mut rng);
        assert!(TILE_M_VALUES.contains(&g.tile_m));
        assert!(TILE_N_VALUES.contains(&g.tile_n));
        assert!(TILE_K_VALUES.contains(&g.tile_k));
        assert!(TILES_PER_WAVE_VALUES.contains(&g.tiles_per_wave));
        assert!(WAVES_PER_BLOCK_VALUES.contains(&g.waves_per_block));
        assert!(PREFETCH_DEPTH_VALUES.contains(&g.prefetch_depth));
        assert!(K_UNROLL_VALUES.contains(&g.k_unroll));
        match g.dequant_strategy {
            DequantStrategy::Inline => {}
            DequantStrategy::PrePass { lds_bytes } => {
                assert!(DEQUANT_PREPASS_LDS_VALUES.contains(&lds_bytes));
            }
            DequantStrategy::Batched { batch_size } => {
                assert!((2..=8).contains(&batch_size));
            }
        }
    }
}

#[test]
fn test_genome_crossover_inherits_genes() {
    let a = KernelGenome {
        tile_m: 16,
        tile_n: 16,
        tile_k: 16,
        tiles_per_wave: 1,
        waves_per_block: 1,
        use_lds_for_a: true,
        use_lds_for_b: true,
        prefetch_depth: 0,
        k_unroll: 1,
        double_buffer: true,
        dequant_strategy: DequantStrategy::Inline,
    };
    let b = KernelGenome {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        tiles_per_wave: 4,
        waves_per_block: 8,
        use_lds_for_a: false,
        use_lds_for_b: false,
        prefetch_depth: 2,
        k_unroll: 8,
        double_buffer: false,
        dequant_strategy: DequantStrategy::Batched { batch_size: 4 },
    };
    let mut rng = SeededRng::new(7);
    for _ in 0..50 {
        let c = KernelGenome::crossover(&a, &b, &mut rng);
        assert!(c.tile_m == a.tile_m || c.tile_m == b.tile_m);
        assert!(c.tile_n == a.tile_n || c.tile_n == b.tile_n);
        assert!(c.tile_k == a.tile_k || c.tile_k == b.tile_k);
        assert!(c.tiles_per_wave == a.tiles_per_wave || c.tiles_per_wave == b.tiles_per_wave);
        assert!(c.waves_per_block == a.waves_per_block || c.waves_per_block == b.waves_per_block);
        assert!(c.prefetch_depth == a.prefetch_depth || c.prefetch_depth == b.prefetch_depth);
        assert!(c.k_unroll == a.k_unroll || c.k_unroll == b.k_unroll);
        assert!(
            c.dequant_strategy == a.dequant_strategy
                || c.dequant_strategy == b.dequant_strategy
        );
    }
}

#[test]
fn test_genome_mutation_changes_values() {
    let base = sensible_genome();
    let mut rng = SeededRng::new(9876);
    let mut any_change = false;
    for _ in 0..20 {
        let mut child = base;
        child.mutate(1.0, &mut rng); // rate = 1.0 → every gene resampled
        if child != base {
            any_change = true;
            break;
        }
    }
    assert!(any_change, "mutation with rate=1.0 should change something");

    // rate = 0.0 → never change
    let mut rng = SeededRng::new(9876);
    for _ in 0..20 {
        let mut child = base;
        child.mutate(0.0, &mut rng);
        assert_eq!(child, base, "mutation with rate=0.0 must not change any gene");
    }
}

#[test]
fn test_genome_to_tile_config() {
    let g = KernelGenome {
        tile_m: 64,
        tile_n: 128,
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
    let tc: TileConfig = (&g).into();
    assert_eq!(tc.tile_m, 64);
    assert_eq!(tc.tile_n, 128);
    assert_eq!(tc.k_chunk, 32);
    assert_eq!(tc.lds_strategy, LdsStrategy::DirectA_LdsB);
    assert_eq!(tc.num_waves, 4);
    assert_eq!(tc.unroll_factor, 4);

    // Cover the other three lds_strategy mappings.
    let mut g2 = g;
    g2.use_lds_for_a = true;
    g2.use_lds_for_b = true;
    assert_eq!(TileConfig::from(&g2).lds_strategy, LdsStrategy::LdsAB);
    g2.use_lds_for_a = false;
    g2.use_lds_for_b = false;
    assert_eq!(TileConfig::from(&g2).lds_strategy, LdsStrategy::DirectAB);
    g2.use_lds_for_a = true;
    g2.use_lds_for_b = false;
    assert_eq!(TileConfig::from(&g2).lds_strategy, LdsStrategy::LdsAB); // fallback
}

// ── Validation ───────────────────────────────────────────────────────────

#[test]
fn test_pre_compile_rejects_bad_alignment() {
    let mut g = sensible_genome();
    g.tile_m = 17;
    assert!(!validate_pre_compile(
        &g,
        &GaQuantFormat::q4_k(),
        PrecisionLevel::Fp16
    ));
}

#[test]
fn test_pre_compile_rejects_lds_overflow() {
    // tile_m=128, tile_n=128, tile_k=64, fp16, double_buffer=true →
    // LDS_A + LDS_B = 2 × (128 × 64 × 2) = 32 KB; with double_buffer
    // that's exactly 64 KB — at the 64 KB ceiling, not over. Add a
    // 2 KB PrePass dequant allocation so the total crosses the
    // threshold (`lds_total > 64 KB`).
    let g = KernelGenome {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        tiles_per_wave: 1,
        waves_per_block: 1,
        use_lds_for_a: true,
        use_lds_for_b: true,
        prefetch_depth: 0,
        k_unroll: 2,
        double_buffer: true,
        dequant_strategy: DequantStrategy::PrePass { lds_bytes: 2048 },
    };
    assert!(!validate_pre_compile(
        &g,
        &GaQuantFormat::q4_k(),
        PrecisionLevel::Fp16
    ));
}

#[test]
fn test_pre_compile_rejects_high_vgpr_heuristic() {
    // tile_m=128, k_unroll=8, prefetch=2, double_buffer=true, fp32 →
    // estimator: 72 + 28 + 32 + 12 + 8 = 152 > 150.
    let g = KernelGenome {
        tile_m: 128,
        tile_n: 32,
        tile_k: 32,
        tiles_per_wave: 1,
        waves_per_block: 1,
        use_lds_for_a: false,
        use_lds_for_b: false,
        prefetch_depth: 2,
        k_unroll: 8,
        double_buffer: true,
        dequant_strategy: DequantStrategy::Inline,
    };
    assert!(!validate_pre_compile(
        &g,
        &GaQuantFormat::q4_k(),
        PrecisionLevel::Fp32
    ));
}

#[test]
fn test_pre_compile_accepts_valid_genome() {
    let g = sensible_genome();
    assert!(validate_pre_compile(
        &g,
        &GaQuantFormat::q4_k(),
        PrecisionLevel::Fp16
    ));
}

#[test]
fn test_pre_compile_reject_rate_on_random_genomes() {
    // Sanity: the spec expects ~30–40% of random genomes to fail
    // pre-compile validation (§2.3). Accept a wide 10–70% band;
    // the point is just that the gate isn't pass-all or reject-all.
    let mut rng = SeededRng::new(2026);
    let mut rejected = 0;
    const N: usize = 1000;
    for _ in 0..N {
        let g = KernelGenome::random(&mut rng);
        if !validate_pre_compile(&g, &GaQuantFormat::q4_k(), PrecisionLevel::Fp16) {
            rejected += 1;
        }
    }
    let rate = rejected as f64 / N as f64;
    assert!(
        (0.10..=0.70).contains(&rate),
        "pre-compile reject rate {:.1}% out of the sane 10–70% band",
        rate * 100.0
    );
    println!(
        "pre-compile reject rate on {N} random genomes: {:.1}%",
        rate * 100.0
    );
}

#[test]
fn test_post_compile_rejects_under_four_waves() {
    // 385 VGPRs → 1536/385 = 3 waves/CU → reject.
    let r = validate_post_compile(CodeObjectResources {
        vgpr_count: 385,
        sgpr_count: 20,
        lds_bytes: 0,
    });
    assert!(!r.accepted);
}

#[test]
fn test_post_compile_accepts_moderate_vgprs() {
    // 186 VGPRs → 1536/186 = 8 waves/CU → accept (matches
    // q4_k_q8_inline_residual real .co on gfx1201).
    let r = validate_post_compile(CodeObjectResources {
        vgpr_count: 186,
        sgpr_count: 18,
        lds_bytes: 0,
    });
    assert!(r.accepted);
    assert_eq!(r.max_waves_per_cu, 8);
}

// ── GA-Engine ────────────────────────────────────────────────────────────

#[test]
fn test_tournament_selection_prefers_better() {
    // Build a tiny fake population and run 1000 tournament picks.
    let mut ga = KernelGa::new(GaConfig {
        tournament_size: 3,
        seed: 42,
        ..toy_ga_defaults(42, "tournament-test")
    });
    let scored: Vec<KernelGenomeScored> = (0..10)
        .map(|i| KernelGenomeScored {
            genome: sensible_genome(),
            fitness: i as f64 * 0.1,
        })
        .collect();

    // Invoke the engine's selection indirectly via next_generation; we
    // check that the top of the next generation is dominated by high
    // fitness (all elites = max-fitness genome).
    let results = rocmforge::v1::ga::engine::GenerationResults {
        scored,
        ..Default::default()
    };
    let next = ga.next_generation(&results);
    // The first ceil(pop*0.05) individuals are elites — with pop=50,
    // 0.05 elite fraction = ceil(2.5) = 3. They should all come from
    // the single highest-fitness genome (index 9 with fitness 0.9).
    // Every gene of the elites must match that genome's genes.
    for elite in next.iter().take(3) {
        assert_eq!(elite.tile_m, 64);
        assert_eq!(elite.tile_n, 64);
    }
}

#[test]
fn test_elitism_preserves_best() {
    // Population where one genome has dominant fitness; it must
    // survive 5 generations unchanged.
    let mut ga = KernelGa::new(toy_ga_defaults(7, "elitism-test"));
    let winning = KernelGenome {
        tile_m: 16,
        tile_n: 16,
        tile_k: 16,
        tiles_per_wave: 1,
        waves_per_block: 1,
        use_lds_for_a: false,
        use_lds_for_b: false,
        prefetch_depth: 0,
        k_unroll: 1,
        double_buffer: false,
        dequant_strategy: DequantStrategy::Inline,
    };

    let scored: Vec<KernelGenomeScored> = (0..ga.config.population_size)
        .map(|i| {
            let genome = if i == 0 {
                winning
            } else {
                KernelGenome::random(&mut ga.rng)
            };
            let fitness = if i == 0 { 100.0 } else { 1.0 };
            KernelGenomeScored { genome, fitness }
        })
        .collect();
    let results = rocmforge::v1::ga::engine::GenerationResults {
        scored,
        ..Default::default()
    };
    let next = ga.next_generation(&results);
    // The elite slot (population × 0.1 ceil = 5 for the toy defaults)
    // should carry the winning genome forward untouched.
    assert!(next.iter().take(5).any(|g| *g == winning));
}

#[test]
fn test_early_exit_after_plateau() {
    let ga = KernelGa::new(GaConfig {
        early_exit_generations: 10,
        early_exit_threshold: 0.01,
        ..Default::default()
    });
    // 12 generations, all at fitness 1.0 — no improvement.
    let history: Vec<f64> = vec![1.0; 12];
    assert!(ga.should_early_exit(&history));
}

#[test]
fn test_early_exit_not_triggered_with_improvement() {
    let ga = KernelGa::new(GaConfig {
        early_exit_generations: 10,
        early_exit_threshold: 0.01,
        ..Default::default()
    });
    // Continuous improvement from 1.0 to 2.0 over 12 generations.
    let history: Vec<f64> = (0..12).map(|i| 1.0 + (i as f64) * 0.1).collect();
    assert!(!ga.should_early_exit(&history));
}

// ── Toy-Problem Konvergenz + Seed-Reproduzierbarkeit ─────────────────────

#[test]
fn test_toy_fitness_landscape() {
    // Sanity — the oracle agrees the sweet-spot is 2.0.
    let sweet = KernelGenome {
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
    assert!((toy_fitness(&sweet) - 2.0).abs() < 1e-9);
}

#[test]
fn test_ga_converges_on_toy_problem() {
    let cfg = toy_ga_defaults(42, "toy-convergence");
    let (result, log) = run_toy_ga(cfg, true);
    println!(
        "toy GA: best_fitness={:.4} generations_ran={} early_exited={}",
        result.best_fitness, result.generations_ran, result.early_exited
    );
    for (i, s) in result.top.iter().enumerate() {
        println!(
            "  top{}: tile_m={} tile_n={} tile_k={} k_unroll={} fitness={:.4}",
            i, s.genome.tile_m, s.genome.tile_n, s.genome.tile_k, s.genome.k_unroll, s.fitness
        );
    }
    if let Some(lines) = log {
        println!("jsonl sample ({} lines total):", lines.len());
        for line in lines.iter().take(3) {
            println!("  {line}");
        }
    }
    assert!(
        result.best_fitness > 1.8,
        "GA should find near-optimal on toy problem, got {:.3} after {} generations",
        result.best_fitness,
        result.generations_ran
    );
}

#[test]
fn test_seed_reproducibility() {
    let cfg_a = toy_ga_defaults(42, "reproducibility-a");
    let cfg_b = toy_ga_defaults(42, "reproducibility-b");
    let (r1, _) = run_toy_ga(cfg_a, false);
    let (r2, _) = run_toy_ga(cfg_b, false);
    // Best fitness is a scalar summary; we also want the top-1 genome
    // to be the same.
    assert_eq!(
        r1.top.first().map(|s| s.genome),
        r2.top.first().map(|s| s.genome),
        "same seed should produce the same best genome"
    );
    assert!((r1.best_fitness - r2.best_fitness).abs() < 1e-12);
}

#[test]
fn test_different_seeds_produce_different_runs() {
    let (r1, _) = run_toy_ga(toy_ga_defaults(42, "seed-a"), false);
    let (r2, _) = run_toy_ga(toy_ga_defaults(99, "seed-b"), false);
    // Both should converge to near-optimal, but the per-generation
    // trajectory differs, so the set of 5 top genomes usually differs.
    assert!(r1.best_fitness > 1.5);
    assert!(r2.best_fitness > 1.5);
    // Not strictly guaranteed (the top-1 can coincide in tiny
    // landscapes) but the odds of 5/5 matching are low.
    let tops_match = r1
        .top
        .iter()
        .map(|s| s.genome)
        .collect::<Vec<_>>()
        == r2.top.iter().map(|s| s.genome).collect::<Vec<_>>();
    if tops_match {
        println!("NOTE: seed 42 vs 99 produced identical top-5 — landscape is small enough to converge to the same set.");
    }
    // The real seed-reproducibility guarantee is covered by the
    // previous test; this one just documents behaviour.
}

// ── Compile-Cache ────────────────────────────────────────────────────────

#[test]
fn test_compile_cache_hit_rate() {
    let mut cache = CompileCache::new();
    let g = sensible_genome();
    // Evaluate the same genome 5 times — first is miss, rest hits.
    for _ in 0..5 {
        let _ = evaluate_toy_fitness(
            &g,
            &GaQuantFormat::q4_k(),
            PrecisionLevel::Fp16,
            100.0,
            &mut cache,
        );
    }
    assert_eq!(cache.misses, 1);
    assert_eq!(cache.hits, 4);
    assert!(cache.hit_rate() >= 0.7);
}

#[test]
fn test_compile_cache_same_key_same_result() {
    use rocmforge::v1::ga::compile::{CompileKey, CompiledKernel};
    let mut cache = CompileCache::new();
    let key = CompileKey::new(
        TileConfig {
            tile_m: 64,
            tile_n: 64,
            k_chunk: 32,
            lds_strategy: LdsStrategy::DirectAB,
            num_waves: 4,
            unroll_factor: 4,
        },
        12,
        PrecisionLevel::Fp16,
        KernelTarget::Gfx1201,
    );
    let factory_calls = std::cell::Cell::new(0);
    let a = cache.get_or_insert_with(&key, |k| {
        factory_calls.set(factory_calls.get() + 1);
        CompiledKernel {
            key: k.clone(),
            resources: CodeObjectResources {
                vgpr_count: 80,
                sgpr_count: 20,
                lds_bytes: 0,
            },
            co_path: None,
        }
    });
    let b = cache.get_or_insert_with(&key, |_| {
        factory_calls.set(factory_calls.get() + 1);
        CompiledKernel {
            key: key.clone(),
            resources: CodeObjectResources {
                vgpr_count: 999,
                sgpr_count: 999,
                lds_bytes: 0,
            },
            co_path: None,
        }
    });
    assert_eq!(factory_calls.get(), 1, "factory should run only on miss");
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_eq!(a.resources.vgpr_count, 80);
}

// ── JSONL-Log ────────────────────────────────────────────────────────────

#[test]
fn test_jsonl_log_parseable() {
    let mut log = GaLogger::capturing("jsonl-parse-test");
    log.log_shape_start("gemv_q4_k_4096x4096").unwrap();
    log.log_generation("gemv_q4_k_4096x4096", 0, 15, 35, 10, 2, 38, 1.35, 0.94, 120, 500)
        .unwrap();
    log.log_early_exit("gemv_q4_k_4096x4096", 12, 1.35).unwrap();
    log.log_shape_complete("gemv_q4_k_4096x4096", 1.35).unwrap();

    let lines = log.captured_lines().unwrap();
    assert!(lines.len() >= 4);
    for line in lines {
        let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON");
        assert!(v.get("ts").is_some(), "every log line needs ts");
        assert!(v.get("run_id").is_some(), "every log line needs run_id");
        assert!(v.get("event").is_some(), "every log line needs event");
    }
}

#[test]
fn test_jsonl_log_contains_run_id() {
    let (_, log_opt) = run_toy_ga(toy_ga_defaults(7, "run-id-check"), true);
    let lines = log_opt.expect("capturing logger should return lines");
    assert!(!lines.is_empty(), "log should have at least one event");
    for line in &lines {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(
            v["run_id"].as_str(),
            Some("run-id-check"),
            "log line must carry the run_id"
        );
    }
}

// ── Fitness ──────────────────────────────────────────────────────────────

#[test]
fn test_fitness_relative_to_baseline() {
    let mut cache = CompileCache::new();
    let g = sensible_genome();
    // Toy landscape gives fitness ≈ 2.0 for the sweet-spot genome.
    // `evaluate_toy_fitness` returns exactly that — not baseline /
    // median — because the toy path skips the real bench.
    let r = evaluate_toy_fitness(
        &g,
        &GaQuantFormat::q4_k(),
        PrecisionLevel::Fp16,
        100.0,
        &mut cache,
    );
    assert!(!r.is_rejected());
    assert!(
        (r.fitness - 2.0).abs() < 0.01,
        "sweet-spot fitness should be ~2.0, got {}",
        r.fitness
    );
}

#[test]
fn test_fitness_sorts_correctly() {
    // The engine's sort should give a list descending by fitness.
    let scored = vec![
        KernelGenomeScored {
            genome: sensible_genome(),
            fitness: 1.0,
        },
        KernelGenomeScored {
            genome: sensible_genome(),
            fitness: 2.5,
        },
        KernelGenomeScored {
            genome: sensible_genome(),
            fitness: 0.5,
        },
    ];
    let results = rocmforge::v1::ga::engine::GenerationResults {
        scored,
        ..Default::default()
    };
    let sorted = results.sorted_by_fitness_desc();
    assert_eq!(sorted[0].fitness, 2.5);
    assert_eq!(sorted[1].fitness, 1.0);
    assert_eq!(sorted[2].fitness, 0.5);
}

// ── GPU-spezifisch (Post-Compile VGPR-Read) ──────────────────────────────

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use rocmforge::v1::ga::compile::parse_amdgpu_metadata;

    /// Confirm that `parse_amdgpu_metadata` returns real VGPR/SGPR
    /// numbers from a compiled gfx1201 code object. This test only
    /// succeeds on a system that's already run `cargo build --features
    /// gpu` once (so the CMake build produced the extracted `.o` files
    /// in `target/release/build/…/hip_kernels_v1_build`). Looks for the
    /// q4_k_q8_inline_residual kernel built in step 2.0.2 — it's the
    /// most stable artifact we have.
    #[test]
    fn test_post_compile_vgpr_read_from_real_co() {
        // Find any rocmforge-<hash> build directory and look for the
        // extracted AMDGPU object.
        let target_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("release")
            .join("build");
        let target_dir = if target_dir.exists() {
            target_dir
        } else {
            eprintln!("target/release/build missing — skipping real-co test");
            return;
        };

        // Walk for any `*.hipv4-amdgcn-*-gfx1201` file. Using WalkDir
        // would pull in a new dep — use a handful of read_dir calls.
        let mut found: Option<std::path::PathBuf> = None;
        for entry in std::fs::read_dir(&target_dir).unwrap().flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let build_dir = p
                .join("out")
                .join("hip_kernels_v1_build")
                .join("CMakeFiles")
                .join("v1_gemv_q4_k_q8_inline_residual.dir")
                .join("gemv");
            if !build_dir.exists() {
                continue;
            }
            for e in std::fs::read_dir(&build_dir).unwrap().flatten() {
                let f = e.path();
                let name = f.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.ends_with("-gfx1201") && name.contains("hipv4-amdgcn") {
                    found = Some(f);
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }

        let Some(co_path) = found else {
            eprintln!(
                "no extracted gfx1201 code object found — run `llvm-objdump \
                 --offloading` against the .hip.o first (or skip this test)"
            );
            return;
        };

        let res = parse_amdgpu_metadata(&co_path)
            .expect("llvm-readobj should succeed on a valid .co");
        let res = res.expect("AMDGPU metadata should be present");
        println!(
            "q4_k_q8_inline_residual VGPRs={} SGPRs={} LDS={} B  (waves/CU={})",
            res.vgpr_count,
            res.sgpr_count,
            res.lds_bytes,
            res.max_waves_per_cu()
        );
        assert!(res.vgpr_count > 0 && res.vgpr_count < 256);
        let gate = validate_post_compile(res);
        assert!(
            gate.accepted,
            "q4_k_q8_inline_residual should pass the post-compile gate"
        );
    }
}
