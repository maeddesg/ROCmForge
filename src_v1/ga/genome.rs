//! `KernelGenome` + GA operators (1:1 from `architecture_v1.2.0-draft §4.2`
//! and `ga_tuning_spec §2.2`).
//!
//! The genome is the GA search-space representation; `TileConfig`
//! (conversion via `impl From<&KernelGenome>`) is the codegen-facing
//! subset.

use super::rng::SeededRng;
use super::types::{
    LdsStrategy, PrecisionLevel, TileConfig, LDS_BUDGET_BYTES, MAX_TILES_PER_BLOCK,
    PRE_COMPILE_VGPR_LIMIT,
};
use super::validation::{estimate_vgprs, GaQuantFormat};

/// Quant-format-specific dequant packaging. `Inline` is the default —
/// dequant one element at a time inside the K-loop. `PrePass` uses
/// shared LDS to stage a whole block's elements first (pays back for
/// large `tile_k`). `Batched` packs several blocks into one dequant
/// call (amortises the block prolog across blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DequantStrategy {
    Inline,
    PrePass { lds_bytes: u16 },
    Batched { batch_size: u8 },
}

impl DequantStrategy {
    /// Extra LDS bytes this strategy adds on top of A/B tile LDS.
    pub fn lds_overhead(&self) -> u32 {
        match *self {
            Self::Inline => 0,
            Self::PrePass { lds_bytes } => lds_bytes as u32,
            Self::Batched { .. } => 0,
        }
    }
}

/// Valid values per genome slot (`architecture §4.2`).
pub const TILE_M_VALUES: &[u8] = &[16, 32, 64, 128];
pub const TILE_N_VALUES: &[u8] = &[16, 32, 64, 128];
pub const TILE_K_VALUES: &[u8] = &[16, 32, 64];
pub const TILES_PER_WAVE_VALUES: &[u8] = &[1, 2, 4];
pub const WAVES_PER_BLOCK_VALUES: &[u8] = &[1, 2, 4, 8];
pub const PREFETCH_DEPTH_VALUES: &[u8] = &[0, 1, 2];
pub const K_UNROLL_VALUES: &[u8] = &[1, 2, 4, 8];
pub const DEQUANT_PREPASS_LDS_VALUES: &[u16] = &[1024, 2048, 4096, 8192];
pub const DEQUANT_BATCH_SIZES: &[u8] = &[2, 4, 8];

/// Kernel configuration as a GA individual. Field set mirrors
/// `architecture_v1.2.0-draft §4.2` exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelGenome {
    // Tile structure
    pub tile_m: u8,
    pub tile_n: u8,
    pub tile_k: u8,
    pub tiles_per_wave: u8,
    pub waves_per_block: u8,

    // Memory strategy
    pub use_lds_for_a: bool,
    pub use_lds_for_b: bool,
    pub prefetch_depth: u8,

    // K-loop
    pub k_unroll: u8,
    pub double_buffer: bool,

    // Quant-specific
    pub dequant_strategy: DequantStrategy,
}

impl KernelGenome {
    /// Random genome — every slot drawn from the legal value set, then
    /// passed through [`Self::sanitize`] so cross-field constraints
    /// (LDS budget, VGPR heuristic, sub-block alignment, workgroup-tile
    /// product) are resolved. The output always passes
    /// `validate_pre_compile` at the default sanitize target
    /// (`Q4_K` + `Fp16`).
    pub fn random(rng: &mut SeededRng) -> Self {
        let mut g = Self::random_unsanitized(rng);
        g.sanitize();
        g
    }

    /// Raw random genome without the post-pass repair. Kept public so
    /// the pre-compile-gate reject-rate test can measure how many
    /// cross-field constraints the unsanitized draw violates — one
    /// calibration point for the heuristic.
    pub fn random_unsanitized(rng: &mut SeededRng) -> Self {
        let dequant_strategy = match rng.gen_index(3) {
            0 => DequantStrategy::Inline,
            1 => DequantStrategy::PrePass {
                lds_bytes: *rng.choose(DEQUANT_PREPASS_LDS_VALUES),
            },
            _ => DequantStrategy::Batched {
                batch_size: rng.choose_u8(DEQUANT_BATCH_SIZES),
            },
        };

        Self {
            tile_m: rng.choose_u8(TILE_M_VALUES),
            tile_n: rng.choose_u8(TILE_N_VALUES),
            tile_k: rng.choose_u8(TILE_K_VALUES),
            tiles_per_wave: rng.choose_u8(TILES_PER_WAVE_VALUES),
            waves_per_block: rng.choose_u8(WAVES_PER_BLOCK_VALUES),
            use_lds_for_a: rng.gen_bool(),
            use_lds_for_b: rng.gen_bool(),
            prefetch_depth: rng.choose_u8(PREFETCH_DEPTH_VALUES),
            k_unroll: rng.choose_u8(K_UNROLL_VALUES),
            double_buffer: rng.gen_bool(),
            dequant_strategy,
        }
    }

    /// Uniform crossover: each gene independently picked from parent
    /// `a` or `b` with 50/50 probability (`ga_tuning_spec §2.5`). The
    /// child is sanitised before return so any cross-field constraint
    /// that a mixed inheritance happens to violate (LDS overflow,
    /// VGPR overflow, sub-block misalignment, workgroup-tile product)
    /// is repaired — individual-gene inheritance is preserved where
    /// it's valid, only the minimum set of fields that would
    /// otherwise fail pre-compile validation gets adjusted.
    pub fn crossover(a: &Self, b: &Self, rng: &mut SeededRng) -> Self {
        let pick_u8 = |rng: &mut SeededRng, x: u8, y: u8| if rng.gen_bool() { x } else { y };
        let pick_bool = |rng: &mut SeededRng, x: bool, y: bool| if rng.gen_bool() { x } else { y };
        let pick_dq =
            |rng: &mut SeededRng, x: DequantStrategy, y: DequantStrategy| {
                if rng.gen_bool() { x } else { y }
            };

        let mut child = Self {
            tile_m: pick_u8(rng, a.tile_m, b.tile_m),
            tile_n: pick_u8(rng, a.tile_n, b.tile_n),
            tile_k: pick_u8(rng, a.tile_k, b.tile_k),
            tiles_per_wave: pick_u8(rng, a.tiles_per_wave, b.tiles_per_wave),
            waves_per_block: pick_u8(rng, a.waves_per_block, b.waves_per_block),
            use_lds_for_a: pick_bool(rng, a.use_lds_for_a, b.use_lds_for_a),
            use_lds_for_b: pick_bool(rng, a.use_lds_for_b, b.use_lds_for_b),
            prefetch_depth: pick_u8(rng, a.prefetch_depth, b.prefetch_depth),
            k_unroll: pick_u8(rng, a.k_unroll, b.k_unroll),
            double_buffer: pick_bool(rng, a.double_buffer, b.double_buffer),
            dequant_strategy: pick_dq(rng, a.dequant_strategy, b.dequant_strategy),
        };
        child.sanitize();
        child
    }

    /// Gaussian-style perturbation: each gene is re-drawn from its
    /// value set with probability `mutation_rate` (`ga_tuning_spec §2.5`
    /// — "Gauss-Perturbation einzelner Gene, Rate 0.1 pro Gen"). The
    /// GA spec uses "gauss" loosely for discrete value sets; the
    /// effective behaviour is "with probability p, resample uniformly
    /// from the legal set". After the perturbation pass the genome is
    /// sanitised so cross-field constraints stay intact.
    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut SeededRng) {
        if rng.gen_f64() < mutation_rate {
            self.tile_m = rng.choose_u8(TILE_M_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.tile_n = rng.choose_u8(TILE_N_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.tile_k = rng.choose_u8(TILE_K_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.tiles_per_wave = rng.choose_u8(TILES_PER_WAVE_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.waves_per_block = rng.choose_u8(WAVES_PER_BLOCK_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.use_lds_for_a = rng.gen_bool();
        }
        if rng.gen_f64() < mutation_rate {
            self.use_lds_for_b = rng.gen_bool();
        }
        if rng.gen_f64() < mutation_rate {
            self.prefetch_depth = rng.choose_u8(PREFETCH_DEPTH_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.k_unroll = rng.choose_u8(K_UNROLL_VALUES);
        }
        if rng.gen_f64() < mutation_rate {
            self.double_buffer = rng.gen_bool();
        }
        if rng.gen_f64() < mutation_rate {
            self.dequant_strategy = match rng.gen_index(3) {
                0 => DequantStrategy::Inline,
                1 => DequantStrategy::PrePass {
                    lds_bytes: *rng.choose(DEQUANT_PREPASS_LDS_VALUES),
                },
                _ => DequantStrategy::Batched {
                    batch_size: rng.choose_u8(DEQUANT_BATCH_SIZES),
                },
            };
        }
        self.sanitize();
    }

    /// Default sanitization target — Q4_K + Fp16, the primary GA focus
    /// from `ga_tuning_spec §2.6` (gate_up_swiglu, q8_inline on Q4_K
    /// layers). Wraps [`Self::sanitize_for`] so the common case
    /// doesn't have to thread the format/level through every caller.
    pub fn sanitize(&mut self) {
        self.sanitize_for(&GaQuantFormat::q4_k(), PrecisionLevel::Fp16);
    }

    /// Comprehensive repair — makes the genome pass `validate_pre_compile`
    /// for the given `(fmt, level)` by adjusting the minimum set of
    /// fields. Field-level snapping is cheap; the cross-field work
    /// (sub-block alignment, workgroup-tile product, LDS budget, VGPR
    /// heuristic) is resolved in fixed-iteration loops that always
    /// terminate (proofs next to each loop).
    ///
    /// Ordering matters: later steps never re-introduce violations
    /// fixed by earlier ones, so a single top-to-bottom pass is
    /// sufficient.
    pub fn sanitize_for(&mut self, fmt: &GaQuantFormat, level: PrecisionLevel) {
        // 1. Field-level snap. Defensive — random / mutate already pick
        //    from legal sets, but a hand-built genome or a future
        //    caller might not.
        self.tile_m = snap_to_set(self.tile_m, TILE_M_VALUES);
        self.tile_n = snap_to_set(self.tile_n, TILE_N_VALUES);
        self.tile_k = snap_to_set(self.tile_k, TILE_K_VALUES);
        self.tiles_per_wave = snap_to_set(self.tiles_per_wave, TILES_PER_WAVE_VALUES);
        self.waves_per_block = snap_to_set(self.waves_per_block, WAVES_PER_BLOCK_VALUES);
        self.k_unroll = snap_to_set(self.k_unroll, K_UNROLL_VALUES);
        self.prefetch_depth = snap_to_set(self.prefetch_depth, PREFETCH_DEPTH_VALUES);

        // 2. Sub-block alignment. Q4_K/Q4_0/Q4_1/Q8_0 have
        //    `sub_block_size = 32` so `tile_k = 16` must be promoted
        //    to 32; Q6_K is already 16-aligned.
        if (self.tile_k as u32) % fmt.sub_block_size != 0 {
            self.tile_k = TILE_K_VALUES
                .iter()
                .copied()
                .filter(|&v| (v as u32) % fmt.sub_block_size == 0)
                .find(|&v| v >= self.tile_k)
                .unwrap_or(64);
        }

        // 3. Workgroup tile product. Halve whichever factor is larger
        //    until the product fits. Max iterations ≤ log₂(32) = 5.
        for _ in 0..8 {
            if (self.tiles_per_wave as u32) * (self.waves_per_block as u32)
                <= MAX_TILES_PER_BLOCK
            {
                break;
            }
            if self.waves_per_block >= self.tiles_per_wave && self.waves_per_block > 1 {
                self.waves_per_block = half_in_set(self.waves_per_block, WAVES_PER_BLOCK_VALUES);
            } else if self.tiles_per_wave > 1 {
                self.tiles_per_wave = half_in_set(self.tiles_per_wave, TILES_PER_WAVE_VALUES);
            } else {
                break;
            }
        }

        // 4. LDS budget. Reductions in order of preference — giving up
        //    double-buffering costs less than dropping LDS altogether,
        //    and the caller's bool choices matter most so we only flip
        //    them if necessary. Max iterations = 4 (db + prepass + 2
        //    lds flags).
        for _ in 0..6 {
            if compute_lds_bytes(self, level) <= LDS_BUDGET_BYTES {
                break;
            }
            if self.double_buffer {
                self.double_buffer = false;
                continue;
            }
            if matches!(self.dequant_strategy, DequantStrategy::PrePass { .. }) {
                self.dequant_strategy = DequantStrategy::Inline;
                continue;
            }
            if self.use_lds_for_a {
                self.use_lds_for_a = false;
                continue;
            }
            if self.use_lds_for_b {
                self.use_lds_for_b = false;
                continue;
            }
            break; // all LDS-contributing fields disabled
        }

        // 5. VGPR heuristic. Same preference order as LDS — drop the
        //    cheapest savings first (double_buffer, prefetch, Batched
        //    dequant) before touching tile_m or k_unroll which affect
        //    throughput.
        for _ in 0..16 {
            if estimate_vgprs(self, fmt, level) <= PRE_COMPILE_VGPR_LIMIT {
                break;
            }
            if self.double_buffer {
                self.double_buffer = false;
                continue;
            }
            if self.prefetch_depth > 0 {
                self.prefetch_depth = 0;
                continue;
            }
            if matches!(self.dequant_strategy, DequantStrategy::Batched { .. }) {
                self.dequant_strategy = DequantStrategy::Inline;
                continue;
            }
            if self.k_unroll > 1 {
                let current = self.k_unroll;
                self.k_unroll = K_UNROLL_VALUES
                    .iter()
                    .copied()
                    .filter(|&v| v < current)
                    .max()
                    .unwrap_or(1);
                continue;
            }
            if self.tile_m > 16 {
                let current = self.tile_m;
                self.tile_m = TILE_M_VALUES
                    .iter()
                    .copied()
                    .filter(|&v| v < current)
                    .max()
                    .unwrap_or(16);
                continue;
            }
            break; // minimal config reached; VGPR estimate for (16, Inline,
            // k_unroll=1, no prefetch, no db) is ≤ 96 across all
            // levels (Fp32 worst case: 72 + 4 + 4 = 80), so this
            // branch is unreachable in practice.
        }
    }
}

/// Snap `val` to the nearest value in `set` (by absolute difference).
/// Ties break towards the smaller value because ordered sets here are
/// monotonic ascending.
fn snap_to_set(val: u8, set: &[u8]) -> u8 {
    *set.iter()
        .min_by_key(|&&v| ((v as i16) - (val as i16)).unsigned_abs() as u16)
        .unwrap()
}

/// Return the largest value in `set` that's strictly smaller than
/// `val`. If no such value exists, return `val` (caller's
/// responsibility to break the loop).
fn half_in_set(val: u8, set: &[u8]) -> u8 {
    set.iter()
        .copied()
        .filter(|&v| v < val)
        .max()
        .unwrap_or(val)
}

/// Concrete LDS-byte computation — mirrors the expression from
/// `validation::validate_pre_compile`. Kept here so `sanitize_for` can
/// iterate toward a fixed point without bouncing between modules.
fn compute_lds_bytes(g: &KernelGenome, level: PrecisionLevel) -> u32 {
    let bpe = level.bytes_per_elem();
    let lds_a = (g.tile_m as u32) * (g.tile_k as u32) * bpe * (g.use_lds_for_a as u32);
    let lds_b = (g.tile_n as u32) * (g.tile_k as u32) * bpe * (g.use_lds_for_b as u32);
    let base = lds_a + lds_b + g.dequant_strategy.lds_overhead();
    if g.double_buffer { base * 2 } else { base }
}

/// `ga_tuning_spec §7.2` — GA → Codegen converter.
impl From<&KernelGenome> for TileConfig {
    fn from(g: &KernelGenome) -> TileConfig {
        TileConfig {
            tile_m: g.tile_m as usize,
            tile_n: g.tile_n as usize,
            k_chunk: g.tile_k as usize,
            lds_strategy: match (g.use_lds_for_a, g.use_lds_for_b) {
                (true, true) => LdsStrategy::LdsAB,
                (false, true) => LdsStrategy::DirectA_LdsB,
                (false, false) => LdsStrategy::DirectAB,
                // Spec says "(true, false) → fallback LdsAB" — that corner
                // is rare (LDS-A but direct-global-B) and the emitter
                // currently only supports the three canonical layouts.
                (true, false) => LdsStrategy::LdsAB,
            },
            num_waves: g.waves_per_block as usize,
            unroll_factor: g.k_unroll as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_values_are_in_legal_sets() {
        let mut rng = SeededRng::new(1);
        for _ in 0..200 {
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
                    assert!(DEQUANT_BATCH_SIZES.contains(&batch_size));
                }
            }
        }
    }
}
