//! `KernelGenome` + GA operators (1:1 from `architecture_v1.2.0-draft §4.2`
//! and `ga_tuning_spec §2.2`).
//!
//! The genome is the GA search-space representation; `TileConfig`
//! (conversion via `impl From<&KernelGenome>`) is the codegen-facing
//! subset.

use super::rng::SeededRng;
use super::types::{LdsStrategy, TileConfig};

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
    /// Random genome with every slot drawn from the legal value set.
    /// Suitable for building the initial population.
    pub fn random(rng: &mut SeededRng) -> Self {
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

    /// Uniform crossover: each gene independently picked from parent `a`
    /// or `b` with 50/50 probability (`ga_tuning_spec §2.5`).
    pub fn crossover(a: &Self, b: &Self, rng: &mut SeededRng) -> Self {
        let pick_u8 = |rng: &mut SeededRng, x: u8, y: u8| if rng.gen_bool() { x } else { y };
        let pick_bool = |rng: &mut SeededRng, x: bool, y: bool| if rng.gen_bool() { x } else { y };
        let pick_dq =
            |rng: &mut SeededRng, x: DequantStrategy, y: DequantStrategy| {
                if rng.gen_bool() { x } else { y }
            };

        Self {
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
        }
    }

    /// Gaussian-style perturbation: each gene is re-drawn from its value
    /// set with probability `mutation_rate` (`ga_tuning_spec §2.5` —
    /// "Gauss-Perturbation einzelner Gene, Rate 0.1 pro Gen"). The GA
    /// spec uses the term "gauss" loosely for discrete value sets; the
    /// effective behaviour is "with probability p, resample uniformly
    /// from the legal set".
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
    }
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
