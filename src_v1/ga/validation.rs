//! Two-stage GA validation (`ga_tuning_spec §2.3` + `§2.3.1`).
//!
//! Stage 1 — pre-compile heuristic. WMMA-alignment, LDS budget,
//! heuristic VGPR ceiling, workgroup limit, sub-block alignment. Rules
//! are intentionally generous (150 VGPRs, not 104) because LLVM's
//! register allocator routinely surprises by ±30 %; the real check is
//! Stage 2.
//!
//! Stage 2 — post-compile VGPR gate. Reads the actual VGPR/SGPR counts
//! out of the AMDGPU ELF-note section of a compiled `.co` file. If
//! `max_waves_per_cu < 4`, the candidate is rejected before the
//! expensive 20-run benchmark kicks in (`§2.3.1`).

use super::genome::{DequantStrategy, KernelGenome};
use super::types::{
    CodeObjectResources, PrecisionLevel, LDS_BUDGET_BYTES, MAX_TILES_PER_BLOCK, MIN_WAVES_PER_CU,
    PRE_COMPILE_VGPR_LIMIT,
};

/// Minimum sub-block alignment per quant format. Passed into the
/// validator so step 2.1.1 doesn't need to pull in the full
/// `QuantFormat` type — a single `u32` carries all the info Stage 1
/// needs.
#[derive(Debug, Clone, Copy)]
pub struct GaQuantFormat {
    pub sub_block_size: u32,
}

impl GaQuantFormat {
    pub fn q4_k() -> Self {
        Self { sub_block_size: 32 }
    }
    pub fn q6_k() -> Self {
        Self { sub_block_size: 16 }
    }
    pub fn q8_0() -> Self {
        Self { sub_block_size: 32 }
    }
}

/// Stage 1 — pre-compile heuristic (`ga_tuning_spec §2.3`).
pub fn validate_pre_compile(
    g: &KernelGenome,
    fmt: &GaQuantFormat,
    level: PrecisionLevel,
) -> bool {
    // 1. WMMA-Alignment.
    if g.tile_m % 16 != 0 || g.tile_n % 16 != 0 || g.tile_k % 16 != 0 {
        return false;
    }

    // 2. LDS budget.
    let bytes_per_elem = level.bytes_per_elem();
    let lds_a =
        (g.tile_m as u32) * (g.tile_k as u32) * bytes_per_elem * (g.use_lds_for_a as u32);
    let lds_b =
        (g.tile_n as u32) * (g.tile_k as u32) * bytes_per_elem * (g.use_lds_for_b as u32);
    let lds_base = lds_a + lds_b + g.dequant_strategy.lds_overhead();
    let lds_total = if g.double_buffer { lds_base * 2 } else { lds_base };
    if lds_total > LDS_BUDGET_BYTES {
        return false;
    }

    // 3. Heuristic VGPR budget — generous, real check post-compile.
    if estimate_vgprs(g, fmt, level) > PRE_COMPILE_VGPR_LIMIT {
        return false;
    }

    // 4. Workgroup tile limit.
    if (g.tiles_per_wave as u32) * (g.waves_per_block as u32) > MAX_TILES_PER_BLOCK {
        return false;
    }

    // 5. Sub-block alignment (format-dependent).
    if (g.tile_k as u32) % fmt.sub_block_size != 0 {
        return false;
    }

    true
}

/// Heuristic VGPR estimator. Per `ga_tuning_spec §2.3` the estimate is
/// "typically 30 % daneben" — the real count comes from the ELF note
/// after compile. Values here are derived from the Dequant-IR register-
/// budget tables (`dequant_ir_spec §5.2`) and a linear model over
/// `k_unroll`, `prefetch_depth`, and `tile_m`.
pub fn estimate_vgprs(g: &KernelGenome, _fmt: &GaQuantFormat, level: PrecisionLevel) -> u32 {
    let base = match level {
        PrecisionLevel::Fp8 => 40,
        PrecisionLevel::Fp16 | PrecisionLevel::Bf16 => 56,
        PrecisionLevel::Fp32 => 72,
    };
    let tile_m_cost = match g.tile_m {
        16 => 4,
        32 => 8,
        64 => 16,
        128 => 28,
        _ => 16,
    };
    let unroll_cost = (g.k_unroll as u32) * 4;
    let prefetch_cost = (g.prefetch_depth as u32) * 6;
    let doublebuf_cost = if g.double_buffer { 8 } else { 0 };
    let dequant_cost = match g.dequant_strategy {
        DequantStrategy::Inline => 0,
        DequantStrategy::PrePass { .. } => 4,
        DequantStrategy::Batched { batch_size } => (batch_size as u32) * 2,
    };
    base + tile_m_cost + unroll_cost + prefetch_cost + doublebuf_cost + dequant_cost
}

/// Post-compile gate result. `accepted = false` means the candidate's
/// code object is too VGPR-heavy to sustain ≥ 4 waves/CU.
#[derive(Debug, Clone, Copy)]
pub struct PostCompileResult {
    pub accepted: bool,
    pub actual_vgprs: u32,
    pub actual_sgprs: u32,
    pub max_waves_per_cu: u32,
    pub lds_bytes: u32,
}

impl PostCompileResult {
    pub fn from_resources(res: CodeObjectResources) -> Self {
        let max_waves = res.max_waves_per_cu();
        Self {
            accepted: max_waves >= MIN_WAVES_PER_CU,
            actual_vgprs: res.vgpr_count as u32,
            actual_sgprs: res.sgpr_count as u32,
            max_waves_per_cu: max_waves,
            lds_bytes: res.lds_bytes,
        }
    }
}

/// Stage 2 — post-compile VGPR gate (`ga_tuning_spec §2.3.1`).
pub fn validate_post_compile(res: CodeObjectResources) -> PostCompileResult {
    PostCompileResult::from_resources(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::ga::genome::KernelGenome;

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

    #[test]
    fn valid_genome_passes() {
        let g = sensible_genome();
        assert!(validate_pre_compile(
            &g,
            &GaQuantFormat::q4_k(),
            PrecisionLevel::Fp16
        ));
    }

    #[test]
    fn bad_alignment_rejected() {
        let mut g = sensible_genome();
        g.tile_m = 17;
        assert!(!validate_pre_compile(
            &g,
            &GaQuantFormat::q4_k(),
            PrecisionLevel::Fp16
        ));
    }

    #[test]
    fn bad_subblock_rejected() {
        let mut g = sensible_genome();
        // Q6_K sub_block_size = 16, but 16-multiples are OK. Use Q4_K
        // (sub_block_size = 32) with tile_k = 16: 16 % 32 != 0.
        g.tile_k = 16;
        assert!(!validate_pre_compile(
            &g,
            &GaQuantFormat::q4_k(),
            PrecisionLevel::Fp16
        ));
    }

    #[test]
    fn post_compile_gates_on_4_waves() {
        // 385 VGPRs → 1536/385 = 3.99 → rejected.
        let r = validate_post_compile(CodeObjectResources {
            vgpr_count: 385,
            sgpr_count: 20,
            lds_bytes: 0,
        });
        assert!(!r.accepted);
        assert_eq!(r.max_waves_per_cu, 3);

        // 256 VGPRs → 6 waves/CU → accepted.
        let r = validate_post_compile(CodeObjectResources {
            vgpr_count: 256,
            sgpr_count: 20,
            lds_bytes: 0,
        });
        assert!(r.accepted);
        assert_eq!(r.max_waves_per_cu, 6);
    }
}
