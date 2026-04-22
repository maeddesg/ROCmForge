//! Shared GA types.
//!
//! `TileConfig`, `LdsStrategy`, `KernelShape`, and `PrecisionLevel` mirror
//! the upcoming Dequant-IR spec (`dequant_ir_spec §5.1`, `§4.4`). They
//! live here because the GA is the first module that needs them;
//! when codegen starts consuming `TileConfig` in step 2.1.3 these
//! definitions move to `src_v1/ir/` and this module re-exports.
//!
//! `KernelTarget` is part of the cache key from day one so the Zen4
//! AVX-512 path can be added without breaking existing cache entries
//! (`ga_tuning_spec §6.3`).
//!
//! `CodeObjectResources` is the post-compile snapshot of VGPR/SGPR/LDS
//! numbers (`ga_tuning_spec §2.3.1`). It's filled by parsing the AMDGPU
//! metadata note of the compiled `.co` file.

/// Shape of a single GEMM/GEMV op the GA is optimising. Mirrors
/// `dequant_ir_spec §5.1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl KernelShape {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}

/// Tile-layout strategy (LDS vs direct-global). `DirectAB` is the
/// GEMV-with-small-N default; `LdsAB` is the GEMM default. Variant
/// names are taken verbatim from `ga_tuning_spec §7.2` — the
/// underscore in `DirectA_LdsB` is deliberate for readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum LdsStrategy {
    LdsAB,
    DirectA_LdsB,
    DirectAB,
}

/// Tile configuration consumed by the codegen. The GA's `KernelGenome`
/// converts to this via `impl From<&KernelGenome>` (`ga_tuning_spec §7.2`).
/// Additional genome fields (`prefetch_depth`, `double_buffer`,
/// `dequant_strategy`, `tiles_per_wave`) are passed separately via the
/// extended codegen context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub k_chunk: usize,
    pub lds_strategy: LdsStrategy,
    pub num_waves: usize,
    pub unroll_factor: usize,
}

/// Precision level from `dequant_ir_spec §4.4`. Level 3 (Fp32 VALU) is
/// the safety fallback and is **not** a GA target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionLevel {
    Fp8,
    Fp16,
    Bf16,
    Fp32,
}

impl PrecisionLevel {
    pub fn bytes_per_elem(self) -> u32 {
        match self {
            Self::Fp8 => 1,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Fp32 => 4,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
        }
    }
}

/// Codegen target. GPU is today; CPU is the Zen4 follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelTarget {
    Gfx1201,
    Zen4Avx512,
}

impl KernelTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gfx1201 => "gfx1201",
            Self::Zen4Avx512 => "zen4_avx512",
        }
    }
}

/// Post-compile resource snapshot (`ga_tuning_spec §2.3.1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeObjectResources {
    pub vgpr_count: u16,
    pub sgpr_count: u16,
    pub lds_bytes: u32,
}

impl CodeObjectResources {
    /// Waves-per-CU ceiling on gfx1201 (1536 VGPRs/CU, Wave=32).
    pub fn max_waves_per_cu(self) -> u32 {
        1536 / (self.vgpr_count.max(1) as u32)
    }
}

/// Max `tiles_per_wave × waves_per_block` per workgroup (gfx1201
/// launch-setup limit, `ga_tuning_spec §2.3`).
pub const MAX_TILES_PER_BLOCK: u32 = 16;

/// gfx1201 LDS budget per workgroup (`ga_tuning_spec §2.3`).
pub const LDS_BUDGET_BYTES: u32 = 64 * 1024;

/// gfx1201 VGPRs per CU.
pub const VGPRS_PER_CU: u32 = 1536;

/// Minimum acceptable waves-per-CU after compile (`ga_tuning_spec §2.3.1`).
/// Below this, memory-latency hiding collapses.
pub const MIN_WAVES_PER_CU: u32 = 4;

/// Pre-compile VGPR heuristic threshold (`ga_tuning_spec §2.3`). Set
/// generous (150 instead of the ~104 post-compile target) because the
/// estimator is routinely ~30 % off; the real check is `MIN_WAVES_PER_CU`
/// after hipcc has had its say.
pub const PRE_COMPILE_VGPR_LIMIT: u32 = 150;
