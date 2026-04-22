//! Säule 4 — Kernel-Tuning-GA-Framework (Phase 2 Schritt 2.1.1).
//!
//! Implements the GA engine described in `docs/v1.0/ga_tuning_spec.md §2`
//! and `docs/v1.0/architecture_v1.2.0-draft.md §4.2`. Struct names and
//! GA parameters are 1:1 from the specs.
//!
//! At this stage the framework is validated on a toy fitness landscape
//! (see `toy`). Real kernel compilation + benchmarking will be wired up
//! in step 2.1.3 when the Dequant-IR codegen accepts a `TileConfig`
//! directly.
//!
//! Module layout follows the prompt for step 2.1.1:
//!   * [`types`]      — shared GA types (TileConfig, PrecisionLevel, …)
//!   * [`rng`]        — deterministic seeded RNG (xorshift64*)
//!   * [`genome`]     — `KernelGenome`, `DequantStrategy`, GA operators
//!   * [`validation`] — pre-compile heuristic + post-compile VGPR gate
//!   * [`fitness`]    — 5-phase fitness flow, `FitnessResult`
//!   * [`compile`]    — `CompileCache`, `CompileKey`, compile pipeline
//!   * [`engine`]     — `GaConfig`, `KernelGa`, tournament / crossover / elitism
//!   * [`logger`]     — JSONL event log (`ga_tuning_spec §5.10`)
//!   * [`toy`]        — toy fitness for framework validation

pub mod compile;
pub mod engine;
pub mod fitness;
pub mod genome;
pub mod logger;
pub mod rng;
pub mod toy;
pub mod types;
pub mod validation;

pub use compile::{CompileCache, CompileKey, CompiledKernel};
pub use engine::{GaConfig, GaResult, GenerationResults, KernelGa, KernelGenomeScored};
pub use fitness::{evaluate_toy_fitness, FitnessResult};
pub use genome::{DequantStrategy, KernelGenome};
pub use logger::GaLogger;
pub use rng::SeededRng;
pub use toy::{run_toy_ga, toy_fitness};
pub use types::{
    CodeObjectResources, KernelShape, KernelTarget, LdsStrategy, PrecisionLevel, TileConfig,
};
pub use validation::{estimate_vgprs, validate_post_compile, validate_pre_compile, PostCompileResult};
