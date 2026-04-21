//! Säule 2 — Computation Graph + Fusion-Passes.
//!
//! Builds a typed operator DAG from GGUF metadata, applies rule-based
//! fusion patterns, and feeds the fused graph into the kernel-codegen
//! pipeline. See architecture_v1.2.0-draft §2.3.

pub mod buffer_plan;
pub mod builder;
#[cfg(feature = "gpu")]
pub mod executor;
pub mod nodes;

pub use buffer_plan::{BufferPlan, BufferSpec, KvCacheLayout};
pub use builder::{BuildError, ComputationGraph, GraphBuildContext, GraphBuilder};
#[cfg(feature = "gpu")]
pub use executor::GraphExecutor;
pub use nodes::{BufferId, GraphNode, WeightRef};
