//! Core types, error handling, and shared infrastructure.
//!
//! Hosts the cross-pillar primitives: `GpuResult`, `PrecisionHint`,
//! `TensorShape`, hardware constants for gfx1201 and Zen4. No dependencies
//! on other `src_v1` pillars; every pillar may depend on `core`.

pub mod gguf;
pub mod model_config;
pub mod model_loader;
pub mod telemetry;
pub mod tensor_info;
