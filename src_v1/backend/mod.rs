//! Backend emitters — gfx1201 (GPU) and Zen4 (CPU).
//!
//! Each target owns a dedicated codegen emitter for the Dequant IR. No
//! hardware-abstraction layer: kernels know their target and use native
//! intrinsics directly. See architecture_v1.2.0-draft §3.

pub mod gpu;
pub mod cpu;
