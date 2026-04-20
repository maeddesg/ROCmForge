//! Zen4 backend — AVX-512 double-pumped, VNNI, BF16 dot products.
//!
//! Targets the Ryzen 9 7945HX directly: 512-bit ops over two 256-bit
//! execution ports, no AVX-512 downclock. MXCSR prolog sets FTZ/DAZ off
//! for IEEE parity with the GPU path. See architecture_v1.2.0-draft §3.3.
