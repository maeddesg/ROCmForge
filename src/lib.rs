pub mod bench;
pub mod cli;
pub mod config;
pub mod cpu;
pub mod hardware;
pub mod loader;
pub mod logging;
pub mod tokenizer;

#[cfg(feature = "gpu")]
pub mod gpu;

// v1.0 rebuild — lives in sibling src_v1/ and is pulled in under the
// `v1` feature without breaking the v0.x default-off build.
#[cfg(feature = "v1")]
#[path = "../src_v1/lib.rs"]
pub mod v1;
