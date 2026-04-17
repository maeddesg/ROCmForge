//! Internal structured logging for ROCmForge.
//!
//! This is **not** user-facing output. The interactive banner, token
//! streaming, and post-turn metrics keep writing to stdout via plain
//! `print!`/`println!`. This module routes `tracing` events to stderr,
//! filtered by `RUST_LOG`:
//!
//! * default: `warn` — silent on a successful run.
//! * `RUST_LOG=debug` — surfaces every kernel-dispatch decision, KV-cache
//!   state change, and feature-flag snapshot.
//! * `RUST_LOG=trace` — adds per-layer prefill timing (hot path, but the
//!   `trace!` macros compile to no-ops when this level is filtered out).
//!
//! `init()` is idempotent in the sense that a double call is a no-op;
//! the underlying subscriber registration is guarded by `try_init`.

use std::sync::atomic::{AtomicBool, Ordering};

use tracing_subscriber::EnvFilter;

static INITIALISED: AtomicBool = AtomicBool::new(false);

/// Install the stderr subscriber. Safe to call from multiple entry
/// points — only the first call wins.
pub fn init() {
    if INITIALISED.swap(true, Ordering::SeqCst) {
        return;
    }
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    // `try_init` fails silently if another subscriber is already set
    // (e.g. in a test harness); that's the behaviour we want.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .try_init();
}
