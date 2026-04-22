//! Compile-Cache + AMDGPU metadata parser (`ga_tuning_spec §2.7.1`).
//!
//! The cache is keyed on `(TileConfig, QuantFormatId, PrecisionLevel,
//! KernelTarget)`. Two genomes that convert to the same `TileConfig`
//! (common — genomes carry extra metadata like `prefetch_depth` that
//! the codegen hashes separately) share a compiled kernel; the hit
//! rate after Generation 5+ is typically 50–70 % per
//! `ga_tuning_spec §2.7.1`.
//!
//! For step 2.1.1 the compile pipeline itself is **stubbed**. The
//! `compile_stub_for_test` entry point feeds known `CodeObjectResources`
//! into the cache so the framework tests exercise cache-hit behaviour
//! deterministically. Real hipcc invocation lands in step 2.1.3 when
//! the Dequant-IR codegen accepts a `TileConfig`.
//!
//! The `parse_amdgpu_metadata` helper is real and works today — it
//! shells out to `llvm-readobj --notes` on an extracted gfx1201 code
//! object and greps the AMDGPU metadata for `.vgpr_count` /
//! `.sgpr_count` / `.group_segment_fixed_size`. That powers the
//! `gpu`-feature-gated test that confirms VGPR reading from a real
//! `.co` is wired up end-to-end.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use super::types::{CodeObjectResources, KernelTarget, PrecisionLevel, TileConfig};

/// Cache-Key per `ga_tuning_spec §6.3`. `engine_git_hash` is carried
/// abstractly so cache invalidation on an engine rebuild is a single
/// field change.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CompileKey {
    pub tile_config: TileConfig,
    pub format_id: u32,
    pub precision_level: PrecisionLevel,
    pub target: KernelTarget,
}

impl CompileKey {
    pub fn new(
        tile_config: TileConfig,
        format_id: u32,
        precision_level: PrecisionLevel,
        target: KernelTarget,
    ) -> Self {
        Self {
            tile_config,
            format_id,
            precision_level,
            target,
        }
    }
}

/// Compiled-kernel handle. Real binary lives on disk or in a cache
/// allocator; the GA holds an `Arc` so multiple genomes that share a
/// `CompileKey` don't pay the compile cost twice.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub key: CompileKey,
    pub resources: CodeObjectResources,
    /// Path to the `.co` file on disk (or `None` for stubbed test
    /// kernels that never hit the FS).
    pub co_path: Option<std::path::PathBuf>,
}

/// Cache of compiled kernels within one GA run (`ga_tuning_spec §2.7.1`).
#[derive(Default)]
pub struct CompileCache {
    entries: HashMap<CompileKey, Arc<CompiledKernel>>,
    pub hits: u64,
    pub misses: u64,
}

impl CompileCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up or stub-insert. Real `codegen_and_compile` arrives in
    /// step 2.1.3; for step 2.1.1 the factory closure builds the
    /// `CompiledKernel`.
    pub fn get_or_insert_with<F>(&mut self, key: &CompileKey, factory: F) -> Arc<CompiledKernel>
    where
        F: FnOnce(&CompileKey) -> CompiledKernel,
    {
        if let Some(cached) = self.entries.get(key) {
            self.hits += 1;
            return cached.clone();
        }
        self.misses += 1;
        let kernel = Arc::new(factory(key));
        self.entries.insert(key.clone(), kernel.clone());
        kernel
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Parse an AMDGPU code-object's metadata note section to extract VGPR,
/// SGPR, and LDS byte counts. Shells out to `llvm-readobj --notes` —
/// `ga_tuning_spec §2.3.1` lists this as the recommended approach.
///
/// Input is an **extracted** gfx1201 ELF (produced by
/// `llvm-objdump --offloading <bundled-hip-object>`). Passing the
/// bundled host-object yields no AMDGPU notes and this function returns
/// `Ok(None)`.
pub fn parse_amdgpu_metadata(path: &Path) -> io::Result<Option<CodeObjectResources>> {
    let output = Command::new("llvm-readobj")
        .arg("--notes")
        .arg(path)
        .output()?;

    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "llvm-readobj --notes failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Look for the first kernel block that carries `.vgpr_count:` etc.
    let vgpr = extract_numeric(&stdout, ".vgpr_count:");
    let sgpr = extract_numeric(&stdout, ".sgpr_count:");
    let lds = extract_numeric(&stdout, ".group_segment_fixed_size:");

    match (vgpr, sgpr, lds) {
        (Some(v), Some(s), Some(l)) => Ok(Some(CodeObjectResources {
            vgpr_count: v as u16,
            sgpr_count: s as u16,
            lds_bytes: l,
        })),
        _ => Ok(None),
    }
}

fn extract_numeric(haystack: &str, needle: &str) -> Option<u32> {
    for line in haystack.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(needle) {
            let digits: String = rest.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                return digits.parse::<u32>().ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v1::ga::types::{LdsStrategy, TileConfig};

    fn sample_key() -> CompileKey {
        CompileKey::new(
            TileConfig {
                tile_m: 64,
                tile_n: 64,
                k_chunk: 32,
                lds_strategy: LdsStrategy::DirectAB,
                num_waves: 4,
                unroll_factor: 4,
            },
            12, // Q4_K
            PrecisionLevel::Fp16,
            KernelTarget::Gfx1201,
        )
    }

    #[test]
    fn cache_hit_on_identical_key() {
        let mut cache = CompileCache::new();
        let k = sample_key();
        let a = cache.get_or_insert_with(&k, |key| CompiledKernel {
            key: key.clone(),
            resources: CodeObjectResources {
                vgpr_count: 80,
                sgpr_count: 20,
                lds_bytes: 0,
            },
            co_path: None,
        });
        let b = cache.get_or_insert_with(&k, |_| panic!("factory should not run on hit"));
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn extract_numeric_works() {
        let text = "
        .group_segment_fixed_size: 0
        .sgpr_count:     18
        .vgpr_count:     186
        .wavefront_size: 32
        ";
        assert_eq!(extract_numeric(text, ".vgpr_count:"), Some(186));
        assert_eq!(extract_numeric(text, ".sgpr_count:"), Some(18));
        assert_eq!(extract_numeric(text, ".group_segment_fixed_size:"), Some(0));
    }
}
