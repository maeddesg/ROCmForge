//! Startup validation + banner data for the interactive chat CLI.
//!
//! Collects GPU identity, model metadata, quantisation type, and post-load
//! VRAM usage, plus a set of non-fatal warnings (e.g. non-Q4_0 weights,
//! non-gfx12 GPU) that get surfaced in the banner so the user can tell at
//! a glance why inference might be slower than advertised.

use std::path::Path;

use crate::config::ModelConfig;
use crate::loader::GgufFile;

#[cfg(feature = "gpu")]
use crate::gpu;

/// Ask `rocminfo` for the first `gfx*` target. The HIP FFI currently
/// returns a placeholder `arch_name`, so we fall back to this external
/// lookup to surface the real architecture in the banner. Returns `None`
/// silently if `rocminfo` is missing or empty.
fn detect_gpu_arch_via_rocminfo() -> Option<String> {
    let output = std::process::Command::new("rocminfo").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Everything the chat banner wants to display after a successful startup.
pub struct StartupInfo {
    pub gpu_name: String,
    pub gpu_arch: String,
    pub hip_driver: String,
    pub model_name: String,
    pub model_file: String,
    pub quant_type: String,
    pub model_size_gb: f64,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub wmma_active: bool,
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub warnings: Vec<String>,
}

/// Best-effort ROCm release version. `hipRuntimeGetVersion` returns the HIP
/// runtime/driver version which is unrelated to the user-facing ROCm release
/// string, so prefer `/opt/rocm/.info/version` when it exists and fall back
/// to whatever `hip_driver_version` gave us.
fn read_rocm_version() -> Option<String> {
    let candidates = ["/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"];
    for path in candidates {
        if let Ok(s) = std::fs::read_to_string(path) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Decode the HIP driver version packed as `major << 22 | minor << 12 | patch`.
fn format_hip_driver(v: u32) -> String {
    if v == 0 {
        return "unknown".to_string();
    }
    let major = (v >> 22) & 0x3FF;
    let minor = (v >> 12) & 0x3FF;
    let patch = v & 0xFFF;
    format!("{}.{}.{}", major, minor, patch)
}

/// Best-effort quantisation tag for the banner. We read the first attention
/// Q-projection tensor — every supported architecture has one under a name
/// containing `attn_q` — and report its `GgmlType`. Not authoritative for
/// mixed-quant files; Phase 5 Step 1 is a human-readable banner, not a
/// policy decision.
fn detect_quant_type(file: &GgufFile) -> String {
    let mut attn_q: Option<&str> = None;
    for name in file.tensor_names() {
        if name.contains("attn_q.weight") {
            attn_q = Some(name);
            break;
        }
    }
    let probe = attn_q.or_else(|| file.tensor_names().find(|n| n.ends_with(".weight")));
    if let Some(n) = probe {
        if let Ok(Some(t)) = file.tensor(n) {
            return t.ggml_type.to_string();
        }
    }
    "unknown".to_string()
}

/// Run the pre-load validation and collect banner data. `vram_used_gb` and
/// `vram_total_gb` are snapshotted *before* weights are loaded; the chat
/// driver is expected to refresh them after load and pass the deltas back
/// (see `refresh_vram_usage`).
pub fn validate_before_load(model_path: &str) -> Result<(StartupInfo, GgufFile, ModelConfig), String> {
    let path = Path::new(model_path);
    if !path.exists() {
        return Err(format!("Model file not found: {}", model_path));
    }
    let model_file = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.to_string());

    let model_size_gb = std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);

    let file = GgufFile::open(model_path).map_err(|e| format!("open GGUF: {}", e))?;
    let config = ModelConfig::from_gguf(&file).map_err(|e| format!("parse model config: {}", e))?;
    let model_name = file
        .metadata
        .raw
        .get("general.name")
        .cloned()
        .unwrap_or_else(|| model_file.clone());
    let quant_type = detect_quant_type(&file);

    let mut warnings: Vec<String> = Vec::new();

    #[cfg(feature = "gpu")]
    let (gpu_name, gpu_arch, hip_driver, vram_total_gb, vram_free_gb, wmma_active) = {
        let caps = gpu::detect()
            .ok_or_else(|| "No AMD GPU detected (chat mode requires --gpu).".to_string())?;
        // The HIP FFI layer currently hands us a placeholder `arch_name`; if
        // the parsed architecture is `Unknown`, ask rocminfo directly so the
        // banner reflects the real hardware.
        let effective_arch = match caps.architecture {
            gpu::GpuArchitecture::Unknown(_) => detect_gpu_arch_via_rocminfo()
                .and_then(|s| gpu::GpuArchitecture::from_name(&s))
                .unwrap_or(caps.architecture),
            other => other,
        };
        let arch_name = match effective_arch {
            gpu::GpuArchitecture::Gfx1201 => "gfx1201 (RDNA 4)".to_string(),
            gpu::GpuArchitecture::Gfx1100 => "gfx1100 (RDNA 3)".to_string(),
            gpu::GpuArchitecture::Gfx1030 => "gfx1030 (RDNA 2)".to_string(),
            gpu::GpuArchitecture::Gfx90a => "gfx90a (CDNA 2)".to_string(),
            gpu::GpuArchitecture::Gfx908 => "gfx908 (CDNA 1)".to_string(),
            gpu::GpuArchitecture::Gfx900 => "gfx900 (Vega)".to_string(),
            gpu::GpuArchitecture::Unknown(id) => format!("gfx{:x} (unknown)", id),
        };
        let arch_is_gfx12 = matches!(effective_arch, gpu::GpuArchitecture::Gfx1201);
        // WMMA is implemented for Q4_0, Q4_1, and Q4_K. Mixed-precision
        // GGUFs (e.g. Q4_K_M with Q6_K V / ffn_down) still go through the
        // WMMA path for the Q4_K tensors; the Q6_K layers fall back to
        // GEMV-loop but that is a separate dispatch-time decision, not a
        // reason to say "WMMA inactive" on the banner.
        let quant_supports_wmma = matches!(
            quant_type.as_str(),
            "Q4_0" | "Q4_1" | "Q4_K"
        );
        let wmma_active = arch_is_gfx12 && quant_supports_wmma;
        if !arch_is_gfx12 {
            warnings.push(format!(
                "GPU is {} — WMMA kernels require gfx12 (RDNA 4). Inference falls back to scalar kernels.",
                arch_name
            ));
        }
        if !quant_supports_wmma {
            warnings.push(format!(
                "Model weights are {} — WMMA kernels are implemented for Q4_0, Q4_1, and Q4_K. Expect slower inference.",
                quant_type
            ));
        }
        (
            caps.device_name.clone(),
            arch_name,
            format_hip_driver(caps.hip_driver_version),
            caps.total_vram_gb(),
            caps.free_vram_gb(),
            wmma_active,
        )
    };

    #[cfg(not(feature = "gpu"))]
    let (gpu_name, gpu_arch, hip_driver, vram_total_gb, vram_free_gb, wmma_active) = {
        warnings.push(
            "Binary built without --features gpu — chat mode runs on CPU only.".to_string(),
        );
        ("CPU only".to_string(), "n/a".to_string(), "n/a".to_string(), 0.0, 0.0, false)
    };

    let info = StartupInfo {
        gpu_name,
        gpu_arch,
        hip_driver,
        model_name,
        model_file,
        quant_type,
        model_size_gb,
        num_layers: config.num_layers,
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        wmma_active,
        vram_used_gb: (vram_total_gb - vram_free_gb).max(0.0),
        vram_total_gb,
        warnings,
    };
    Ok((info, file, config))
}

/// Query the GPU for current VRAM usage so the banner can show the delta
/// after weights have been uploaded.
#[cfg(feature = "gpu")]
pub fn refresh_vram_usage(info: &mut StartupInfo) {
    if let Some(caps) = gpu::detect() {
        info.vram_used_gb = (caps.total_vram_gb() - caps.free_vram_gb()).max(0.0);
        info.vram_total_gb = caps.total_vram_gb();
    }
}

#[cfg(not(feature = "gpu"))]
pub fn refresh_vram_usage(_info: &mut StartupInfo) {}

pub fn print_banner(info: &StartupInfo) {
    let version = env!("CARGO_PKG_VERSION");
    let wmma_line = if info.wmma_active {
        "active (GEMM + Attention)".to_string()
    } else {
        "inactive (requires gfx12 + Q4_0 / Q4_1 / Q4_K)".to_string()
    };
    let rocm_line = read_rocm_version().unwrap_or_else(|| info.hip_driver.clone());
    println!();
    println!("  ROCmForge v{}", version);
    println!("  ─────────────────────────────────────");
    println!(
        "  GPU:       {} ({})",
        info.gpu_name, info.gpu_arch
    );
    println!("  ROCm:      {}", rocm_line);
    println!(
        "  Model:     {} ({}, {:.1} GB)",
        info.model_name, info.quant_type, info.model_size_gb
    );
    println!(
        "  Layers:    {}   Hidden: {}   Vocab: {}",
        info.num_layers, info.hidden_size, info.vocab_size
    );
    println!("  WMMA:      {}", wmma_line);
    println!(
        "  VRAM:      {:.1} / {:.1} GB used",
        info.vram_used_gb, info.vram_total_gb
    );
    for w in &info.warnings {
        println!("  ⚠ {}", w);
    }
    println!("  ─────────────────────────────────────");
    println!("  Type /help for commands, /quit to exit.");
    println!();
}
