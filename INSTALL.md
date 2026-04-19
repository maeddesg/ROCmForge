# Installation

ROCmForge has been developed and tested exclusively on **Arch Linux
(CachyOS)** with **ROCm 7.2.1 / 7.2.2** on an **AMD Radeon RX 9070 XT
(gfx1201, RDNA 4)**. Notes for Ubuntu / Debian / Fedora are included
as a starting point but are **untested** — if you try them, please
report your results as a GitHub issue.

## Requirements

- AMD GPU with RDNA 4 architecture (gfx1201). See
  [Compatibility](README.md#compatibility) in the README for the
  status of other architectures.
- ROCm 7.2.x (7.2.1 or 7.2.2 verified).
- Rust 1.81 or newer (install via [rustup](https://rustup.rs)).
- CMake 3.21 or newer.
- Git.
- 64-bit Linux. Windows is not supported by ROCm on consumer GPUs.

## Arch Linux / CachyOS (tested)

### 1. Install ROCm

```bash
sudo pacman -S rocm-core hip-runtime-amd hipblas rocm-llvm rocm-hip-libraries
```

Verify the install finds your GPU:

```bash
rocminfo | grep "Name:" | grep gfx
# Expected: Name: gfx1201
```

If `rocminfo` does not list your GPU, check that the `amdgpu` kernel
module is loaded (`lsmod | grep amdgpu`) and that your user is in the
`render` and `video` groups (`groups $USER`). A reboot may be required
after first install.

### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version   # should be 1.81 or newer
```

### 3. Clone and build

```bash
git clone https://github.com/maeddesg/ROCmForge.git
cd ROCmForge
cargo build --release --features gpu
```

Cold build takes about 60 seconds. The binary is at
`target/release/rocmforge`.

### 4. Download models

```bash
mkdir -p ~/models
```

**Target model — required, pick one**:

- **Qwen2.5-7B-Instruct Q4_0** (~4.1 GB, v0.1.0 reference, 102 tok/s
  decode, near-parity prefill with llama.cpp):
  <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF>. Save to
  `~/models/Qwen2.5-7B-Instruct-Q4_0.gguf`.
- **Qwen3-8B Q4_K_M** (~4.7 GB, new in v0.2.0, ~30 tok/s decode):
  <https://huggingface.co/Qwen/Qwen3-8B-GGUF>. Save to
  `~/models/Qwen3-8B-Q4_K_M.gguf`.
- **Meta-Llama-3.1-8B-Instruct Q4_K_M** (~4.6 GB, new in v0.2.0,
  ~31 tok/s decode):
  <https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF>.
  Save to `~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.

**Draft model — optional** (Qwen2.5-0.5B-Instruct Q4_0, ~0.4 GB,
only needed for speculative decoding with the Qwen2.5-7B target):

- Hugging Face page: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF>
- Download the Q4_0 file
- Save to `~/models/qwen2.5-0.5b-instruct-q4_0.gguf`

### 5. Verify

One-shot inference:

```bash
./target/release/rocmforge \
    --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
    --prompt "Hello" --max-tokens 10 --gpu
```

Expected: coherent text, `Prefill: …` line and `N tokens in … = ~100 tok/s`
line on stderr.

Interactive chat:

```bash
./target/release/rocmforge chat \
    --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf
```

Expected banner:

```
  ROCmForge v0.1.0
  ─────────────────────────────────────
  GPU:       AMD Radeon RX 9070 XT (gfx1201 (RDNA 4))
  Model:     Qwen2.5 7B Instruct (Q4_0, 4.1 GB)
  WMMA:      active (GEMM + Attention)
  VRAM:      8.9 / 15.9 GB used
  ─────────────────────────────────────
```

Type a message at the `>` prompt. `/help` lists commands, `/quit`
exits. Full CLI reference: [`docs/cli-guide.md`](docs/cli-guide.md).

## Ubuntu 24.04 / Debian 13 (untested)

These instructions follow the official ROCm installation guide but
have not been exercised on ROCmForge. If you try this path, please
file a GitHub issue with the result.

### 1. Install ROCm

Follow the official instructions at
<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>.
Typical flow (adapt to the current stable release):

```bash
# Add AMD repository per official docs (varies by release)
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs
```

### 2. Install Rust and build

Same as Arch (sections 2–5 above).

### 3. Likely friction points

- ROCm may install to `/opt/rocm-7.2.2/` instead of `/opt/rocm/`.
  If the build fails at the HIP kernel compile step, try:
  ```bash
  export ROCM_PATH=/opt/rocm-7.2.2
  export PATH=$ROCM_PATH/bin:$PATH
  cargo clean && cargo build --release --features gpu
  ```
- `hipcc` may not be in `PATH`. Prepend `/opt/rocm/bin` (or the
  versioned directory) to your shell profile.
- Ubuntu's default kernel may need the `amdgpu-dkms` package. Again,
  follow the official ROCm instructions.
- User must be in `render` and `video` groups.

## Fedora 41+ / RHEL 10 (untested)

### 1. Install ROCm

Follow <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>.
Typical flow:

```bash
sudo dnf install rocm-hip-sdk rocm-libs
```

### 2. Likely friction points

- SELinux may block GPU access. `dmesg | grep -i avc` shows denials.
  A permissive mode or correct context is typically required.
- Fedora's default kernel may need `linux-firmware` updates for
  current AMD GPUs. Reboot after installing.
- Same PATH / group considerations as on Ubuntu.

## Troubleshooting

### Build fails — "hipcc not found"

```bash
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
```

Add these to your shell profile. On distros that use a versioned path,
substitute `/opt/rocm-7.2.2`.

### Build fails — "WMMA intrinsic not found" / LLVM headers missing

Your ROCm install may not include gfx12 support. Verify:

```bash
grep -r "wmma_f32_16x16x16_f16_w32_gfx12" /opt/rocm/lib/llvm/ | head -3
```

If nothing matches, update to ROCm ≥ 7.2. On older ROCm, gfx12
intrinsics simply don't exist and no amount of patching can fix that
from the ROCmForge side.

### Runtime error — "No GPU detected" / `hipGetDeviceCount` returns 0

```bash
rocminfo | grep "Name:.*gfx"
```

If nothing shows up:

1. `lsmod | grep amdgpu` — is the module loaded?
2. `groups $USER` — do you see `render` and `video`?
3. `ls -l /dev/kfd /dev/dri/render*` — are the device files readable
   by your user?

If the groups are missing, add them and log out / back in:

```bash
sudo usermod -a -G render,video $USER
```

### Runtime warning — "feature flag wmma_prefill=false" / scalar fallback

```bash
RUST_LOG=rocmforge::gpu::ops=debug ./target/release/rocmforge \
    --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
    --prompt "Hello" --max-tokens 5 --gpu 2>&1 | grep "GEMM dispatch"
```

You should see `path="wmma_q4_0"`, `path="wmma_q4_0_fused_gate_up"`,
and (for mixed-precision models) `path="wmma_q4_1"`. If you see
`path="gemv"` or `path="hipblas"` for prefill dispatches, either:

- your GPU is not gfx1201 (the WMMA path only engages there), or
- you have one of the `ROCMFORGE_DISABLE_WMMA_*` environment flags
  set. Check `env | grep ROCMFORGE_`.

### VRAM — "out of memory"

Qwen2.5-7B Q4_0 needs ~8.9 GB of VRAM. A running browser with GPU
acceleration, desktop compositor, or another ML process can easily
consume the rest. Check:

```bash
rocm-smi --showmeminfo vram
```

Free VRAM before running the 7B model. The 0.5B model needs only
~0.6 GB and is useful for testing in constrained environments:

```bash
./target/release/rocmforge chat \
    --model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf
```

### Chat CLI — Ctrl+C quits the whole terminal

That means the signal handler installation failed. The chat CLI uses
`ctrlc::set_handler` to catch SIGINT and interrupt generation without
exiting. If the handler cannot register (for example, a parent shell
swallows signals), you'll get the default behaviour. Run in a plain
terminal outside `tmux` / `screen` as a first test.

### `cargo build` is stuck at "Compiling rocmforge (build script)"

The build script invokes `hipcc` and `cmake` to compile the HIP
kernels. On a cold build this takes ~60 seconds. If it hangs longer
than 2 minutes:

- `ps auxf | grep -E 'cmake|hipcc'` — confirm something is running.
- Check `target/build/` for CMake output files.
- Re-run with `CARGO_LOG=cargo::util::command=debug` to see exact
  commands.

## Next steps

- [CLI guide](docs/cli-guide.md) — chat commands, one-shot inference,
  slash commands, environment flags, logging.
- [Changelog](CHANGELOG.md) — what changed in each phase.
- [Phase 4 final analysis](benches/results/phase4_final_analysis.md)
  — full performance breakdown and optimisation arc.
