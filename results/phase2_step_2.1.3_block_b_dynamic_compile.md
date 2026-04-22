# Phase 2 Schritt 2.1.3 Block B — Dynamic Kernel Compile + Load Infrastructure

**Date:** 2026-04-22
**Branch:** v1.0-dev (on top of 2.1.3 Block A `5bc635e`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Scope:** Thin vertical slice — hipcc runtime invocation, `HipModule`
FFI + RAII wrappers, und ein `num_waves`-parametrischer
`gate_up_swiglu`-Codegen. Weitere Tile-Config-Achsen
(`tile_m`, `lds_strategy`, `unroll`, FP8-GEMV) folgen in späteren
Sessions; Block B ist der End-to-End-Beweis der Pipeline.

## Kurzfassung

Die GA kann jetzt **echte Kernel** kompilieren und ausführen:

```
KernelGenome(num_waves=N)
   → emit_q4_k_gemv_gate_up_swiglu_parametric(N) = (source, symbol)
   → compile_hip_source(source, kernel_name)
     → hipcc --offload-arch=gfx1201 --genco -O3 → bundle
     → clang-offload-bundler --unbundle → gfx1201 AMDGPU ELF
   → HipModule::load(co_bytes)
   → module.get_function(symbol) → HipFunction
   → DynamicKernel::launch_gate_up_swiglu(weights, input, output, …)
   → GPU-Execution
```

**Kritischer Gate bestanden:** Dynamischer Kernel mit `num_waves=8`
liefert **bit-identischen Output** zum statischen Phase-1-Kernel
(max_abs_err = 0.000e0). Verschiedene `num_waves` produzieren
verschiedene Timings (`num_waves=2 → 56 µs, num_waves=8 → 69 µs`).

Tests: **10/10 grün** (3 CPU + 7 GPU). Regression auf framework,
parity, fp8, wmma, v0.x — alle grün.

## Neue / Geänderte Dateien

| Datei | LOC | Was |
|---|---:|---|
| `src_v1/backend/gpu/hip_ffi.rs` | +40 | `hipModule_t`, `hipFunction_t`, 4 FFI-Bindings (`hipModuleLoadData` / `GetFunction` / `LaunchKernel` / `Unload`) |
| `src_v1/backend/gpu/module.rs` | 122 | **NEU** — `HipModule` (RAII, Drop→Unload), `HipFunction<'m>` (Lifetime-bound) + `launch()` |
| `src_v1/backend/gpu/mod.rs` | +1 | `pub mod module;` |
| `src_v1/ga/compile.rs` | +170 | `CompileError`, `find_hipcc()`, `compile_hip_source()` mit `--genco` + `clang-offload-bundler --unbundle` |
| `src_v1/ga/dynamic.rs` | 153 | **NEU** — `DynamicKernel`, `GateUpSwigluGeometry`, `launch_gate_up_swiglu()` mit `**void` arg-marshalling |
| `src_v1/ga/mod.rs` | +5 | GPU-gated re-exports |
| `src_v1/ir/codegen_gpu.rs` | +150 | `emit_q4_k_gemv_gate_up_swiglu_parametric(num_waves)` + `ga_gate_up_swiglu_symbol(num_waves)` |
| `tests_v1/dynamic_kernel_test.rs` | 447 | **NEU** — 10 Tests (3 CPU + 7 GPU) |
| `Cargo.toml` | +9 | `[[test]] v1_dynamic_kernel_test` |

**Neue Code-LOC:** 886. Neue Test-LOC: 447.

## Component 1 — hipcc Runtime-Invocation

`find_hipcc()` sucht in Reihenfolge:
1. `$HIP_PATH/bin/hipcc`
2. `$ROCM_PATH/bin/hipcc`
3. `/opt/rocm/bin/hipcc` (Arch/CachyOS Default)
4. `hipcc` im `$PATH`

Auf dem Dev-System: `/opt/rocm/bin/hipcc` (verifiziert durch
`test_hipcc_available`).

### hipcc-Flags + zweistufige Extraktion

Die erste naive Lösung (`hipcc --offload-arch=gfx1201 -shared -fPIC -O3`)
produzierte ein **Host-Shared-Object** mit embedded Offload-Bundle —
`hipModuleLoadData` lehnte das ab mit `hipErrorInvalidImage (200)
— "device kernel image is invalid"`. `hipModuleLoadData` erwartet
ein nacktes **AMDGPU-ELF**, nicht ein Host-Wrapper.

**Finale Lösung (zweistufig):**

```rust
// Stage 1: hipcc produziert einen Offload-Bundle
hipcc --offload-arch=gfx1201 --genco -O3 -o <kernel>.co <kernel>.hip

// Stage 2: clang-offload-bundler extrahiert den gfx1201-Teil
clang-offload-bundler \
    --type=o \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
    --input=<kernel>.co \
    --output=<kernel>.gfx1201.co \
    --unbundle
```

Der extrahierte `<kernel>.gfx1201.co` ist ein AMDGPU-ELF (`file` zeigt
"ELF 64-bit LSB shared object, AMD GPU architecture version 1,
dynamically linked, not stripped"), den `hipModuleLoadData`
akzeptiert.

### Compile-Zeiten (median über `num_waves ∈ {1, 2, 4, 8}`)

| Szenario | Wallclock |
|---|---:|
| Trivial `__global__ void noop()` | ~400 ms |
| `gate_up_swiglu` parametrisch | ~1.0 s |
| 4× parametric (alle num_waves) inkl. Load + Symbol-Lookup | 3.92 s |

**Compile-Zeit pro Kernel ≈ 1 Sekunde** (hipcc Launch + LLVM Backend
+ Bundler). Das sprengt den §2.7.1-Budget-Rahmen (50–100 ms
erwartet) — für Block C wird entweder
- Per-Compile-Cache auf Disk (die `.gfx1201.co`-Dateien werden
  bereits nicht gelöscht — ein File-Hash-Cache wäre trivial
  nachzuschieben), oder
- Parallele hipcc-Invocation über Rayon
…in `§2.7.2` kommen. Für Block B ist der sequenzielle Pfad OK,
weil der Vertical-Slice-Test bewusst nur wenige Kernel kompiliert.

### CompileError-Taxonomie

```rust
pub enum CompileError {
    Io(io::Error),
    HipccNotFound,
    HipccFailed { kernel_name: String, stderr: String, status: Option<i32> },
    MissingCodeObject(PathBuf),
}
```

Fehler propagieren den hipcc-Stderr wörtlich — `test_compile_error_reported`
verifiziert, dass syntaktisch kaputtes HIP den Stderr liefert (nicht
Panic, nicht stilles Schlucken).

## Component 2 — `HipModule` / `HipFunction` FFI + RAII

Vier neue FFI-Bindings in `hip_ffi.rs`:

```rust
pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
pub fn hipModuleGetFunction(function: *mut hipFunction_t, module: hipModule_t,
                            name: *const c_char) -> hipError_t;
pub fn hipModuleLaunchKernel(
    f: hipFunction_t,
    gridDimX: u32, gridDimY: u32, gridDimZ: u32,
    blockDimX: u32, blockDimY: u32, blockDimZ: u32,
    sharedMemBytes: u32,
    stream: hipStream_t,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> hipError_t;
pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
```

RAII in `src_v1/backend/gpu/module.rs`:

- `HipModule` — owns `hipModule_t`, `Drop` → `hipModuleUnload`
- `HipFunction<'m>` — Lifetime-bound auf das Parent-Modul
  (kein Drop nötig; die Funktion wird durch `hipModuleUnload`
  invalidiert)
- `HipFunction::launch(grid, block, shared, stream, args)` —
  argumenten-Array-Layout folgt HIP-Konvention (`**void` pointing
  at each argument's value)

## Component 3 — Parametrischer Codegen (num_waves)

`emit_q4_k_gemv_gate_up_swiglu_parametric(num_waves: u32) -> (String, String)`
ist eine **1:1-strukturelle Kopie** von `emit_q4_k_gemv_gate_up_swiglu`
mit nur einem substanziellen Unterschied:

```c
#define Q4_K_FIXED_WAVES       {num_waves}   // war: 8 (konstant)
```

Plus:
- `extern "C" __global__` auf dem Kernel (damit
  `hipModuleGetFunction` den Symbolnamen unmangled findet)
- Eindeutiger Symbolname pro Config: `rf_v1_ga_gate_up_swiglu_w{N}_kernel`
- Eindeutige Helper-Namen (`rf_v1_ga_*` statt `rf_v1_gu_*`) um
  Kollisionen mit dem statischen Kernel zu vermeiden
- Kein extern "C" launcher-Wrapper (wird nicht gebraucht — wir
  rufen direkt `hipModuleLaunchKernel` auf)

**Kernel-Logik unverändert.** Der Dequant-Loop, die Super-Block-
Dot-Product-Funktion, die SwiGLU-Sequenz, der Warp-Reduce — alles
Byte-gleich zum Phase-1-Kernel. Nur die Werte, nicht die Struktur,
werden parametrisiert.

## Component 4 — `DynamicKernel` + `launch_gate_up_swiglu`

```rust
pub struct DynamicKernel {
    module: HipModule,
    symbol: String,
    geometry: GateUpSwigluGeometry,
}

pub struct GateUpSwigluGeometry {
    pub num_waves: u32,
    pub multi_row_cols: u32, // 4 (fest für Block B)
    pub warp_size: u32,      // 32 (gfx1201)
}
```

Launch-Parameter werden aus der `geometry` abgeleitet:
- `threads_per_block = num_waves × 32`
- `cols_per_block = num_waves × 4`
- `grid_x = ⌈ncols_dst / cols_per_block⌉`
- `shared_mem_bytes = n_rows × 4` (FP32-Input-Staging)

Argument-Marshalling für `hipModuleLaunchKernel` — 6 Args × `*mut c_void`
pointer-to-value:

```rust
let gate_ptr = weights_gate.as_ptr();
// ... und analog für up/in/out
let mut args: [*mut c_void; 6] = [
    &gate_ptr as *const _ as *mut c_void,  // *mut *const c_void
    &up_ptr   as *const _ as *mut c_void,
    &in_ptr   as *const _ as *mut c_void,
    &out_ptr  as *const _ as *mut c_void,
    &n_rows   as *const _ as *mut c_void,  // *mut *const i32
    &ncols    as *const _ as *mut c_void,
];
func.launch(grid, block, shared, stream, &mut args)?;
```

Die lokalen `*_ptr`-Variablen leben lang genug — sie sind
Funktions-Stack-Variablen, die den `launch()`-Call überdauern.

## Ergebnisse

### Critical Gate: Dynamisch == Statisch bei num_waves=8

```
dynamic(w=8) vs static: max_abs_err = 0.000e0
```

**Byte-für-Byte identisch.** Das ist das fundamentale Safety-Gate
— wenn es brechen würde, wäre der parametrische Codegen strukturell
defekt. Test: `test_dynamic_equals_static_kernel`.

### Parametrisierung wirkt: verschiedene num_waves → verschiedene Timings

```
gate_up_swiglu median (n_rows=4096, ncols=512):
  num_waves=2 → 56 µs
  num_waves=8 → 69 µs
  → Ratio 1.23× (>5 % Gate bestanden)
```

Test: `test_different_num_waves_different_timings`. Die 1.23× weisen
darauf hin, dass die GA-Search-Space `num_waves ∈ {1, 2, 4, 8}`
tatsächlich unterschiedliche Kernel-Binaries produziert und die
Launch-Geometrie unterschiedlich wirkt (nicht nur den Cache).
`num_waves=2` ist auf dieser Shape schneller — der GA würde das in
Block C finden.

### Post-Compile-VGPR-Read auf dynamischen `.co`

| num_waves | .co Bytes | VGPRs | SGPRs | Waves/CU |
|---:|---:|---:|---:|---:|
| 1 | 72 192 | 189 | 17 | 8 |
| 2 | 72 448 | 189 | 16 | 8 |
| 4 | 72 448 | 189 | 16 | 8 |
| 8 | 72 448 | 189 | 16 | 8 |

Identische VGPR-Zahl über alle Configs (erwartet — der Kernel-Body
ist unverändert, `num_waves` beeinflusst nur Launch-Geometrie). Der
2.1.1 `parse_amdgpu_metadata`-Pfad funktioniert unverändert auf den
dynamisch produzierten ELFs.

### Stubbed-Lookup, dann launch: Trivial-Kernel

```
noop .co: 3952 bytes
Fill-Kernel (writes 0xCAFE to 128 slots) — all 128 slots verified.
```

Tests `test_compile_trivial_kernel` + `test_module_launch_trivial_kernel`
decken den Minimalpfad separat ab — wenn die echten Tests brechen,
isoliert dieser Minimalpfad sofort, ob der Fehler im Codegen oder in
der Infrastruktur liegt.

## Test-Ergebnisse

### Dynamic-Kernel-Tests (10/10, alle grün)

| Test | Scope |
|---|---|
| `test_hipcc_available` | CPU-only — find_hipcc |
| `test_parametric_symbol_unique_per_num_waves` | CPU — Symbol-Mangling |
| `test_parametric_source_contains_symbol_and_num_waves` | CPU — Emit-Content |
| `gpu_tests::test_compile_trivial_kernel` | GPU — compile → .co |
| `gpu_tests::test_compile_error_reported` | GPU — error propagation |
| `gpu_tests::test_module_launch_trivial_kernel` | GPU — end-to-end fill → verify |
| `gpu_tests::test_parametric_codegen_hipcc_compiles` | GPU — alle num_waves kompilieren + loaden |
| `gpu_tests::test_dynamic_equals_static_kernel` | GPU — **bit-parity** |
| `gpu_tests::test_different_num_waves_different_timings` | GPU — Parametrisierung wirkt |
| `gpu_tests::test_post_compile_vgpr_from_dynamic_co` | GPU — VGPR-Read auf dynamischem .co |

### Regression

| Suite | Status |
|---|:---:|
| `v1_ga_framework_test` (30 Tests) | ✅ |
| `v1_ga_parity_test` (21 Tests) | ✅ |
| `v1_fp8_pair_packing_test` (5 Tests) | ✅ |
| `v1_wmma_test` (15 Tests inkl. FP8 Parity + Performance) | ✅ |
| `v1_codegen_wmma_emit_test` (Drift-Check) | ✅ |
| lib `v1::ga::` inline (23 Tests) | ✅ |
| v0.x Build (`cargo build --release --features gpu`) | ✅ |

## Design-Entscheidungen

- **Zweistufige Extraktion (hipcc + clang-offload-bundler) statt direkter ELF-Emission.**
  `--genco` produziert einen `__CLANG_OFFLOAD_BUNDLE__`-Wrapper;
  `hipModuleLoadData` lehnt das ab. Der Bundler ist in jeder
  ROCm-Installation vorhanden (`/opt/rocm/lib/llvm/bin/clang-offload-bundler`
  auf Arch, System-wide als Fallback) — also keine neue Abhängigkeit,
  nur ein zweiter Subprozess. Vorteil: wir bekommen das "natürliche"
  hipcc-Output-Format und müssen nicht mit `-fgpu-rdc` und `--hip-link`
  jonglieren.

- **`extern "C" __global__` auf dem Kernel.** Ohne das würde HIP
  den `__global__`-Symbol C++-mangeln und `hipModuleGetFunction`
  nach dem unmangled Namen schlagen. Das `extern "C"` macht den
  Symbolnamen deterministisch und beobachtbar (steht 1:1 im emittierten
  Source — im Test per `test_parametric_source_contains_symbol_and_num_waves`
  verifiziert).

- **Eindeutige Symbol- und Helper-Namen pro Config
  (`rf_v1_ga_*_w{N}_kernel`).** Der statische Phase-1-Kernel
  verwendet `rf_v1_gemv_q4_k_gate_up_swiglu_kernel` und Helper
  `rf_v1_gu_*`. Der parametrische emittiert `rf_v1_ga_*` — keine
  Namenskollision wenn jemand später beide im selben Modul linken
  will.

- **`HipFunction<'m>` statt `HipFunction`-mit-Drop.** HIP hat kein
  `hipFunctionDestroy` — Kernel sind Handles in den Modul-Tabellen,
  werden beim `hipModuleUnload` invalidiert. Rust's Lifetime-System
  drückt das präziser aus als ein Drop-Hook.

- **Temporäre Dateien bleiben auf Disk.** `compile_hip_source`
  räumt den `$TMPDIR/rocmforge_ga/`-Inhalt nicht auf. Grund: Block
  C wird einen File-Hash-basierten Compile-Cache brauchen (unser
  CompileCache aus 2.1.1 ist ein In-Process-Cache; für Multi-Run-
  Amortisation brauchen wir die Bytes auf Disk). Den Aufräumpfad
  kann man nach der Cache-Persistenz nachziehen, falls nötig.

- **Kein FP8-Pfad in Block B.** Die `GateUpSwigluGeometry` und der
  Codegen-Emitter enthalten **keinen** FP8-Branch. Der bestehende
  gate_up_swiglu ist FP32-only (scalar accumulation, nicht WMMA);
  FP8-GEMV ist ein from-scratch-Emitter (~400 LOC) und kommt in
  einer Folge-Session. Der Block-A-Fix (FP8 Pair-Packing) wirkt
  daher heute nur auf WMMA-Prefill — das ist bewusster Trade-off.

## Beschränkungen (Follow-up-Scope)

Was Block B **nicht** liefert (explizit deferred):

1. **Weitere `TileConfig`-Achsen.** `tile_m`, `tile_n`, `k_chunk`,
   `lds_strategy`, `unroll_factor`, `prefetch_depth`, `double_buffer`
   werden vom Codegen entgegengenommen, aber ignoriert. Der GA kann
   sie in Genomen variieren; nur `num_waves` ändert tatsächlich den
   emittierten Code. Follow-up: entweder den bestehenden Kernel für
   jede Achse parametrisieren (graduell), oder einen IR-getriebenen
   Emitter bauen.

2. **FP8-GEMV.** Der gate_up_swiglu-Kernel läuft FP32-scalar; Block
   A hat FP8 pair-packing nur für den WMMA-GEMM-Pfad addressiert.
   Der echte FP8-Decode-Gewinn (aus 2.0.3 Arch-Prioritäten) braucht
   einen dedizierten FP8-GEMV-Emitter.

3. **Compile-Cache auf Disk.** Die
   `$TMPDIR/rocmforge_ga/*.gfx1201.co`-Dateien werden nicht
   hashgelesen — ein zweiter Lauf re-kompiliert. Trivial nachzurüsten.

4. **Parallele hipcc-Invocation.** `§2.7.1` schlägt 8 Threads vor;
   Block B ist sequenziell. Wird in Block C relevant wenn Pop 100 ×
   Gen 50 × ~1 s Compile = ~1 Stunde sequenziell nicht in das
   8-min-Budget passt.

5. **Echter GA-Lauf.** Block C.

## Was Block C ermöglicht

Mit Block B's Infrastruktur kann ein Mini-GA-Lauf (Pop 30, Gen 15)
auf `num_waves ∈ {1, 2, 4, 8}` jetzt tatsächlich funktionieren:

- `KernelGenome::random()` → `num_waves` aus Legal-Set
- `sanitize_for(Q4_K, Fp16)` → clamped auf 1/2/4/8
- `evaluate_kernel_fitness()` (real GPU statt Toy):
  1. `validate_pre_compile` → ok
  2. `compile_hip_source` (~1 s) → `.co` Bytes
  3. `parse_amdgpu_metadata` → VGPRs 189, waves/CU=8 → Post-Compile-Gate passes
  4. `check_parity_known_kernel` (angepasst auf dynamische Kernel) → max_err=0 gegen
     `valu_reference_gemv`
  5. Warmup 5 + Benchmark 20 Runs → median_us
  6. Fitness = baseline(432.8 µs) / median_us

Der Search-Space ist heute nur 1-D (num_waves), also wird der GA
schnell konvergieren — typisch in < 5 Generationen. Das ist OK für
einen ersten End-to-End-Beweis; echte Optimierung braucht die
weiteren Achsen aus Follow-ups.

## Commit

Prefix: `feat(v1/ga):`

```
feat(v1/ga): Phase 2 step 2.1.3 Block B — dynamic compile + load
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
