# Phase 6 Step 2 — Fused FFN + Q4_1 GEMV Check

**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2
**Model:** Qwen2.5-7B-Instruct Q4_0
**Ground truth:** 102.5 tok/s

**Outcome:** Q4_1 GEMV is already correct. The fused norm+gate+up+SwiGLU
kernel exists but is broken when routed into the active decode path —
reverted, documented, handed to Step 3.

---

## 1. Q4_1 GEMV dispatch check (2-minute verification)

**Finding: no bug, no fix needed.**

Dispatch log at decode (`RUST_LOG=rocmforge::gpu::ops=debug`):

```
GEMM dispatch: decode-path GEMV seq_len=1 n=3584 k=18944 wtype=Q4_1 path="gemv_decode"
```

The 3 Q4_1 `ffn_down` layers dispatch to `gemv_q4_1_f32_residual_on_stream_*`
(with autotune support in `src/gpu/ops.rs::gpu_dispatch_gemv_residual_on_stream`).
A real Q4_1 GEMV kernel is already in `hip_kernels/quant/q4_1_gemv.hip` —
no scalar fallback, no hidden overhead. The suspicion in the Step 1
analysis that "the 3 Q4_1 layers might be on a slow fallback" is ruled
out.

---

## 2. Fused FFN routing attempt — blocked by correctness bug

### 2.1 Kernel located

- Source: `hip_kernels/quant/q4_0_fused_norm_gate_up.hip`
- FFI wrapper: `gemv_norm_gate_up_swiglu_q4_0_f32_on_stream`
  (`src/gpu/kernels/quant.rs:2728`)
- Rust dispatcher: `gpu_dispatch_fused_norm_gate_up_on_stream`
  (`src/gpu/ops.rs:1305`)

Kernel signature:
```
fused_norm_gate_up(raw_hidden, norm_weight, eps,
                   w_gate_Q4_0, w_up_Q4_0, out_swiglu,
                   n_rows, n_ff)
```

Phases inside one kernel launch:
1. RMSNorm over `raw_hidden` into shared memory
2. Quantise the normalised activation to Q8_0 in shared memory
3. Q4_0 × Q8_0 Gate + Up GEMV with inline SiLU(gate) * up → write `out_swiglu`

### 2.2 Preconditions match Qwen2.5-7B

The dispatcher bails out early if any precondition fails:

| Check | Qwen2.5-7B value | Pass? |
|---|---|---|
| `gate_meta.wtype == Q4_0` | Q4_0 | ✅ |
| `up_meta.wtype == Q4_0` | Q4_0 | ✅ |
| `h % 32 == 0` | 3584 | ✅ |
| `ff_size % 4 == 0` | 18944 | ✅ |
| `shared_mem ≤ 32768` | 14464 B | ✅ |

All preconditions pass. The kernel should be usable on this model.

### 2.3 Where the kernel is currently used

The kernel is wired into `gpu_layer_forward_from_state_on_stream`
(`src/gpu/forward.rs:2065`). That function is only called by
`gpu_launch_full_greedy_decode_on_stream`, which is only called during
full-decode graph capture — a path that is **unconditionally
short-circuited** in `gpu_try_full_greedy_decode_graph`
(line 2192: `return Ok(None);`) because of the known RDNA-4 device-pointer
stale-read bug documented in `hip_graph_device_pointer_bug.md`.

**Net effect: the fused kernel was never executed** on this hardware.
It compiled, it existed, it was dead code in practice. No test covers
it.

### 2.4 Routing attempt

Replaced the separate `FfnNorm` + `GateUp` dispatches in the active
`gpu_layer_forward_hybrid` with a try of
`gpu_dispatch_fused_norm_gate_up_on_stream`, gated by an opt-out env
flag `ROCMFORGE_DISABLE_FUSED_FFN`. Minimal diff, just swapped two
`profile_decode_stage` calls for one.

### 2.5 Result: correctness bug

First token is correct, output diverges catastrophically from token 2
onward:

| Tokens | Fallback (separate norm + gate_up) | Fused (new path) |
|---|---|---|
| 1 | `,` | `,` |
| 2 | `, I` | `,!` |
| 3 | `, I'm` | `,!!` |
| 5 | `, I'm trying to` | `,!!!!` |
| 10 | `, I'm trying to create a function that takes` | `,!!!!!!!!!` |
| 20 | `, I'm trying to create a function that takes a string and returns a list of all the words` | `,!!!!!!!!!!! strugg!!!!!!!` |

The `!` token is ID 0 or ID 1 — the argmax collapses to a low-vocab
token after one good step. This is not FP16 accumulation drift (which
would show fluent drift, not character spam); it's a real
state-corruption bug.

Step 1's first token matches, so the initial RMS stats, Q8 quantisation,
and Q4_0 × Q8_0 dot products are producing the right per-layer FFN
output on the very first activation vector. Something later — probably
between decode steps — corrupts state.

**Candidates** (not confirmed, this is a kernel-level debug task):

- Shared-memory aliasing. The kernel reinterprets the same `s_raw`
  buffer as `float[n_rows]` in Phase 1 and `Q8_0_block[n_blocks]` in
  Phase 2. The two layouts overlap byte-wise (`Q8_0_block` = 34 B
  stride vs. `f32` = 128 B stride), so the first few Q8 writes land
  inside the f32 region that Phase 2 is simultaneously reading from.
  The author's `__syncthreads()` and "read-into-registers-first"
  comment suggests they were aware of the hazard, but they may only
  have tested on a smaller `n_rows` / different wave count where the
  race didn't fire.
- Missing `__syncwarp()` after the `__shfl_down` / `__shfl` reductions
  — gfx12 wave-level shuffle semantics on RDNA 4 can differ from
  what the kernel was originally tested on (RDNA 3?).
- Inter-block norm coherence. Each thread block recomputes `inv_rms`
  from its own `raw_hidden` read into its own shared memory. If the
  reads race with writes to `hidden` from some other in-flight kernel
  on the stream, different blocks could see different norm statistics.
  This would explain step 1 being OK (hidden is stable at that point)
  and step 2+ diverging (hidden is being written/read around this
  kernel).

Debugging any of these requires isolated kernel testing with a
controlled activation vector — out of scope for this step.

### 2.6 Reverted

The routing change was reverted. Zero code changes from this step
land on disk. `git diff HEAD src/gpu/forward.rs` → empty.

---

## 3. Next-step recommendation

**Step 3 should be direct GEMV kernel tuning for `gate_up` + `ffn_down`**
(5 ms combined = 52 % of the decode budget — Step 1 Table 2).
Concrete levers:

1. **Re-tune `Q4_0_FIXED_WAVES` and `Q4_0_MULTI_ROW_COLS`** for the
   Qwen2.5-7B ncols_dst = 18,944 (gate/up) and 3,584 (down) shapes.
   The current 8-wave / 4-col heuristic has a comment noting a 0.5B
   regression with 4-wave; nothing has been tried for 7B.
2. **Activate the existing `ffn_gate_up_interleaved` buffer** in the
   decode path. It's built at model load but only consumed by the Q8
   fastpath today. Interleaved Q4_0 blocks halve the L2 round-trips
   for the shared input.
3. **Compare against hipBLAS-Hgemv** as a ceiling. If we're within
   10 % of the bandwidth-limited floor, further GEMV tuning is
   yield-diminishing and the decode-path HIP-graph workaround (the
   other Step 1 recommendation) becomes the only remaining large
   lever.

**Not recommended:** fixing the fused-norm kernel in-place. Its
structural bug needs a targeted correctness pass (probably an isolated
CPU-reference test with hand-constructed activations), which is a
standalone ticket.

---

## 4. Kurz-Report

**Phase 6 Step 2 — Fused FFN + Q4_1-GEMV-Check:**

- **Q4_1-GEMV-Status: Kernel vorhanden, korrekt dispatcht** (kein Bug, 3× `wtype=Q4_1 path="gemv_decode"` im Log). Keine Code-Änderung nötig.
- **Fused-Kernel-Status: Q4_0-kompatibel, aber Correctness-Bug beim Routing in den aktiven Decode-Pfad.** First token identisch, ab Token 2 kollabiert die Ausgabe in `!`-Spam. Routing-Änderung reverted.
- **Launch-Count: unverändert 255 pro Token** (kein Merge).

| Metrik | Vorher | Nachher | Veränderung |
|---|---:|---:|---|
| Decode tok/s | 102.5 | 102.5 | ±0 (revert) |
| ffn_norm + gate_up µs (bereinigt) | 3,467 | 3,467 | unchanged |
| Q4_1 ffn_down Dispatch | `gemv_q4_1` | `gemv_q4_1` | bereits korrekt |
| Launches pro Token | 255 | 255 | unchanged |

- **Q4_1-Fix: nicht nötig** — die 3 Q4_1 `ffn_down`-Layer laufen bereits auf einem echten GEMV-Kernel mit Autotune-Support.
- **Fused-Kernel Blocker:** `q4_0_fused_norm_gate_up.hip`-Kernel hat ein Zustandskorruptions-Bug (Verdacht: Shared-Memory-Aliasing in Phase 2 oder fehlende `__syncwarp` nach den Reduktionen auf RDNA 4). Nie gegen einen Korrektheitstest exekutiert, weil der einzige Aufrufer (`gpu_launch_full_greedy_decode_on_stream`) im deaktivierten Graph-Capture-Pfad lebt.
- **Nächster Schritt: Schritt 3 — direktes GEMV-Tuning** (`gate_up` + `ffn_down` Wave/Column-Heuristik, `ffn_gate_up_interleaved` aktivieren, hipBLAS-Hgemv als Bandwidth-Ceiling-Referenz). Der Fused-Kernel-Bug bleibt als eigenständiges Ticket offen.
