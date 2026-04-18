# `q4_0_fused_norm_gate_up.hip` — state corruption from token 2+

**Status:** OPEN — not blocking. Workaround: keep the separate
`rms_norm` + `fused_gate_up` kernels on the active decode path.

**Discovered:** 2026-04-18 during Phase 6 Step 2.

## Summary

The fused RMSNorm + Gate + Up + SwiGLU kernel in
`hip_kernels/quant/q4_0_fused_norm_gate_up.hip` produces correct output
for the **first** decode token but corrupts state from token 2 onward.
The model's argmax collapses to low-vocab tokens (`!` on Qwen2.5), so
the generation degenerates into character spam after a single coherent
step.

## Reproduction

1. Route the active decode path through
   `gpu_dispatch_fused_norm_gate_up_on_stream`. The exact change tried
   in Phase 6 Step 2 is captured in commit `742091a` (investigation
   report, no code change); the routing diff that triggered the bug
   was reverted before commit. To re-trigger: in
   `src/gpu/forward.rs::gpu_layer_forward_hybrid`, replace the
   sequential `DecodeStage::FfnNorm` + `DecodeStage::GateUp` calls
   with a single call to `super::ops::gpu_dispatch_fused_norm_gate_up_on_stream`
   and use `scratch.swiglu` as the output.

2. Build and run:
   ```
   cargo build --release --features gpu
   ./target/release/rocmforge \
       --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
       --prompt "Hello" --max-tokens 20 --gpu \
       --temperature 0.0 --top-p 1.0 --no-template
   ```

3. Observed output at 20 tokens:
   ```
   ,!!!!!!!!!!! strugg!!!!!!!
   ```
   Unfused path produces:
   ```
   , I'm trying to create a function that takes a string and returns a list of all the words
   ```

The first comma is correct in both cases. The divergence starts at
token 2 and propagates.

## Why the bug was never caught before

The kernel was wired into `gpu_layer_forward_from_state_on_stream`
(`src/gpu/forward.rs:2065`), which is only called by
`gpu_launch_full_greedy_decode_on_stream` during full-decode graph
capture. That capture path is unconditionally short-circuited in
`gpu_try_full_greedy_decode_graph` (`src/gpu/forward.rs:2192`,
`return Ok(None);`) because of the separate RDNA-4 device-pointer
stale-read bug documented in `hip_graph_device_pointer_bug.md`.

Net effect: this kernel has never executed on real hardware on this
project. No unit test exercises it.

## Root-cause candidates (unconfirmed)

1. **Shared-memory aliasing between Phase 1 and Phase 2.** The kernel
   reinterprets `s_raw` as both `float[n_rows]` (Phase 1, RMS norm)
   and `Q8_0_block[n_blocks]` (Phase 2, activation quantisation) with
   the same base pointer. `float` has 128 B stride per block, `Q8_0_block`
   has 34 B stride, so the two layouts overlap byte-wise at block 0.
   The author's `__syncthreads()` before Phase 2 plus the
   "read-into-registers-first" comment at line 128 suggest they were
   aware of the hazard — but on RDNA 4 with 8 waves × 32 threads the
   interleaving of reads and writes across waves may still race.

2. **Missing `__syncwarp()` after `__shfl_down` / `__shfl` reductions.**
   The kernel uses wave-level shuffles for sum-reduction (`local_sum_sq`)
   and for broadcasting the quantisation scale. On RDNA 4 the shuffle
   semantics can differ from RDNA 3 where the kernel was presumably
   developed — in particular, non-synced shuffles inside a loop may
   not produce deterministic results.

3. **Inter-block norm-statistics divergence.** Each thread block
   independently reads `raw_hidden` into its own shared memory and
   computes its own `inv_rms`. Different blocks should compute the
   same value, but if `scratch.hidden` is being written by an earlier
   kernel on the same stream that is not yet complete when the fused
   kernel launches, different blocks would observe different hidden
   values and diverge. The `device.stream()` is supposed to serialise
   these launches, so this is low-probability, but worth ruling out
   with a `hipStreamSynchronize` before the fused launch as a
   sanity check.

## Impact

At Qwen2.5-7B Q4_0 decode, the unfused path costs ~407 µs `ffn_norm` +
~3,060 µs `gate_up` = 3,467 µs per token, rescaled to the 9.76 ms
ground-truth budget. A working fused kernel would save the launch
overhead of one additional kernel dispatch (~3–5 µs per layer × 28
layers = ~100 µs) and the VRAM round-trip for `scratch.normed` (~14 KB
read + 14 KB write per layer × 28 = ~800 KB of extra L2 traffic).
Rough expected savings: **400–500 µs per decode token (~5 %)** → 102
tok/s → ~107 tok/s.

Not a blocker for shipping. The unfused path is correct and the
performance gap relative to llama.cpp is dominated by total launch
overhead (13 %), not this specific fusion.

## Recommended fix approach

A standalone correctness pass, not an attempted re-route:

1. Write an isolated correctness test with a CPU reference — 2 decode
   steps on a hand-constructed activation vector, compare fused
   vs. sequential norm + gate_up byte-for-byte. Place next to
   `tests/wmma_q4_0_correctness.rs`.

2. Instrument the kernel to dump shared memory after Phase 1 and
   Phase 2 of the first and second blocks. Compare what each block
   saw — if they diverge, root cause 3 is confirmed.

3. Add `__syncwarp()` (or `__syncthreads()` if needed) after every
   shuffle reduction and re-test. If that fixes it, root cause 2 is
   confirmed.

4. If neither works, the shared-memory aliasing is the remaining
   suspect. Either split into two allocations (Phase 1 buffer and
   Phase 2 buffer at distinct offsets, trading off shared-memory
   budget), or add a `__syncthreads()` barrier immediately before
   the Phase 2 writes begin.

5. Once the isolated test passes, re-route the active decode path
   behind the env flag `ROCMFORGE_ENABLE_FUSED_FFN=1` (opt-in) for
   a full 128-token regression run before flipping the default.

## Context links

- Phase 6 Step 1 decode profiling:
  `profiling/results/decode_profiling_analysis.md`
- Phase 6 Step 2 investigation report:
  `profiling/results/decode_fused_ffn_investigation.md`
- Commit documenting the discovery: `742091a`
