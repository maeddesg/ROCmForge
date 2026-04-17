# Profiling Results Index

Chronological index of profiling artifacts. Each file captures a specific
experiment; the analysis `.md` files interpret the raw JSON data.

## 2026-04-17 — Intermediate Buffer Traffic Validation (Fused FFN)

- [`BUFFER_TRAFFIC_ANALYSIS.md`](BUFFER_TRAFFIC_ANALYSIS.md) — Micro-benchmark verdict: fused FFN saves ~200 µs/step (1.2%), falsifying the intermediate-buffer-traffic hypothesis. Decision: do not build fused FFN.
- [`buffer_traffic_validation_{92d037a}_{1776402443}.json`](buffer_traffic_validation_%7B92d037a%7D_%7B1776402443%7D.json) — Raw 100-iter HIP-event timings across 3 variants (A separate, B1 LDS-fused, B2 cooperative-VRAM) × 4 fma_depths × 2 batches.

## 2026-04-16 — Launch-Overhead Sub-Phase Breakdown

- [`LAUNCH_OVERHEAD_ANALYSIS.md`](LAUNCH_OVERHEAD_ANALYSIS.md) — HIP Event timings for verify sub-phases (FFN 67.3%, attention 32.7%, 448 kernel launches/step). Initial estimate of fused-FFN savings (1.500–2.500 µs) that was later falsified by the buffer-traffic experiment.
- [`launch_overhead_{d8d1579}_{1776352604}_{code_01}.json`](launch_overhead_%7Bd8d1579%7D_%7B1776352604%7D_%7Bcode_01%7D.json) — Raw sub-phase timings, code prompt.
- [`launch_overhead_{d8d1579}_{1776352604}_{chat_01}.json`](launch_overhead_%7Bd8d1579%7D_%7B1776352604%7D_%7Bchat_01%7D.json) — Raw sub-phase timings, chat prompt.
- [`launch_overhead_{d8d1579}_{1776352604}_{prose_03}.json`](launch_overhead_%7Bd8d1579%7D_%7B1776352604%7D_%7Bprose_03%7D.json) — Raw sub-phase timings, prose prompt.

## 2026-04-16 — Spec-Step Cost Breakdown

- [`ANALYSIS.md`](ANALYSIS.md) — 5-phase spec-step timing (draft 10.2%, target_verify 88.6%, accept_reject 1.2%). Identified target_verify as the optimization target.
- [`spec_step_profile_{f360a32}_{1776346172}.json`](spec_step_profile_%7Bf360a32%7D_%7B1776346172%7D.json) — Raw spec-step profile data.

## 2026-04-15 — HIP Submission Latency Benchmark

- [`hip_submission_latency_d13246c.json`](hip_submission_latency_d13246c.json) — Per-dispatch submission-latency measurements across kernel depths. Basis for the "~2.7 µs dispatch overhead" figure used in subsequent cost models.

## Related artifacts outside `results/`

- **Benchmark sweeps** live in [`../../benches/results/`](../../benches/results/):
  - `batched_lm_head_sweep_2d595f8_1776356792.json` + `batched_lm_head_analysis.md`
  - `tiled_sweep_313efdf.json`
  - `depth_threshold_sweep.md`
  - `baseline_40b480e_1776343249.json`
- **Micro-benchmark source** — [`../buffer_traffic/bench.hip`](../buffer_traffic/bench.hip), [`../buffer_traffic/run.fish`](../buffer_traffic/run.fish)
- **Profiling harness scripts** — [`../profile_spec_step.fish`](../profile_spec_step.fish), [`../profile_launch_overhead.fish`](../profile_launch_overhead.fish)
- **Milestone summary** — [`../../docs/spec_decode_milestone_summary.md`](../../docs/spec_decode_milestone_summary.md)
- **Architectural findings** — [`../../docs/architecture_notes.md`](../../docs/architecture_notes.md)
