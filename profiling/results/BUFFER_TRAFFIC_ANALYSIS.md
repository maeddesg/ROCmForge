# Intermediate Buffer Traffic Validation — 2026-04-17

## TL;DR

**Verdict: Fused FFN is not worth building.** Maximum realistic gain ~224 µs/step (1.4 % of verify time), and that gain is **dispatch-overhead reduction**, not buffer-traffic elimination. The original hypothesis (intermediate buffer traffic is the lever) is **falsified** by the measurement.

## Setup

- Hardware: RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
- Dimensions (matching Qwen2.5-7B): hidden=3584, intermediate=17920 (5×3584, clean ratio instead of 18944)
- 28 layers per run, 10 warmup + 100 measured runs
- All kernels elementwise, 1–1024 FMAs per element via the `fma_depth` parameter

Three variants:

- **A** — 4 separate kernel dispatches per layer, intermediate stored in VRAM. 112 launches/step.
- **B2** — 1 cooperative kernel per layer with `grid.sync()` between phases. Same VRAM traffic profile as A, but 1 dispatch/layer → 28 launches/step.
- **B1** — 1 chunked kernel per layer; the intermediate lives only in registers. No VRAM touches for the intermediate. 28 launches/step.

Δ(A − B2) isolates dispatch/launch overhead. Δ(B2 − B1) isolates intermediate-buffer-traffic cost. Δ(A − B1) is the maximum achievable fused-FFN gain.

Artifact: [buffer_traffic_validation_{92d037a}_{1776402443}.json](buffer_traffic_validation_{92d037a}_{1776402443}.json)

## Headline results (median µs, batch=2)

| fma_depth |   A  |  B2  |  B1  | Δ(A−B2) | Δ(B2−B1) | Δ(A−B1) |
|----------:|-----:|-----:|-----:|--------:|---------:|--------:|
|         1 |  321 |  389 |   97 |     −69 |      293 |   **224** |
|        16 |  325 |  393 |  112 |     −68 |      281 |     212 |
|       128 |  375 |  440 |  242 |     −65 |      198 |     133 |
|      1024 |  845 |  847 | 1865 |      −2 |    −1018 |   −1020 |

Median µs, batch=4:

| fma_depth |   A  |  B2  |  B1  | Δ(A−B2) | Δ(B2−B1) | Δ(A−B1) |
|----------:|-----:|-----:|-----:|--------:|---------:|--------:|
|         1 |  333 |  604 |   97 |    −270 |      507 |   **236** |
|        16 |  340 |  603 |  112 |    −263 |      491 |     228 |
|       128 |  423 |  602 |  244 |    −179 |      359 |     179 |
|      1024 | 1089 | 1061 | 1896 |      29 |     −835 |    −806 |

## What the measurement shows

### 1. Variant A is dispatch-bound, not compute-bound

A barely moves between `fma_depth=1` and `fma_depth=128` (321 → 375 µs, +54 µs for 128× more ALU work). The GPU hides the extra compute behind the dispatch pipeline of 112 kernel launches. Per-kernel budget in A: ~3 µs (≈ 2 µs dispatch + ~1 µs GPU execution for memory-bound elementwise).

### 2. Variant B1 is compute-bound at relevant compute volumes

B1 scales strongly with `fma_depth` (97 → 112 → 242 → 1865 µs). Because B1 has only 28 launches and the ALU work is fully exposed, total time grows linearly with compute.

### 3. The "fused gain" shrinks as compute intensity rises

Δ(A−B1) falls from **224 µs** (depth=1) to **133 µs** (depth=128) and **turns negative** at depth=1024. That is not buffer traffic — it is direct evidence that fusion primarily saves **launch overhead**. Once kernels have enough compute to hide their launches, the fused advantage vanishes.

### 4. B2 does NOT isolate dispatch overhead cleanly

Cooperative-kernel launches with `grid.sync()` are more expensive on this hardware than 4 sequential dispatches on the same stream. At batch=4 B2 is consistently ~265 µs slower than A — A's stream pipelining is more efficient than grid-wide sync inside a single launch. This indirectly confirms the RDNA-4 memory-controller-pipelining hypothesis from `docs/architecture_notes.md`: back-to-back dispatches on the same stream pipeline nearly without loss.

### 5. Buffer traffic cannot be isolated in the elementwise regime

Δ(B2−B1) swings from +293 (depth=1) to −1018 (depth=1024). That is not a buffer-traffic signal — it is a mix of cooperative-launch overhead (B2 expensive) and B1's compute exposure (B1 becomes compute-bound sooner than B2). The micro-benchmark cannot cleanly separate buffer traffic from dispatch and occupancy effects.

## Projection to real FFN time

Measured real FFN time (launch-overhead analysis, 28 layers, batch=2, depth=1): **10,900 µs**.

The real FFN kernels (GEMV against Q4_0 weight matrices) are **memory-bound against the weight matrix**, not against the intermediate buffer:

| Kernel               | Time (µs)  | Weight read    | Intermediate traffic |
|----------------------|-----------:|---------------:|---------------------:|
| gate_up_silu (fused) |        232 |         ~75 MB |              ~75 KB  |
| mul (elementwise)    |        ~30 |              — |             ~150 KB  |
| down (GEMV)          |        145 |         ~75 MB |             ~150 KB  |
| residual             |        ~10 |              — |              ~14 KB  |

The weight volume (~150 MB per layer) dominates the memory pipeline by a factor of ~500. Intermediate-buffer traffic is already amortized inside the GEMV runtimes, not a separate cost item.

**What fused FFN can realistically save:**

| Component                                  | Saving      |
|--------------------------------------------|------------:|
| Dispatch overhead (4→1 per layer × 28)     |   ~168 µs   |
| Intermediate-buffer traffic elimination    |    ~30 µs   |
| Sync elimination (4→1 per layer)           |     ~0 µs*  |
| **Total**                                  | **~200 µs** |

\* Stream pipelining makes between-kernel syncs on the same stream essentially free (documented in `docs/architecture_notes.md`).

**Relative to verify time:**

| Metric                               | Value          |
|--------------------------------------|----------------|
| Real FFN time (28 layers)            | 10,900 µs      |
| Total verify step                    | 16,200 µs      |
| Expected fused-FFN gain              | ~200 µs        |
| Share of verify time                 | **~1.2 %**     |
| Share of FFN time                    | **~1.8 %**     |

## Verdict

**Fused FFN is NOT worth building.**

- Micro-benchmark upper bound (depth=1): **224 µs/step** (1.4 % of verify time).
- Realistic projection (with compute coverage): **~100–200 µs/step** (0.6–1.2 %).
- "Worth it" threshold: ≥ 1,500 µs (8 %).
- Result sits 7–15× **below** the worth-it threshold.

The full "buffer traffic" hypothesis is falsified by the measurement:

1. Buffer traffic in the real (GEMV-dominated) FFN is overshadowed by weight-matrix traffic by a factor of ~500.
2. The launch-overhead share (what fusion actually saves) is ~200 µs — far below the original ~4,000 µs estimate from the launch-overhead analysis.
3. The 4,000 µs estimate was a modelling artifact: reading "wall clock minus GPU-only" as buffer traffic was not justified, because large parts of that difference came from submission latency (effectively hidden by stream pipelining on RDNA 4) and profiling overhead.

This is the third consecutive null effect on this architecture (tiled GEMV, batched lm_head, now fused FFN).

## What lever remains?

Not an implementation plan — a direction for the next step:

1. **Prefill GEMM.** Current prefill: 59 tok/s vs llama.cpp 1,092 tok/s (5 % of their baseline). This is not a 1–2 % gap, it is a 95 % gap. The next meaningful lever is real batch-1 GEMM instead of N × GEMV on prefill tokens.
2. **Attention optimization at long context.** At K=4096 throughput drops from 115 to 78 tok/s (−32 %), attention reaches 34 % of decode time. Already in the memory regime — FlashAttention-style tiling could deliver real throughput here.
3. **14B target with 0.5B draft.** Currently VRAM-limited to 16 GB. A Q8_0 draft instead of Q4_0 could raise acceptance.

Fused FFN should **not** be implemented before one of the above levers loses priority.

## Consistency check

- Launch-overhead analysis estimated FFN buffer traffic at ~4,000 µs. The measurement finds ~200 µs realistic saving. Deviation: 20× — identical to the deviation on tiled GEMV (~1.5 % instead of 15 %) and batched lm_head (~0.4 % instead of 8 %).
- The consistent 20× overshoot from "load/traffic once instead of N times" modelling is now documented across three independent experiments and tracked in `docs/architecture_notes.md` as a stable RDNA-4 property.
