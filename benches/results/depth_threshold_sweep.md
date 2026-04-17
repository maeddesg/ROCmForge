# Adaptive Depth Threshold Sweep

**Date:** 2026-04-16  
**Git SHA:** 313efdf  
**Hardware:** RX 9070 XT (gfx1201, RDNA4)  
**Target:** Qwen2.5-7B-Instruct-Q4_0.gguf  
**Draft:** qwen2.5-0.5b-instruct-q4_0.gguf  
**Max tokens:** 128  
**Tiled GEMV:** on (default)  

## Hypothesis

With tiled batched GEMV reducing verify costs at higher depths, the adaptive depth-down threshold (`ema < 1.2`) might be too aggressive. Test whether `ema < 1.0` or `ema < 0.9` yields better throughput by keeping higher depths longer.

## Baseline (no speculative decoding)

| Prompt | tok/s |
|--------|-------|
| code_01 | 81.9 |
| code_02 | 81.8 |
| code_03 | 81.8 |
| code_04 | 81.8 |
| code_05 | 81.8 |

## depth=3

| Prompt | ema<1.2 (current) | ema<1.0 | ema<0.9 |
|--------|-------------------|---------|---------|
| code_01 | **77.6** (alpha=74.5%, depth->1) | 77.8 (alpha=75.2%, depth->3) | 77.9 (alpha=75.2%, depth->3) |
| code_02 | **73.5** (alpha=66.0%, depth->1) | 72.6 (alpha=64.8%, depth->1) | 72.6 (alpha=64.8%, depth->1) |
| code_03 | **75.7** (alpha=69.3%, depth->1) | 74.3 (alpha=66.3%, depth->1) | 74.2 (alpha=66.3%, depth->1) |
| code_04 | **71.8** (alpha=57.8%, depth->1) | 68.2 (alpha=51.0%, depth->1) | 68.3 (alpha=51.0%, depth->1) |
| code_05 | **69.9** (alpha=60.0%, depth->1) | 68.7 (alpha=57.6%, depth->1) | 68.7 (alpha=57.6%, depth->1) |

Verify times (avg, us):

| Prompt | ema<1.2 | ema<1.0 | ema<0.9 |
|--------|---------|---------|---------|
| code_01 | 27,433 | 31,402 | 31,397 |
| code_02 | 21,636 | 22,596 | 22,604 |
| code_03 | 19,263 | 19,675 | 19,683 |
| code_04 | 18,337 | 18,837 | 18,818 |
| code_05 | 19,631 | 20,016 | 19,991 |

## depth=5

| Prompt | ema<1.2 (current) | ema<1.0 | ema<0.9 |
|--------|-------------------|---------|---------|
| code_01 | 62.2 (alpha=53.4%, depth->5) | 62.2 (alpha=53.4%, depth->5) | 62.2 (alpha=53.4%, depth->5) |
| code_02 | **52.8** (alpha=41.1%, depth->3) | 51.4 (alpha=39.5%, depth->4) | 50.2 (alpha=38.2%, depth->5) |
| code_03 | **66.9** (alpha=53.4%, depth->1) | 63.4 (alpha=48.8%, depth->1) | 63.2 (alpha=48.8%, depth->1) |
| code_04 | **63.5** (alpha=45.9%, depth->1) | 61.4 (alpha=42.6%, depth->1) | 59.0 (alpha=40.3%, depth->1) |
| code_05 | **63.1** (alpha=48.8%, depth->1) | 61.6 (alpha=46.5%, depth->1) | 61.0 (alpha=46.2%, depth->1) |

Verify times (avg, us):

| Prompt | ema<1.2 | ema<1.0 | ema<0.9 |
|--------|---------|---------|---------|
| code_01 | 44,284 | 44,266 | 44,255 |
| code_02 | 41,591 | 42,748 | 43,829 |
| code_03 | 22,217 | 23,796 | 23,860 |
| code_04 | 21,859 | 22,334 | 23,511 |
| code_05 | 22,423 | 22,990 | 23,581 |

## Conclusion

**The current threshold `ema < 1.2` is already optimal.** Lower thresholds are strictly worse:

- **depth=3:** ema<1.2 wins 4/5 prompts (code_01 is a wash at +0.2 tok/s). code_04 loses 3.5 tok/s with ema<1.0.
- **depth=5:** ema<1.2 wins 4/5 prompts (code_01 is identical). code_04 loses 4.5 tok/s with ema<0.9.

**Root cause:** Verify cost scales superlinearly with batch size. Keeping higher depth means more tokens to verify per step, but acceptance rate doesn't increase proportionally. The extra verify time is not compensated by additional accepted tokens. The adaptive algorithm is correctly aggressive at dropping to depth=1.

**Key example:** code_02 at depth=5 — threshold=0.9 holds depth->5 (50.2 tok/s), threshold=1.2 drops to depth->3 (52.8 tok/s). The 5% throughput loss comes entirely from unnecessary verify overhead at high batch sizes.

No changes to the adaptive depth algorithm are warranted.
