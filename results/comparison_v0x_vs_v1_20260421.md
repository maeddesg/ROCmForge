# ROCmForge v0.x vs v1.0 — Engine Comparison Report

- **Date:** 2026-04-21
- **GPU:** RX 9070 XT (gfx1201)
- **Sampling:** temperature=0.6, top_p=0.9 (v0.x has no repeat_penalty flag; v1.0 runs with repeat_penalty=1.05)
- **Chat template:** both use ChatML + `/no_think` appended to user turn

## Diagnostic question

After the third 15-prompt re-run (2 korrekt / 3 teilweise / 10 falsch-müll) we
need to isolate whether the v1.0 quality issues are (a) model-specific,
(b) engine-specific, or (c) Qwen3-specific. Two tests:

1. v0.x engine + Qwen3-8B-Q4_K_M, same 4 prompts
2. v1.0 engine + Llama-3.1-8B-Q4_K_M, same 4 prompts

## Test 2 — v1.0 on Llama-3.1-8B-Q4_K_M

**Result: v1.0 cannot run Llama-3.1 at all.**

```
[ 1/4] Greeting — smoke (trivial/64)
Error: generate prompt 1: HIP error 1 (invalid argument) in gemv
```

The model loads fine (`--list-tensors` reports
architecture=llama, 32 layers, vocab=128256, `has_rope_freqs=true`),
so the GGUF parser and tensor inventory are OK. The failure is in the
GEMV dispatch of the first decode step. This is a separate v1.0 bug
that blocks the intended cross-model comparison.

## Test 1 — v0.x on Qwen3-8B-Q4_K_M

Same model, same sampling knobs the v0.x CLI exposes. All four prompts
produce clean, structured, technically correct output.

### Prompt #1 — "Hallo /no_think"

**v0.x:**
```
Hallo! 😊 Wie kann ich dir heute helfen?
```
15 tokens, 38.2 tok/s decode. Clean.

**v1.0 (prior run):**
```
Hallo! Wie kann ich Ihnen heute helfen? 😊
```
15 tokens. Clean. ✅ Both engines agree.

### Prompt #3 — "Write a Python function that checks if a number is prime. /no_think"

**v0.x:**
```python
def is_prime(n):
    """
    Check if a number is prime.
    ...
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
Textbook-correct: `i*i <= n` sqrt bound, 6k±1 optimisation, all edge
cases. 40.0 tok/s decode.

**v1.0 (prior run):**
```
Okay, I need to write a Python function that checks if a number is
prime. Let me think about how to do this.

First, I remember that a prime number is one that has exactly two
divisors 1 and itself. So the function should return True if the input
is prime.
[… 200 more tokens of degraded thinking, no actual code …]
If n<2return False
But wait: ifn
```
No code emitted. 18.4 tok/s decode.

### Prompt #6 — "Explain what a mutex is in one paragraph. /no_think"

**v0.x:**
> A mutex, short for "mutual exclusion," is a synchronization mechanism
> used in concurrent programming to prevent multiple threads or processes
> from accessing a shared resource simultaneously. It ensures that only
> one thread can execute a critical section of code at any given time,
> thereby avoiding race conditions and maintaining data integrity. When
> a thread acquires a mutex, it locks the resource, preventing other
> threads from accessing it until the mutex is released. Mutexes are
> essential for coordinating access to shared resources in multi-threaded
> or multi-process environments.

106 tokens, hit EOS. Textbook-correct.

**v1.0 (prior run):**
> A mutex, short for "mutual exclusion," is a synchronization mechanism
> used in concurrent programming to prevent multiple threads or processes
> from accessing shared resources simultaneously, ensuring that only one
> thread or process at a time, preventing race conditions and ensuring
> data integrity and consistency in multi-threaded programs from accessing
> the same resource or data structure that allows only one thread or
> process to access shared resources, which is used in programming. It
> ensures that at any given moment. A mutex (mutual exclusion by
> providing the same time, which can be executed at any point in the
> system to access to prevent race conditions and avoid data corruption.

128 tokens, hit cap. Technically starts correct but loses coherence
around token ~40 — sentences fragment, phrases repeat with slight
reordering. No EOS.

### Prompt #7 — "Compare and contrast TCP and UDP protocols. … /no_think"

**v0.x:** (first 512 tokens, clean structured comparison)
```
### 1. Reliability
- TCP: Reliable, three-way handshake, flow control, ACKs, retransmissions.
- UDP: Unreliable, no handshake, best-effort only.

### 2. Speed
- TCP: Slower (overhead of handshake + retransmission).
- UDP: Faster, minimal overhead, low latency.

### 3. Use Cases
- TCP: HTTP/HTTPS, SMTP, FTP, VoIP with jitter buffer.
- UDP: Streaming media, online gaming, DNS, real-time sensor data.

### 4. Header Differences
- TCP Header: 20–60 bytes. Fields: Source/Dest Port, Sequence …
```
512 tokens, hit cap. Every fact correct.

**v1.0 (prior run):**
```
TCP (Transmission Control Protocol) and UDP (User Datagram Protocol)
are two core transport layer protocols used in the internet. They have
different characteristics, use cases, header structure.

### 1. Reliability:
- TCP is reliable protocol that ensures data is delivered in order …
[… starts correctly, then devolves …]
20012456778910102389853106567789: on the same as they will return a
request: e.g., but not if you can get this is no which would not have
any which may have more than it's so that you can't get at once …
```
512 tokens, hit cap. First ~30 tokens correct, then degrades into
fragmented phrases interleaved with random number sequences.

## Verdict

| Finding | Signal |
|---|---|
| v0.x on Qwen3: all 4 prompts produce correct, structured output | Qwen3-Q4_K_M weights are fine |
| v1.0 on Qwen3: only short prompts clean; long prompts degrade with incoherent prose + number soup | v1.0 engine bug |
| Short prompts (#1) agree between engines | Forward-pass for first ~15 tokens is correct |
| Degradation scales with decode length (#6 @ 40 tok, #7 @ 50 tok, #3 @ 100 tok, #11 @ 200+ tok) | KV-cache / attention-drift hypothesis |
| Number-soup pattern `20012456778910102389853` | Logit distribution skews toward digit tokens as context grows |
| v1.0 decode ~18 tok/s vs v0.x ~39 tok/s | Separate: Phase-1 non-fused kernels, not the quality issue |
| v1.0 cannot run Llama-3.1 (HIP GEMV invalid-argument at first decode) | Separate bug — architecture-conditional codepath missing |

**Conclusion:** the 0/15 → 2/15 evaluation result is NOT caused by the
Qwen3 thinking mode or by the quantisation format. v0.x runs the exact
same Qwen3-Q4_K_M weights and produces clean output on every prompt.
**v1.0 has an engine bug that corrupts the logit distribution as decode
length grows.** Likely suspects (in descending probability):

1. **KV-cache index bug** — something wrong with position offsets once
   past ~15–40 decode steps; off-by-one in `execute_decode(tok, pos)`
   would explain gradual drift.
2. **Attention-score accumulation** — FP16 accumulator overflow or
   missing causal mask beyond some position threshold.
3. **RoPE frequency table** — Qwen3 uses the standard RoPE but with
   `has_qk_norm=true`; a wrong indexing into the frequency table would
   corrupt queries/keys with position.

The separate Llama-3.1 GEMV failure suggests an additional
architecture-conditional bug in the v1 dispatch.

## Next diagnostic steps (not started)

- Compare the logit vector at decode step 1, 10, 50, 100 between v0.x
  and v1.0 on the same seeded prompt. Point where top-1 diverges is
  the first sign of the bug.
- Dump KV cache at step 50 and compare head-by-head.
- Fix v1.0 Llama-3.1 GEMV so we can replicate the drift test on a
  non-Qwen3 model.
