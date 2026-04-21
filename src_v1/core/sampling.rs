//! Sampling strategies for next-token selection.
//!
//! Phase 1 supports greedy (temperature = 0) and the standard trio of
//! top-k / top-p / temperature plus an optional repeat-penalty. The
//! 15-prompt validation run uses greedy exclusively so its output is
//! reproducible across runs.

/// Sampler configuration.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn default_chat() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: 0xCAFE,
        }
    }
}

/// Pick the next token id from logits. Mutates `logits` (in-place
/// repeat-penalty / softmax); callers should not reuse it afterwards.
pub fn sample_token(
    logits: &mut [f32],
    cfg: &SamplingConfig,
    previous: &[u32],
) -> u32 {
    debug_assert!(!logits.is_empty());

    // 1) Repeat penalty — divide logits of previously-seen tokens.
    if cfg.repeat_penalty > 1.0 {
        for &tid in previous {
            let idx = tid as usize;
            if idx < logits.len() {
                let v = logits[idx];
                logits[idx] = if v > 0.0 {
                    v / cfg.repeat_penalty
                } else {
                    v * cfg.repeat_penalty
                };
            }
        }
    }

    // 2) Greedy fast path.
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }

    // 3) Temperature scaling.
    for v in logits.iter_mut() {
        *v /= cfg.temperature;
    }

    // 4) Top-K filter — keep only the k largest.
    if cfg.top_k > 0 && cfg.top_k < logits.len() {
        let kth = nth_largest(logits, cfg.top_k);
        for v in logits.iter_mut() {
            if *v < kth {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // 5) Softmax.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum <= 0.0 {
        return argmax(logits);
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }

    // 6) Top-P nucleus filter.
    if cfg.top_p < 1.0 {
        apply_top_p(logits, cfg.top_p);
    }

    // 7) Multinomial draw.
    let mut rng = SimpleRng::new(cfg.seed);
    let r: f32 = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    argmax(logits)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

fn nth_largest(values: &[f32], k: usize) -> f32 {
    let mut buf: Vec<f32> = values.to_vec();
    buf.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    buf[k.min(buf.len() - 1)]
}

fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> =
        probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut cutoff = indexed.len();
    for (rank, (_, p)) in indexed.iter().enumerate() {
        cum += *p;
        if cum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> =
        indexed.iter().take(cutoff).map(|(i, _)| *i).collect();
    let mut sum = 0.0f32;
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// xorshift64 — good enough for stochastic sampling, fully deterministic
/// across runs when seeded identically.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_F00D } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32 / u64::MAX as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let cfg = SamplingConfig::greedy();
        assert_eq!(sample_token(&mut logits, &cfg, &[]), 3);
    }

    #[test]
    fn greedy_is_deterministic() {
        let mut a = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut b = a.clone();
        let cfg = SamplingConfig::greedy();
        assert_eq!(
            sample_token(&mut a, &cfg, &[]),
            sample_token(&mut b, &cfg, &[]),
        );
    }

    #[test]
    fn temperature_sampling_introduces_variety() {
        let base = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
        let mut seen = std::collections::HashSet::new();
        for seed in 0..100u64 {
            let mut l = base.clone();
            let cfg = SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                seed,
            };
            seen.insert(sample_token(&mut l, &cfg, &[]));
        }
        assert!(seen.len() > 1, "temperature sampling should spread");
    }
}
