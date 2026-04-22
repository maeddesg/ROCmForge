//! Deterministic seeded RNG (xorshift64*).
//!
//! The GA spec requires bit-identical runs for a given `GA_SEED` on a
//! given engine git hash (`ga_tuning_spec §2.5`). A custom xorshift64*
//! gives that guarantee without pulling the `rand` crate into the
//! dependency graph, and keeps the GA engine ~200 LOC as the prompt
//! asks.
//!
//! `next_u64` is the standard xorshift64* from Marsaglia (2003):
//!
//!   s ^= s << 13
//!   s ^= s >> 7
//!   s ^= s << 17
//!   return s * 0x2545F4914F6CDD1D
//!
//! Quality is sufficient for a GA search where each generation draws
//! ~200 random numbers. Period is 2^64−1. Not cryptographic.

/// Seeded PRNG used throughout the GA module.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        // Zero seed would lock the xorshift state at 0; map it to 1.
        let state = if seed == 0 { 0xDEAD_BEEF_CAFE_F00D } else { seed };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform u64 in `[low, hi)` (half-open). Panics if `hi <= low`.
    pub fn gen_range_u64(&mut self, low: u64, hi: u64) -> u64 {
        assert!(hi > low, "gen_range_u64: empty interval [{low}, {hi})");
        low + self.next_u64() % (hi - low)
    }

    /// Uniform usize in `[0, bound)`. Panics on bound == 0.
    pub fn gen_index(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "gen_index: bound == 0");
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform f64 in `[0.0, 1.0)`.
    pub fn gen_f64(&mut self) -> f64 {
        // Top 53 bits of next_u64 → [0, 1).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    pub fn gen_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    /// Pick an element uniformly at random. Panics on empty slice.
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        &slice[self.gen_index(slice.len())]
    }

    /// Pick one of a fixed small set of u8 values.
    pub fn choose_u8(&mut self, set: &[u8]) -> u8 {
        *self.choose(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(43);
        let mut any_diff = false;
        for _ in 0..10 {
            if a.next_u64() != b.next_u64() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different seeds should not produce identical streams");
    }

    #[test]
    fn gen_f64_within_unit_interval() {
        let mut r = SeededRng::new(7);
        for _ in 0..1000 {
            let x = r.gen_f64();
            assert!((0.0..1.0).contains(&x), "gen_f64 out of [0,1): {x}");
        }
    }

    #[test]
    fn gen_index_within_bound() {
        let mut r = SeededRng::new(99);
        for _ in 0..1000 {
            let i = r.gen_index(17);
            assert!(i < 17);
        }
    }
}
