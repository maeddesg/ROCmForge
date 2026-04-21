//! Token-level repetition detection (Phase-1 heuristic).
//!
//! The simplest loop detector: if the same token id shows up in
//! five consecutive decode steps, flag it. 5-gram detection is a
//! Phase-2 candidate once the Quality Monitor drives precision
//! escalation — right now it would just duplicate the role of
//! `repeat_penalty` in the sampler.

use super::types::{DriftReason, QualityMonitor};

/// How many consecutive identical tokens trip the Phase-1 detector.
const REPEAT_THRESHOLD: usize = 5;
/// Ring-buffer cap for the recent-token window. 100 keeps the
/// check O(REPEAT_THRESHOLD) while still leaving room for future
/// n-gram detection.
const RECENT_TOKENS_CAP: usize = 100;

impl QualityMonitor {
    /// Record a newly decoded token. Must be called once per decode
    /// step so `should_check` and `check_repetition` work.
    pub fn record_token(&mut self, token_id: u32) {
        if self.recent_tokens.len() >= RECENT_TOKENS_CAP {
            self.recent_tokens.remove(0);
        }
        self.recent_tokens.push(token_id);
        self.tokens_since_check += 1;
    }

    /// Returns true once `sample_rate` tokens have been recorded
    /// since the last hidden-state check — caller resets via
    /// `reset_check_counter`.
    pub fn should_check(&self) -> bool {
        self.tokens_since_check >= self.sample_rate as u64
    }

    pub fn reset_check_counter(&mut self) {
        self.tokens_since_check = 0;
    }

    /// Fires once the last `REPEAT_THRESHOLD` tokens are all
    /// `token_id`. The caller passes the *just-recorded* token so
    /// the detector can cheaply verify it was the most recent
    /// push, but the check reads the ring buffer directly —
    /// `record_token` must have already been called.
    pub fn check_repetition(&self, token_id: u32) -> Option<DriftReason> {
        if self.recent_tokens.len() < REPEAT_THRESHOLD {
            return None;
        }
        let tail = &self.recent_tokens[self.recent_tokens.len() - REPEAT_THRESHOLD..];
        if tail.iter().all(|&t| t == token_id) {
            Some(DriftReason::RepetitionDetected {
                token_id,
                count: REPEAT_THRESHOLD as u32,
            })
        } else {
            None
        }
    }
}
