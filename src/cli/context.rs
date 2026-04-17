//! Conversation state for the interactive chat CLI.
//!
//! Phase 5 Step 3: multi-turn history, truncation, and session-level
//! accumulators. The tokenizer dependency stays opaque — truncation
//! accepts any callable that maps a string to a token count — so this
//! module has no deps on `rocmforge::tokenizer` directly and is fully
//! unit-testable.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

pub struct ChatContext {
    pub system_prompt: String,
    pub conversation_history: Vec<Turn>,
    pub user_input: String,
}

impl ChatContext {
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            conversation_history: Vec::new(),
            user_input: String::new(),
        }
    }

    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    pub fn push_turn(&mut self, role: Role, content: String) {
        self.conversation_history.push(Turn { role, content });
    }

    pub fn set_system_prompt(&mut self, new_prompt: String) {
        self.system_prompt = new_prompt;
        // A fresh system prompt implies a fresh conversation — the
        // previous turns were generated under the old contract.
        self.conversation_history.clear();
    }

    pub fn turn_count(&self) -> usize {
        self.conversation_history.len() / 2
    }
}

/// Drop the oldest (user, assistant) pairs until a formatted prompt fits
/// inside `max_tokens`. `format` renders the current context to a string;
/// `count_tokens` returns the tokenizer's count for that string. Returns
/// the number of pairs that were dropped.
///
/// The function never drops the most recent pair — if even the last pair
/// (plus the new user input + system prompt) does not fit, the caller has
/// to handle that at a higher level (return an error, or prune the user
/// input itself, which is outside this module's scope).
pub fn truncate_if_needed<F, G>(
    ctx: &mut ChatContext,
    max_tokens: usize,
    mut format: F,
    mut count_tokens: G,
) -> usize
where
    F: FnMut(&ChatContext) -> String,
    G: FnMut(&str) -> usize,
{
    let mut dropped_pairs = 0usize;
    loop {
        let formatted = format(ctx);
        let n = count_tokens(&formatted);
        if n <= max_tokens {
            return dropped_pairs;
        }
        if ctx.conversation_history.len() < 4 {
            // At most one prior pair left (2 entries) — keep it so the
            // current exchange still has *some* context. Caller handles
            // the rest.
            return dropped_pairs;
        }
        // Remove the oldest user+assistant pair.
        ctx.conversation_history.drain(0..2);
        dropped_pairs += 1;
    }
}

/// Session-wide accumulators for the `/stats` command.
pub struct SessionStats {
    pub turn_count: u32,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub total_ttft_ms: f64,
    pub total_decode_time_s: f64,
    pub session_start: Instant,
}

impl SessionStats {
    pub fn new() -> Self {
        Self {
            turn_count: 0,
            total_prompt_tokens: 0,
            total_generated_tokens: 0,
            total_ttft_ms: 0.0,
            total_decode_time_s: 0.0,
            session_start: Instant::now(),
        }
    }

    pub fn record(
        &mut self,
        prompt_tokens: usize,
        generated_tokens: usize,
        ttft_ms: f64,
        decode_time_s: f64,
    ) {
        self.turn_count += 1;
        self.total_prompt_tokens += prompt_tokens as u64;
        self.total_generated_tokens += generated_tokens as u64;
        self.total_ttft_ms += ttft_ms;
        self.total_decode_time_s += decode_time_s;
    }

    pub fn avg_ttft_ms(&self) -> f64 {
        if self.turn_count == 0 {
            0.0
        } else {
            self.total_ttft_ms / self.turn_count as f64
        }
    }

    pub fn avg_decode_tps(&self) -> f64 {
        if self.total_decode_time_s == 0.0 {
            0.0
        } else {
            self.total_generated_tokens as f64 / self.total_decode_time_s
        }
    }

    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }
}

impl Default for SessionStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_history_keeps_system_prompt() {
        let mut ctx = ChatContext::new("You are helpful.".to_string());
        ctx.push_turn(Role::User, "hi".to_string());
        ctx.push_turn(Role::Assistant, "hello".to_string());
        ctx.clear_history();
        assert_eq!(ctx.conversation_history.len(), 0);
        assert_eq!(ctx.system_prompt, "You are helpful.");
    }

    #[test]
    fn set_system_prompt_clears_history() {
        let mut ctx = ChatContext::new("Old.".to_string());
        ctx.push_turn(Role::User, "q".to_string());
        ctx.push_turn(Role::Assistant, "a".to_string());
        ctx.set_system_prompt("New.".to_string());
        assert_eq!(ctx.system_prompt, "New.");
        assert_eq!(ctx.conversation_history.len(), 0);
    }

    #[test]
    fn turn_count_counts_pairs() {
        let mut ctx = ChatContext::new("S".to_string());
        assert_eq!(ctx.turn_count(), 0);
        ctx.push_turn(Role::User, "u1".to_string());
        ctx.push_turn(Role::Assistant, "a1".to_string());
        assert_eq!(ctx.turn_count(), 1);
        ctx.push_turn(Role::User, "u2".to_string());
        ctx.push_turn(Role::Assistant, "a2".to_string());
        assert_eq!(ctx.turn_count(), 2);
    }

    #[test]
    fn truncate_drops_oldest_pair_first() {
        let mut ctx = ChatContext::new("S".to_string());
        for i in 0..5 {
            ctx.push_turn(Role::User, format!("u{}", i));
            ctx.push_turn(Role::Assistant, format!("a{}", i));
        }
        ctx.user_input = "new question".to_string();
        // Count tokens as bytes for simplicity; force aggressive truncation.
        let dropped = truncate_if_needed(
            &mut ctx,
            20,
            |c| {
                let mut s = c.system_prompt.clone();
                for t in &c.conversation_history {
                    s.push_str(&t.content);
                }
                s.push_str(&c.user_input);
                s
            },
            |s| s.len(),
        );
        assert!(dropped >= 1, "should drop at least one pair");
        // Newest pair preserved.
        let last = ctx.conversation_history.last().unwrap();
        assert_eq!(last.content, "a4");
    }

    #[test]
    fn truncate_noop_when_under_limit() {
        let mut ctx = ChatContext::new("S".to_string());
        ctx.push_turn(Role::User, "hi".to_string());
        ctx.push_turn(Role::Assistant, "ok".to_string());
        let dropped = truncate_if_needed(&mut ctx, 4096, |c| c.system_prompt.clone(), |s| s.len());
        assert_eq!(dropped, 0);
        assert_eq!(ctx.conversation_history.len(), 2);
    }

    #[test]
    fn session_stats_accumulate() {
        let mut stats = SessionStats::new();
        stats.record(20, 15, 50.0, 0.15);
        stats.record(30, 10, 70.0, 0.10);
        assert_eq!(stats.turn_count, 2);
        assert_eq!(stats.total_prompt_tokens, 50);
        assert_eq!(stats.total_generated_tokens, 25);
        assert!((stats.avg_ttft_ms() - 60.0).abs() < 1e-9);
        let expected_tps = 25.0 / 0.25;
        assert!((stats.avg_decode_tps() - expected_tps).abs() < 1e-6);
    }
}
