//! Phase 2.4 — streaming output helper.
//!
//! Live token emission with `<think>…</think>` filtering. Qwen3 is a
//! reasoning model: when prompted without `/no_think` it emits a
//! multi-hundred-token reasoning block wrapped in `<think>` tags
//! before the final answer. Users don't want to see that; this emitter
//! buffers potential tag boundaries and only forwards the final text.
//!
//! Token-boundary hazards:
//!   1. A single Qwen3 BPE piece rarely crosses a tag boundary but it
//!      can happen (e.g. `<think>` is split across two pieces on some
//!      variants). The emitter buffers anything that could be the start
//!      of a `<think>` open tag and only flushes once it's certain the
//!      text is not a tag opener.
//!   2. The closing `</think>` is always a single piece in Qwen3's
//!      vocab but we still use a substring search — robust against
//!      vocab differences in other reasoning models we might add later.

/// Longest suffix of `s` that is a strict prefix of `pat`.
/// Used to decide how many trailing bytes of the buffer COULD still
/// become a `<think>` opener — those are kept buffered; the rest is
/// safe to emit.
fn longest_prefix_suffix(s: &str, pat: &str) -> usize {
    let bytes = s.as_bytes();
    let pb = pat.as_bytes();
    let max = bytes.len().min(pb.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if bytes[bytes.len() - k..] == pb[..k] {
            // Confirm this cut falls on a UTF-8 boundary so the result
            // is a valid str slice.
            if s.is_char_boundary(bytes.len() - k) {
                return k;
            }
        }
    }
    0
}

/// Streaming emitter that filters out `<think>...</think>` blocks.
///
/// Pattern:
///   ```ignore
///   let mut em = StreamingEmitter::new(true);
///   for piece in detokenize_one_by_one(...) {
///       if let Some(visible) = em.process(&piece) {
///           print!("{visible}");
///       }
///   }
///   if let Some(tail) = em.flush() {
///       print!("{tail}");
///   }
///   ```
pub struct StreamingEmitter {
    /// Buffer of already-decoded tokens that haven't been emitted yet.
    /// Either because they might start a `<think>` tag, or because
    /// we're inside a think block.
    buffer: String,
    /// `true` when the buffer content is inside a `<think>…</think>`
    /// span and must be discarded until the closing tag arrives.
    in_think_block: bool,
    /// When `false`, the emitter acts as a passthrough (still flushes
    /// whole-token strings, never partial byte sequences). Used by the
    /// `--show-think` CLI flag.
    filter_think: bool,
}

impl StreamingEmitter {
    pub fn new(filter_think: bool) -> Self {
        Self {
            buffer: String::with_capacity(64),
            in_think_block: false,
            filter_think,
        }
    }

    /// Consume one detokenized token. Returns the substring (if any)
    /// that is safe to emit to the user at this point.
    pub fn process(&mut self, piece: &str) -> Option<String> {
        self.buffer.push_str(piece);

        // Drive a small state machine: inside → seek close; outside →
        // seek open. Loop because a single token could close a block
        // AND open the next (unlikely but cheap).
        let mut out = String::new();
        loop {
            if self.in_think_block {
                if let Some(rel) = self.buffer.find("</think>") {
                    self.buffer = self.buffer[rel + "</think>".len()..].to_string();
                    self.in_think_block = false;
                    continue;
                } else {
                    // Still inside; drop everything buffered so far.
                    // Keep only the longest suffix that could grow into
                    // `</think>` so we don't miss a split close tag.
                    let keep = longest_prefix_suffix(&self.buffer, "</think>");
                    // Anything beyond `keep` is interior-of-think and
                    // discarded — no user-visible side effects.
                    let new_len = self.buffer.len() - keep;
                    self.buffer.drain(..new_len);
                    break;
                }
            }

            // Out-of-think state. If filtering is off we just flush
            // tokens as they come; we never emit partial bytes.
            if !self.filter_think {
                if !self.buffer.is_empty() {
                    out.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for the opening tag.
            if let Some(rel) = self.buffer.find("<think>") {
                // Everything before the tag is user-visible.
                out.push_str(&self.buffer[..rel]);
                self.buffer = self.buffer[rel + "<think>".len()..].to_string();
                self.in_think_block = true;
                continue;
            }

            // No complete open tag. Emit whatever cannot still become
            // one: keep the longest suffix that is a prefix of `<think>`.
            let keep = longest_prefix_suffix(&self.buffer, "<think>");
            let safe_end = self.buffer.len() - keep;
            if safe_end > 0 {
                out.push_str(&self.buffer[..safe_end]);
                self.buffer.drain(..safe_end);
            }
            break;
        }

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Emit any remaining buffer content. Called on EOS or when the
    /// generation loop hits `max_tokens`. If we're still inside a
    /// think block (no closing tag ever arrived — truncated output),
    /// the buffer is dropped silently.
    pub fn flush(&mut self) -> Option<String> {
        if self.in_think_block {
            self.buffer.clear();
            return None;
        }
        if self.buffer.is_empty() {
            None
        } else {
            let out = std::mem::take(&mut self.buffer);
            Some(out)
        }
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(em: &mut StreamingEmitter, pieces: &[&str]) -> String {
        let mut out = String::new();
        for p in pieces {
            if let Some(s) = em.process(p) {
                out.push_str(&s);
            }
        }
        if let Some(s) = em.flush() {
            out.push_str(&s);
        }
        out
    }

    #[test]
    fn emitter_passthrough_no_tags() {
        let mut em = StreamingEmitter::new(true);
        let got = feed(&mut em, &["Hello", " ", "world", "!"]);
        assert_eq!(got, "Hello world!");
    }

    #[test]
    fn emitter_full_think_block_single_token_tags() {
        let mut em = StreamingEmitter::new(true);
        let got = feed(
            &mut em,
            &["Pre", "<think>", "reasoning ", "here", "</think>", "Answer"],
        );
        assert_eq!(got, "PreAnswer");
    }

    #[test]
    fn emitter_split_open_tag_across_pieces() {
        let mut em = StreamingEmitter::new(true);
        let got = feed(
            &mut em,
            &["Say ", "<", "th", "ink", ">", "secret", "</think>", " hi"],
        );
        assert_eq!(got, "Say  hi");
    }

    #[test]
    fn emitter_partial_tag_is_not_suppressed_when_not_a_tag() {
        // "<type" looks like it could become "<think>" for the first
        // two chars, then diverges at "y". The "ty" should appear as
        // soon as we see enough to know it's NOT a think tag.
        let mut em = StreamingEmitter::new(true);
        let got = feed(&mut em, &["<", "ty", "pe>"]);
        assert_eq!(got, "<type>");
    }

    #[test]
    fn emitter_split_close_tag() {
        let mut em = StreamingEmitter::new(true);
        let got = feed(
            &mut em,
            &[
                "<think>", "r1 ", "r2", "<", "/th", "ink", ">", "answer",
            ],
        );
        assert_eq!(got, "answer");
    }

    #[test]
    fn emitter_flush_drops_unterminated_think() {
        let mut em = StreamingEmitter::new(true);
        em.process("<think>");
        em.process("never closes");
        // No `</think>` — flush must not leak the reasoning text.
        assert_eq!(em.flush().unwrap_or_default(), "");
    }

    #[test]
    fn emitter_filter_off_passes_tags_through() {
        let mut em = StreamingEmitter::new(false);
        let got = feed(&mut em, &["<think>", "x", "</think>", "y"]);
        assert_eq!(got, "<think>x</think>y");
    }

    #[test]
    fn longest_prefix_suffix_basic() {
        assert_eq!(longest_prefix_suffix("abc<t", "<think>"), 2); // "<t"
        assert_eq!(longest_prefix_suffix("abc<thi", "<think>"), 4); // "<thi"
        assert_eq!(longest_prefix_suffix("abcd", "<think>"), 0);
        assert_eq!(longest_prefix_suffix("abc<think", "<think>"), 6); // "<think"
    }
}
