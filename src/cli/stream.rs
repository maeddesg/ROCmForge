//! Streaming chat-output emitter.
//!
//! Handles two concerns that are each per-token cross-cutting and bite
//! the naive `decode_token` + `print!` loop:
//!
//! * **UTF-8 multi-byte sequences split across tokens.** A 4-byte emoji
//!   is usually tokenised into 2–3 byte-level BPE tokens (`<0xF0>`
//!   `<0x9F>` `<0x98>` `<0x8A>`). `String::from_utf8_lossy` on a single
//!   token yields `�`. We buffer raw bytes and only hand complete
//!   codepoints to stdout.
//!
//! * **Qwen3 `<think>…</think>` chain-of-thought blocks.** The
//!   instruct-tuned Qwen3 models emit a reasoning trace between the
//!   tags before the real reply. Users don't want to see it by default.
//!   We swallow tokens while inside the block. A small prefix buffer
//!   holds output that could still be the start of an opening or
//!   closing tag.
//!
//! Stop markers from `template::STOP_MARKERS` (`<|im_end|>`,
//! `<|endoftext|>`) are checked on the full visible text once emitted.

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// Streaming assistant-output decoder with UTF-8 buffering and
/// `<think>` block suppression.
pub struct StreamingEmitter {
    byte_buf: Vec<u8>,
    pending: String,
    in_think: bool,
    show_thinking: bool,
    /// True after a `</think>` has just closed; swallow leading
    /// whitespace from the next emitted chunk so the visible reply
    /// starts cleanly.
    trim_leading_ws: bool,
    accumulated: String,
}

impl StreamingEmitter {
    pub fn new(show_thinking: bool) -> Self {
        Self {
            byte_buf: Vec::new(),
            pending: String::new(),
            in_think: false,
            show_thinking,
            trim_leading_ws: false,
            accumulated: String::new(),
        }
    }

    /// Push the raw bytes of a newly decoded token and return the
    /// substring (if any) that is now safe to write to stdout. Internal
    /// state tracks partial UTF-8 codepoints and partial `<think>` /
    /// `</think>` matches across token boundaries.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.byte_buf.extend_from_slice(bytes);
        let mut visible = String::new();

        // Drain whatever prefix of `byte_buf` is valid UTF-8.
        let drained = loop {
            match std::str::from_utf8(&self.byte_buf) {
                Ok(s) => {
                    let owned = s.to_string();
                    self.byte_buf.clear();
                    break owned;
                }
                Err(e) => {
                    let valid_up_to = e.valid_up_to();
                    if valid_up_to == 0 {
                        if let Some(_invalid_len) = e.error_len() {
                            // A definitively invalid byte sequence at the
                            // start: skip one byte and try again so we
                            // don't lock up on a lone 0xF0.
                            self.byte_buf.remove(0);
                            continue;
                        } else {
                            // Incomplete but still potentially valid —
                            // wait for more bytes.
                            break String::new();
                        }
                    }
                    let valid = std::str::from_utf8(&self.byte_buf[..valid_up_to])
                        .expect("valid_up_to comes from Utf8Error")
                        .to_string();
                    self.byte_buf.drain(..valid_up_to);
                    break valid;
                }
            }
        };

        if drained.is_empty() {
            return visible;
        }

        self.pending.push_str(&drained);
        loop {
            if self.in_think {
                // Swallow until we see the closing tag.
                if let Some(idx) = self.pending.find(CLOSE_TAG) {
                    self.pending.drain(..idx + CLOSE_TAG.len());
                    self.in_think = false;
                    self.trim_leading_ws = true;
                    continue;
                }
                // Keep only the last CLOSE_TAG.len()-1 chars as a
                // rolling suffix window; everything before that can't
                // overlap with the closing tag any more.
                let max_overlap = CLOSE_TAG.len().saturating_sub(1);
                if self.pending.len() > max_overlap {
                    let cut = self.pending.len() - max_overlap;
                    // align `cut` to a char boundary
                    let cut = self
                        .pending
                        .char_indices()
                        .rev()
                        .find(|(i, _)| *i <= cut)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.pending.drain(..cut);
                }
                return visible;
            }

            // Not in a think block. Either:
            //   (a) an opening tag starts inside `pending` → swallow up
            //       to and including the tag, then re-enter the loop in
            //       in_think mode.
            //   (b) `pending` could still be the start of an opening
            //       tag (prefix match) → hold the trailing prefix,
            //       emit the rest.
            //   (c) no prefix match at all → emit everything.
            if !self.show_thinking {
                if let Some(idx) = self.pending.find(OPEN_TAG) {
                    let head = &self.pending[..idx];
                    if !head.is_empty() {
                        visible.push_str(head);
                        self.accumulated.push_str(head);
                    }
                    self.pending.drain(..idx + OPEN_TAG.len());
                    self.in_think = true;
                    continue;
                }
            }

            // If we just closed a think block, eat any leading
            // whitespace in `pending` before computing what to emit.
            // This only fires ONCE — after the first non-whitespace
            // char (or a char that prevents further whitespace) we
            // clear the flag.
            if self.trim_leading_ws {
                let trim_len = self
                    .pending
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
                if trim_len > 0 {
                    self.pending.drain(..trim_len);
                }
                if !self.pending.is_empty() {
                    // Next char is non-whitespace — trimming done.
                    self.trim_leading_ws = false;
                } else {
                    // Still all whitespace so far; wait for more bytes.
                    return visible;
                }
            }

            let keep = if self.show_thinking {
                0
            } else {
                longest_open_tag_suffix_match(&self.pending)
            };
            let emit_len = self.pending.len() - keep;
            if emit_len > 0 {
                let emitted: String = self.pending.drain(..emit_len).collect();
                self.accumulated.push_str(&emitted);
                visible.push_str(&emitted);
            }
            return visible;
        }
    }

    /// Text emitted so far (with `<think>` blocks already stripped
    /// when `show_thinking` is false). Used for stop-marker detection.
    pub fn accumulated(&self) -> &str {
        &self.accumulated
    }

    /// Truncate the accumulated visible text — used after a stop
    /// marker is detected so it isn't kept in the conversation history.
    pub fn truncate_accumulated(&mut self, new_len: usize) {
        self.accumulated.truncate(new_len);
    }
}

/// Length (in bytes) of the longest suffix of `s` that is also a prefix
/// of `<think>`. Lets the emitter hold back `"<"` / `"<t"` / `"<thi"` /
/// etc. until we know whether a tag is actually starting. Slicing only
/// happens at char boundaries, so any multi-byte codepoint in `s` can
/// never be split.
fn longest_open_tag_suffix_match(s: &str) -> usize {
    // OPEN_TAG is ASCII, so any multi-byte char in `s` can't match a
    // non-zero prefix of it. Walk backwards over char boundaries and
    // only check ASCII-suffix slices.
    for (idx, _) in s.char_indices() {
        let suffix = &s[idx..];
        if OPEN_TAG.starts_with(suffix) {
            return suffix.len();
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_all(e: &mut StreamingEmitter, input: &str) -> String {
        let mut out = String::new();
        for ch in input.chars() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            out.push_str(&e.push_bytes(s.as_bytes()));
        }
        out
    }

    #[test]
    fn plain_text_passes_through() {
        let mut e = StreamingEmitter::new(false);
        let got = push_all(&mut e, "Hello, world!");
        assert_eq!(got, "Hello, world!");
        assert_eq!(e.accumulated(), "Hello, world!");
    }

    #[test]
    fn think_block_is_stripped() {
        let mut e = StreamingEmitter::new(false);
        let got = push_all(
            &mut e,
            "<think>Reasoning about 2+2…</think>The answer is 4.",
        );
        assert_eq!(got, "The answer is 4.");
        assert_eq!(e.accumulated(), "The answer is 4.");
    }

    #[test]
    fn show_thinking_keeps_block() {
        let mut e = StreamingEmitter::new(true);
        let got = push_all(&mut e, "<think>a</think>b");
        assert_eq!(got, "<think>a</think>b");
    }

    #[test]
    fn emoji_split_across_byte_tokens_is_complete() {
        let mut e = StreamingEmitter::new(false);
        let smiley = "😊".as_bytes();
        let mut out = String::new();
        for byte in smiley {
            out.push_str(&e.push_bytes(std::slice::from_ref(byte)));
        }
        assert_eq!(out, "😊");
    }

    #[test]
    fn partial_open_tag_is_held_until_resolved() {
        let mut e = StreamingEmitter::new(false);
        // A message starting with `<t` that is NOT the opening tag.
        let out = push_all(&mut e, "<thanks>");
        assert_eq!(out, "<thanks>");
    }

    #[test]
    fn leading_whitespace_after_close_tag_is_trimmed() {
        let mut e = StreamingEmitter::new(false);
        let got = push_all(&mut e, "<think>x</think>\n\nHello");
        assert_eq!(got, "Hello");
    }
}
