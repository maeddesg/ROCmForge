//! Qwen2.5 ChatML template for the interactive chat CLI.
//!
//! Phase 5 Step 3: multi-turn rendering. Conversation history is pulled
//! from [`ChatContext`]; [`format_single_turn`] is retained as a thin
//! wrapper so existing single-turn callers (and unit tests from Step 2)
//! keep working.

use super::context::{ChatContext, Role};

/// Render the full ChatML prompt ending with `<|im_start|>assistant\n`.
/// Both single-turn and multi-turn flows go through this function — the
/// distinguishing factor is whether `ctx.conversation_history` is empty.
pub fn format_multi_turn(ctx: &ChatContext) -> String {
    let mut out = String::with_capacity(
        ctx.system_prompt.len()
            + ctx.user_input.len()
            + ctx
                .conversation_history
                .iter()
                .map(|t| t.content.len() + 32)
                .sum::<usize>()
            + 64,
    );
    out.push_str("<|im_start|>system\n");
    out.push_str(&ctx.system_prompt);
    out.push_str("<|im_end|>\n");

    for turn in &ctx.conversation_history {
        let role = match turn.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&turn.content);
        out.push_str("<|im_end|>\n");
    }

    out.push_str("<|im_start|>user\n");
    out.push_str(&ctx.user_input);
    out.push_str("<|im_end|>\n");
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Backward-compatible single-turn wrapper.
pub fn format_single_turn(ctx: &ChatContext) -> String {
    debug_assert!(
        ctx.conversation_history.is_empty(),
        "format_single_turn called with non-empty history — use format_multi_turn"
    );
    format_multi_turn(ctx)
}

/// Literal stop markers the assistant may emit. `BpeTokenizer::is_eog`
/// already recognises `<|im_end|>` and `<|endoftext|>` by name, so the
/// primary stop mechanism is the numeric ID check. This list is a
/// string-level safety net for cases where the marker arrives as
/// multiple non-EOG tokens.
pub const STOP_MARKERS: &[&str] = &["<|im_end|>", "<|endoftext|>"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::context::{ChatContext, Role};

    #[test]
    fn qwen_chatml_single_turn() {
        let mut ctx = ChatContext::new("You are a helpful assistant.".to_string());
        ctx.user_input = "Hello".to_string();
        let result = format_single_turn(&ctx);
        assert_eq!(
            result,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn template_ends_with_assistant_prompt() {
        let mut ctx = ChatContext::new("S".to_string());
        ctx.user_input = "U".to_string();
        let result = format_multi_turn(&ctx);
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn multi_turn_preserves_history_order() {
        let mut ctx = ChatContext::new("You are helpful.".to_string());
        ctx.push_turn(Role::User, "Hi".to_string());
        ctx.push_turn(Role::Assistant, "Hello!".to_string());
        ctx.user_input = "How are you?".to_string();
        let result = format_multi_turn(&ctx);
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHow are you?<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
        // Order check: user "Hi" comes before assistant "Hello!" which comes
        // before "How are you?".
        let idx_hi = result.find("Hi").unwrap();
        let idx_hello = result.find("Hello!").unwrap();
        let idx_how = result.find("How are you?").unwrap();
        assert!(idx_hi < idx_hello && idx_hello < idx_how);
    }
}
