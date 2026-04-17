//! Qwen2.5 ChatML template for the interactive chat CLI.
//!
//! Phase 5 Step 2: single-turn only. Multi-turn conversation history
//! wraps into the same template in Step 3 by appending additional
//! `<|im_start|>user` / `<|im_start|>assistant` blocks.

pub struct ChatContext {
    pub system_prompt: String,
    pub user_input: String,
}

/// Render a ChatContext as a ChatML string ending with
/// `<|im_start|>assistant\n`. The generator continues from there.
pub fn format_single_turn(ctx: &ChatContext) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        ctx.system_prompt, ctx.user_input,
    )
}

/// Literal stop markers the assistant may emit. `BpeTokenizer::is_eog`
/// already recognises `<|im_end|>` and `<|endoftext|>` by name, so the
/// primary stop mechanism is the numeric ID check. This list is a
/// string-level safety net: if the model produced the marker as two or
/// more tokens that individually are not EOG, we stop when the
/// accumulated output ends with one of these patterns.
pub const STOP_MARKERS: &[&str] = &["<|im_end|>", "<|endoftext|>"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_chatml_single_turn() {
        let ctx = ChatContext {
            system_prompt: "You are a helpful assistant.".to_string(),
            user_input: "Hello".to_string(),
        };
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
        let ctx = ChatContext {
            system_prompt: "S".to_string(),
            user_input: "U".to_string(),
        };
        let result = format_single_turn(&ctx);
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }
}
