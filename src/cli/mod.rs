//! Interactive chat CLI.
//!
//! `rocmforge chat --model <path>` routes here; the classic
//! `rocmforge --model <path> --prompt "…"` single-shot path remains
//! entirely untouched. Phase 5 Step 1 only scaffolds the subcommand:
//! startup validation + banner + input loop with `/help` and `/quit`.
//! Inference arrives in Step 2.

pub mod chat;
pub mod template;
pub mod validate;
