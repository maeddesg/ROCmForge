#!/usr/bin/env bash
set -u
BIN=./target/release/rocmforge
PROMPT_DIR=./benches/prompts
RUNS=3

MODELS=(
  "Qwen2.5-7B-Instruct-Q4_0.gguf"
  "Qwen3-8B-Q4_K_M.gguf"
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

for model in "${MODELS[@]}"; do
  echo ""
  echo "========== $model =========="
  for f in "$PROMPT_DIR"/*.txt; do
    name=$(basename "$f" .txt)
    prompt=$(cat "$f")
    echo "--- $name ---"
    for run in $(seq 1 "$RUNS"); do
      RUST_LOG=warn $BIN --model "$HOME/models/$model" \
                         --prompt "$prompt" --max-tokens 128 --gpu \
                         --temperature 0.0 --top-p 1.0 --no-template \
                         2>&1 | grep -E "^Prompt:|^Prefill|^128 tokens in" || true
    done
  done
done
