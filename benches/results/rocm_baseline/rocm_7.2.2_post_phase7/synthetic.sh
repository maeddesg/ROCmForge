#!/usr/bin/env bash
set -u
BIN=./target/release/rocmforge
MODELS=(
  "Qwen2.5-7B-Instruct-Q4_0.gguf"
  "Qwen3-8B-Q4_K_M.gguf"
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
PP_SIZES=(64 128 256 512)
RUNS=3

for model in "${MODELS[@]}"; do
  echo ""
  echo "========== $model =========="
  for pp in "${PP_SIZES[@]}"; do
    echo "--- pp=$pp ---"
    prompt=$(yes 'word' | head -n "$pp" | tr '\n' ' ')
    for run in $(seq 1 "$RUNS"); do
      $BIN --model "$HOME/models/$model" \
           --prompt "$prompt" \
           --max-tokens 1 --gpu --temperature 0.0 --top-p 1.0 --no-template \
           2>&1 | grep -E "^Prefill|^[0-9]+ tokens in" || true
    done
  done
  echo "--- decode 128 ---"
  for run in $(seq 1 "$RUNS"); do
    RUST_LOG=warn $BIN --model "$HOME/models/$model" \
                       --prompt "Hello" --max-tokens 128 --gpu \
                       --temperature 0.0 --top-p 1.0 --no-template \
                       2>&1 | grep -E "^Prefill|^[0-9]+ tokens in" || true
  done
done
