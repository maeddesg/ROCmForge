# Benchmarks

## Speculative Decode Benchmark (`bench_spec.sh`)

Runs 15 prompts (5 prose, 5 code, 5 chat) across 4 modes:
- **baseline**: Target model only (no draft)
- **spec_depth_1/3/5**: Speculative decoding with draft model

### Usage

```bash
cargo build --release --features gpu
./benches/bench_spec.sh
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROCMFORGE_BENCH_TARGET_MODEL` | `~/models/Qwen2.5-7B-Instruct-Q4_0.gguf` | Target model path |
| `ROCMFORGE_BENCH_DRAFT_MODEL` | `~/models/qwen2.5-0.5b-instruct-q4_0.gguf` | Draft model path |

### Output JSON Schema

```json
{
  "meta": {
    "git_sha": "string",
    "timestamp": "int (unix epoch)",
    "date": "string (ISO 8601)",
    "rocm_version": "string",
    "target_model": "string (file path)",
    "draft_model": "string (file path)",
    "max_tokens": "int",
    "gpu": "string (GPU model name)"
  },
  "runs": [
    {
      "mode": "baseline | spec_depth_1 | spec_depth_3 | spec_depth_5",
      "spec_depth": "int (0 for baseline)",
      "prompt": "string (filename without extension)",
      "tok_s": "float (tokens per second)",
      "acceptance_pct": "float (% of draft tokens accepted, 0 for baseline)",
      "avg_accepted_per_step": "float (avg draft tokens accepted per spec step)",
      "e_n": "float (expected committed tokens per target forward)",
      "n_steps": "int (number of spec steps, 0 for baseline)",
      "n_generated": "int (total tokens generated)",
      "gpu_power_before_w": "float | null (GPU power in watts before run)",
      "gpu_power_after_w": "float | null (GPU power in watts after run)"
    }
  ]
}
```

### Results Directory

Results are saved to `benches/results/` as `baseline_{gitsha}_{timestamp}.json`.
Compare across commits to track performance regressions.
