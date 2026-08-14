# MLLM benchmarks

Zero-shot evaluation and synthetic latency benchmarks for Gemma 3 4B and Gemma 4 E2B.

```bash
uv sync

uv run python scripts/benchmark_synthetic.py \
  --model gemma3 --batch-size 1 --runs 10 --warmup 3

uv run python scripts/evaluate.py \
  --model gemma4 --data-json ../test_data/train.json --batch-size 1
```

Model IDs, extraction fields, prompts, image size, and defaults are defined in `src/mllm/constants.py`. Results are written to `results/`.
