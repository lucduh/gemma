# Gemma document extraction

Zero-shot evaluation, LoRA fine-tuning, and latency benchmarks for Gemma 3 4B and Gemma 4 E2B.

```bash
uv sync

# Inspect one prediction.
uv run python scripts/debug_prediction.py \
  --model gemma3 --data-json DATA.json --index 0

# Evaluate zero-shot, optionally on a small subset.
uv run python scripts/evaluate.py \
  --model gemma3 --data-json DATA.json --limit 10

# Fine-tune a LoRA adapter.
uv run python scripts/train_lora.py \
  --model gemma3 --data-json TRAIN.json --run-name gemma3-lora \
  --batch-size 4 --gradient-accumulation 2

# Inspect or evaluate the fine-tuned model.
uv run python scripts/debug_prediction.py \
  --model gemma3 --adapter results/training/gemma3-lora/best \
  --data-json TEST.json --index 0

uv run python scripts/evaluate.py \
  --model gemma3 --adapter results/training/gemma3-lora/best \
  --data-json TEST.json

# Synthetic latency benchmark.
uv run python scripts/benchmark_synthetic.py \
  --model gemma3 --batch-size 1 --runs 10 --warmup 3
```

All paths and defaults are defined in `src/mllm/constants.py`. Artifacts use stable paths under `results/`:

```text
results/
├── benchmarks/
├── evaluation/
└── training/<run-name>/
    ├── best/
    ├── last/
    └── train.json
```
