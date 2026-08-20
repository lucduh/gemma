# Multimodal document extraction

Zero-shot evaluation, LoRA fine-tuning, and latency benchmarks for Gemma 3, Gemma 4, and InternVL 3 across multiple document datasets.

## Setup

```bash
uv sync
```

Datasets are selected by name and live under `/domino/datasets/mllm/data` by default:

```text
/domino/datasets/mllm/data/
├── BR/
├── KARAPASS_DEATH/
└── KARAPASS_ID/
```

Set `MLLM_DATA_ROOT` to use another root directory.

## Prepare and analyze data

The preparation command discovers fields from `train.json` and `test.json`, resolves duplicate annotations, excludes rows with missing images, and writes clean Parquet splits.

```bash
# Prepare all registered datasets.
uv run python scripts/data/prepare.py

# Prepare one dataset again.
uv run python scripts/data/prepare.py --dataset BR --overwrite

# Analyze the resulting Parquet files.
uv run python scripts/data/analyze.py --dataset BR
```

## Run inference and training

```bash
# Inspect one prediction.
uv run python scripts/debug_prediction.py \
  --model gemma3 --dataset BR --split test --index 0

# Evaluate zero-shot, optionally on a small subset.
uv run python scripts/evaluate.py \
  --model internvl --dataset KARAPASS_DEATH --split test --limit 10

# Fine-tune a LoRA adapter.
uv run python scripts/train_lora.py \
  --model gemma3 --dataset KARAPASS_ID --run-name gemma3-karapass-id \
  --batch-size 4 --gradient-accumulation 2

# Evaluate the adapter.
uv run python scripts/evaluate.py \
  --model gemma3 --dataset KARAPASS_ID \
  --adapter results/training/gemma3-karapass-id/best

# Synthetic latency benchmark.
uv run python scripts/benchmark_synthetic.py \
  --model internvl --batch-size 1 --runs 10 --warmup 3
```

Model aliases and paths are in `src/mllm/constants.py`. Dataset registration and prompts are intentionally separate:

- `src/mllm/datasets.py` — dataset registry and Parquet loading
- `src/mllm/prompts.py` — all dataset-specific prompts in one place
- `src/mllm/constants.py` — model IDs, paths, and runtime defaults

InternVL uses the Transformers-native `OpenGVLab/InternVL3-8B-hf` checkpoint.

Artifacts use stable paths under `results/`:

```text
results/
├── benchmarks/
├── evaluation/
└── training/<run-name>/
    ├── best/
    ├── last/
    └── train.json
```
