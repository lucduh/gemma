# Multimodal document extraction

Zero-shot evaluation and LoRA training for Gemma 3, Gemma 4, and InternVL 3 across multiple document datasets. Synthetic benchmarking is awaiting migration to the new runtime API.

## Setup

```bash
uv sync
```

Production datasets are registered with explicit paths in `src/mllm/config.py`:

```text
/domino/datasets/local/MLLM/data/
├── BR/
├── KARAPASS_DEATH/
└── KARAPASS_ID/
```

Each prepared Parquet split contains `source_index`, an absolute `image_path`, and the extraction fields.

## Prepare and analyze data

The preparation command discovers fields from `train.json` and `test.json`, resolves duplicate annotations, excludes rows with missing images, and writes clean Parquet splits.

```bash
# Prepare all production datasets under the configured data root.
uv run python scripts/data/prepare.py

# Prepare one dataset again.
uv run python scripts/data/prepare.py --dataset BR --overwrite

# Analyze the resulting Parquet files.
uv run python scripts/data/analyze.py --dataset BR
```

`--data-root` can be passed to the preparation and analysis commands without changing the registered inference paths.

## Run inference and training

```bash
# Inspect one prediction.
uv run python scripts/debug_prediction.py \
  --model gemma3 --dataset BR --split test --index 0

# Evaluate zero-shot, optionally on a small subset.
uv run python scripts/evaluate.py \
  --model internvl --dataset KARAPASS_DEATH --split test --limit 10

# Evaluate a Gemma 4 variant.
uv run python scripts/evaluate.py \
  --model gemma4-e2b --dataset BR --split test

# Fine-tune a LoRA adapter.
uv run python scripts/train_lora.py \
  --model gemma3 --dataset KARAPASS_ID --run-name gemma3-karapass-id

# Evaluate the adapter.
uv run python scripts/evaluate.py \
  --model gemma3 --dataset KARAPASS_ID \
  --adapter results/training/gemma3-karapass-id/best
```

Evaluation artifacts include per-batch profiling for image loading, preprocessing, prefill, decoding, generation overhead, postprocessing, and end-to-end latency. `image_load_ms` measures the wait for the DataLoader; `end_to_end_ms` starts after a batch is returned and ends when its prediction records are complete. For batches larger than one, latency values describe the whole synchronous batch, while amortized milliseconds per document and documents per second describe throughput. Output filenames include the batch size so batch sweeps do not overwrite each other.

The tiny `TEST` dataset and weight-free `test` model run the inference code on CPU:

```bash
uv run python scripts/debug_prediction.py --model test --device cpu --dataset TEST
uv run python scripts/evaluate.py --model test --device cpu --dataset TEST
```

Configuration and implementation are separated as follows:

- `src/mllm/config.py` — model paths, dataset paths, and runtime defaults
- `src/mllm/dataset.py` — Parquet dataset loading
- `src/mllm/inference.py` — model loading, input preparation, and generation
- `src/mllm/prompts.py` — dataset-specific prompts

InternVL uses the Transformers-native `OpenGVLab/InternVL3-8B-hf` checkpoint.

Evaluation artifacts are written under `results/evaluation/`; training adapters and metrics are written under `results/training/`.
