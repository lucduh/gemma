# Gemma 4 visual-token pooling experiment

This experiment compares native low-resolution image encoding with high-resolution encoding followed by spatial pooling. Both evaluators use Gemma 4 E4B, greedy decoding, and batch size 1.

## 1. Smoke test

Run one document first to confirm the model and dataset paths are available:

```bash
uv run python scripts/evaluate_gemma4_manual.py \
  --dataset BR \
  --image-tokens 280 \
  --limit 1 \
  --warmup 0

uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 280 \
  --limit 1 \
  --warmup 0
```

## 2. Run the comparison

Use the same dataset, split, and limit for both commands. The scripts evaluate the first `N` documents, so matching limits select matching images.

```bash
# Native 280-token-budget baseline.
uv run python scripts/evaluate_gemma4_manual.py \
  --dataset BR \
  --image-tokens 280 \
  --limit 10

# Encode at the 1120-token budget and average-pool to the 280-budget grid.
uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 280 \
  --pool-method average \
  --limit 10

# Select one intact token near the center of each spatial region.
uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 280 \
  --pool-method spatial-select \
  --limit 10

# Merge nearby tokens with similar embeddings.
uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 560 \
  --pool-method similarity-merge \
  --limit 10
```

The token budgets are maxima. Document aspect ratio determines the actual source and target token counts. `average` aggregates each spatial region, `spatial-select` preserves one evenly spaced token per region, and `similarity-merge` performs local cosine-similarity matching before restoring spatial order.

## 3. Compare results

Results are saved under `results/evaluation/`. For a ten-document BR run, the files are:

```text
gemma4-e4b_manual_BR_test_280imgtok_n10.json
gemma4-e4b_pool_average_1120to280_BR_test_n10.json
gemma4-e4b_pool_spatial-select_1120to280_BR_test_n10.json
gemma4-e4b_pool_similarity-merge_1120to560_BR_test_n10.json
```

Compare these fields:

- `metrics.micro_f1`
- `metrics.document_accuracy`
- `timing.mean_ms.vision_encoder_ms`
- `timing.mean_ms.llm_prefill_ms`
- `timing.mean_ms.end_to_end_ms`
- `timing.mean_ms.pooling_ms` in the pooling result
- per-document visual grids and token counts under `timing.documents`

The pooling run still pays the full 1120-budget vision-encoder cost. Its expected savings are in LLM prefill and visual-token KV-cache usage.

## 4. LoRA comparison

Train the native 560-token-budget baseline:

```bash
uv run python scripts/train_lora.py \
  --model gemma4-e4b \
  --dataset BR \
  --run-name gemma4-e4b-BR-native560 \
  --image-tokens 560 \
  --batch-size 4
```

Evaluate its best adapter:

```bash
uv run python scripts/evaluate_gemma4_manual.py \
  --dataset BR \
  --image-tokens 560 \
  --adapter results/training/gemma4-e4b-BR-native560/best \
  --limit 10
```

Train LoRA with fixed 1120-to-560 spatial selection:

```bash
uv run python scripts/train_lora_pooling.py \
  --dataset BR \
  --run-name gemma4-e4b-BR-select1120to560 \
  --source-image-tokens 1120 \
  --target-image-tokens 560 \
  --pool-method spatial-select \
  --batch-size 4
```

Evaluate the pooled adapter using the same reduction configuration:

```bash
uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 560 \
  --pool-method spatial-select \
  --adapter results/training/gemma4-e4b-BR-select1120to560/best \
  --limit 10
```

The pooled training script supports configurable training batches and freezes the vision encoder, projector, and fixed reducer. Only language-model LoRA parameters are trained. Reduce `--batch-size` and increase `--gradient-accumulation` if GPU memory is limited.

## 5. Learned local resampler

Train the local attention resampler jointly with language-model LoRA:

```bash
uv run python scripts/train_lora_resampler.py \
  --dataset BR \
  --run-name gemma4-e4b-BR-resampler1120to560 \
  --source-image-tokens 1120 \
  --target-image-tokens 560 \
  --neighbors 4 \
  --batch-size 4
```

Evaluate the best LoRA and resampler checkpoint:

```bash
uv run python scripts/evaluate_gemma4_pooling.py \
  --dataset BR \
  --source-image-tokens 1120 \
  --target-image-tokens 560 \
  --pool-method local-resampler \
  --adapter results/training/gemma4-e4b-BR-resampler1120to560/best \
  --limit 10
```

The evaluator loads `resampler.pt` and `resampler_config.json` from the adapter directory automatically. Use `--resampler PATH` only when the resampler and LoRA adapter are stored separately.

The resampler emits one token for every target-grid position. Each output attends to the nearest source tokens using learned content scores and a spatial-distance bias. It is initialized close to fixed spatial selection. The vision encoder and Gemma multimodal projector remain frozen; the resampler and language-model LoRA are optimized with separate learning rates.
