#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for image_tokens in 70 140 280 560 1120; do
    uv run python scripts/benchmark_synthetic.py \
        --model gemma4 \
        --batch-size 16 \
        --image-tokens "$image_tokens"
done
