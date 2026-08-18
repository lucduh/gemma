#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for batch_size in {1..16}; do
    uv run python scripts/benchmark_synthetic.py \
        --model gemma4 \
        --batch-size "$batch_size"
done
