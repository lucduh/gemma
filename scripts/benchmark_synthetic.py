import json
import statistics
import time
from datetime import UTC, datetime

import torch
import typer
from PIL import Image

from mllm.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    MODELS,
    RESULTS_DIR,
    SYNTHETIC_IMAGE_SIZE,
    SYNTHETIC_MAX_NEW_TOKENS,
)
from mllm.inference import generate, load_model, prepare_inputs


def main(
    model: str = typer.Option(..., help=f"Model alias: {', '.join(MODELS)}"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
):
    if model not in MODELS:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODELS)}")
    if batch_size < 1 or runs < 1 or warmup < 0:
        raise typer.BadParameter(
            "batch-size and runs must be positive; warmup cannot be negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    height, width = SYNTHETIC_IMAGE_SIZE
    images = [Image.new("RGB", (width, height), "white") for _ in range(batch_size)]
    loaded_model, processor = load_model(model)
    inputs = prepare_inputs(processor, images)

    for _ in range(warmup):
        generate(
            loaded_model,
            processor,
            inputs,
            SYNTHETIC_MAX_NEW_TOKENS,
            fixed_length=True,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms = []
    generated_tokens = 0
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, tokens = generate(
            loaded_model,
            processor,
            inputs,
            SYNTHETIC_MAX_NEW_TOKENS,
            fixed_length=True,
        )
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)
        generated_tokens += tokens

    total_s = sum(latencies_ms) / 1000
    record = {
        "config": {
            "model": model,
            "model_id": MODELS[model],
            "batch_size": batch_size,
            "runs": runs,
            "warmup": warmup,
            "source_image_size": [height, width],
            "max_new_tokens": SYNTHETIC_MAX_NEW_TOKENS,
            "input_shapes": {
                key: list(value.shape)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            },
        },
        "timing": {
            "mean_batch_ms": statistics.mean(latencies_ms),
            "median_batch_ms": statistics.median(latencies_ms),
            "mean_ms_per_document": statistics.mean(latencies_ms) / batch_size,
            "documents_per_second": batch_size * runs / total_s,
            "generated_tokens_per_second": generated_tokens / total_s,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "raw_batch_ms": latencies_ms,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / f"synthetic_{model}_{timestamp}.json"
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record["timing"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    typer.run(main)
