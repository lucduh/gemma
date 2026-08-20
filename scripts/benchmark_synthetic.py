import json
import statistics
import time
from pathlib import Path

import torch
import typer
from PIL import Image

from mllm.constants import (
    BENCHMARK_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    MODELS,
    SYNTHETIC_IMAGE_SIZE,
    SYNTHETIC_OUTPUT_TOKENS,
)
from mllm.inference import load_model, prepare_inputs


def main(
    model: str = typer.Option(..., help=f"Model alias: {', '.join(MODELS)}"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
    height: int = SYNTHETIC_IMAGE_SIZE[0],
    width: int = SYNTHETIC_IMAGE_SIZE[1],
    output_tokens: int = SYNTHETIC_OUTPUT_TOKENS,
    image_tokens: int = DEFAULT_GEMMA4_IMAGE_TOKENS,
    adapter: Path | None = None,
):
    images = [Image.new("RGB", (width, height), "white") for _ in range(batch_size)]
    loaded_model, processor = load_model(
        model, adapter=str(adapter) if adapter else None
    )
    inputs = prepare_inputs(
        processor,
        images,
        "Describe the image briefly.",
        image_tokens=image_tokens,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            loaded_model.generate(
                **inputs,
                min_new_tokens=output_tokens,
                max_new_tokens=output_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies_ms = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loaded_model.generate(
                **inputs,
                min_new_tokens=output_tokens,
                max_new_tokens=output_tokens,
                do_sample=False,
            )
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000)

    total_s = sum(latencies_ms) / 1000
    record = {
        "config": {
            "model": model,
            "model_id": MODELS[model],
            "attention_implementation": getattr(
                loaded_model.config, "_attn_implementation", None
            ),
            "adapter": str(adapter) if adapter else None,
            "batch_size": batch_size,
            "runs": runs,
            "warmup": warmup,
            "source_image_size": [height, width],
            "output_tokens": output_tokens,
            "image_tokens": image_tokens if model == "gemma4" else 256,
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
            "generated_tokens_per_second": batch_size * runs * output_tokens / total_s,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "raw_batch_ms": latencies_ms,
        },
    }

    BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variant = f"{adapter.parent.name}_{adapter.name}" if adapter else "zero_shot"
    image_variant = f"_{image_tokens}imgtok" if model == "gemma4" else ""
    output = BENCHMARK_RESULTS_DIR / (
        f"{model}_{variant}_bs={batch_size}_{height}x{width}"
        f"_{output_tokens}tok{image_variant}.json"
    )
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record["timing"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    typer.run(main)
