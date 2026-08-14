import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
from PIL import Image
from tqdm import tqdm

from mllm.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WARMUP,
    FIELDS,
    MODELS,
    RESULTS_DIR,
)
from mllm.inference import generate, load_model, parse_json, prepare_inputs
from mllm.metrics import calculate_metrics


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ground_truth(sample: dict) -> dict[str, str | None]:
    values = {
        field["field_name"].split("/")[-1]: field.get("annotator_text", "").strip()
        for field in sample.get("fields", [])
    }
    return {field: values.get(field) or None for field in FIELDS}


def load_images(samples: list[dict], data_dir: Path) -> list[Image.Image]:
    return [Image.open(data_dir / sample["image"]).convert("RGB") for sample in samples]


def main(
    model: Annotated[str, typer.Option(help=f"Model alias: {', '.join(MODELS)}")],
    data_json: Annotated[Path, typer.Option(exists=True)],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    warmup: int = DEFAULT_WARMUP,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
):
    if model not in MODELS:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODELS)}")
    if batch_size < 1 or max_new_tokens < 1 or warmup < 0:
        raise typer.BadParameter(
            "batch-size and max-new-tokens must be positive; warmup cannot be negative"
        )
    if limit is not None and limit < 1:
        raise typer.BadParameter("limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")

    samples = json.loads(data_json.read_text())
    if limit is not None:
        samples = samples[:limit]
    loaded_model, processor = load_model(model)

    if warmup and samples:
        warmup_samples = samples[:batch_size]
        warmup_images = load_images(warmup_samples, data_json.parent)
        inputs = prepare_inputs(processor, warmup_images)
        for _ in range(warmup):
            generate(loaded_model, processor, inputs, max_new_tokens)
        for image in warmup_images:
            image.close()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    timings = []

    for batch_samples in tqdm(list(batches(samples, batch_size)), desc="evaluating"):
        batch_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        images = load_images(batch_samples, data_json.parent)
        inputs = prepare_inputs(processor, images)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        texts, generated_tokens = generate(
            loaded_model, processor, inputs, max_new_tokens
        )
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_start) * 1000

        for sample, text in zip(batch_samples, texts):
            gt = ground_truth(sample)
            pred = parse_json(text)
            field_results = {
                field: {
                    "ground_truth": gt[field],
                    "prediction": pred[field],
                    "correct": gt[field] == pred[field],
                }
                for field in FIELDS
            }
            predictions.append(
                {
                    "image": sample["image"],
                    "text": text,
                    "ground_truth": gt,
                    "prediction": pred,
                    "fields": field_results,
                    "document_correct": all(
                        value["correct"] for value in field_results.values()
                    ),
                }
            )

        end_to_end_ms = (time.perf_counter() - batch_start) * 1000
        timings.append(
            {
                "documents": len(batch_samples),
                "preprocess_ms": preprocess_ms,
                "generation_ms": generation_ms,
                "end_to_end_ms": end_to_end_ms,
                "generated_tokens": generated_tokens,
            }
        )
        for image in images:
            image.close()

    metrics = calculate_metrics(predictions)
    total_docs = len(predictions)
    total_end_to_end_s = sum(item["end_to_end_ms"] for item in timings) / 1000
    total_generation_s = sum(item["generation_ms"] for item in timings) / 1000
    total_tokens = sum(item["generated_tokens"] for item in timings)
    end_to_end_values = [item["end_to_end_ms"] for item in timings]

    timing_summary = {
        "mean_batch_latency_ms": statistics.mean(end_to_end_values) if timings else 0.0,
        "median_batch_latency_ms": statistics.median(end_to_end_values)
        if timings
        else 0.0,
        "mean_ms_per_document": total_end_to_end_s * 1000 / total_docs
        if total_docs
        else 0.0,
        "documents_per_second": total_docs / total_end_to_end_s
        if total_end_to_end_s
        else 0.0,
        "generated_tokens_per_second": total_tokens / total_generation_s
        if total_generation_s
        else 0.0,
        "total_preprocess_s": sum(item["preprocess_ms"] for item in timings) / 1000,
        "total_generation_s": total_generation_s,
        "total_end_to_end_s": total_end_to_end_s,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "batches": timings,
    }
    record = {
        "config": {
            "model": model,
            "model_id": MODELS[model],
            "data_json": str(data_json),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "limit": limit,
        },
        "metrics": metrics,
        "timing": timing_summary,
        "predictions": predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output = output_dir / f"evaluate_{model}_{timestamp}.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    print("\nPer-field strict F1:")
    for field, values in metrics["per_field"].items():
        print(f"  {field}: {values['f1']:.3f}")
    print(f"\nStrict micro F1: {metrics['micro_f1']:.3f}")
    print(f"Document accuracy: {metrics['document_accuracy']:.3f}")
    print(f"Mean latency: {timing_summary['mean_ms_per_document']:.1f} ms/document")
    print(f"Throughput: {timing_summary['documents_per_second']:.2f} documents/s")
    print(f"Peak GPU memory: {timing_summary['peak_gpu_memory_gb']:.2f} GB")
    print(f"Saved: {output}")


if __name__ == "__main__":
    typer.run(main)
