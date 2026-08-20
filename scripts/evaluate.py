import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from mllm.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WARMUP,
    EVALUATION_RESULTS_DIR,
    MODELS,
)
from mllm.datasets import DATASETS, ground_truth, image_path, load_split
from mllm.inference import generate, load_model, parse_json, prepare_inputs
from mllm.metrics import calculate_metrics


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_images(samples: list[dict], data_dir: Path) -> list[Image.Image]:
    return [
        Image.open(image_path(sample, data_dir)).convert("RGB") for sample in samples
    ]


def main(
    model: str,
    dataset: str,
    split: str,
    batch_size: int,
    max_new_tokens: int,
    image_tokens: int,
    warmup: int,
    adapter: Path | None,
    limit: int | None,
    output_dir: Path,
):
    dataset_split = load_split(dataset, split)
    samples = dataset_split.samples
    fields = dataset_split.fields
    if limit is not None:
        samples = samples[:limit]
    loaded_model, processor = load_model(
        model, adapter=str(adapter) if adapter else None
    )

    if warmup and samples:
        warmup_samples = samples[:batch_size]
        warmup_images = load_images(warmup_samples, dataset_split.directory)
        inputs = prepare_inputs(
            processor,
            warmup_images,
            dataset_split.prompt,
            image_tokens=image_tokens,
        )
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
        images = load_images(batch_samples, dataset_split.directory)
        inputs = prepare_inputs(
            processor, images, dataset_split.prompt, image_tokens=image_tokens
        )
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        texts, generated_tokens = generate(
            loaded_model, processor, inputs, max_new_tokens
        )
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_start) * 1000

        for sample, text in zip(batch_samples, texts):
            gt = ground_truth(sample, fields)
            pred = parse_json(text, fields)
            field_results = {
                field: {
                    "ground_truth": gt[field],
                    "prediction": pred[field],
                    "correct": gt[field] == pred[field],
                }
                for field in fields
            }
            predictions.append(
                {
                    "image": sample["image_path"],
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

    metrics = calculate_metrics(predictions, fields)
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
            "attention_implementation": getattr(
                loaded_model.config, "_attn_implementation", None
            ),
            "adapter": str(adapter) if adapter else None,
            "dataset": dataset,
            "split": split,
            "data_path": str(dataset_split.path),
            "fields": list(fields),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "image_tokens": image_tokens if model == "gemma4" else 256,
            "warmup": warmup,
            "limit": limit,
        },
        "metrics": metrics,
        "timing": timing_summary,
        "predictions": predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{adapter.parent.name}_{adapter.name}" if adapter else "zero_shot"
    image_variant = f"_{image_tokens}imgtok" if model == "gemma4" else ""
    output = output_dir / (f"{model}_{variant}_{dataset}_{split}{image_variant}.json")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_GEMMA4_IMAGE_TOKENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
