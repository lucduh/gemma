import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mllm.config import (
    DATASETS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WARMUP,
    EVALUATION_RESULTS_DIR,
    MODELS,
)
from mllm.dataset import Dataset
from mllm.inference import load_model, parse_json
from mllm.metrics import calculate_metrics


def main(
    model: str,
    device: str,
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
    model_name = model
    dataset_name = dataset

    dataset = Dataset(dataset_name, split)
    model = load_model(model_name, device, adapter)

    if limit is not None:
        dataset.samples = dataset.samples.iloc[:limit]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda samples: tuple(zip(*samples)),
    )

    if warmup:
        images, _, _ = next(iter(dataloader))
        inputs = model.prepare_inputs(images, dataset.prompt, image_tokens)
        for _ in range(warmup):
            model.generate(inputs, max_new_tokens)
        model.synchronize()

    model.reset_peak_memory_stats()
    predictions = []
    timings = []

    for images, paths, gts in tqdm(dataloader, desc="evaluating"):
        batch_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        inputs = model.prepare_inputs(images, dataset.prompt, image_tokens)
        model.synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        generation_start = time.perf_counter()
        texts, generated_tokens = model.generate(inputs, max_new_tokens)
        model.synchronize()
        generation_ms = (time.perf_counter() - generation_start) * 1000

        for gt, path, text in zip(gts, paths, texts, strict=True):
            prediction = parse_json(text, dataset.fields)
            field_results = {
                field: {
                    "ground_truth": gt[field],
                    "prediction": prediction[field],
                    "correct": gt[field] == prediction[field],
                }
                for field in dataset.fields
            }
            predictions.append(
                {
                    "image": path,
                    "text": text,
                    "ground_truth": gt,
                    "prediction": prediction,
                    "fields": field_results,
                    "document_correct": all(
                        value["correct"] for value in field_results.values()
                    ),
                }
            )

        timings.append(
            {
                "documents": len(gts),
                "preprocess_ms": preprocess_ms,
                "generation_ms": generation_ms,
                "end_to_end_ms": (time.perf_counter() - batch_start) * 1000,
                "generated_tokens": generated_tokens,
            }
        )

    metrics = calculate_metrics(predictions, dataset.fields)
    total_documents = len(predictions)
    total_end_to_end_s = sum(item["end_to_end_ms"] for item in timings) / 1000
    total_generation_s = sum(item["generation_ms"] for item in timings) / 1000
    total_tokens = sum(item["generated_tokens"] for item in timings)
    end_to_end_values = [item["end_to_end_ms"] for item in timings]

    timing_summary = {
        "mean_batch_latency_ms": statistics.mean(end_to_end_values),
        "median_batch_latency_ms": statistics.median(end_to_end_values),
        "mean_ms_per_document": total_end_to_end_s * 1000 / total_documents,
        "documents_per_second": total_documents / total_end_to_end_s,
        "generated_tokens_per_second": total_tokens / total_generation_s,
        "total_preprocess_s": sum(item["preprocess_ms"] for item in timings) / 1000,
        "total_generation_s": total_generation_s,
        "total_end_to_end_s": total_end_to_end_s,
        "peak_gpu_memory_gb": model.peak_memory_gb(),
        "batches": timings,
    }
    record = {
        "config": {
            "model": model_name,
            "model_path": str(model.model_id),
            "adapter": str(adapter) if adapter else None,
            "dataset": dataset_name,
            "split": split,
            "data_path": str(dataset.path),
            "fields": list(dataset.fields),
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "image_tokens": image_tokens if model.supports_image_tokens else None,
            "warmup": warmup,
            "limit": limit,
            "device": str(model.device),
        },
        "metrics": metrics,
        "timing": timing_summary,
        "predictions": predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{adapter.parent.name}_{adapter.name}" if adapter else "zero_shot"
    image_variant = f"_{image_tokens}imgtok" if model.supports_image_tokens else ""
    output = output_dir / (
        f"{model_name}_{variant}_{dataset_name}_{split}{image_variant}.json"
    )
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    parser.add_argument("--device", default="cuda")
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
