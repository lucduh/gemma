import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import transformers
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
from mllm.metrics import calculate_metrics, values_match


class GenerationTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.first_token_recorded = False
        self.ended = False
        self.start_time = None
        self.first_token_time = None
        self.end_time = None
        self.start_event = None
        self.first_token_event = None
        self.end_event = None

    def start(self) -> None:
        if self.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.first_token_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self.start_time = time.perf_counter()

    def __call__(self, input_ids, scores):
        if not self.first_token_recorded:
            if self.device.type == "cuda":
                self.first_token_event.record(torch.cuda.current_stream(self.device))
            else:
                self.first_token_time = time.perf_counter()
            self.first_token_recorded = True
        return scores

    def end(self) -> None:
        if self.device.type == "cuda":
            self.end_event.record(torch.cuda.current_stream(self.device))
        else:
            self.end_time = time.perf_counter()
        self.ended = True

    def durations(self, generation_ms: float) -> tuple[float, float]:
        if not self.first_token_recorded or not self.ended:
            return generation_ms, 0.0

        if self.device.type == "cuda":
            prefill_ms = self.start_event.elapsed_time(self.first_token_event)
            decode_ms = self.first_token_event.elapsed_time(self.end_event)
        else:
            prefill_ms = (self.first_token_time - self.start_time) * 1000
            decode_ms = (self.end_time - self.first_token_time) * 1000
        return prefill_ms, decode_ms


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_stage(timings: list[dict], field: str) -> dict:
    values = [item[field] for item in timings]
    return {
        "mean_batch_ms": statistics.mean(values),
        "p50_batch_ms": percentile(values, 0.50),
        "p95_batch_ms": percentile(values, 0.95),
        "total_ms": sum(values),
    }


def summarize_timings(timings: list[dict], model, model_allocated_gb: float) -> dict:
    total_documents = sum(item["documents"] for item in timings)
    total_end_to_end_ms = sum(item["end_to_end_ms"] for item in timings)
    total_including_load_ms = sum(
        item["image_load_ms"] + item["end_to_end_ms"] for item in timings
    )
    total_generation_ms = sum(item["generation_ms"] for item in timings)
    total_decode_ms = sum(item["decode_ms"] for item in timings)
    token_counts = [count for item in timings for count in item["generated_tokens"]]
    reached_limits = [
        reached for item in timings for reached in item["reached_max_new_tokens"]
    ]
    total_tokens = sum(token_counts)
    total_decode_tokens = sum(max(count - 1, 0) for count in token_counts)

    latency = {
        name: summarize_stage(timings, field)
        for name, field in {
            "image_load": "image_load_ms",
            "preprocess": "preprocess_ms",
            "prefill": "prefill_ms",
            "decode": "decode_ms",
            "generation_overhead": "generation_overhead_ms",
            "generation": "generation_ms",
            "postprocess": "postprocess_ms",
            "end_to_end": "end_to_end_ms",
        }.items()
    }
    throughput = {
        "documents_per_second": (
            total_documents * 1000 / total_end_to_end_ms if total_end_to_end_ms else 0.0
        ),
        "documents_per_second_including_load": (
            total_documents * 1000 / total_including_load_ms
            if total_including_load_ms
            else 0.0
        ),
        "generation_tokens_per_second": (
            total_tokens * 1000 / total_generation_ms if total_generation_ms else 0.0
        ),
        "decode_tokens_per_second": (
            total_decode_tokens * 1000 / total_decode_ms if total_decode_ms else 0.0
        ),
        "amortized_ms_per_document": total_end_to_end_ms / total_documents,
    }
    generation = {
        "total_tokens": total_tokens,
        "mean_tokens_per_document": statistics.mean(token_counts),
        "reached_max_new_tokens": sum(reached_limits),
        "reached_max_new_tokens_rate": sum(reached_limits) / total_documents,
    }
    memory = {
        "model_allocated_gb": model_allocated_gb,
        "peak_allocated_gb": model.peak_memory_gb(),
        "peak_reserved_gb": model.peak_reserved_memory_gb(),
    }
    return {
        "latency": latency,
        "throughput": throughput,
        "generation": generation,
        "memory": memory,
        "batches": timings,
    }


def environment(model) -> dict:
    gpu = None
    if model.device.type == "cuda":
        gpu = torch.cuda.get_device_name(model.device)

    dtype = None
    if hasattr(model, "model"):
        parameter = next(model.model.parameters(), None)
        if parameter is not None:
            dtype = str(parameter.dtype)

    return {
        "gpu": gpu,
        "dtype": dtype,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
    }


def validate_arguments(
    batch_size: int, max_new_tokens: int, warmup: int, limit: int | None
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")


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
    validate_arguments(batch_size, max_new_tokens, warmup, limit)
    model_name = model
    dataset_name = dataset

    dataset = Dataset(dataset_name, split)
    if limit is not None:
        dataset.samples = dataset.samples.iloc[:limit]
    if not len(dataset):
        raise ValueError("The selected dataset contains no documents")

    model = load_model(model_name, device, adapter)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda samples: tuple(zip(*samples)),
    )

    if warmup:
        images, _, _ = next(iter(dataloader))
        for _ in range(warmup):
            inputs = model.prepare_inputs(images, dataset.prompt, image_tokens)
            model.generate(inputs, max_new_tokens)
            model.synchronize()

    model.synchronize()
    model.reset_peak_memory_stats()
    model_allocated_gb = model.allocated_memory_gb()
    predictions = []
    timings = []
    iterator = iter(dataloader)
    progress = tqdm(total=len(dataset), desc="evaluating")

    while True:
        image_load_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        image_load_ms = elapsed_ms(image_load_start)

        end_to_end_start = time.perf_counter()
        images, paths, gts = batch

        model.synchronize()
        preprocess_start = time.perf_counter()
        inputs = model.prepare_inputs(images, dataset.prompt, image_tokens)
        model.synchronize()
        preprocess_ms = elapsed_ms(preprocess_start)

        model.synchronize()
        generation_start = time.perf_counter()
        generation_timer = GenerationTimer(model.device)
        generation_timer.start()
        texts, token_counts, reached_token_limits = model.generate(
            inputs, max_new_tokens, observer=generation_timer
        )
        batch_documents = len(gts)
        if not (
            len(texts)
            == len(token_counts)
            == len(reached_token_limits)
            == batch_documents
        ):
            raise RuntimeError("Model output count does not match the batch size")
        model.synchronize()
        generation_ms = elapsed_ms(generation_start)
        prefill_ms, decode_ms = generation_timer.durations(generation_ms)
        generation_overhead_ms = max(generation_ms - prefill_ms - decode_ms, 0.0)

        postprocess_start = time.perf_counter()
        for gt, path, text in zip(gts, paths, texts, strict=True):
            prediction = parse_json(text, dataset.fields)
            field_results = {
                field: {
                    "ground_truth": gt[field],
                    "prediction": prediction[field],
                    "correct": values_match(gt[field], prediction[field]),
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
        postprocess_ms = elapsed_ms(postprocess_start)
        end_to_end_ms = elapsed_ms(end_to_end_start)

        timings.append(
            {
                "documents": batch_documents,
                "image_load_ms": image_load_ms,
                "preprocess_ms": preprocess_ms,
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "generation_overhead_ms": generation_overhead_ms,
                "generation_ms": generation_ms,
                "postprocess_ms": postprocess_ms,
                "end_to_end_ms": end_to_end_ms,
                "generated_tokens": token_counts,
                "reached_max_new_tokens": reached_token_limits,
            }
        )
        progress.update(batch_documents)

    progress.close()
    metrics = calculate_metrics(predictions, dataset.fields)
    timing_summary = summarize_timings(timings, model, model_allocated_gb)
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
            "environment": environment(model),
        },
        "metrics": metrics,
        "timing": timing_summary,
        "predictions": predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{adapter.parent.name}_{adapter.name}" if adapter else "zero_shot"
    image_variant = f"_{image_tokens}imgtok" if model.supports_image_tokens else ""
    output = output_dir / (
        f"{model_name}_{variant}_{dataset_name}_{split}{image_variant}_b{batch_size}.json"
    )
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    end_to_end = timing_summary["latency"]["end_to_end"]
    throughput = timing_summary["throughput"]
    memory = timing_summary["memory"]
    print("\nPer-field F1:")
    for field, values in metrics["per_field"].items():
        print(f"  {field}: {values['f1']:.3f}")
    print(f"\nMicro F1: {metrics['micro_f1']:.3f}")
    print(f"Document accuracy: {metrics['document_accuracy']:.3f}")
    print(f"P50 batch latency: {end_to_end['p50_batch_ms']:.1f} ms")
    print(
        f"Amortized latency: {throughput['amortized_ms_per_document']:.1f} ms/document"
    )
    print(f"Throughput: {throughput['documents_per_second']:.2f} documents/s")
    print(f"Peak GPU memory: {memory['peak_allocated_gb']:.2f} GB")
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
