import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from mllm.config import (
    DATASETS,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WARMUP,
    EVALUATION_RESULTS_DIR,
    MODELS,
)
from mllm.dataset import Dataset
from mllm.gemma4_manual import (
    build_llm_inputs,
    encode_vision,
    encoded_soft_grid,
    greedy_decode,
    move_to_device,
    prefill,
    prepare_inputs,
    project_vision,
)
from mllm.inference import parse_json
from mllm.metrics import calculate_metrics, values_match

MODEL_NAME = "gemma4-e4b"
DEFAULT_LIMIT = 10
STAGES = (
    "image_load_ms",
    "processor_ms",
    "device_transfer_ms",
    "vision_encoder_ms",
    "vision_projector_ms",
    "embedding_setup_ms",
    "llm_prefill_ms",
    "llm_decode_ms",
    "text_decode_ms",
    "postprocess_ms",
    "end_to_end_ms",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def run_inference(model, processor, image, prompt, image_tokens, max_new_tokens):
    device = next(model.parameters()).device
    timings = {}

    start = time.perf_counter()
    inputs = prepare_inputs(processor, image, prompt, image_tokens)
    visual_grid = encoded_soft_grid(
        inputs["image_position_ids"], processor.image_processor.pooling_kernel_size
    )
    timings["processor_ms"] = elapsed_ms(start)

    synchronize(device)
    start = time.perf_counter()
    inputs = move_to_device(inputs, device)
    synchronize(device)
    timings["device_transfer_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    vision_tokens = encode_vision(model, inputs)
    synchronize(device)
    timings["vision_encoder_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    image_features = project_vision(model, vision_tokens)
    synchronize(device)
    timings["vision_projector_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    inputs_embeds, per_layer_inputs, visual_tokens = build_llm_inputs(
        model, inputs, image_features
    )
    synchronize(device)
    timings["embedding_setup_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    prefill_output = prefill(model, inputs, inputs_embeds, per_layer_inputs)
    synchronize(device)
    timings["llm_prefill_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    generated_ids = greedy_decode(
        model,
        prefill_output,
        inputs["attention_mask"],
        max_new_tokens,
    )
    synchronize(device)
    timings["llm_decode_ms"] = elapsed_ms(start)

    start = time.perf_counter()
    text = processor.decode(generated_ids[0], skip_special_tokens=True)
    timings["text_decode_ms"] = elapsed_ms(start)
    visual_token_info = {
        "visual_grid": list(visual_grid),
        "visual_tokens": visual_tokens,
    }
    return text, generated_ids.shape[1], visual_token_info, timings


def summarize_timings(timings: list[dict], peak_memory_gb: float) -> dict:
    total_generation_ms = sum(
        item["llm_prefill_ms"] + item["llm_decode_ms"] for item in timings
    )
    total_generated_tokens = sum(item["generated_tokens"] for item in timings)
    return {
        "mean_ms": {
            stage: statistics.mean(item[stage] for item in timings) for stage in STAGES
        },
        "median_ms": {
            stage: statistics.median(item[stage] for item in timings)
            for stage in STAGES
        },
        "total_end_to_end_s": sum(item["end_to_end_ms"] for item in timings) / 1000,
        "generated_tokens_per_second": (
            total_generated_tokens * 1000 / total_generation_ms
            if total_generation_ms
            else 0.0
        ),
        "peak_gpu_memory_gb": peak_memory_gb,
        "documents": timings,
    }


def main(
    device: str,
    dataset: str,
    split: str,
    image_tokens: int,
    max_new_tokens: int,
    adapter: Path | None,
    warmup: int,
    limit: int,
    output_dir: Path,
):

    dataset_name = dataset
    dataset = Dataset(dataset_name, split)
    document_count = min(limit, len(dataset))
    if document_count == 0:
        raise ValueError("The selected dataset contains no documents")

    _, model_id = MODELS[MODEL_NAME]
    torch_device = torch.device(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(
        torch_device
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()

    if warmup:
        image, _, _ = dataset[0]
        for _ in range(warmup):
            with torch.inference_mode():
                run_inference(
                    model,
                    processor,
                    image,
                    dataset.prompt,
                    image_tokens,
                    max_new_tokens,
                )

    synchronize(torch_device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    predictions = []
    timings = []
    for index in tqdm(range(document_count), desc="evaluating"):
        end_to_end_start = time.perf_counter()

        start = time.perf_counter()
        image, image_path, ground_truth = dataset[index]
        image_load_ms = elapsed_ms(start)

        with torch.inference_mode():
            text, generated_tokens, visual_token_info, document_timings = run_inference(
                model,
                processor,
                image,
                dataset.prompt,
                image_tokens,
                max_new_tokens,
            )

        start = time.perf_counter()
        prediction = parse_json(text, dataset.fields)
        field_results = {
            field: {
                "ground_truth": ground_truth[field],
                "prediction": prediction[field],
                "correct": values_match(ground_truth[field], prediction[field]),
            }
            for field in dataset.fields
        }
        predictions.append(
            {
                "image": image_path,
                "text": text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "fields": field_results,
                "document_correct": all(
                    value["correct"] for value in field_results.values()
                ),
            }
        )
        postprocess_ms = elapsed_ms(start)

        document_timings.update(
            {
                "image": image_path,
                "image_load_ms": image_load_ms,
                "postprocess_ms": postprocess_ms,
                "end_to_end_ms": elapsed_ms(end_to_end_start),
                "generated_tokens": generated_tokens,
                **visual_token_info,
            }
        )
        timings.append(document_timings)

    peak_memory_gb = (
        torch.cuda.max_memory_allocated(torch_device) / 1024**3
        if torch_device.type == "cuda"
        else 0.0
    )
    metrics = calculate_metrics(predictions, dataset.fields)
    timing_summary = summarize_timings(timings, peak_memory_gb)
    record = {
        "config": {
            "model": MODEL_NAME,
            "model_path": str(model_id),
            "adapter": str(adapter) if adapter else None,
            "dataset": dataset_name,
            "split": split,
            "data_path": str(dataset.path),
            "fields": list(dataset.fields),
            "batch_size": 1,
            "image_tokens": image_tokens,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "limit": limit,
            "documents_evaluated": document_count,
            "device": str(torch_device),
        },
        "metrics": metrics,
        "timing": timing_summary,
        "predictions": predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    variant = f"{adapter.parent.name}_" if adapter else ""
    output = output_dir / (
        f"{MODEL_NAME}_{variant}manual_{dataset_name}_{split}_{image_tokens}imgtok_"
        f"n{document_count}.json"
    )
    output.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nMicro F1: {metrics['micro_f1']:.3f}")
    print(f"Document accuracy: {metrics['document_accuracy']:.3f}")
    print(
        "Mean end-to-end latency: "
        f"{timing_summary['mean_ms']['end_to_end_ms']:.1f} ms/document"
    )
    print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    print(f"Saved: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manually evaluate Gemma 4 E4B with batch size 1."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_GEMMA4_IMAGE_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
