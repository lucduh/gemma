import argparse
import json
import time
from pathlib import Path

import torch

from mllm.config import (
    DATASETS,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    MODELS,
)
from mllm.dataset import Dataset
from mllm.inference import load_model, parse_json
from mllm.metrics import values_match


def main(
    model: str,
    device: str,
    dataset: str,
    split: str,
    index: int,
    max_new_tokens: int,
    image_tokens: int,
    adapter: Path | None,
):
    model_name = model
    dataset_name = dataset

    dataset = Dataset(dataset_name, split)
    model = load_model(model_name, device, adapter)

    image, image_path, gt = dataset[index]
    inputs = model.prepare_inputs([image], dataset.prompt, image_tokens)

    input_shapes = {
        key: list(value.shape)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    model.synchronize()
    start = time.perf_counter()
    texts, _, _ = model.generate(inputs, max_new_tokens)
    model.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    text = texts[0]
    prediction = parse_json(text, dataset.fields)
    field_results = {
        field: {
            "ground_truth": gt[field],
            "prediction": prediction[field],
            "correct": values_match(gt[field], prediction[field]),
        }
        for field in dataset.fields
    }

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}/{split}")
    print(f"Image: {image_path}")
    print(f"Input shapes: {input_shapes}")
    print(f"Generation latency: {latency_ms:.1f} ms")
    print("\nRaw response:")
    print(text)
    print("\nParsed fields:")
    print(json.dumps(field_results, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one model prediction.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_GEMMA4_IMAGE_TOKENS)
    parser.add_argument("--adapter", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
