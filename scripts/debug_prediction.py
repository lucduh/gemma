import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from mllm.constants import DEFAULT_MAX_NEW_TOKENS, MODELS
from mllm.datasets import DATASETS, ground_truth, image_path, load_split
from mllm.inference import generate, load_model, parse_json, prepare_inputs


def main(
    model: str,
    dataset: str,
    split: str,
    index: int,
    max_new_tokens: int,
    adapter: Path | None,
):
    dataset_split = load_split(dataset, split)
    sample = dataset_split.samples[index]
    path = image_path(sample, dataset_split.directory)
    with Image.open(path) as source:
        image = source.convert("RGB")

    loaded_model, processor = load_model(
        model, adapter=str(adapter) if adapter else None
    )
    inputs = prepare_inputs(processor, [image], dataset_split.prompt)
    image.close()

    input_shapes = {
        key: list(value.shape)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    torch.cuda.synchronize()
    start = time.perf_counter()
    texts, _ = generate(loaded_model, processor, inputs, max_new_tokens)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    text = texts[0]
    fields = dataset_split.fields
    truth = ground_truth(sample, fields)
    prediction = parse_json(text, fields)
    field_results = {
        field: {
            "ground_truth": truth[field],
            "prediction": prediction[field],
            "correct": truth[field] == prediction[field],
        }
        for field in fields
    }

    print(f"Dataset: {dataset}/{split}")
    print(f"Image: {path}")
    print(f"Input shapes: {input_shapes}")
    print(f"Generation latency: {latency_ms:.1f} ms")
    print("\nRaw response:")
    print(text)
    print("\nParsed fields:")
    print(json.dumps(field_results, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one model prediction.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--adapter", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
