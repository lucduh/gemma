import json
import time
from typing import Annotated

import torch
import typer
from PIL import Image

from mllm.constants import DEFAULT_MAX_NEW_TOKENS, MODELS
from mllm.datasets import DATASETS, ground_truth, image_path, load_split
from mllm.inference import generate, load_model, parse_json, prepare_inputs


def main(
    model: Annotated[str, typer.Option(help=f"Model alias: {', '.join(MODELS)}")],
    dataset: Annotated[str, typer.Option(help=f"Dataset name: {', '.join(DATASETS)}")],
    split: str = "test",
    index: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    adapter: str | None = None,
):
    if model not in MODELS:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODELS)}")
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASETS)}")

    dataset_split = load_split(dataset, split)
    samples = dataset_split.samples
    if not 0 <= index < len(samples):
        raise typer.BadParameter(f"index must be between 0 and {len(samples) - 1}")

    sample = samples[index]
    path = image_path(sample, dataset_split.directory)
    image = Image.open(path).convert("RGB")

    loaded_model, processor = load_model(model, adapter=adapter)
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
    gt = ground_truth(sample, fields)
    prediction = parse_json(text, fields)
    field_results = {
        field: {
            "ground_truth": gt[field],
            "prediction": prediction[field],
            "correct": gt[field] == prediction[field],
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


if __name__ == "__main__":
    typer.run(main)
