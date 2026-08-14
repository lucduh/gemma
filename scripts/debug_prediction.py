import json
import time
from pathlib import Path
from typing import Annotated

import torch
import typer
from PIL import Image

from mllm.constants import DEFAULT_MAX_NEW_TOKENS, FIELDS, MODELS
from mllm.inference import generate, load_model, parse_json, prepare_inputs


def ground_truth(sample: dict) -> dict[str, str | None]:
    values = {
        field["field_name"].split("/")[-1]: field.get("annotator_text", "").strip()
        for field in sample.get("fields", [])
    }
    return {field: values.get(field) or None for field in FIELDS}


def main(
    model: Annotated[str, typer.Option(help=f"Model alias: {', '.join(MODELS)}")],
    data_json: Annotated[Path, typer.Option(exists=True)],
    index: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    adapter: Path | None = None,
):
    if model not in MODELS:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODELS)}")

    samples = json.loads(data_json.read_text())
    if not 0 <= index < len(samples):
        raise typer.BadParameter(f"index must be between 0 and {len(samples) - 1}")

    sample = samples[index]
    image_path = data_json.parent / sample["image"]
    image = Image.open(image_path).convert("RGB")

    loaded_model, processor = load_model(
        model, adapter=str(adapter) if adapter else None
    )
    inputs = prepare_inputs(processor, [image])
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
    gt = ground_truth(sample)
    prediction = parse_json(text)
    fields = {
        field: {
            "ground_truth": gt[field],
            "prediction": prediction[field],
            "correct": gt[field] == prediction[field],
        }
        for field in FIELDS
    }

    print(f"Image: {image_path}")
    print(f"Input shapes: {input_shapes}")
    print(f"Generation latency: {latency_ms:.1f} ms")
    print("\nRaw response:")
    print(text)
    print("\nParsed fields:")
    print(json.dumps(fields, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
