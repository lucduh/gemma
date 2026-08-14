import json

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)

from mllm.constants import FIELDS, MODELS, PROMPT


def load_model(name: str, device: str = "cuda"):
    model_id = MODELS[name]
    processor = AutoProcessor.from_pretrained(model_id)
    model_class = (
        AutoModelForMultimodalLM if name == "gemma4" else AutoModelForImageTextToText
    )
    model = model_class.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    return model, processor


def prepare_inputs(processor, images: list[Image.Image], device: str = "cuda"):
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        for image in images
    ]
    template_kwargs = (
        {"enable_thinking": False}
        if processor.__class__.__name__ == "Gemma4Processor"
        else {}
    )
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        **template_kwargs,
    )
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to(device=device, dtype=dtype)
    return inputs


def generate(model, processor, inputs, max_new_tokens: int, fixed_length=False):
    input_length = inputs["input_ids"].shape[1]
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if fixed_length:
        kwargs["min_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)

    generated_ids = output_ids[:, input_length:]
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    pad_token_id = processor.tokenizer.pad_token_id
    token_count = generated_ids.numel()
    if pad_token_id is not None:
        token_count = generated_ids.ne(pad_token_id).sum().item()
    return texts, int(token_count)


def parse_json(text: str) -> dict[str, str | None]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        value = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        value = {}

    result = {}
    for field in FIELDS:
        field_value = value.get(field)
        result[field] = None if field_value is None else str(field_value).strip()
    return result
