import json

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)

from mllm.constants import DEFAULT_GEMMA4_IMAGE_TOKENS, MODELS


def load_model(name: str, adapter: str | None = None, device: str = "cuda"):
    model_id = MODELS[name]
    processor = AutoProcessor.from_pretrained(model_id)
    model_class = (
        AutoModelForMultimodalLM if name == "gemma4" else AutoModelForImageTextToText
    )
    model = model_class.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


def prepare_inputs(
    processor,
    images: list[Image.Image],
    prompt: str,
    device: str = "cuda",
    image_tokens: int = DEFAULT_GEMMA4_IMAGE_TOKENS,
):
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]
    is_gemma4 = processor.__class__.__name__ == "Gemma4Processor"
    template_kwargs = {"enable_thinking": False} if is_gemma4 else {}
    processor_kwargs = {"padding": True}
    if is_gemma4:
        processor_kwargs["images_kwargs"] = {"max_soft_tokens": image_tokens}

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
        **template_kwargs,
    )
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to(device=device, dtype=dtype)
    return inputs


def generate(model, processor, inputs, max_new_tokens: int):
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, input_length:]
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    pad_token_id = processor.tokenizer.pad_token_id
    token_count = generated_ids.numel()
    if pad_token_id is not None:
        token_count = generated_ids.ne(pad_token_id).sum().item()
    return texts, int(token_count)


def parse_json(text: str, fields: tuple[str, ...]) -> dict[str, str | None]:
    value = {}
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            value = candidate
            break

    if isinstance(value.get("fields"), dict):
        value = value["fields"]

    result = {}
    for field in fields:
        field_value = value.get(field)
        result[field] = None if field_value is None else str(field_value).strip()
    return result
