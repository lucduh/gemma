import json

import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)


def move_inputs(inputs, device):
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to(device=device, dtype=dtype)
    return inputs


class BaseModel:
    supports_image_tokens = False

    def __init__(self, model_id, device, adapter=None):
        self.model_id = model_id
        self.device = torch.device(device)

    def generate(self, inputs, max_new_tokens, observer=None):
        input_length = inputs["input_ids"].shape[1]
        generation_kwargs = {}
        if observer is not None:
            generation_kwargs["logits_processor"] = [observer]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **generation_kwargs,
            )
        if observer is not None:
            observer.end()

        generated_ids = output_ids[:, input_length:]
        token_counts, reached_token_limits = self._generation_stats(
            generated_ids, max_new_tokens
        )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return texts, token_counts, reached_token_limits

    def _generation_stats(self, generated_ids, max_new_tokens):
        generation_config = self.model.generation_config
        eos_token_ids = generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = set()
        elif isinstance(eos_token_ids, int):
            eos_token_ids = {eos_token_ids}
        else:
            eos_token_ids = set(eos_token_ids)
        pad_token_id = generation_config.pad_token_id

        token_counts = []
        reached_token_limits = []
        for row in generated_ids.tolist():
            eos_index = next(
                (index for index, token in enumerate(row) if token in eos_token_ids),
                None,
            )
            if eos_index is not None:
                count = eos_index + 1
                reached_limit = False
            else:
                count = len(row)
                if pad_token_id is not None:
                    while count and row[count - 1] == pad_token_id:
                        count -= 1
                reached_limit = count >= max_new_tokens
            token_counts.append(count)
            reached_token_limits.append(reached_limit)

        return token_counts, reached_token_limits

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_peak_memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def allocated_memory_gb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated(self.device) / 1024**3

    def peak_memory_gb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / 1024**3

    def peak_reserved_memory_gb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_reserved(self.device) / 1024**3


class Gemma3(BaseModel):
    def __init__(self, model_id, device, adapter=None):
        super().__init__(model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16
        ).to(self.device)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def prepare_inputs(self, images, prompt, image_tokens):
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
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return move_inputs(inputs, self.device)


class Gemma4(BaseModel):
    supports_image_tokens = True

    def __init__(self, model_id, device, adapter=None):
        super().__init__(model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.bfloat16
        ).to(self.device)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def prepare_inputs(self, images, prompt, image_tokens):
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
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "images_kwargs": {"max_soft_tokens": image_tokens},
            },
            enable_thinking=False,
        )
        return move_inputs(inputs, self.device)


class InternVL(BaseModel):
    def __init__(self, model_id, device, adapter=None):
        super().__init__(model_id, device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16
        ).to(self.device)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def prepare_inputs(self, images, prompt, image_tokens):
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
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return move_inputs(inputs, self.device)


class TestModel(BaseModel):
    def prepare_inputs(self, images, prompt, image_tokens):
        return {
            "input_ids": torch.zeros(
                (len(images), 1), dtype=torch.long, device=self.device
            )
        }

    def generate(self, inputs, max_new_tokens, observer=None):
        batch_size = inputs["input_ids"].shape[0]
        if observer is not None:
            observer.end()
        return ["{}"] * batch_size, [0] * batch_size, [False] * batch_size


def load_model(model_name, device, adapter):
    from mllm.config import MODELS

    model_class, model_id = MODELS[model_name]
    return model_class(model_id, device, adapter)


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
