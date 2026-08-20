import argparse
import json
import random

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm

from mllm.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_VALIDATION_FRACTION,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MODELS,
    TRAINING_RESULTS_DIR,
)
from mllm.datasets import DATASETS, image_path, load_split
from mllm.inference import load_model


def make_target(sample, fields):
    target = {field: sample[field] for field in fields if sample.get(field)}
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def apply_template(processor, messages, add_generation_prompt):
    extra = (
        {"enable_thinking": False}
        if processor.__class__.__name__ == "Gemma4Processor"
        else {}
    )
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        **extra,
    )


def make_batches(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def prepare_batch(samples, data_dir, processor, prompt, fields, device="cuda"):
    """Create a padded batch and mask everything before each answer."""
    images = [
        Image.open(image_path(sample, data_dir)).convert("RGB") for sample in samples
    ]
    users = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
        for image in images
    ]
    prompts = [[user] for user in users]
    conversations = [
        [
            user,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": make_target(sample, fields)}],
            },
        ]
        for user, sample in zip(users, samples)
    ]

    prompt_inputs = apply_template(processor, prompts, add_generation_prompt=True)
    inputs = apply_template(processor, conversations, add_generation_prompt=False)
    for image in images:
        image.close()

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
    full_lengths = inputs["attention_mask"].sum(dim=1)
    left_padding = processor.tokenizer.padding_side == "left"
    for row, prompt_length in enumerate(prompt_lengths):
        start = labels.shape[1] - full_lengths[row].item() if left_padding else 0
        labels[row, start : start + prompt_length.item()] = -100
    inputs["labels"] = labels

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to(device, dtype=dtype)
    return inputs


def get_validation_loss(
    model, samples, batch_size, data_dir, processor, prompt, fields
):
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in tqdm(
            list(make_batches(samples, batch_size)), desc="validation", leave=False
        ):
            inputs = prepare_batch(batch, data_dir, processor, prompt, fields)
            losses.append(model(**inputs).loss.item())
    return sum(losses) / len(losses)


def main(
    model: str,
    dataset: str,
    run_name: str,
    split: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation: int,
    validation_fraction: float,
    seed: int,
):
    dataset_split = load_split(dataset, split)
    samples = dataset_split.samples.copy()
    fields = dataset_split.fields
    rng = random.Random(seed)
    rng.shuffle(samples)

    validation_size = min(
        len(samples) - 1, max(1, round(len(samples) * validation_fraction))
    )
    validation_samples = samples[:validation_size]
    training_samples = samples[validation_size:]

    base_model, processor = load_model(model)

    # Full module names keep LoRA out of the frozen vision tower.
    projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets = [
        name
        for name, _ in base_model.named_modules()
        if "language_model" in name and name.endswith(projections)
    ]
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    trained_model = get_peft_model(base_model, config)
    trained_model.config.use_cache = False
    trained_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
    )
    run_dir = TRAINING_RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        rng.shuffle(training_samples)
        trained_model.train()
        optimizer.zero_grad()
        losses = []

        batches = list(make_batches(training_samples, batch_size))
        progress = tqdm(batches, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            loss = trained_model(
                **prepare_batch(
                    batch,
                    dataset_split.directory,
                    processor,
                    dataset_split.prompt,
                    fields,
                )
            ).loss
            (loss / gradient_accumulation).backward()
            losses.append(loss.item())

            if step % gradient_accumulation == 0 or step == len(batches):
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = sum(losses) / len(losses)
        val_loss = get_validation_loss(
            trained_model,
            validation_samples,
            batch_size,
            dataset_split.directory,
            processor,
            dataset_split.prompt,
            fields,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trained_model.save_pretrained(run_dir / "best")

    trained_model.save_pretrained(run_dir / "last")
    record = {
        "model": model,
        "model_id": MODELS[model],
        "attention_implementation": getattr(
            base_model.config, "_attn_implementation", None
        ),
        "dataset": dataset,
        "split": split,
        "data_path": str(dataset_split.path),
        "fields": list(fields),
        "training_documents": len(training_samples),
        "validation_documents": len(validation_samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "seed": seed,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "best_validation_loss": best_loss,
        "history": history,
    }
    (run_dir / "train.json").write_text(json.dumps(record, indent=2))
    print(f"Saved training results to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a LoRA adapter.")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
