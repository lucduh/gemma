import json
import random
from pathlib import Path
from typing import Annotated

import torch
import typer
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm

from mllm.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    FIELDS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MODELS,
    PROMPT,
    TRAINING_RESULTS_DIR,
)
from mllm.inference import load_model


def make_target(sample):
    values = {
        item["field_name"].split("/")[-1]: item.get("annotator_text", "").strip()
        for item in sample.get("fields", [])
    }
    target = {field: values[field] for field in FIELDS if values.get(field)}
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
        **extra,
    )


def prepare_sample(sample, data_dir, processor):
    """Create one training example and mask everything before the answer."""
    image = Image.open(data_dir / sample["image"]).convert("RGB")
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": make_target(sample)}],
    }

    prompt = apply_template(processor, [user], add_generation_prompt=True)
    inputs = apply_template(processor, [user, assistant], add_generation_prompt=False)
    image.close()

    labels = inputs["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[1]] = -100
    inputs["labels"] = labels

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to("cuda", dtype=dtype)
    return inputs


def get_validation_loss(model, samples, data_dir, processor):
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc="validation", leave=False):
            inputs = prepare_sample(sample, data_dir, processor)
            losses.append(model(**inputs).loss.item())
    return sum(losses) / len(losses)


def main(
    model: Annotated[str, typer.Option(help=f"Model alias: {', '.join(MODELS)}")],
    data_json: Annotated[Path, typer.Option(exists=True)],
    run_name: Annotated[str, typer.Option(help="Result directory name")],
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
):
    samples = json.loads(data_json.read_text())
    rng = random.Random(seed)
    rng.shuffle(samples)

    validation_size = max(1, round(len(samples) * validation_fraction))
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

        progress = tqdm(training_samples, desc=f"epoch {epoch}/{epochs}")
        for step, sample in enumerate(progress, start=1):
            loss = trained_model(
                **prepare_sample(sample, data_json.parent, processor)
            ).loss
            (loss / gradient_accumulation).backward()
            losses.append(loss.item())

            if step % gradient_accumulation == 0 or step == len(training_samples):
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = sum(losses) / len(losses)
        val_loss = get_validation_loss(
            trained_model, validation_samples, data_json.parent, processor
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
        "data_json": str(data_json),
        "training_documents": len(training_samples),
        "validation_documents": len(validation_samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "seed": seed,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "best_validation_loss": best_loss,
        "history": history,
    }
    (run_dir / "train.json").write_text(json.dumps(record, indent=2))
    print(f"Saved training results to {run_dir}")


if __name__ == "__main__":
    typer.run(main)
