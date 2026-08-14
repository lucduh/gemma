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


def target_json(sample: dict) -> str:
    """Build the same compact JSON representation used during inference."""
    annotated = {
        field["field_name"].split("/")[-1]: field.get("annotator_text", "").strip()
        for field in sample.get("fields", [])
    }
    target = {field: annotated[field] for field in FIELDS if annotated.get(field)}
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def prepare_sample(
    sample: dict, data_dir: Path, processor, device: str = "cuda"
) -> dict:
    """Process one document and mask the prompt so loss uses only the answer."""
    image = Image.open(data_dir / sample["image"]).convert("RGB")
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
    prompt_messages = [user_message]
    full_messages = [
        user_message,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_json(sample)}],
        },
    ]

    template_kwargs = (
        {"enable_thinking": False}
        if processor.__class__.__name__ == "Gemma4Processor"
        else {}
    )
    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    inputs = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    image.close()

    labels = inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    inputs["labels"] = labels

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            inputs[key] = value.to(device=device, dtype=dtype)
    return inputs


def validation_loss(model, samples: list[dict], data_dir: Path, processor) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc="validation", leave=False):
            inputs = prepare_sample(sample, data_dir, processor)
            losses.append(model(**inputs).loss.item())
    return sum(losses) / len(losses)


def language_lora_targets(model) -> list[str]:
    """Select attention projections from the language model, not the vision tower."""
    projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    return [
        name
        for name, _ in model.named_modules()
        if "language_model" in name and name.endswith(projections)
    ]


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
    if model not in MODELS:
        raise typer.BadParameter(f"Choose one of: {', '.join(MODELS)}")
    if epochs < 1 or gradient_accumulation < 1 or learning_rate <= 0:
        raise typer.BadParameter(
            "epochs, learning-rate and accumulation must be positive"
        )
    if not 0 < validation_fraction < 1:
        raise typer.BadParameter("validation-fraction must be between 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA")

    samples = json.loads(data_json.read_text())
    if len(samples) < 2:
        raise ValueError("Training requires at least two documents")

    # Make a deterministic validation split. The held-out test set is never used here.
    rng = random.Random(seed)
    rng.shuffle(samples)
    validation_size = max(1, round(len(samples) * validation_fraction))
    validation_samples = samples[:validation_size]
    training_samples = samples[validation_size:]

    base_model, processor = load_model(model)
    targets = language_lora_targets(base_model)
    if not targets:
        raise RuntimeError("Could not find language-model attention projections")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model_with_lora = get_peft_model(base_model, lora_config)
    model_with_lora.config.use_cache = False
    model_with_lora.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model_with_lora.parameters()
            if parameter.requires_grad
        ),
        lr=learning_rate,
    )
    run_dir = TRAINING_RESULTS_DIR / run_name
    best_dir = run_dir / "best"
    last_dir = run_dir / "last"
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    data_dir = data_json.parent
    for epoch in range(1, epochs + 1):
        rng.shuffle(training_samples)
        model_with_lora.train()
        optimizer.zero_grad()
        epoch_losses = []

        progress = tqdm(training_samples, desc=f"epoch {epoch}/{epochs}")
        for step, sample in enumerate(progress, start=1):
            inputs = prepare_sample(sample, data_dir, processor)
            loss = model_with_lora(**inputs).loss
            (loss / gradient_accumulation).backward()
            epoch_losses.append(loss.item())

            if step % gradient_accumulation == 0 or step == len(training_samples):
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = sum(epoch_losses) / len(epoch_losses)
        val_loss = validation_loss(
            model_with_lora, validation_samples, data_dir, processor
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            model_with_lora.save_pretrained(best_dir)

    model_with_lora.save_pretrained(last_dir)
    record = {
        "model": model,
        "model_id": MODELS[model],
        "data_json": str(data_json),
        "documents": len(samples),
        "training_documents": len(training_samples),
        "validation_documents": len(validation_samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "best_validation_loss": best_loss,
        "history": history,
    }
    (run_dir / "train.json").write_text(json.dumps(record, indent=2))
    print(f"Saved training results to {run_dir}")


if __name__ == "__main__":
    typer.run(main)
