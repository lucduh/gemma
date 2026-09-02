import argparse
import copy
import gc
import json
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from mllm.config import (
    DATASETS,
    DEFAULT_EPOCHS,
    DEFAULT_GEMMA4_IMAGE_TOKENS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    MODELS,
    TRAINING_RESULTS_DIR,
)
from mllm.dataset import Dataset
from mllm.gemma4_manual import prepare_inputs
from mllm.gemma4_pruning import prune_vision_tower, save_vision_checkpoint

MODEL_NAME = "gemma4-e4b"
DEFAULT_VISION_KEEP_RATIO = 0.75
DEFAULT_LEARNING_RATE = 1e-5
COSINE_WEIGHT = 1.0
SMOOTH_L1_WEIGHT = 1.0


def prepare_vision_inputs(processor, image, prompt, image_tokens, device):
    inputs = prepare_inputs(processor, image, prompt, image_tokens)
    return {
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "pixel_position_ids": inputs["image_position_ids"].to(device=device),
        "return_dict": True,
    }


def distillation_loss(student_tokens, teacher_tokens):
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            "Teacher and student token shapes differ: "
            f"{tuple(teacher_tokens.shape)} and {tuple(student_tokens.shape)}"
        )

    student_tokens = student_tokens.float()
    teacher_tokens = teacher_tokens.float()
    cosine_loss = 1 - F.cosine_similarity(student_tokens, teacher_tokens, dim=-1).mean()
    smooth_l1_loss = F.smooth_l1_loss(student_tokens, teacher_tokens)
    loss = COSINE_WEIGHT * cosine_loss + SMOOTH_L1_WEIGHT * smooth_l1_loss
    return loss, cosine_loss, smooth_l1_loss


def evaluate_loss(
    teacher,
    student,
    dataset,
    samples,
    processor,
    image_tokens,
    device,
):
    student.eval()
    totals = {"loss": 0.0, "cosine_loss": 0.0, "smooth_l1_loss": 0.0}
    with torch.inference_mode():
        for index in tqdm(samples, desc="validation", leave=False):
            image, _, _ = dataset[index]
            inputs = prepare_vision_inputs(
                processor, image, dataset.prompt, image_tokens, device
            )
            teacher_tokens = teacher(**inputs).last_hidden_state
            student_tokens = student(**inputs).last_hidden_state
            loss, cosine_loss, smooth_l1_loss = distillation_loss(
                student_tokens, teacher_tokens
            )
            totals["loss"] += loss.item()
            totals["cosine_loss"] += cosine_loss.item()
            totals["smooth_l1_loss"] += smooth_l1_loss.item()

    return {name: value / len(samples) for name, value in totals.items()}


def main(
    device: str,
    dataset: str,
    run_name: str,
    split: str,
    image_tokens: int,
    vision_keep_ratio: float,
    epochs: int,
    learning_rate: float,
    gradient_accumulation: int,
    validation_fraction: float,
    seed: int,
    limit: int | None,
):
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be greater than zero")

    dataset_name = dataset
    dataset = Dataset(dataset_name, split)
    sample_count = len(dataset) if limit is None else min(limit, len(dataset))
    samples = list(range(sample_count))
    if len(samples) < 2:
        raise ValueError("Vision distillation requires at least two documents")

    rng = random.Random(seed)
    rng.shuffle(samples)
    validation_size = min(
        len(samples) - 1, max(1, round(len(samples) * validation_fraction))
    )
    validation_samples = samples[:validation_size]
    training_samples = samples[validation_size:]

    _, model_id = MODELS[MODEL_NAME]
    torch_device = torch.device(device)
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading the teacher vision tower")
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    )
    teacher = base_model.model.vision_tower
    student = copy.deepcopy(teacher)
    pruning_info = prune_vision_tower(student, vision_keep_ratio)
    del base_model
    gc.collect()

    teacher.requires_grad_(False).eval().to(torch_device)
    student.train().to(torch_device)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in student.parameters()
        if parameter.requires_grad
    )
    print(
        f"Student depth: {pruning_info.retained_layers}/"
        f"{pruning_info.original_layers}; "
        f"trainable parameters: {trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    run_dir = TRAINING_RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        rng.shuffle(training_samples)
        student.train()
        optimizer.zero_grad()
        totals = {"loss": 0.0, "cosine_loss": 0.0, "smooth_l1_loss": 0.0}

        progress = tqdm(training_samples, desc=f"epoch {epoch}/{epochs}")
        for step, index in enumerate(progress, start=1):
            image, _, _ = dataset[index]
            inputs = prepare_vision_inputs(
                processor, image, dataset.prompt, image_tokens, torch_device
            )
            with torch.no_grad():
                teacher_tokens = teacher(**inputs).last_hidden_state
            student_tokens = student(**inputs).last_hidden_state
            loss, cosine_loss, smooth_l1_loss = distillation_loss(
                student_tokens, teacher_tokens
            )
            (loss / gradient_accumulation).backward()

            totals["loss"] += loss.item()
            totals["cosine_loss"] += cosine_loss.item()
            totals["smooth_l1_loss"] += smooth_l1_loss.item()
            if step % gradient_accumulation == 0 or step == len(training_samples):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = {
            name: value / len(training_samples) for name, value in totals.items()
        }
        validation_metrics = evaluate_loss(
            teacher,
            student,
            dataset,
            validation_samples,
            processor,
            image_tokens,
            torch_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"Epoch {epoch}: train={train_metrics['loss']:.4f} "
            f"validation={validation_metrics['loss']:.4f}"
        )

        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_vision_checkpoint(student, pruning_info, run_dir / "best")

    save_vision_checkpoint(student, pruning_info, run_dir / "last")
    record = {
        "model": MODEL_NAME,
        "model_id": str(model_id),
        "dataset": dataset_name,
        "split": split,
        "data_path": str(dataset.path),
        "training_documents": len(training_samples),
        "validation_documents": len(validation_samples),
        "image_tokens": image_tokens,
        "vision_pruning": pruning_info.to_dict(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": gradient_accumulation,
        "cosine_weight": COSINE_WEIGHT,
        "smooth_l1_weight": SMOOTH_L1_WEIGHT,
        "seed": seed,
        "device": str(torch_device),
        "best_validation_loss": best_loss,
        "history": history,
    }
    (run_dir / "train.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved training results to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a pruned Gemma 4 vision tower from the full-depth tower."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_GEMMA4_IMAGE_TOKENS)
    parser.add_argument(
        "--vision-keep-ratio", type=float, default=DEFAULT_VISION_KEEP_RATIO
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
