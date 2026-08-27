import argparse
import json
import random

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

from mllm.config import (
    DATASETS,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MODELS,
    TRAINING_RESULTS_DIR,
)
from mllm.dataset import Dataset
from mllm.gemma4_manual import (
    POOL_METHODS,
    build_llm_inputs,
    compact_image_placeholders,
    encode_vision,
    encoded_soft_grid,
    move_to_device,
    project_vision,
    reduce_visual_tokens,
    target_soft_grid,
)

MODEL_NAME = "gemma4-e4b"
DEFAULT_SOURCE_IMAGE_TOKENS = 1120
DEFAULT_TARGET_IMAGE_TOKENS = 560
DEFAULT_POOL_METHOD = "spatial-select"


def make_target(ground_truth: dict, fields: tuple[str, ...]) -> str:
    target = {field: ground_truth[field] for field in fields if ground_truth.get(field)}
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def apply_template(processor, messages, image_tokens, add_generation_prompt):
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "padding": True,
            "images_kwargs": {"max_soft_tokens": image_tokens},
        },
        enable_thinking=False,
    )


def prepare_example(
    dataset,
    index,
    processor,
    model,
    device,
    source_image_tokens,
    target_image_tokens,
    pool_method,
):
    image, _, ground_truth = dataset[index]
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": dataset.prompt},
        ],
    }
    prompt = [[user]]
    conversation = [
        [
            user,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": make_target(ground_truth, dataset.fields),
                    }
                ],
            },
        ]
    ]

    prompt_inputs = apply_template(
        processor,
        prompt,
        image_tokens=source_image_tokens,
        add_generation_prompt=True,
    )
    inputs = apply_template(
        processor,
        conversation,
        image_tokens=source_image_tokens,
        add_generation_prompt=False,
    )

    target_grid = target_soft_grid(
        image, processor.image_processor, target_image_tokens
    )
    target_count = target_grid[0] * target_grid[1]
    prompt_inputs = compact_image_placeholders(
        prompt_inputs, model.config.image_token_id, target_count
    )
    inputs = compact_image_placeholders(
        inputs, model.config.image_token_id, target_count
    )
    source_grid = encoded_soft_grid(
        inputs["image_position_ids"], processor.image_processor.pooling_kernel_size
    )

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    prompt_length = int(prompt_inputs["attention_mask"].sum().item())
    full_length = int(inputs["attention_mask"].sum().item())
    start = (
        labels.shape[1] - full_length
        if processor.tokenizer.padding_side == "left"
        else 0
    )
    labels[:, start : start + prompt_length] = -100
    inputs["labels"] = labels
    inputs = move_to_device(inputs, device)

    with torch.no_grad():
        vision_tokens = encode_vision(model, inputs)
        reduced_tokens = reduce_visual_tokens(
            vision_tokens, source_grid, target_grid, pool_method
        )
        image_features = project_vision(model, reduced_tokens)
        inputs_embeds, per_layer_inputs, visual_tokens = build_llm_inputs(
            model, inputs, image_features
        )

    if visual_tokens != target_count:
        raise ValueError(
            f"Expected {target_count} visual tokens, produced {visual_tokens}"
        )

    return {
        "inputs_embeds": inputs_embeds,
        "per_layer_inputs": per_layer_inputs,
        "attention_mask": inputs["attention_mask"],
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
        "labels": inputs["labels"],
        "use_cache": False,
        "return_dict": True,
    }


def validation_loss(
    trained_model,
    base_model,
    dataset,
    samples,
    processor,
    device,
    source_image_tokens,
    target_image_tokens,
    pool_method,
):
    trained_model.eval()
    losses = []
    with torch.inference_mode():
        for index in tqdm(samples, desc="validation", leave=False):
            inputs = prepare_example(
                dataset,
                index,
                processor,
                base_model,
                device,
                source_image_tokens,
                target_image_tokens,
                pool_method,
            )
            losses.append(trained_model(**inputs).loss.item())
    return sum(losses) / len(losses)


def main(
    device: str,
    dataset: str,
    run_name: str,
    split: str,
    source_image_tokens: int,
    target_image_tokens: int,
    pool_method: str,
    epochs: int,
    learning_rate: float,
    gradient_accumulation: int,
    validation_fraction: float,
    seed: int,
):
    dataset_name = dataset
    dataset = Dataset(dataset_name, split)
    samples = list(range(len(dataset)))
    if len(samples) < 2:
        raise ValueError("LoRA training requires at least two documents")

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
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(torch_device)
    base_model.eval()

    projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    targets = [
        name
        for name, _ in base_model.named_modules()
        if "language_model" in name and name.endswith(projections)
    ]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    trained_model = get_peft_model(base_model, lora_config)
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
        base_model.model.vision_tower.eval()
        base_model.model.embed_vision.eval()
        optimizer.zero_grad()
        losses = []

        progress = tqdm(training_samples, desc=f"epoch {epoch}/{epochs}")
        for step, index in enumerate(progress, start=1):
            inputs = prepare_example(
                dataset,
                index,
                processor,
                base_model,
                torch_device,
                source_image_tokens,
                target_image_tokens,
                pool_method,
            )
            loss = trained_model(**inputs).loss
            (loss / gradient_accumulation).backward()
            losses.append(loss.item())

            if step % gradient_accumulation == 0 or step == len(training_samples):
                optimizer.step()
                optimizer.zero_grad()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = sum(losses) / len(losses)
        val_loss = validation_loss(
            trained_model,
            base_model,
            dataset,
            validation_samples,
            processor,
            torch_device,
            source_image_tokens,
            target_image_tokens,
            pool_method,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trained_model.save_pretrained(run_dir / "best")

    trained_model.save_pretrained(run_dir / "last")
    record = {
        "model": MODEL_NAME,
        "model_id": str(model_id),
        "dataset": dataset_name,
        "split": split,
        "data_path": str(dataset.path),
        "fields": list(dataset.fields),
        "training_documents": len(training_samples),
        "validation_documents": len(validation_samples),
        "source_image_tokens": source_image_tokens,
        "target_image_tokens": target_image_tokens,
        "pool_method": pool_method,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": 1,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": gradient_accumulation,
        "seed": seed,
        "device": str(torch_device),
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "best_validation_loss": best_loss,
        "history": history,
    }
    (run_dir / "train.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved training results to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 4 E4B LoRA with fixed visual-token reduction."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument(
        "--source-image-tokens", type=int, default=DEFAULT_SOURCE_IMAGE_TOKENS
    )
    parser.add_argument(
        "--target-image-tokens", type=int, default=DEFAULT_TARGET_IMAGE_TOKENS
    )
    parser.add_argument(
        "--pool-method", choices=POOL_METHODS, default=DEFAULT_POOL_METHOD
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
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
