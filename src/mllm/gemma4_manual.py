import torch
import torch.nn.functional as F
from transformers.models.gemma4.image_processing_gemma4 import (
    get_aspect_ratio_preserving_size,
)

SUPPORTED_IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def prepare_inputs(processor, image, prompt: str, image_tokens: int):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        [conversation],
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


def move_to_device(inputs, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value
    return moved


def encode_vision(model, inputs: dict) -> torch.Tensor:
    output = model.model.vision_tower(
        pixel_values=inputs["pixel_values"],
        pixel_position_ids=inputs["image_position_ids"],
        return_dict=True,
    )
    return output.last_hidden_state


def project_vision(model, vision_tokens: torch.Tensor) -> torch.Tensor:
    return model.model.embed_vision(inputs_embeds=vision_tokens)


def build_llm_inputs(
    model, inputs: dict, image_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    base_model = model.model
    input_ids = inputs["input_ids"]
    image_mask = input_ids == model.config.image_token_id
    image_token_count = int(image_mask.sum().item())

    if image_features.shape[0] != image_token_count:
        raise ValueError(
            "Image features and placeholders do not match: "
            f"{image_features.shape[0]} features and {image_token_count} placeholders"
        )

    pad_token_id = model.config.text_config.pad_token_id
    llm_input_ids = torch.where(image_mask, pad_token_id, input_ids)
    inputs_embeds = base_model.get_input_embeddings()(llm_input_ids)

    per_layer_inputs = None
    language_model = base_model.language_model
    if model.config.text_config.hidden_size_per_layer_input:
        per_layer_inputs = language_model.get_per_layer_inputs(
            llm_input_ids, inputs_embeds
        )

    expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        expanded_image_mask, image_features.to(inputs_embeds.dtype)
    )
    return inputs_embeds, per_layer_inputs, image_token_count


def prefill(
    model,
    inputs: dict,
    inputs_embeds: torch.Tensor,
    per_layer_inputs: torch.Tensor | None,
):
    return model(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        attention_mask=inputs["attention_mask"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )


def eos_token_ids(model) -> set[int]:
    token_ids = model.generation_config.eos_token_id
    if token_ids is None:
        token_ids = model.config.text_config.eos_token_id
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return set(token_ids)


def greedy_decode(
    model,
    prefill_output,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    next_token = prefill_output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    past_key_values = prefill_output.past_key_values
    stop_tokens = eos_token_ids(model)

    while len(generated) < max_new_tokens and next_token.item() not in stop_tokens:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1), dtype=attention_mask.dtype, device=attention_mask.device
                ),
            ],
            dim=1,
        )
        position_ids = torch.tensor(
            [[attention_mask.shape[1] - 1]],
            dtype=torch.long,
            device=attention_mask.device,
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=1)


def compact_image_placeholders(inputs, image_token_id: int, target_count: int) -> dict:
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("Placeholder compaction only supports batch size 1")

    image_positions = torch.where(input_ids[0] == image_token_id)[0]
    if target_count <= 0:
        raise ValueError("target_count must be greater than zero")
    if image_positions.numel() == 0:
        raise ValueError("The input contains no image placeholders")
    if image_positions.numel() > 1 and not torch.all(
        image_positions[1:] == image_positions[:-1] + 1
    ):
        raise ValueError(
            "The input must contain one contiguous image-placeholder block"
        )
    if target_count > image_positions.numel():
        raise ValueError(
            f"Cannot retain {target_count} placeholders from {image_positions.numel()}"
        )

    keep = torch.ones(input_ids.shape[1], dtype=torch.bool)
    keep[image_positions[target_count:]] = False
    compacted = dict(inputs)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
        value = compacted.get(key)
        if isinstance(value, torch.Tensor):
            compacted[key] = value[:, keep]
    return compacted


def target_soft_grid(image, image_processor, image_tokens: int) -> tuple[int, int]:
    patch_size = image_processor.patch_size
    pooling_kernel_size = image_processor.pooling_kernel_size
    target_height, target_width = get_aspect_ratio_preserving_size(
        height=image.height,
        width=image.width,
        patch_size=patch_size,
        max_patches=image_tokens * pooling_kernel_size**2,
        pooling_kernel_size=pooling_kernel_size,
    )
    return (
        target_height // patch_size // pooling_kernel_size,
        target_width // patch_size // pooling_kernel_size,
    )


def encoded_soft_grid(
    image_position_ids: torch.Tensor, pooling_kernel_size: int
) -> tuple[int, int]:
    positions = image_position_ids[0]
    positions = positions[(positions != -1).all(dim=-1)]
    patch_width = int(positions[:, 0].max().item()) + 1
    patch_height = int(positions[:, 1].max().item()) + 1
    return (
        patch_height // pooling_kernel_size,
        patch_width // pooling_kernel_size,
    )


def validate_grid(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> None:
    source_height, source_width = source_grid
    target_height, target_width = target_grid
    if tokens.shape[0] != source_height * source_width:
        raise ValueError(
            f"Token count {tokens.shape[0]} does not match source grid "
            f"{source_height}x{source_width}"
        )
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Target grid dimensions must be greater than zero")
    if target_height > source_height or target_width > source_width:
        raise ValueError("Target grid cannot be larger than source grid")


def spatial_average_pool(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    validate_grid(tokens, source_grid, target_grid)
    source_height, source_width = source_grid
    target_height, target_width = target_grid

    original_dtype = tokens.dtype
    grid = tokens.reshape(source_height, source_width, -1)
    grid = grid.permute(2, 0, 1).unsqueeze(0).float()
    pooled = F.adaptive_avg_pool2d(grid, (target_height, target_width))
    return pooled.flatten(2).transpose(1, 2).squeeze(0).to(original_dtype)


def spatial_select(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    validate_grid(tokens, source_grid, target_grid)
    source_height, source_width = source_grid
    target_height, target_width = target_grid
    grid = tokens.reshape(source_height, source_width, -1)

    rows = (
        (2 * torch.arange(target_height, device=tokens.device) + 1)
        * source_height
        // (2 * target_height)
    )
    columns = (
        (2 * torch.arange(target_width, device=tokens.device) + 1)
        * source_width
        // (2 * target_width)
    )
    selected = grid[rows[:, None], columns[None, :]]
    return selected.reshape(target_height * target_width, -1)


def similarity_merge(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
    local_neighbors: int = 4,
) -> torch.Tensor:
    validate_grid(tokens, source_grid, target_grid)
    if local_neighbors <= 0:
        raise ValueError("local_neighbors must be greater than zero")

    source_height, source_width = source_grid
    target_count = target_grid[0] * target_grid[1]
    original_dtype = tokens.dtype
    positions = torch.stack(
        torch.meshgrid(
            torch.arange(source_height, device=tokens.device),
            torch.arange(source_width, device=tokens.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    positions = positions.float()
    weights = torch.ones(tokens.shape[0], 1, device=tokens.device)

    while tokens.shape[0] > target_count:
        source_tokens = tokens[::2]
        destination_tokens = tokens[1::2]
        source_positions = positions[::2]
        destination_positions = positions[1::2]
        source_weights = weights[::2]
        destination_weights = weights[1::2]

        merge_count = min(tokens.shape[0] - target_count, destination_tokens.shape[0])
        distances = torch.cdist(source_positions, destination_positions)
        neighbor_count = min(local_neighbors, destination_tokens.shape[0])
        neighbors = distances.topk(neighbor_count, dim=1, largest=False).indices

        normalized_sources = F.normalize(source_tokens.float(), dim=-1)
        normalized_destinations = F.normalize(destination_tokens.float(), dim=-1)
        neighbor_features = normalized_destinations[neighbors]
        neighbor_similarities = (
            normalized_sources[:, None, :] * neighbor_features
        ).sum(dim=-1)
        best_neighbor = neighbor_similarities.argmax(dim=1, keepdim=True)
        match_scores = neighbor_similarities.gather(1, best_neighbor).squeeze(1)
        matched_destinations = neighbors.gather(1, best_neighbor).squeeze(1)

        merged_sources = match_scores.topk(merge_count).indices
        merge_mask = torch.zeros(
            source_tokens.shape[0], dtype=torch.bool, device=tokens.device
        )
        merge_mask[merged_sources] = True
        destination_indices = matched_destinations[merged_sources]

        destination_sums = destination_tokens.float() * destination_weights
        destination_position_sums = destination_positions * destination_weights
        destination_sums.index_add_(
            0,
            destination_indices,
            source_tokens[merged_sources].float() * source_weights[merged_sources],
        )
        destination_position_sums.index_add_(
            0,
            destination_indices,
            source_positions[merged_sources] * source_weights[merged_sources],
        )
        destination_weights.index_add_(
            0, destination_indices, source_weights[merged_sources]
        )
        destination_tokens = destination_sums / destination_weights
        destination_positions = destination_position_sums / destination_weights

        tokens = torch.cat(
            [source_tokens[~merge_mask].float(), destination_tokens], dim=0
        )
        positions = torch.cat(
            [source_positions[~merge_mask], destination_positions], dim=0
        )
        weights = torch.cat([source_weights[~merge_mask], destination_weights], dim=0)
        spatial_order = torch.argsort(
            positions[:, 0] * (source_width + 1) + positions[:, 1]
        )
        tokens = tokens[spatial_order]
        positions = positions[spatial_order]
        weights = weights[spatial_order]

    return tokens.to(original_dtype)
