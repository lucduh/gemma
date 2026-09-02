import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class VisionPruningInfo:
    requested_keep_ratio: float
    original_layers: int
    retained_layers: int
    retained_indices: tuple[int, ...]
    original_parameters: int
    retained_parameters: int

    def to_dict(self) -> dict:
        record = asdict(self)
        record["retained_indices"] = list(self.retained_indices)
        record["actual_keep_ratio"] = self.retained_layers / self.original_layers
        record["removed_parameters"] = (
            self.original_parameters - self.retained_parameters
        )
        return record


def uniform_layer_indices(total_layers: int, keep_ratio: float) -> tuple[int, ...]:
    """Select a uniformly spaced subset that includes the first and last blocks."""
    if total_layers <= 0:
        raise ValueError("total_layers must be greater than zero")
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be greater than zero and at most one")

    retained_layers = min(total_layers, max(1, int(total_layers * keep_ratio + 0.5)))
    if retained_layers == total_layers:
        return tuple(range(total_layers))
    if retained_layers == 1:
        return (total_layers - 1,)

    intervals = retained_layers - 1
    return tuple(
        (index * (total_layers - 1) + intervals // 2) // intervals
        for index in range(retained_layers)
    )


def prune_vision_tower(vision_tower, keep_ratio: float) -> VisionPruningInfo:
    """Physically remove Gemma 4 vision blocks while preserving their order."""
    try:
        encoder = vision_tower.encoder
        layers = encoder.layers
    except AttributeError as error:
        raise ValueError(
            "Expected a Gemma 4 vision tower with encoder.layers"
        ) from error

    if not isinstance(layers, nn.ModuleList):
        raise TypeError("The Gemma 4 vision encoder layers must be an nn.ModuleList")

    original_layers = len(layers)
    retained_indices = uniform_layer_indices(original_layers, keep_ratio)
    original_parameters = sum(
        parameter.numel() for parameter in vision_tower.parameters()
    )

    encoder.layers = nn.ModuleList([layers[index] for index in retained_indices])
    encoder.num_layers = len(retained_indices)
    encoder.config.num_hidden_layers = len(retained_indices)

    retained_parameters = sum(
        parameter.numel() for parameter in vision_tower.parameters()
    )
    return VisionPruningInfo(
        requested_keep_ratio=keep_ratio,
        original_layers=original_layers,
        retained_layers=len(retained_indices),
        retained_indices=retained_indices,
        original_parameters=original_parameters,
        retained_parameters=retained_parameters,
    )


def prune_vision_encoder(model, keep_ratio: float) -> VisionPruningInfo:
    try:
        vision_tower = model.model.vision_tower
    except AttributeError as error:
        raise ValueError("Expected a Gemma 4 model with model.vision_tower") from error
    return prune_vision_tower(vision_tower, keep_ratio)


def save_vision_checkpoint(
    vision_tower, pruning_info: VisionPruningInfo, directory: str | Path
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(vision_tower.state_dict(), directory / "vision_tower.pt")
    metadata = {"format_version": 1, "pruning": pruning_info.to_dict()}
    (directory / "vision_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_vision_checkpoint(
    model, directory: str | Path, requested_keep_ratio: float | None = None
) -> VisionPruningInfo:
    directory = Path(directory)
    metadata = json.loads(
        (directory / "vision_config.json").read_text(encoding="utf-8")
    )
    checkpoint_pruning = metadata["pruning"]
    checkpoint_ratio = float(checkpoint_pruning["requested_keep_ratio"])
    pruning_info = prune_vision_encoder(model, checkpoint_ratio)

    expected_indices = tuple(checkpoint_pruning["retained_indices"])
    if pruning_info.retained_indices != expected_indices:
        raise ValueError(
            "Checkpoint layer selection does not match the loaded base model: "
            f"expected {expected_indices}, got {pruning_info.retained_indices}"
        )
    if requested_keep_ratio is not None:
        requested_indices = uniform_layer_indices(
            pruning_info.original_layers, requested_keep_ratio
        )
        if requested_indices != expected_indices:
            raise ValueError("--vision-keep-ratio does not match the vision checkpoint")

    state = torch.load(
        directory / "vision_tower.pt", map_location="cpu", weights_only=True
    )
    model.model.vision_tower.load_state_dict(state)
    return pruning_info
