import math
from dataclasses import asdict, dataclass

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


def prune_vision_encoder(model, keep_ratio: float) -> VisionPruningInfo:
    """Physically remove Gemma 4 vision blocks while preserving their order."""
    try:
        vision_tower = model.model.vision_tower
        encoder = vision_tower.encoder
        layers = encoder.layers
    except AttributeError as error:
        raise ValueError(
            "Expected a Gemma 4 model with model.vision_tower.encoder.layers"
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
