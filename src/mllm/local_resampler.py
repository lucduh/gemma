import json
import math
from pathlib import Path

import torch
from torch import nn


class LocalResampler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        neighbors: int = 4,
        score_hidden_size: int = 64,
        position_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if neighbors <= 0:
            raise ValueError("neighbors must be greater than zero")
        if position_scale <= 0:
            raise ValueError("position_scale must be greater than zero")

        self.hidden_size = hidden_size
        self.neighbors = neighbors
        self.score_hidden_size = score_hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.content_score = nn.Sequential(
            nn.Linear(hidden_size, score_hidden_size),
            nn.GELU(),
            nn.Linear(score_hidden_size, 1),
        )
        self.log_position_scale = nn.Parameter(
            torch.tensor(math.log(position_scale), dtype=torch.float32)
        )

        # Start as a sharp spatial selector. Content-based selection is learned.
        nn.init.zeros_(self.content_score[-1].weight)
        nn.init.zeros_(self.content_score[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        source_height, source_width = source_grid
        target_height, target_width = target_grid
        if tokens.shape != (source_height * source_width, self.hidden_size):
            raise ValueError(
                f"Expected tokens with shape "
                f"({source_height * source_width}, {self.hidden_size}), "
                f"got {tuple(tokens.shape)}"
            )
        if target_height <= 0 or target_width <= 0:
            raise ValueError("Target grid dimensions must be greater than zero")
        if target_height > source_height or target_width > source_width:
            raise ValueError("Target grid cannot be larger than source grid")

        device = tokens.device
        source_positions = torch.stack(
            torch.meshgrid(
                torch.arange(source_height, device=device),
                torch.arange(source_width, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        source_positions = source_positions.float()

        target_rows = (
            (2 * torch.arange(target_height, device=device) + 1)
            * source_height
            // (2 * target_height)
        ).float()
        target_columns = (
            (2 * torch.arange(target_width, device=device) + 1)
            * source_width
            // (2 * target_width)
        ).float()
        target_positions = torch.stack(
            torch.meshgrid(target_rows, target_columns, indexing="ij"), dim=-1
        ).reshape(-1, 2)

        distances = torch.cdist(target_positions, source_positions).square()
        neighbor_count = min(self.neighbors, source_positions.shape[0])
        neighbor_indices = distances.topk(neighbor_count, dim=1, largest=False).indices
        neighbor_distances = distances.gather(1, neighbor_indices)

        candidates = tokens[neighbor_indices].float()
        content_scores = self.content_score(self.norm(candidates)).squeeze(-1)
        position_scores = -self.log_position_scale.exp() * neighbor_distances
        weights = torch.softmax(content_scores + position_scores, dim=-1)
        output = (weights.unsqueeze(-1) * candidates).sum(dim=1)
        return output.to(tokens.dtype)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "resampler.pt")
        config = {
            "hidden_size": self.hidden_size,
            "neighbors": self.neighbors,
            "score_hidden_size": self.score_hidden_size,
        }
        (directory / "resampler_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, device: torch.device
    ) -> "LocalResampler":
        directory = Path(directory)
        config = json.loads(
            (directory / "resampler_config.json").read_text(encoding="utf-8")
        )
        resampler = cls(**config)
        state = torch.load(
            directory / "resampler.pt", map_location="cpu", weights_only=True
        )
        resampler.load_state_dict(state)
        return resampler.to(device=device, dtype=torch.float32)
