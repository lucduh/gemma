from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mllm.constants import DATA_ROOT
from mllm.prompts import (
    BR_PROMPT,
    KARAPASS_DEATH_PROMPT,
    KARAPASS_ID_PROMPT,
    render_prompt,
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prompt_template: str

    def directory(self, data_root: Path = DATA_ROOT) -> Path:
        return data_root / self.name


@dataclass(frozen=True)
class DatasetSplit:
    directory: Path
    path: Path
    fields: tuple[str, ...]
    prompt: str
    samples: list[dict]


DATASETS = {
    "BR": DatasetConfig("BR", BR_PROMPT),
    "KARAPASS_DEATH": DatasetConfig("KARAPASS_DEATH", KARAPASS_DEATH_PROMPT),
    "KARAPASS_ID": DatasetConfig("KARAPASS_ID", KARAPASS_ID_PROMPT),
}


def load_split(dataset: str, split: str, data_root: Path = DATA_ROOT) -> DatasetSplit:
    config = DATASETS[dataset]
    directory = config.directory(data_root)
    path = directory / f"{split}.parquet"
    frame = pd.read_parquet(path)
    fields = tuple(
        column
        for column in frame.columns
        if column not in {"source_index", "image_path"}
    )
    samples = frame.where(frame.notna(), None).to_dict(orient="records")
    return DatasetSplit(
        directory=directory,
        path=path,
        fields=fields,
        prompt=render_prompt(config.prompt_template, fields),
        samples=samples,
    )


def image_path(sample: dict, directory: Path) -> Path:
    path = Path(sample["image_path"])
    return path if path.is_absolute() else directory / path


def ground_truth(sample: dict, fields: tuple[str, ...]) -> dict[str, str | None]:
    return {field: sample[field] for field in fields}
