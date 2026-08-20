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

_METADATA_COLUMNS = {"source_index", "image_path"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    field_prefix: str
    prompt_template: str

    def directory(self, data_root: Path = DATA_ROOT) -> Path:
        return data_root / self.name


@dataclass(frozen=True)
class DatasetSplit:
    config: DatasetConfig
    split: str
    directory: Path
    path: Path
    fields: tuple[str, ...]
    prompt: str
    samples: list[dict]


DATASETS = {
    "BR": DatasetConfig("BR", "BR/COMISSION_PAYMENET/", BR_PROMPT),
    "KARAPASS_DEATH": DatasetConfig(
        "KARAPASS_DEATH",
        "FR_CD/SUCCESSIONS/",
        KARAPASS_DEATH_PROMPT,
    ),
    "KARAPASS_ID": DatasetConfig(
        "KARAPASS_ID",
        "FR_KARAPASS/CLAIMS/",
        KARAPASS_ID_PROMPT,
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    try:
        return DATASETS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset {name!r}; choose one of: {', '.join(DATASETS)}"
        ) from error


def load_split(
    dataset: str,
    split: str,
    data_root: Path = DATA_ROOT,
) -> DatasetSplit:
    config = get_dataset(dataset)
    directory = config.directory(data_root)
    path = directory / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run scripts/data/prepare.py first"
        )

    frame = pd.read_parquet(path)
    missing_columns = _METADATA_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    fields = tuple(
        column for column in frame.columns if column not in _METADATA_COLUMNS
    )
    if not fields:
        raise ValueError(f"{path} has no extracted field columns")

    samples = []
    for record in frame.to_dict(orient="records"):
        samples.append(
            {key: None if pd.isna(value) else value for key, value in record.items()}
        )

    return DatasetSplit(
        config=config,
        split=split,
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
    return {
        field: str(sample[field]).strip() if sample.get(field) is not None else None
        for field in fields
    }
