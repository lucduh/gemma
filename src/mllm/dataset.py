from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from mllm.config import DATASETS


class Dataset(TorchDataset):
    def __init__(self, name: str, split: str) -> None:
        config = DATASETS[name]
        self.name = name
        self.split = split
        self.directory = config.directory
        self.path = self.directory / f"{split}.parquet"
        self.samples = pd.read_parquet(self.path)
        self.fields = tuple(
            column
            for column in self.samples.columns
            if column not in {"source_index", "image_path"}
        )
        self.prompt = config.prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, dict]:
        sample = self.samples.iloc[index]
        path = Path(sample.image_path)
        with Image.open(path) as source:
            image = source.convert("RGB")

        gt = sample[list(self.fields)].to_dict()
        return image, str(path), gt
