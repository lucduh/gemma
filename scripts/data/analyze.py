import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from mllm.constants import DATA_ROOT
from mllm.datasets import DATASETS


def inspect_images(paths: pd.Series, dataset_dir: Path) -> dict:
    missing = []
    corrupt = []
    sizes = []
    for value in paths.dropna().unique():
        path = Path(value) if Path(value).is_absolute() else dataset_dir / value
        if not path.is_file():
            missing.append(value)
            continue
        try:
            with Image.open(path) as image:
                sizes.append((image.width, image.height))
                image.verify()
        except OSError:
            corrupt.append(value)
    return {
        "missing": missing,
        "corrupt": corrupt,
        "min_width": min((width for width, _ in sizes), default=None),
        "max_width": max((width for width, _ in sizes), default=None),
        "min_height": min((height for _, height in sizes), default=None),
        "max_height": max((height for _, height in sizes), default=None),
    }


def main(dataset: str | None, data_root: Path, check_images: bool):
    names = [dataset] if dataset else DATASETS
    for name in names:
        dataset_dir = data_root / name
        frames = {
            split: pd.read_parquet(dataset_dir / f"{split}.parquet")
            for split in ("train", "test")
        }
        split_reports = {}
        for split, frame in frames.items():
            fields = [
                column
                for column in frame.columns
                if column not in {"source_index", "image_path"}
            ]
            duplicates = frame["image_path"].duplicated(keep=False)
            split_reports[split] = {
                "rows": len(frame),
                "fields": fields,
                "missing_values": {
                    field: int(frame[field].isna().sum()) for field in fields
                },
                "duplicate_images": sorted(
                    frame.loc[duplicates, "image_path"].unique().tolist()
                ),
            }
            if check_images:
                split_reports[split]["images"] = inspect_images(
                    frame["image_path"], dataset_dir
                )

        overlap = sorted(
            set(frames["train"]["image_path"]) & set(frames["test"]["image_path"])
        )
        report = {
            "dataset": name,
            "splits": split_reports,
            "train_test_image_overlap": overlap,
        }
        output = dataset_dir / "analysis_report.json"
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"\n{name}")
        for split, values in split_reports.items():
            print(
                f"  {split}: {values['rows']} rows, {len(values['fields'])} fields, "
                f"{len(values['duplicate_images'])} duplicate images"
            )
        print(f"  train/test image overlap: {len(overlap)}")
        print(f"Report: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze prepared Parquet datasets.")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--no-check-images", dest="check_images", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
