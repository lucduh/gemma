import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from PIL import Image

from mllm.constants import DATA_ROOT
from mllm.datasets import DATASETS


def selected_datasets(dataset: str | None) -> list[str]:
    if dataset is None:
        return list(DATASETS)
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASETS)}")
    return [dataset]


def inspect_images(paths: pd.Series, dataset_dir: Path) -> dict:
    missing = []
    corrupt = []
    dimensions = {}
    for value in paths.dropna().unique():
        path = Path(value)
        path = path if path.is_absolute() else dataset_dir / path
        if not path.is_file():
            missing.append(str(value))
            continue
        try:
            with Image.open(path) as image:
                dimensions[str(value)] = [image.width, image.height]
                image.verify()
        except (OSError, ValueError) as error:
            corrupt.append({"image_path": str(value), "error": str(error)})
    return {
        "missing_count": len(missing),
        "missing": missing,
        "corrupt_count": len(corrupt),
        "corrupt": corrupt,
        "dimensions": dimensions,
    }


def main(
    dataset: Annotated[
        str | None,
        typer.Option(help="Dataset name. Omit to analyze every registered dataset."),
    ] = None,
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing one directory per dataset."),
    ] = DATA_ROOT,
    check_images: bool = True,
):
    """Analyze canonical train/test Parquet files and their referenced images."""
    for name in selected_datasets(dataset):
        dataset_dir = data_root / name
        frames = {}
        split_reports = {}

        for split in ("train", "test"):
            path = dataset_dir / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} does not exist; run scripts/data/prepare.py first"
                )
            frame = pd.read_parquet(path)
            frames[split] = frame
            field_columns = [
                column
                for column in frame.columns
                if column not in {"source_index", "image_path"}
            ]
            duplicate_images = frame["image_path"].dropna().duplicated(keep=False)
            split_report = {
                "rows": len(frame),
                "fields": field_columns,
                "missing_values": {
                    field: int(frame[field].isna().sum()) for field in field_columns
                },
                "present_values": {
                    field: int(frame[field].notna().sum()) for field in field_columns
                },
                "duplicate_image_rows": int(duplicate_images.sum()),
                "duplicate_images": sorted(
                    frame.loc[duplicate_images, "image_path"].unique().tolist()
                ),
                "exact_duplicate_rows": int(
                    frame.drop(columns="source_index", errors="ignore")
                    .duplicated(keep=False)
                    .sum()
                ),
            }
            if check_images:
                split_report["images"] = inspect_images(
                    frame["image_path"], dataset_dir
                )
            split_reports[split] = split_report

        train_images = set(frames["train"]["image_path"].dropna())
        test_images = set(frames["test"]["image_path"].dropna())
        overlap = sorted(train_images & test_images)
        report = {
            "dataset": name,
            "directory": str(dataset_dir),
            "splits": split_reports,
            "train_test_image_overlap_count": len(overlap),
            "train_test_image_overlap": overlap,
        }

        output = dataset_dir / "analysis_report.json"
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"\n{name}")
        for split, values in split_reports.items():
            typer.echo(
                f"  {split}: {values['rows']} rows, "
                f"{len(values['fields'])} fields, "
                f"{values['duplicate_image_rows']} duplicate-image rows"
            )
        typer.echo(f"  train/test image overlap: {len(overlap)}")
        typer.echo(f"Report: {output}")


if __name__ == "__main__":
    typer.run(main)
