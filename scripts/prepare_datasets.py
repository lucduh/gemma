import json
import random
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from mllm.constants import DATA_ROOT, DATASET_PREFIXES
from mllm.dataset_tools import (
    inspect_samples,
    issue_dicts,
    load_samples,
    samples_to_frame,
)


def selected_datasets(dataset: str | None) -> list[str]:
    if dataset is None:
        return list(DATASET_PREFIXES)
    if dataset not in DATASET_PREFIXES:
        raise typer.BadParameter(f"Choose one of: {', '.join(DATASET_PREFIXES)}")
    return [dataset]


def main(
    dataset: Annotated[
        str | None,
        typer.Option(help="Dataset name. Omit to prepare every registered dataset."),
    ] = None,
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing one directory per dataset."),
    ] = DATA_ROOT,
    overwrite: bool = False,
    seed: int = 42,
):
    """Create clean Parquet splits while discovering fields automatically."""
    for name in selected_datasets(dataset):
        dataset_dir = data_root / name
        split_samples = {}
        all_fields: set[str] = set()
        all_full_names: set[str] = set()
        all_issues = []

        for split in ("train", "test"):
            source = dataset_dir / f"{split}.json"
            if not source.is_file():
                raise FileNotFoundError(source)
            samples = load_samples(source)
            fields, full_names, issues = inspect_samples(
                samples, split, DATASET_PREFIXES[name]
            )
            split_samples[split] = samples
            all_fields.update(fields)
            all_full_names.update(full_names)
            all_issues.extend(issues)

        fields = sorted(all_fields)
        if not fields:
            raise ValueError(f"No valid fields discovered in {dataset_dir}")

        outputs = {split: dataset_dir / f"{split}.parquet" for split in split_samples}
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            paths = ", ".join(map(str, existing))
            raise FileExistsError(f"Refusing to overwrite {paths}; pass --overwrite")

        split_rows = {}
        cleaning = {}
        for split, samples in split_samples.items():
            rng = random.Random(f"{seed}:{name}:{split}")
            frame, split_cleaning = samples_to_frame(samples, fields, dataset_dir, rng)
            frame.to_parquet(outputs[split], index=False)
            split_rows[split] = len(frame)
            cleaning[split] = split_cleaning
            typer.echo(
                f"Wrote {outputs[split]} ({len(frame)} rows, {len(fields)} fields; "
                f"excluded {split_cleaning['excluded_row_count']} missing/invalid "
                f"images; resolved {split_cleaning['resolved_duplicate_count']} "
                "duplicate fields)"
            )

        issue_counts = Counter(issue.code for issue in all_issues)
        report = {
            "dataset": name,
            "directory": str(dataset_dir),
            "expected_prefix": DATASET_PREFIXES[name],
            "fields": fields,
            "full_field_names": sorted(all_full_names),
            "rows": split_rows,
            "duplicate_selection_seed": seed,
            "cleaning": cleaning,
            "issue_counts": dict(sorted(issue_counts.items())),
            "issues": issue_dicts(all_issues),
        }
        report_path = dataset_dir / "preparation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"Report: {report_path}")


if __name__ == "__main__":
    typer.run(main)
