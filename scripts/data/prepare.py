import argparse
import json
import random
from pathlib import Path

from mllm.constants import DATA_ROOT
from mllm.dataset_tools import discover_fields, load_samples, samples_to_frame
from mllm.datasets import DATASETS


def main(
    dataset: str | None,
    data_root: Path,
    overwrite: bool,
    seed: int,
):
    names = [dataset] if dataset else DATASETS
    for name in names:
        dataset_dir = data_root / name
        split_samples = {
            split: load_samples(dataset_dir / f"{split}.json")
            for split in ("train", "test")
        }
        fields = sorted(
            set().union(
                *(discover_fields(samples) for samples in split_samples.values())
            )
        )
        outputs = {split: dataset_dir / f"{split}.parquet" for split in split_samples}
        existing = [path for path in outputs.values() if path.exists()]
        if existing and not overwrite:
            paths = ", ".join(map(str, existing))
            raise FileExistsError(f"Refusing to overwrite {paths}; pass --overwrite")

        cleaning = {}
        for split, samples in split_samples.items():
            rng = random.Random(f"{seed}:{name}:{split}")
            frame, cleaning[split] = samples_to_frame(samples, fields, dataset_dir, rng)
            frame.to_parquet(outputs[split], index=False)
            print(
                f"Wrote {outputs[split]}: {len(frame)} rows, {len(fields)} fields, "
                f"{len(cleaning[split]['missing_image_rows'])} missing images removed"
            )

        report = {
            "dataset": name,
            "fields": fields,
            "duplicate_selection_seed": seed,
            "cleaning": cleaning,
        }
        report_path = dataset_dir / "preparation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert dataset JSON files to clean Parquet files."
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
