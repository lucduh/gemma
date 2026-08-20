import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_samples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_fields(samples: list[dict]) -> set[str]:
    return {
        annotation["field_name"].rsplit("/", 1)[-1]
        for sample in samples
        for annotation in sample["fields"]
    }


def samples_to_frame(
    samples: list[dict],
    fields: list[str],
    dataset_dir: Path,
    rng: random.Random,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    missing_images = []
    duplicate_fields = 0

    for source_index, sample in enumerate(samples):
        image = sample["image"].strip()
        path = Path(image) if Path(image).is_absolute() else dataset_dir / image
        if not path.is_file():
            missing_images.append({"source_index": source_index, "image_path": image})
            continue

        values = defaultdict(list)
        for annotation in sample["fields"]:
            field = annotation["field_name"].rsplit("/", 1)[-1]
            text = annotation.get("annotator_text")
            text = str(text).strip() if text is not None else ""
            values[field].append(text or None)

        row = {"source_index": source_index, "image_path": image}
        for field in fields:
            populated = [value for value in values[field] if value]
            row[field] = rng.choice(populated) if populated else None
            duplicate_fields += len(values[field]) > 1
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["source_index", "image_path", *fields])
    report = {
        "input_rows": len(samples),
        "output_rows": len(frame),
        "missing_image_rows": missing_images,
        "duplicate_fields_resolved": duplicate_fields,
    }
    return frame, report
