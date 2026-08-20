import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetIssue:
    split: str
    source_index: int | None
    code: str
    detail: str


def load_samples(path: Path) -> list:
    """Load a source JSON file and require a top-level array."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path} must contain a top-level JSON array")
    return value


def inspect_samples(
    samples: list,
    split: str,
    expected_prefix: str | None = None,
) -> tuple[set[str], set[str], list[DatasetIssue]]:
    """Discover terminal field names and report malformed source annotations."""
    fields: set[str] = set()
    full_field_names: set[str] = set()
    short_to_full: dict[str, set[str]] = {}
    issues: list[DatasetIssue] = []

    for source_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            issues.append(
                DatasetIssue(
                    split, source_index, "invalid_sample", "sample is not an object"
                )
            )
            continue

        image = sample.get("image")
        if not isinstance(image, str) or not image.strip():
            issues.append(
                DatasetIssue(
                    split,
                    source_index,
                    "invalid_image",
                    "image is missing or not a non-empty string",
                )
            )

        annotations = sample.get("fields", [])
        if not isinstance(annotations, list):
            issues.append(
                DatasetIssue(
                    split, source_index, "invalid_fields", "fields is not an array"
                )
            )
            continue

        seen_in_sample: set[str] = set()
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "invalid_annotation",
                        f"fields[{annotation_index}] is not an object",
                    )
                )
                continue

            full_name = annotation.get("field_name")
            if not isinstance(full_name, str) or not full_name.strip():
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "invalid_field_name",
                        f"fields[{annotation_index}].field_name is missing or invalid",
                    )
                )
                continue

            full_name = full_name.strip()
            field = full_name.rsplit("/", 1)[-1].strip()
            if not field:
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "empty_terminal_field_name",
                        f"field_name={full_name!r}",
                    )
                )
                continue

            fields.add(field)
            full_field_names.add(full_name)
            short_to_full.setdefault(field, set()).add(full_name)

            if expected_prefix and not full_name.startswith(expected_prefix):
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "unexpected_prefix",
                        f"field_name={full_name!r}; expected prefix={expected_prefix!r}",
                    )
                )
            if field in seen_in_sample:
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "duplicate_field",
                        f"field={field!r}",
                    )
                )
            seen_in_sample.add(field)

            text = annotation.get("annotator_text")
            if text is not None and not isinstance(text, str):
                issues.append(
                    DatasetIssue(
                        split,
                        source_index,
                        "non_string_annotator_text",
                        f"field={field!r}; type={type(text).__name__}",
                    )
                )

    for field, names in short_to_full.items():
        if len(names) > 1:
            issues.append(
                DatasetIssue(
                    split,
                    None,
                    "terminal_name_collision",
                    f"field={field!r}; full names={sorted(names)!r}",
                )
            )

    return fields, full_field_names, issues


def samples_to_frame(
    samples: list,
    fields: list[str],
    dataset_dir: Path,
    rng: random.Random,
) -> tuple[pd.DataFrame, dict]:
    """Flatten samples, resolve duplicate fields, and exclude missing images."""
    rows = []
    excluded_rows = []
    resolved_duplicates = []

    for source_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            excluded_rows.append(
                {"source_index": source_index, "reason": "invalid_sample"}
            )
            continue

        image = sample.get("image")
        if not isinstance(image, str) or not image.strip():
            excluded_rows.append(
                {"source_index": source_index, "reason": "invalid_image_path"}
            )
            continue

        image = image.strip()
        resolved_image = Path(image)
        if not resolved_image.is_absolute():
            resolved_image = dataset_dir / resolved_image
        if not resolved_image.is_file():
            excluded_rows.append(
                {
                    "source_index": source_index,
                    "image_path": image,
                    "reason": "missing_image",
                }
            )
            continue

        row = {"source_index": source_index, "image_path": image}
        row.update(dict.fromkeys(fields))
        candidates = {field: [] for field in fields}
        annotations = sample.get("fields", [])
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                full_name = annotation.get("field_name")
                if not isinstance(full_name, str):
                    continue
                field = full_name.rsplit("/", 1)[-1].strip()
                if field not in candidates:
                    continue
                text = annotation.get("annotator_text")
                if isinstance(text, str):
                    text = text.strip() or None
                elif text is not None:
                    text = str(text).strip() or None
                candidates[field].append(text)

        for field, values in candidates.items():
            populated = [value for value in values if value is not None]
            if populated:
                row[field] = rng.choice(populated)
            if len(values) > 1:
                resolved_duplicates.append(
                    {
                        "source_index": source_index,
                        "field": field,
                        "occurrences": len(values),
                        "populated_occurrences": len(populated),
                    }
                )

        rows.append(row)

    frame = pd.DataFrame(rows, columns=["source_index", "image_path", *fields])
    cleaning = {
        "input_rows": len(samples),
        "output_rows": len(frame),
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows,
        "resolved_duplicate_count": len(resolved_duplicates),
        "resolved_duplicates": resolved_duplicates,
    }
    return frame, cleaning


def issue_dicts(issues: list[DatasetIssue]) -> list[dict]:
    return [asdict(issue) for issue in issues]
