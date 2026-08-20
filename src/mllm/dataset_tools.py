import json
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


def samples_to_frame(samples: list, split: str, fields: list[str]) -> pd.DataFrame:
    """Flatten source samples to one row per image and one column per field."""
    rows = []
    for source_index, sample in enumerate(samples):
        row = {"source_index": source_index, "image_path": None}
        row.update(dict.fromkeys(fields))

        if isinstance(sample, dict):
            image = sample.get("image")
            row["image_path"] = image.strip() if isinstance(image, str) else None
            annotations = sample.get("fields", [])
            if isinstance(annotations, list):
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    full_name = annotation.get("field_name")
                    if not isinstance(full_name, str):
                        continue
                    field = full_name.rsplit("/", 1)[-1].strip()
                    if field not in fields:
                        continue
                    text = annotation.get("annotator_text")
                    if isinstance(text, str):
                        text = text.strip() or None
                    elif text is not None:
                        text = str(text)
                    # Keep the first annotation. Duplicate fields are reported separately.
                    if row[field] is None:
                        row[field] = text

        rows.append(row)

    return pd.DataFrame(rows, columns=["source_index", "image_path", *fields])


def issue_dicts(issues: list[DatasetIssue]) -> list[dict]:
    return [asdict(issue) for issue in issues]
