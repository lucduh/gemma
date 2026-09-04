"""Preserve BR document formatting by redacting pixels in copies of source images."""

import argparse
import base64
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from mllm.config import DATASETS, MODELS
from mllm.inference import load_model

GEMMA_MODELS = tuple(name for name in MODELS if name.startswith("gemma"))
DEFAULT_OUTPUT_DIR = Path("results/anonymization/br-layout")
DEFAULT_IMAGE_TOKENS = 1120
DEFAULT_MAX_NEW_TOKENS = 2048
CATEGORY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

# Separate passes are intentionally redundant. Focused prompts find substantially
# more regions than asking a vision-language model for every privacy category once.
SENSITIVE_PASSES = (
    (
        "names_and_organizations",
        (
            "person names, customer or provider names, legal company names, trade names, "
            "initials, and identifying names inside logos, stamps, or signatures"
        ),
        "person_name, company_name, trade_name, signature, identifying_logo",
    ),
    (
        "addresses_and_contacts",
        (
            "every complete or partial address block, street, building number, "
            "complement, neighborhood, city/state location, postal code or CEP, "
            "telephone number, email address, and identifying website"
        ),
        "address, street, location, cep, phone, email, website",
    ),
    (
        "identifiers",
        (
            "CPF, CNPJ, municipal or state registration, personal or company "
            "identifier, account number, registration number, and any other unique "
            "alphanumeric ID"
        ),
        "cpf, cnpj, tax_identifier, registration, account, other_identifier",
    ),
    (
        "document_and_transaction",
        (
            "invoice or document number, issue or service date and time, order, "
            "contract, protocol, verification or authentication code, access key, "
            "and transaction ID"
        ),
        "document_number, date, order, contract, protocol, verification_code, access_key",
    ),
    (
        "financial",
        (
            "bank, branch, account, PIX or payment details and every monetary value, "
            "tax amount, rate, deduction, total, and other transaction-specific "
            "financial value"
        ),
        "bank_details, payment_details, amount, tax, rate, total",
    ),
    (
        "machine_codes_and_free_text",
        (
            "the complete area of every QR code, barcode, signature, stamp, long "
            "machine code, and free-text service description or note that can reveal "
            "a person, organization, location, project, case, health matter, or "
            "transaction"
        ),
        "qr_code, barcode, signature, stamp, machine_code, sensitive_free_text",
    ),
)


@dataclass(frozen=True)
class Redaction:
    category: str
    box_2d: tuple[int, int, int, int]


def build_prompt(fields: tuple[str, ...]) -> str:
    field_list = ", ".join(fields)
    return f"""Locate every identifying or confidential region in this Brazilian service invoice.

Return exactly one compact JSON object and nothing else, with this schema:
{{"redactions":[{{"category":"cpf_cnpj_tomador","box_2d":[y_min,x_min,y_max,x_max]}},{{"category":"address","box_2d":[y_min,x_min,y_max,x_max]}}]}}

The array must contain a separate object for EVERY sensitive region in the document.
A normal invoice should produce many objects. Scan from top to bottom and do not stop
after the first match.

Rules:
- Coordinates are integers from 0 to 1000, normalized over the entire image.
- Coordinate order is exactly [y_min, x_min, y_max, x_max].
- Draw a tight box around the complete printed VALUE. Include all punctuation.
- For QR codes, barcodes, signatures, stamps, and access keys, box the entire region.
- Do not include transcribed text or field values in the JSON.
- Known dataset categories are: {field_list}.
- Also locate names, company names, tax identifiers, invoice/document numbers, dates,
  full or partial addresses, postal codes, phones, emails, bank/payment details,
  verification codes, QR/barcode payloads, signatures, and identifying free text.
- Use short lowercase category names for other regions.
- Do not box generic labels unless the label itself contains identifying information.
- When uncertain whether a region is identifying, include it.
"""


def build_sensitive_prompt(pass_name: str, targets: str, categories: str) -> str:
    return f"""Perform a focused privacy scan of this Brazilian service invoice.

Scan type: {pass_name}
Find ALL of the following anywhere on the page: {targets}.
Use these category names when applicable: {categories}.

Return exactly one compact JSON object and nothing else:
{{"redactions":[{{"category":"address","box_2d":[y_min,x_min,y_max,x_max]}},{{"category":"cep","box_2d":[y_min,x_min,y_max,x_max]}}]}}

Return one object for every separate matching region. Scan the full page from top to
bottom and do not stop after the first match. Coordinates are integers from 0 to
1000 over the entire image in exact [y_min, x_min, y_max, x_max] order. Cover each
complete value or visual code, including punctuation. For a multiline address or
free-text block, cover the complete block. Do not include transcribed text or field
values. Return an empty redactions list only when none of these targets is visible.
"""


def build_field_prompt(field: str, value: str) -> str:
    encoded_value = json.dumps(value, ensure_ascii=False)
    return f"""Locate the visible value corresponding to one annotated field in this Brazilian service invoice.

Field category: {field}
Annotation value: {encoded_value}

Return exactly this JSON shape and nothing else:
{{"redactions":[{{"category":"{field}","box_2d":[y_min,x_min,y_max,x_max]}}]}}

Coordinates are integers from 0 to 1000 over the entire image, in the exact order
[y_min, x_min, y_max, x_max]. Box the complete visible field value, including any
punctuation or adjacent time component. Use its label and section to disambiguate it.
The annotation can be normalized or formatted slightly differently from the image.
Return an empty redactions list only if the corresponding value is genuinely absent.
Do not return the transcribed value.
"""


def annotation_targets(
    row: pd.Series, fields: tuple[str, ...]
) -> list[tuple[str, str]]:
    targets = []
    for field in fields:
        value = row[field]
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            continue
        rendered = str(value).strip()
        if rendered:
            targets.append((field, rendered))
    return targets


def decode_json_object(response: str) -> dict:
    value = response.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        value = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for start, character in enumerate(value):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(value[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise ValueError("Gemma did not return a JSON object")


def parse_redactions(response: str) -> list[Redaction]:
    payload = decode_json_object(response)
    values = payload.get("redactions")
    if not isinstance(values, list):
        raise TypeError("Gemma response does not contain a redactions list")

    redactions = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        category = value.get("category")
        box = value.get("box_2d")
        if not isinstance(category, str) or not CATEGORY.fullmatch(category):
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        if any(
            isinstance(coordinate, bool) or not isinstance(coordinate, (int, float))
            for coordinate in box
        ):
            continue

        y_min, x_min, y_max, x_max = (
            max(0, min(1000, round(coordinate))) for coordinate in box
        )
        y_min, y_max = sorted((y_min, y_max))
        x_min, x_max = sorted((x_min, x_max))
        if y_min == y_max or x_min == x_max:
            continue
        key = (category, y_min, x_min, y_max, x_max)
        if key not in seen:
            redactions.append(Redaction(category, (y_min, x_min, y_max, x_max)))
            seen.add(key)
    return redactions


def boxes_payload(redactions: list[Redaction]) -> dict:
    return {
        "coordinate_system": "normalized_0_1000",
        "box_order": ["y_min", "x_min", "y_max", "x_max"],
        "redactions": [
            {"category": redaction.category, "box_2d": list(redaction.box_2d)}
            for redaction in redactions
        ],
    }


def read_boxes(path: Path) -> list[Redaction]:
    return parse_redactions(path.read_text(encoding="utf-8"))


def pixel_box(
    box: tuple[int, int, int, int], width: int, height: int, padding: int
) -> tuple[int, int, int, int]:
    y_min, x_min, y_max, x_max = box
    left = max(0, math.floor(x_min * width / 1000) - padding)
    top = max(0, math.floor(y_min * height / 1000) - padding)
    right = min(width, math.ceil(x_max * width / 1000) + padding)
    bottom = min(height, math.ceil(y_max * height / 1000) + padding)
    return left, top, right, bottom


def render_images(
    image: Image.Image, redactions: list[Redaction], padding: int
) -> tuple[Image.Image, Image.Image]:
    review = image.copy()
    redacted = image.copy()
    review_draw = ImageDraw.Draw(review)
    redacted_draw = ImageDraw.Draw(redacted)
    outline_width = max(2, min(image.size) // 400)

    for redaction in redactions:
        left, top, right, bottom = pixel_box(
            redaction.box_2d, image.width, image.height, padding
        )
        review_draw.rectangle(
            (left, top, max(left, right - 1), max(top, bottom - 1)),
            outline="#ff0000",
            width=outline_width,
        )
        redacted_draw.rectangle(
            (left, top, max(left, right - 1), max(top, bottom - 1)), fill="#000000"
        )
    return review, redacted


def image_html(image: Image.Image, title: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
html, body {{ margin: 0; background: #e5e7eb; }}
main {{ max-width: 1200px; margin: 20px auto; padding: 0; background: white; }}
img {{ display: block; width: 100%; height: auto; }}
@media print {{ body {{ background: white; }} main {{ margin: 0; max-width: none; }} }}
</style>
</head>
<body>
<main><img src="data:image/png;base64,{encoded}" alt="Anonymized BR document"></main>
</body>
</html>
"""


def resolve_image_path(value: str, dataset_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def main(
    model: str,
    device: str,
    dataset_dir: Path,
    split: str,
    indices: list[int],
    output_dir: Path,
    max_new_tokens: int,
    image_tokens: int,
    padding: int,
    adapter: Path | None,
    reuse_boxes: bool,
    general_only: bool,
) -> None:
    if len(set(indices)) != 2:
        raise ValueError("--indices must contain two distinct row positions")
    if padding < 0:
        raise ValueError("--padding cannot be negative")

    frame = pd.read_parquet(dataset_dir / f"{split}.parquet")
    fields = tuple(
        column
        for column in frame.columns
        if column not in {"source_index", "image_path"}
    )
    for index in indices:
        if not 0 <= index < len(frame):
            raise IndexError(f"row index {index} is outside 0..{len(frame) - 1}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gemma = None if reuse_boxes else load_model(model, device, adapter)
    prompt = build_prompt(fields)

    for example_number, index in enumerate(indices, start=1):
        stem = f"br_{split}_{index:05d}"
        boxes_path = output_dir / f"{stem}.boxes.json"
        if reuse_boxes:
            if not boxes_path.is_file():
                raise FileNotFoundError(
                    f"cannot reuse missing boxes file: {boxes_path}"
                )
            redactions = read_boxes(boxes_path)
        else:
            image_path = resolve_image_path(
                str(frame.iloc[index]["image_path"]), dataset_dir
            )
            with Image.open(image_path) as source:
                model_image = source.convert("RGB")
            print(f"Locating general sensitive regions in BR/{split} row {index}...")
            inputs = gemma.prepare_inputs([model_image], prompt, image_tokens)
            responses, _ = gemma.generate(inputs, max_new_tokens)
            redactions = parse_redactions(responses[0])

            located_fields = set()
            if not general_only:
                for pass_name, targets, categories in SENSITIVE_PASSES:
                    print(f"  Focused privacy scan: {pass_name}")
                    sensitive_prompt = build_sensitive_prompt(
                        pass_name, targets, categories
                    )
                    inputs = gemma.prepare_inputs(
                        [model_image], sensitive_prompt, image_tokens
                    )
                    responses, _ = gemma.generate(inputs, max_new_tokens)
                    try:
                        pass_redactions = parse_redactions(responses[0])
                    except (TypeError, ValueError) as error:
                        print(f"    WARNING: invalid scan response: {error}")
                        continue
                    if not pass_redactions:
                        print("    WARNING: this scan returned no regions")
                    redactions.extend(pass_redactions)

                for field, value in annotation_targets(frame.iloc[index], fields):
                    print(f"  Targeting annotated field: {field}")
                    field_prompt = build_field_prompt(field, value)
                    inputs = gemma.prepare_inputs(
                        [model_image], field_prompt, image_tokens
                    )
                    responses, _ = gemma.generate(inputs, max_new_tokens)
                    try:
                        field_redactions = parse_redactions(responses[0])
                    except (TypeError, ValueError) as error:
                        print(f"    WARNING: invalid grounding response: {error}")
                        continue
                    if field_redactions:
                        located_fields.add(field)
                    redactions.extend(
                        Redaction(field, redaction.box_2d)
                        for redaction in field_redactions
                    )

                expected_fields = {
                    field for field, _ in annotation_targets(frame.iloc[index], fields)
                }
                missing_fields = sorted(expected_fields - located_fields)
                if missing_fields:
                    print(
                        "  WARNING: no targeted box for: " + ", ".join(missing_fields)
                    )

            redactions = list(dict.fromkeys(redactions))
            boxes_path.write_text(
                json.dumps(boxes_payload(redactions), indent=2), encoding="utf-8"
            )

        if not redactions:
            raise ValueError(
                f"no redaction regions found for row {index}; refusing to copy the source"
            )

        image_path = resolve_image_path(
            str(frame.iloc[index]["image_path"]), dataset_dir
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        review, redacted = render_images(image, redactions, padding)

        review_path = output_dir / f"{stem}.review.png"
        redacted_path = output_dir / f"{stem}.redacted.png"
        html_path = output_dir / f"{stem}.redacted.html"
        review.save(review_path, format="PNG", optimize=True)
        redacted.save(redacted_path, format="PNG", optimize=True)
        html_path.write_text(
            image_html(redacted, f"BR report example {example_number}"),
            encoding="utf-8",
        )

        print(f"Example {example_number}: BR/{split} row {index}")
        print(f"  Source (private): {image_path}")
        print(f"  Boxes:           {boxes_path}")
        print(f"  Review image:    {review_path}")
        print(f"  Redacted image:  {redacted_path}")
        print(f"  Downloadable HTML: {html_path}")
        print(f"  Redaction regions: {len(redactions)}")

    print("\nWARNING: Gemma can miss sensitive regions or return inaccurate boxes.")
    print(
        "Check both review images against the sources and adjust boxes before publishing."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redact two BR images while preserving their original formatting."
    )
    parser.add_argument("--model", choices=GEMMA_MODELS, default="gemma4-e4b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=DATASETS["BR"].directory)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--indices",
        nargs=2,
        type=int,
        metavar=("FIRST", "SECOND"),
        default=[0, 1],
        help="two distinct zero-based Parquet row positions (default: 0 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_IMAGE_TOKENS)
    parser.add_argument(
        "--padding", type=int, default=6, help="extra pixels around every redaction"
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--reuse-boxes",
        action="store_true",
        help="skip Gemma and rerender using edited *.boxes.json files",
    )
    parser.add_argument(
        "--general-only",
        action="store_true",
        help="skip all focused privacy scans and annotated-field grounding passes",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
