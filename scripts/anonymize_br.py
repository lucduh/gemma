"""Create editable HTML transcriptions of two BR documents with Gemma.

The generated ``*.review.html`` files may contain personal data. The corresponding
``*.redacted.html`` files replace every value that Gemma tagged with a
``data-field`` attribute, but they still require a human privacy review.
"""

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from PIL import Image

from mllm.config import DATASETS, MODELS
from mllm.inference import load_model

GEMMA_MODELS = tuple(name for name in MODELS if name.startswith("gemma"))
DEFAULT_OUTPUT_DIR = Path("results/anonymization/br")
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_IMAGE_TOKENS = 1120

ALLOWED_TAGS = {
    "article",
    "b",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "em",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "i",
    "li",
    "main",
    "ol",
    "p",
    "section",
    "small",
    "span",
    "strong",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "time",
    "tr",
    "ul",
}
VOID_TAGS = {"br", "hr"}
DANGEROUS_TAGS = {"iframe", "math", "object", "script", "style", "svg", "template"}
CLASS_TOKEN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
INTEGER_ATTRIBUTE = re.compile(r"^[1-9][0-9]?$|^100$")
SAFE_STYLE_VALUE = re.compile(r"^[a-zA-Z0-9\s#.,()%/'\"_+*-]+$")
SAFE_STYLE_PROPERTIES = {
    "align-items",
    "background-color",
    "border",
    "border-bottom",
    "border-collapse",
    "border-color",
    "border-left",
    "border-radius",
    "border-right",
    "border-style",
    "border-top",
    "border-width",
    "bottom",
    "box-shadow",
    "box-sizing",
    "color",
    "column-gap",
    "display",
    "flex",
    "flex-direction",
    "flex-wrap",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "gap",
    "grid-column",
    "grid-row",
    "grid-template-columns",
    "grid-template-rows",
    "height",
    "justify-content",
    "justify-items",
    "left",
    "letter-spacing",
    "line-height",
    "margin",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "margin-top",
    "max-width",
    "min-height",
    "min-width",
    "overflow-wrap",
    "padding",
    "padding-bottom",
    "padding-left",
    "padding-right",
    "padding-top",
    "position",
    "right",
    "row-gap",
    "table-layout",
    "text-align",
    "text-decoration",
    "text-transform",
    "top",
    "vertical-align",
    "white-space",
    "width",
}
ALLOWED_DISPLAY_VALUES = {
    "block",
    "flex",
    "grid",
    "inline",
    "inline-block",
    "inline-flex",
    "table",
    "table-cell",
    "table-row",
}
ALLOWED_POSITION_VALUES = {"absolute", "relative", "static"}

HTML_STYLE = """
:root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
body { margin: 0; background: #eceff1; color: #17202a; }
.document-page {
  box-sizing: border-box; max-width: 210mm; min-height: 297mm; margin: 18px auto;
  padding: 16mm; background: white; box-shadow: 0 2px 12px #0002;
}
h1, h2, h3, h4 { margin: .5rem 0; }
p { margin: .35rem 0; }
section { margin: .8rem 0; }
table { width: 100%; border-collapse: collapse; margin: .75rem 0; }
th, td { border: 1px solid #9aa4ad; padding: .35rem .45rem; text-align: left; }
.field, .row { display: flex; flex-wrap: wrap; gap: .35rem; margin: .25rem 0; }
.label { font-weight: 700; }
.sensitive { background: #ffe082; outline: 2px solid #f9a825; }
.redacted { background: #111; color: white; padding: 0 .35em; white-space: nowrap; }
@media print {
  body { background: white; }
  .document-page { margin: 0; box-shadow: none; max-width: none; min-height: 0; }
  .sensitive { outline: none; }
}
""".strip()


def build_prompt(fields: tuple[str, ...]) -> str:
    field_list = ", ".join(fields)
    return f"""Transcribe this Brazilian service invoice into an editable HTML fragment.

Rules:
- Return only the HTML fragment, with no Markdown fence, explanation, document type,
  <html>, <head>, <body>, CSS, JavaScript, links, images, SVG, or comments.
- Reproduce only text that is visibly present. Do not infer, correct, translate, or
  invent text. Preserve the document's reading order.
- Reconstruct the visible formatting closely: page sections, columns, table borders,
  alignment, spacing, relative font sizes and weights, and shaded header cells.
- Use semantic headings, sections, paragraphs, lists, and tables. Use CSS grid or
  flexbox where the source has columns. Prefer tables for tabular invoice sections.
- Put formatting in inline style attributes. Use only ordinary layout properties
  such as display, grid/flex properties, width, margin, padding, border,
  background-color, color, font properties, line-height, and text-align. Do not use
  external resources, CSS variables, URLs, hidden content, or a <style> element.
- Use structural classes when helpful: document, section, row, field, label, value.
- Wrap every identifying or confidential VALUE in a dedicated element having
  class="sensitive" and data-field="CATEGORY". Keep its label outside that element.
- Known dataset categories are: {field_list}.
- Also tag names, company names, tax identifiers, invoice/document numbers, dates,
  full or partial addresses, postal codes, phone numbers, email addresses, bank
  details, access keys, QR/barcode payloads, signatures, and identifying free text.
  For these, use a short lowercase category such as person_name, company_name,
  address, phone, email, bank_details, access_key, or other_identifier.
- Never repeat a sensitive value outside its data-field element.
- Do not anonymize the values yet: transcribe them exactly so a human can review the
  private draft before making the publication copy.
"""


def extract_fragment(response: str) -> str:
    """Remove common model wrappers before sanitizing the HTML fragment."""
    value = response.strip()
    fenced = re.fullmatch(
        r"```(?:html)?\s*(.*?)\s*```", value, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        value = fenced.group(1).strip()

    body = re.search(
        r"<body(?:\s[^>]*)?>(.*?)</body\s*>",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if body:
        return body.group(1).strip()
    return value


def sanitize_style(value: str) -> str:
    """Keep formatting-oriented CSS declarations without external resources."""
    declarations = []
    for declaration in value.split(";"):
        property_name, separator, property_value = declaration.partition(":")
        if not separator:
            continue
        property_name = property_name.strip().lower()
        property_value = " ".join(property_value.strip().split())
        lowered_value = property_value.lower()
        if property_name not in SAFE_STYLE_PROPERTIES or not property_value:
            continue
        if not SAFE_STYLE_VALUE.fullmatch(property_value):
            continue
        if any(
            forbidden in lowered_value
            for forbidden in ("url(", "expression(", "javascript", "@import", "var(")
        ):
            continue
        if property_name == "display" and lowered_value not in ALLOWED_DISPLAY_VALUES:
            continue
        if property_name == "position" and lowered_value not in ALLOWED_POSITION_VALUES:
            continue
        declarations.append(f"{property_name}: {property_value}")
    return "; ".join(declarations)


class FragmentSanitizer(HTMLParser):
    """Allow safe structural HTML and optionally redact tagged element contents."""

    def __init__(self, redact_fields: set[str] | None = None, redact_all: bool = False):
        super().__init__(convert_charrefs=True)
        self.redact_fields = redact_fields or set()
        self.redact_all = redact_all
        self.output: list[str] = []
        self.stack: list[str] = []
        self.suppressed_depth = 0
        self.redacted_depth = 0
        self.redacted_tag: str | None = None
        self.detected_fields: set[str] = set()

    def _attributes(
        self, attrs: list[tuple[str, str | None]]
    ) -> tuple[str, str | None]:
        safe: list[tuple[str, str]] = []
        field = None
        classes: list[str] = []
        for name, value in attrs:
            name = name.lower()
            value = value or ""
            if name == "class":
                classes.extend(
                    token for token in value.split() if CLASS_TOKEN.fullmatch(token)
                )
            elif name == "style" and (style := sanitize_style(value)):
                safe.append((name, style))
            elif name == "data-field" and FIELD_NAME.fullmatch(value):
                field = value
            elif (
                name in {"colspan", "rowspan"} and INTEGER_ATTRIBUTE.fullmatch(value)
            ) or (name == "scope" and value in {"row", "col", "rowgroup", "colgroup"}):
                safe.append((name, value))

        if field:
            self.detected_fields.add(field)
            safe.append(("data-field", field))
        if classes:
            safe.insert(0, ("class", " ".join(dict.fromkeys(classes))))
        rendered = "".join(
            f' {name}="{html.escape(value, quote=True)}"' for name, value in safe
        )
        return rendered, field

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.suppressed_depth:
            if tag not in VOID_TAGS:
                self.suppressed_depth += 1
            return
        if tag in DANGEROUS_TAGS:
            self.suppressed_depth = 1
            return
        if self.redacted_depth:
            if tag not in VOID_TAGS:
                self.redacted_depth += 1
            return
        if tag not in ALLOWED_TAGS:
            return

        rendered_attrs, field = self._attributes(attrs)
        should_redact = field is not None and (
            self.redact_all or field in self.redact_fields
        )
        if should_redact:
            if 'class="' in rendered_attrs:
                rendered_attrs = rendered_attrs.replace(
                    'class="', 'class="redacted ', 1
                )
            else:
                rendered_attrs = f' class="redacted"{rendered_attrs}'
            self.output.append(f"<{tag}{rendered_attrs}>[REDACTED]")
            if tag not in VOID_TAGS:
                self.redacted_depth = 1
                self.redacted_tag = tag
            return

        self.output.append(f"<{tag}{rendered_attrs}>")
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.suppressed_depth:
            self.suppressed_depth -= 1
            return
        if self.redacted_depth:
            self.redacted_depth -= 1
            if self.redacted_depth == 0 and self.redacted_tag is not None:
                self.output.append(f"</{self.redacted_tag}>")
                self.redacted_tag = None
            return
        if tag not in self.stack:
            return
        while self.stack:
            open_tag = self.stack.pop()
            self.output.append(f"</{open_tag}>")
            if open_tag == tag:
                break

    def handle_data(self, data: str) -> None:
        if not self.suppressed_depth and not self.redacted_depth:
            self.output.append(html.escape(data, quote=False))

    def close(self) -> None:
        super().close()
        while self.stack:
            self.output.append(f"</{self.stack.pop()}>")

    def result(self) -> str:
        return "".join(self.output).strip()


def sanitize_fragment(
    fragment: str,
    *,
    redact_fields: set[str] | None = None,
    redact_all: bool = False,
) -> tuple[str, set[str]]:
    sanitizer = FragmentSanitizer(redact_fields=redact_fields, redact_all=redact_all)
    sanitizer.feed(fragment)
    sanitizer.close()
    return sanitizer.result(), sanitizer.detected_fields


def make_document(fragment: str, title: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
{HTML_STYLE}
</style>
</head>
<body>
<main class="document-page">
{fragment}
</main>
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
    adapter: Path | None,
) -> None:
    if len(set(indices)) != 2:
        raise ValueError("--indices must contain two distinct row positions")

    parquet_path = dataset_dir / f"{split}.parquet"
    frame = pd.read_parquet(parquet_path)
    fields = tuple(
        column
        for column in frame.columns
        if column not in {"source_index", "image_path"}
    )
    for index in indices:
        if not 0 <= index < len(frame):
            raise IndexError(f"row index {index} is outside 0..{len(frame) - 1}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gemma = load_model(model, device, adapter)
    prompt = build_prompt(fields)

    for example_number, index in enumerate(indices, start=1):
        image_path = resolve_image_path(
            str(frame.iloc[index]["image_path"]), dataset_dir
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        inputs = gemma.prepare_inputs([image], prompt, image_tokens)
        responses, _ = gemma.generate(inputs, max_new_tokens)
        fragment = extract_fragment(responses[0])
        review_fragment, detected_fields = sanitize_fragment(fragment)
        redacted_fragment, _ = sanitize_fragment(fragment, redact_all=True)

        stem = f"br_{split}_{index:05d}"
        review_path = output_dir / f"{stem}.review.html"
        redacted_path = output_dir / f"{stem}.redacted.html"
        title = f"BR report example {example_number}"
        review_path.write_text(make_document(review_fragment, title), encoding="utf-8")
        redacted_path.write_text(
            make_document(redacted_fragment, title), encoding="utf-8"
        )

        print(f"Example {example_number}: BR/{split} row {index}")
        print(f"  Source (private): {image_path}")
        print(f"  Review draft:     {review_path}")
        print(f"  Redacted draft:   {redacted_path}")
        print(f"  Tagged fields:    {', '.join(sorted(detected_fields)) or 'NONE'}")

    print("\nWARNING: Gemma tagging is not a privacy guarantee.")
    print(
        "Compare each redacted draft with its source and inspect the HTML source before publishing."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert exactly two BR dataset images to editable HTML with Gemma."
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
    parser.add_argument("--adapter", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
