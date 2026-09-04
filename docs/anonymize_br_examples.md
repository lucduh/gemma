# Anonymizing two BR examples as editable HTML

This guide creates editable HTML transcriptions of two Brazilian service invoices
from the BR dataset. Gemma reads each source image and reconstructs its visible
content as structural HTML. The script then produces:

1. a **private review draft**, containing Gemma's transcription and highlighted
   sensitive values; and
2. a **redacted draft**, in which every value Gemma tagged as sensitive is replaced
   with `[REDACTED]`.

The redacted draft is only a starting point. A vision-language model can miss,
duplicate, or misclassify text. Human review is mandatory before either example is
included in a report.

## Important distinction: transcription is not pixel redaction

The script does not modify the source invoice image. It creates a new HTML
representation and deliberately does not embed or link the original image. This is
useful for a report because text and table cells can be edited directly, but the
result is a model-produced reconstruction rather than a facsimile. Describe it that
way in the figure caption; for example:

> An anonymized HTML reconstruction of a BR dataset document, transcribed with
> Gemma and manually checked against the source.

Do not place the unredacted scan under the HTML as a background, and do not use CSS
to merely blur or hide source content. Hidden and blurred content can often be
recovered. Values must be removed from the HTML source or replaced with neutral
placeholders.

### Recommended when the original formatting must remain unchanged

Use pixel redaction rather than transcription when the report must retain the exact
source layout:

```bash
uv run python scripts/redact_br_layout.py \
  --model gemma4-e4b \
  --split test \
  --indices 0 1 \
  --image-tokens 1120
```

This first asks Gemma for all sensitive regions, runs six focused privacy scans, and
then makes an additional targeted grounding request for every non-empty annotated BR
field. The focused scans separately cover names and organizations, addresses and
contacts, identifiers, document and transaction details, financial values, and
machine codes/signatures/free text. These redundant passes prevent a short first
response from producing only one redaction. The script then draws opaque black
rectangles into a copy of each image, strips the original image metadata, and
produces:

```text
results/anonymization/br-layout/br_test_00000.boxes.json
results/anonymization/br-layout/br_test_00000.review.png
results/anonymization/br-layout/br_test_00000.redacted.png
results/anonymization/br-layout/br_test_00000.redacted.html
```

The standalone HTML embeds only the already-redacted PNG. It does not contain or
link the original image, so downloading the HTML alone is sufficient. Every pixel
outside the redaction rectangles remains identical to the decoded source image.
The private `review.png` outlines proposed regions in red and must not be published.

Gemma's boxes are suggestions, not a privacy guarantee. The terminal warns when an
annotated field did not receive a targeted box. Compare the review image to the
source even when there are no warnings. To fix a missed or inaccurate region, edit
`box_2d` in the corresponding
`*.boxes.json`. Coordinates use `[y_min, x_min, y_max, x_max]` normalized from 0 to
1000. Add another object when a region was missed:

```json
{
  "category": "other_identifier",
  "box_2d": [120, 640, 155, 910]
}
```

Then regenerate the PNG and HTML without rerunning Gemma:

```bash
uv run python scripts/redact_br_layout.py \
  --split test --indices 0 1 --reuse-boxes
```

Use `--padding 10` if boxes are too tight around character edges. The default is six
extra pixels. Keep increasing or manually correcting boxes until every sensitive
pixel is covered. Download only `*.redacted.html` or `*.redacted.png` after this
review.

The remainder of this tutorial covers editable HTML transcription with
`scripts/anonymize_br.py`. That route intentionally reconstructs the document and
therefore cannot preserve the original formatting exactly.

## 1. Decide what must be removed

Before running the model, define the privacy policy for the report. For BR invoices,
a conservative policy removes at least:

- names of people and companies;
- CPF/CNPJ identifiers;
- invoice numbers, access keys, verification codes, and QR/barcode payloads;
- full and partial addresses and CEP values;
- telephone numbers and email addresses;
- dates when they can identify a transaction;
- bank or payment details;
- signatures;
- free-text service descriptions that name a person, organization, location, case,
  or transaction; and
- any unique combination of otherwise harmless fields.

Amounts and generic service descriptions may or may not be publishable under your
organization's policy. When uncertain, remove them. Anonymization is contextual:
removing a CPF while retaining a unique invoice number and company name is usually
not sufficient.

Use only documents that you are authorized to process and publish. Follow the BR
dataset license, your organization's data-handling rules, and any applicable legal
or ethics-review requirements.

## 2. Set up the project

Install the locked project environment from the repository root:

```bash
uv sync
```

The project is configured to find BR at:

```text
/domino/datasets/local/MLLM/data/BR/
├── train.parquet
├── test.parquet
└── ... source images ...
```

The Gemma checkpoints are registered in `src/mllm/config.py`. The available command
names are:

- `gemma3`
- `gemma4-e2b`
- `gemma4-e4b`

If BR is mounted elsewhere, pass its directory with `--dataset-dir`. The directory
must contain the selected Parquet split, and relative image paths in that file are
resolved against this directory.

Do not copy the dataset into the repository. The default output directory is under
`results/`, which is excluded by `.gitignore`.

## 3. Select exactly two examples

`--indices` takes two distinct, zero-based **row positions in the prepared Parquet
file**. They are not necessarily the values in the `source_index` column.

The default is rows 0 and 1 of the test split. For a scientific report, choose the
selection rule before examining model output—for example, the first two eligible
test rows or two rows selected with a recorded random seed. This avoids silently
selecting only unusually good transcriptions.

When choosing examples, prefer documents that do not contain unnecessary sensitive
free text. Avoid printing complete Parquet rows to the terminal or a notebook,
because that can copy annotations into logs and notebook outputs.

## 4. Generate the HTML drafts

For the strongest configured Gemma 4 model and a larger image-token budget:

```bash
uv run python scripts/anonymize_br.py \
  --model gemma4-e4b \
  --split test \
  --indices 0 1 \
  --image-tokens 1120 \
  --output-dir results/anonymization/br
```

For Gemma 3:

```bash
uv run python scripts/anonymize_br.py \
  --model gemma3 \
  --split test \
  --indices 0 1
```

Gemma 3 ignores `--image-tokens`. If GPU memory is limited, use `gemma4-e2b`, reduce
`--image-tokens`, or use Gemma 3. A lower image-token budget can make small identifiers
and table text easier for the model to miss, so inspect the result especially
carefully.

To use another BR mount:

```bash
uv run python scripts/anonymize_br.py \
  --model gemma4-e4b \
  --dataset-dir /secure/path/to/BR \
  --indices 12 47 \
  --image-tokens 1120
```

Useful optional arguments are:

- `--device cuda:1` to choose a different GPU;
- `--adapter /path/to/adapter` to load a compatible LoRA adapter;
- `--max-new-tokens 6000` if a dense document is being truncated; and
- `--split train` to select from the training split.

The script insists on two distinct indices and fails if either position is outside
the split.

## 5. Understand the output

For indices 0 and 1, the output directory contains:

```text
br_test_00000.review.html
br_test_00000.redacted.html
br_test_00001.review.html
br_test_00001.redacted.html
```

The terminal also reports the private source image path and the categories Gemma
tagged. No raw model response, original image, annotation value, source path, or
base64 image is embedded in the HTML.

Gemma is instructed to emit markup such as:

```html
<div class="field">
  <span class="label">CPF/CNPJ:</span>
  <span class="sensitive" data-field="cpf_cnpj_tomador">12.345.678/0001-90</span>
</div>
```

The review draft highlights the value in yellow. In the automatically redacted
draft, the same element becomes conceptually:

```html
<span class="redacted sensitive" data-field="cpf_cnpj_tomador">[REDACTED]</span>
```

All model HTML passes through an allow-list sanitizer. Scripts, styles, event
handlers, images, links, embedded objects, and unknown attributes are discarded.
The script supplies its own local print CSS. This reduces the risk of opening model
generated HTML, but does not make the transcription accurate or anonymized.

## 6. Perform the human review

Treat every `*.review.html` file as sensitive. Open it only on an approved machine.
Compare it side by side with the source path printed by the command.

For each visible item in the source, ask:

1. Did Gemma transcribe it?
2. If it is sensitive, did Gemma wrap the complete value in `data-field`?
3. Did Gemma repeat the value elsewhere without a sensitive tag?
4. Is sensitive information encoded in a QR code, barcode, signature, logo, stamp,
   free-text block, or long access key?
5. Can the remaining combination of fields identify the issuer, customer, or
   transaction?

Then inspect `*.redacted.html` in both rendered form and source form. Do not approve
it merely because all yellow highlights became black placeholders: untagged text is
not automatically removed.

Edit the redacted file with a plain-text editor. Typical safe edits are:

### Replace only a value

```html
<span class="redacted">[REDACTED]</span>
```

### Remove the entire field

Delete the complete field container when the label itself is unnecessary:

```html
<div class="field"> ... </div>
```

### Replace identifying free text

```html
<p>Descrição do serviço: [DESCRIPTION REMOVED]</p>
```

Use generic, consistent placeholders. Do not use realistic fake CPF/CNPJ values or
invented names, because readers may mistake them for real data. If document layout
matters, a fixed `[REDACTED]` marker or a black rectangle is preferable to a
plausible replacement.

## 7. Search the HTML source for common leaks

Visual inspection is not enough. Search the publication candidates themselves. The
following commands are heuristics, not proofs of anonymity:

```bash
# CPF-like values
rg -n '[0-9]{3}\.?[0-9]{3}\.?[0-9]{3}-?[0-9]{2}' \
  results/anonymization/br/*.redacted.html

# CNPJ-like values
rg -n '[0-9]{2}\.?[0-9]{3}\.?[0-9]{3}/?[0-9]{4}-?[0-9]{2}' \
  results/anonymization/br/*.redacted.html

# CEP-like values
rg -n '[0-9]{5}-?[0-9]{3}' results/anonymization/br/*.redacted.html

# Email addresses
rg -ni '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}' \
  results/anonymization/br/*.redacted.html

# Long access keys, barcode payloads, or account-like digit runs
rg -n '[0-9][0-9 .-]{10,}[0-9]' results/anonymization/br/*.redacted.html
```

Also search explicitly for every name, identifier, address fragment, and distinctive
phrase visible in the source. HTML escaping and line breaks can defeat simple
regular expressions, so combine source searches with manual inspection.

The HTML has no JavaScript and no hidden source image, but inspect for unintended
text in the `<title>`, comments, attributes, and page body anyway. The provided
sanitizer discards comments and uses a generic title.

## 8. Handle missed or poor transcriptions

If the model omits non-sensitive content or badly damages the table structure:

1. rerun with `gemma4-e4b` and `--image-tokens 1120`;
2. increase `--max-new-tokens` if output ends abruptly;
3. select a less dense authorized example if the report's selection protocol allows
   it; or
4. correct the HTML manually while looking at the source.

Never paste a complete OCR dump into an external service unless that service is
explicitly approved for the data. This script uses the locally configured model
checkpoint.

A missing sensitive value is not itself a privacy problem in the reconstructed
HTML, but it is evidence that the reconstruction is incomplete. The report should
not claim that the HTML is an exact visual reproduction.

## 9. Create the report artifact

Keep a final, reviewed copy separate from the private draft:

```bash
cp results/anonymization/br/br_test_00000.redacted.html \
   results/anonymization/br/br_test_00000.final.html
cp results/anonymization/br/br_test_00001.redacted.html \
   results/anonymization/br/br_test_00001.final.html
```

Make all manual edits in the `*.final.html` files. Open each final file in a browser
and use **Print → Save to PDF** if the report tool requires PDF. Disable browser
headers and footers so a local path and timestamp are not added. If a raster figure
is needed, render or screenshot the final HTML—not the original invoice.

After exporting, inspect the exported PDF or image again. For PDF, select/copy its
text or use an approved local text-extraction tool to verify that removed values are
not present in the PDF text layer. Check the filename and PDF metadata as well.

## 10. Record reproducibility information without retaining PII

For each figure, record privately:

- dataset version and split;
- Parquet row position and `source_index` if policy permits;
- Gemma model/checkpoint version;
- script commit;
- image-token and generation-token settings;
- date of human review; and
- the reviewer's approval.

Do not put source paths, annotation dictionaries, raw model responses, or sensitive
review drafts in the report repository. If provenance identifiers themselves are
sensitive, store them in the approved restricted system rather than in the report.

## 11. Clean up

The `*.review.html` files contain sensitive transcribed values. Once review and any
required audit are complete, delete them according to the data-retention policy:

```bash
rm results/anonymization/br/*.review.html
```

Deleting a file from a Git working tree does not remove it from Git history. Keep the
output under the ignored `results/` directory and verify before committing:

```bash
git status --short
```

If a sensitive artifact was accidentally committed, stop sharing the repository and
follow the organization's incident-response process; a normal follow-up deletion
commit is not sufficient.

## Final publication checklist

- [ ] Exactly two intended BR rows were used.
- [ ] The source images and review drafts stayed in approved storage.
- [ ] Every visible identifier and identifying combination was considered.
- [ ] Both redacted files were compared with their source images.
- [ ] The HTML source—not only the rendered page—was checked.
- [ ] Common identifier patterns and known source strings were searched.
- [ ] QR codes, barcodes, access keys, signatures, and metadata were addressed.
- [ ] The exported report artifact was checked again.
- [ ] The caption calls the result an anonymized Gemma HTML reconstruction.
- [ ] A human reviewer approved the final two artifacts.
- [ ] Sensitive review drafts were retained or deleted according to policy.
