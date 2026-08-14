from pathlib import Path

MODELS = {
    "gemma3": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-E2B-it",
}

FIELDS = [
    "E-mail",
    "cpf_cnpj_prestador",
    "cpf_cnpj_tomador",
    "data_emissao",
    "destinataire",
    "numero_da_nota",
    "servico_prestado",
    "valor_da_nota",
]

PROMPT = f"""Extract these fields from the document: {", ".join(FIELDS)}.
Return only a compact valid JSON object. Include only fields found in the document and omit absent fields.
Every value must be a JSON string kept exactly as written in the document. Do not add Markdown or explanations."""

RESULTS_DIR = Path("results")
SYNTHETIC_IMAGE_SIZE = (1280, 960)  # (height, width)
SYNTHETIC_MAX_NEW_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 3
