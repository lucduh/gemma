from pathlib import Path

MODELS = {
    "gemma3": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-E2B-it",
}

FIELDS = [
    "cpf_cnpj_prestador",
    "cpf_cnpj_tomador",
    "data_emissao",
    "destinataire",
    "numero_da_nota",
    "servico_prestado",
    "valor_da_nota",
    "calculo_do_imposto",
]

PROMPT = f"""Extract these fields from the document: {", ".join(FIELDS)}.
Return only a compact valid JSON object. Include only fields found in the document and omit absent fields.
Every value must be a JSON string kept exactly as written in the document. Do not add Markdown or explanations."""

RESULTS_DIR = Path("results")
DATA_DIR = Path("domino/datasets/local/donut/data")
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmarks"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"

SYNTHETIC_IMAGE_SIZE = (1280, 960)  # (height, width)
SYNTHETIC_OUTPUT_TOKENS = 64
DEFAULT_GEMMA4_IMAGE_TOKENS = 280
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_RUNS = 10
DEFAULT_WARMUP = 3

# LoRA training defaults. The vision encoder remains frozen.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION = 2
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SEED = 42
