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

PROMPT = """You are extracting structured information from a Brazilian service invoice (NFS-e).

Return exactly one compact valid JSON object and nothing else. Use only the following keys, and omit any field that is not clearly present:

- "numero_da_nota": Invoice number. Look for labels such as "Número da nota", "Nº", "N°", or "Nota". Return only the identifier, without a label or prefix.
- "data_emissao": Invoice issue date. Look for labels such as "Data de emissão" or "Emitida em". Return only the date in DD/MM/YYYY format, without the time.
- "cpf_cnpj_prestador": CPF or CNPJ from the PRESTADOR/EMITENTE section. This must belong to the service provider.
- "cpf_cnpj_tomador": CPF or CNPJ from the TOMADOR section. This must belong to the service customer.
- "destinataire": Name or company name of the service customer (TOMADOR).
- "servico_prestado": Specific description of the service performed, such as "Agenciamento" or "comissão". Do not return a section heading or the provider's name.
- "valor_da_nota": Total or gross invoice value. Prefer "Valor dos serviços", "Valor bruto", or the invoice total. Do not return taxes, deductions, the tax base, or the net value.
- "calculo_do_imposto": Total federal tax withholding amount associated with the invoice. Prefer a value explicitly associated with "Retenções Federais". Do not return a tax rate, percentage, ISS value, tax base, or net value.

Use the document layout and section headings as well as the text. PRESTADOR is the service provider; TOMADOR is the service customer. Extract a CPF/CNPJ only from an explicitly identified CPF/CNPJ field, never from a telephone number, CEP, municipal registration, or nearby identifier. Return each value without its field label. Preserve the document's spelling and numeric formatting except for the required invoice-number and date normalization. Do not infer, calculate, or reconstruct missing values. Every returned value must be a JSON string. Do not return null values, Markdown, comments, explanations, or text outside the JSON."""

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
