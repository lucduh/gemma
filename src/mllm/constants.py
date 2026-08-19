from pathlib import Path

MODELS = {
    "gemma3": "google/gemma-3-4b-it",
    "gemma4": "google/gemma-4-E2B-it",
}

FIELDS = [
    "cpf_cnpj_prestador",
    "cep_prestador",
    "cpf_cnpj_tomador",
    "data_emissao",
    "destinataire",
    "numero_da_nota",
    "servico_prestado",
    "valor_da_nota",
    "calculo_do_imposto",
]

PROMPT = """Extract structured information from this Brazilian service invoice (NFS-e).

Return exactly one compact JSON object and nothing else. Include only clearly present fields, use only the keys below, and make every value a JSON string:

- "numero_da_nota": value labeled "Número da nota", "Nº", "N°", or "Nota" in the invoice header. Remove the label and prefix.
- "data_emissao": value labeled "Data de emissão" or "Emitida em". Return only the date as DD/MM/YYYY, without the time.
- "cpf_cnpj_prestador": CPF/CNPJ explicitly identified in the PRESTADOR or EMITENTE section.
- "cep_prestador": CEP explicitly identified in the PRESTADOR or EMITENTE section.
- "cpf_cnpj_tomador": CPF/CNPJ explicitly identified in the TOMADOR section.
- "destinataire": customer name or company name in the TOMADOR section.
- "servico_prestado": actual service description written under "Discriminação dos Serviços" or an equivalent description section. Do not return the section heading or a company name.
- "valor_da_nota": gross invoice total, preferably labeled "Valor dos serviços", "Valor bruto", or "Valor total". Do not use the net value, tax base, deductions, or taxes.
- "calculo_do_imposto": amount explicitly labeled "Retenções Federais". Do not use a percentage, tax rate, ISS, tax base, or net value.

Use the layout to keep PRESTADOR/EMITENTE values separate from TOMADOR values. Take CPF/CNPJ and CEP only from their explicitly labeled fields; do not substitute telephone numbers, registrations, or nearby numbers. Preserve spelling and numeric formatting except for the required invoice-number and date normalization. Never infer or calculate a value. Omit missing fields; do not return null, Markdown, or explanations."""

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
