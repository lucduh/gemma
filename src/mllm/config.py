from dataclasses import dataclass
from pathlib import Path

from mllm.inference import Gemma3, Gemma4, InternVL, TestModel
from mllm.prompts import (
    BR_PROMPT,
    KARAPASS_DEATH_PROMPT,
    KARAPASS_ID_PROMPT,
    TEST_PROMPT,
)


@dataclass(frozen=True)
class DatasetConfig:
    directory: Path
    prompt_template: str


DATA_ROOT = Path("/domino/datasets/local/MLLM/data")

MODELS = {
    "gemma3": (
        Gemma3,
        Path("/domino/datasets/ModelHub-model-huggingface-google/gemma-3-4b-it/main"),
    ),
    "gemma4-e2b": (
        Gemma4,
        Path("/domino/datasets/ModelHub-model-huggingface-google/gemma-4-E2B-it/main"),
    ),
    "gemma4-e4b": (
        Gemma4,
        Path("/domino/datasets/ModelHub-model-huggingface-google/gemma-4-E4B-it/main"),
    ),
    "internvl": (InternVL, "OpenGVLab/InternVL3-8B-hf"),
    "test": (TestModel, "test"),
}

DATASETS = {
    "BR": DatasetConfig(DATA_ROOT / "BR", BR_PROMPT),
    "KARAPASS_DEATH": DatasetConfig(
        DATA_ROOT / "KARAPASS_DEATH", KARAPASS_DEATH_PROMPT
    ),
    "KARAPASS_ID": DatasetConfig(DATA_ROOT / "KARAPASS_ID", KARAPASS_ID_PROMPT),
    "TEST": DatasetConfig(Path("test_data"), TEST_PROMPT),
}

RESULTS_DIR = Path("results")
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
