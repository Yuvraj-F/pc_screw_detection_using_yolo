from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"
BASE_WEIGHTS_DIR = MODELS_DIR / "base"
BEST_WEIGHTS_DIR = MODELS_DIR / "best"
DATA_YAML = ROOT_DIR / "datasets" / "data.yaml"
RUNS_DIR = ROOT_DIR / "runs" 
DATASET_DIR = ROOT_DIR / "datasets"
TRAIN_DATA_DIR = DATASET_DIR / "train"
VAL_DATA_DIR = DATASET_DIR / "val"