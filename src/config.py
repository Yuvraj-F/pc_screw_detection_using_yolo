"""
A place to hold project related global values
"""

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

# Optional class names
CLASS_NAMES = {
    0: "2.5 SSD",
    1: "3.5 HDD",
    2: "GPU stand off bolt",
    3: "MB stand off bolt",
    4: "MB",
    5: "PSU",
    6: "bracket",
    7: "fan",
    8: "long fan above PSU",
    9: "short fan above PSU",
    10: "spare",
    11: "stand off bolt tool"
}

# Define colors per class
CLASS_COLORS = {
    0: "#052AFF",
    1: "#0CDBEB",
    2: "#F3F3F3",
    3: "#01DFB8",
    4: "#111F68",
    5: "#FF6FE0",
    6: "#FF434F",
    7: "#CCEC02",
    8: "#00F244",
    9: "#BE00FF",
    10: "#00B4FF",
    11: "#DC00BA"
}