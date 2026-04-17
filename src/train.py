"""
Author: Yuvraj Fagotra
Date: 16/04/2026

ChatGPT was used to fix syntax errors and generate boilerplate where needed. Any generated code was used only as reference and edited, refactored, and structured manually unless stated otherwise. 
"""

import shutil
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_gpu_info

from config import *
import model_loader as ml

def copy_best_weights(model_name):
    dst_path = BEST_WEIGHTS_DIR / model_name
    BEST_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS_DIR / model_name / "weights" / "best.pt", dst=dst_path)

def train_model():
    print(get_gpu_info(0))

    model = None
    while model is None:
        model_name = ml.get_model_name_from_user()
        model_variant = ml.get_model_variant_from_user()
        model = ml.load_model(model_name, model_variant)

    results = model.train(data=DATA_YAML, exist_ok=True, project=RUNS_DIR, name=model_name)
    copy_best_weights(model_name)

if __name__ == "__main__":
    train_model()