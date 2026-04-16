"""
Author: Yuvraj Fagotra
Date: 16/04/2026

ChatGPT was used to fix syntax errors and generate boilerplate where needed. Any generated code was used only as reference and edited, refactored, and structured manually unless stated otherwise. 
"""

from pathlib import Path
from ultralytics import YOLO

from config import *
import model_loader as ml

def train_model():
    model = None
    while model is None:
        model_name = ml.get_model_name_from_user()
        model_variant = ml.get_model_variant_from_user()
        model = ml.load_model(model_name, model_variant)

    results = model.train(data=DATA_YAML, exist_ok=True, project=RUNS_DIR, name=model_variant)

if __name__ == "__main__":
    train_model()