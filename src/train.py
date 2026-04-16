"""
Author: Yuvraj Fagotra
Date: 16/04/2026

ChatGPT was used to fix syntax errors and generate boilerplate where needed. Any generated code was used only as reference and edited, refactored, and structured manually unless stated otherwise. 
"""

from pathlib import Path
from ultralytics import YOLO

from model_loader import load_model_from_user_input as load_model

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
DATA_YAML = DATASET_DIR / "data.yaml"

def train_model(model):
    results = model.train(data=DATA_YAML, exist_ok=True)

if __name__ == "__main__":
    model = load_model()
    results = model.train(data="./object_detection/dataset/data.yaml", exist_ok=True)