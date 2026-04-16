"""
Author: Yuvraj Fagotra
Date: 16/04/2026

ChatGPT was used to fix syntax errors and generate boilerplate where needed. Any generated code was used only as reference and edited, refactored, and structured manually unless stated otherwise. 
"""

from ultralytics import YOLO
from pathlib import Path

from config import *

BASE = "base"
BEST = "best"

def get_model_name_from_user():
    user_input = input("Enter YOLO model name: ")
    return user_input if user_input.endswith(".pt") else user_input + ".pt"

def get_model_variant_from_user():
    while True:
        user_input = input("Please pick an option (1) Base weights or (2) Current best weights: ").strip()

        if user_input == "1":
            return BASE
            break
        elif user_input == "2":
            return BEST
            break
        else:
            print("Invalid input. Please enter 1 or 2.")
    
def load_model(model_name, variant=BEST):
    MODELS_DIR.mkdir(exist_ok=True)

    local_path = None
    if variant == BEST:
        path = BEST_WEIGHTS_DIR / model_name 
        if path.exists():
            print(f"[INFO] Best {model_name} weights found")
            local_path = path
        else:
            print(f"[INFO] Could not find best {model_name} weights. Using base weights instead")
            local_path = BASE_WEIGHTS_DIR / model_name
    else:
        local_path = BASE_WEIGHTS_DIR / model_name

    try:
        print(f"[INFO] Loading {model_name}...")
        return YOLO(str(local_path)) 
    except Exception as e:
        print(f"ERR: Failed to load model {model_name}")
        print(e)
    
def load_model_from_user_input():
    model = None
    while model is None:
        model_name = get_model_name_from_user()
        model_variant = get_model_variant_from_user()
        model = load_model(model_name, model_variant)
    return model

if __name__ == "__main__":
    load_model_from_user_input()