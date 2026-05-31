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
    """
    Assumes model in the runs directory is the "best" (probably should be called last trained or latest run). 
    Takes the model's weights from the runs directory and stores them in the best weights directory. 

    These are the best weights that are used by model loader when asked to load the current best weights.
    """
    dst_path = BEST_WEIGHTS_DIR / model_name
    BEST_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS_DIR / model_name / "weights" / "best.pt", dst=dst_path)

def train_model():
    """
    Trains a given model. It expects the dataset to be setup and does not perform any validation or checks to enforce correctness.
    """
    print(get_gpu_info(0))

    model = None
    while model is None:
        model_name = ml.get_model_name_from_user()
        model_variant = ml.get_model_variant_from_user()
        model = ml.load_model(model_name, model_variant)

    results = model.train(
    data=DATA_YAML,
    project=RUNS_DIR, 
    name=model_name, 
    imgsz=640,
    epochs=100,
    batch=16,
    #classes=[0,1,2,3,4,5,6,7,8,9,10,11],
    # cls_pw=0.2,
    # hsv_h=0.5,
    # hsv_s=0.5,
    # hsv_v=0.4,
    # degrees=180,
    # translate=0.1,
    # flipud=0,
    # fliplr=0.5,
    scale=0,
    mosaic=0, #can cause occlusion which is not ideal
    )
    copy_best_weights(model_name)

if __name__ == "__main__":
    train_model()