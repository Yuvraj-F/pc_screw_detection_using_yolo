"""
This is a utility to get results data from a specific epoch of a training run. Can be used to get for example the loss ,recall, pecision etc. of the 
last epoch since the results graph do not contain those exact values.
"""

import csv
from config import *
from model_loader import get_model_name_from_user

def get_user_input():
    """
    Asks for model name and epoch to fetch from the user.
    It only returns data from models in the runs directory.
    """
    model = get_model_name_from_user()
    model_path = RUNS_DIR / model / "results.csv"
    epoch = int(input("Enter the epoch for which you would like to see the results: "))
    return model_path, epoch

def print_epoch_metrics(file_path, target_epoch):
    """
    Loads data from the model training run file path and prints the results at target_epoch 
    """
    print(f"Loading epoch {target_epoch} metrics from: {file_path}")
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if int(row["epoch"]) == target_epoch:
                for k, v in row.items():
                    if k != "epoch":
                        print(f"{k}: {v}")
                return
    
    print("Epoch not found")

if __name__ == "__main__":
    print_epoch_metrics(*get_user_input())