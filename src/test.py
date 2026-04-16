"""
Author: Yuvraj Fagotra
Date: 16/04/2026

ChatGPT was used to fix syntax errors and generate boilerplate where needed. Any generated code was used only as reference and edited, refactored, and structured manually unless stated otherwise. 
"""

import cv2
from matplotlib import scale
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

from model_loader import load_model_from_user_input as load_model

# Directly from chatGPT
def select_file():
    root = tk.Tk()
    root.withdraw()  # hides the empty main window

    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    return file_path

def infer(image_path):
    if not image_path:
        print("No file selected. Exiting.")
    else:
        model = load_model()
        result = model(image_path)

        # Use YOLO plot function to annotate the input image
        annotated_image = result[0].plot()

        scale = 0.5
        h, w = annotated_image.shape[:2]
        resized = cv2.resize(annotated_image, (int(w * scale), int(h * scale)))

        cv2.imshow("result", resized)
        cv2.waitKey(0)

if __name__ == "__main__":
    infer(select_file())
