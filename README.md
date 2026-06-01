# PC Screw Detection Using YOLO
Using transfer learning to train existing YOLO models for small object detection task, specifically detecting pc screws. 

# How to run
This package provides sripts and utilities to use YOLO models for inference, train YOLO models, and apply simple crop augmentation to your dataset.

## Inference
You can follow these steps to use any YOLO model available through the Ultralytics module and do not need to install or download anything. You can also refer to [Demo Dataset](#demo-dataset) for access to a custom yolov8n model pretrained using a custom pc screw dataset.
- Run the following command to start ```python src/test.py```.
- You  will be asked for the name of the YOLO model you would like to use. For supported models refer to the [official documentation]( https://docs.ultralytics.com/models#featured-models).
- Next you get to pick between the base or best weights. Base refers to the base weights for the given model provide by Ultralytics (usually pre-trained on COCO). Best refers to the weights from the latest training run for that model. The "best" weights are only available if you have trained a model using the [training script](#training). If no best weights are found, it fallbacks to using base weights.

  <img width="695" height="117" alt="image" src="https://github.com/user-attachments/assets/2b8df7a0-8211-4c4e-98f5-f6ac69a9bc2d" />

  
## Training
Training requires a dataset. You can either provide your own or refer to [Demo Dataset](#demo-dataset) to use the demo dataset. 

You can follow these steps to use any YOLO model available through the Ultralytics module and do not need to install or download anything.
```
python src/train.py
```

# Demo Dataset
The dataset branch contains a trained model and a sample of the training dataset. 
- If you do not have the `dataset.zip` file, you can download it from the dataset branch or switch to the dataset branch.
  ```
  git checkout dataset
  ```
- 


## Setup
- This branch contains a `dataset.zip` file. Extract this file into the project root. Ensure that the project now has src, models, datasets directories.
- 
