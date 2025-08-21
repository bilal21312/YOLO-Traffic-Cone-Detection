## YOLO Finetuning Report

### 1. Installation and Training Commands

Below are the exact steps/commands I used to set up and train the YOLOv8 model:

```bash
# Create and activate virtual environment
python -m venv yoloenv
yoloenv\Scripts\activate

# Install Libraries
pip install -r requirements.txt

# Train YOLOv8 on custom dataset
python train_yolo.py
```

### 2. Role of data.yaml

The `data.yaml` file tells YOLO where to find the dataset and what classes exist. It typically looks like:

```yaml
train: Traffic Cones/train/images
val: Traffic Cones/valid/images
nc: 1
names: ["cone"]
```

- **train**: path to training images
- **val**: path to validation images
- **nc**: number of classes (1 = cone)
- **names**: list of class labels

This file is crucial because YOLO uses it to link images with labels and understand what categories to predict.

### 3. Training Results

- **results.png**: Shows training curves for losses (box, cls, dfl) and performance metrics (precision, recall, mAP).
  - Loss curves should steadily decrease.
  - mAP/precision/recall curves should rise and stabilize.
- **confusion_matrix.png**: Displays how often predictions matched ground truth.
  - Diagonal values = correct detections.
  - Off-diagonal values = misclassifications (not a problem here since we only detect cones).
  - Normalized confusion matrix shows percentages instead of raw counts.

### 4. Inference Output

I created an `examples/` folder with test images (`cone1.jpg–cone4.jpg`).

Using the inference script:

```bash
python infer_yolo.py
```

The model loaded `runs/detect/train/weights/best.pt`. Each image was processed, and results saved to:

```
runs/detect/inference/
```

### 5. Transfer Learning in Action

This project demonstrates transfer learning:

- Instead of training YOLO from scratch, we started from `yolov8n.pt` pretrained on COCO (a massive dataset).
- Only a small dataset of traffic cones was needed.
- The pretrained features (edges, shapes, textures) transferred well, so the model quickly adapted to detecting cones with high accuracy.
- This saved both time and computational resources compared to full training from zero.

### 6. Challenges & Solutions

- **CUDA Setup**: Initially, PyTorch installed the CPU version. Fixed by manually installing with `--index-url` for CUDA 12.1.
- **Multiprocessing Error (Windows)**: Resolved by wrapping training code inside:

```python
if __name__ == "__main__":
    train()
```

- **Extra YOLO Weights Download**: YOLO auto-downloads latest versions (`yolo11n.pt`) if not specified. Resolved by explicitly using `yolov8n.pt`.
- **Inference Path**: Needed to adjust script to point to `Traffic Cones/examples/`.


