import glob
import os
from ultralytics import YOLO

def run_inference():
    model = YOLO("runs/detect/train/weights/best.pt")
    example_dir = "Traffic Cones/examples"
    image_paths = glob.glob(os.path.join(example_dir, "*.jpg"))

    for img_path in image_paths:
        print(f"Running inference on: {img_path}")
        results = model(img_path, save=True, project="runs/detect", name="inference")

    print("\nInference completed.")

if __name__ == "__main__":
    run_inference()
