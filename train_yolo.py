from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")  
    model.train(
        data="Traffic Cones/data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        optimizer="AdamW",
        patience=20
    )

if __name__ == "__main__":
    train()