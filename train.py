import torch
from ultralytics import YOLO

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = YOLO("yolov8m")

    # Train the model
    model.train(
        data="configs/oid.yaml",
        epochs=10,
        batch=32,
        device=device,
        project="NutritionTracker",
        name="oid",
        seed=42
    )
