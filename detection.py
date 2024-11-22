import torch
from PIL import Image
from ultralytics import YOLO

def detect_food(model: YOLO, image: str) -> str:
    # Load the image
    img = Image.open(image)

    # Perform object detection
    result = model(img)[0]

    class_names = [result.names[id] for id in result.boxes.cls.tolist()]

    food_items = ", ".join(class_names)

    return food_items

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
