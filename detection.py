import torch
from PIL import Image
from ultralytics import YOLO


def get_food_items(model: YOLO, image: str) -> str:
    # Load the image
    img = Image.open(image)

    # Perform object detection
    results = model(img)

    # Extract the food items
    food_items = results.names

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
