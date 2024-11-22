from ultralytics import YOLO

from detection import detect_food
from llm import generate_response, load_language_model


def main():
    # Load the model
    detection_model = YOLO("weights/yolov8m.pt")

    language_model, tokenizer = load_language_model("meta-llama/Llama-2-7b-chat-hf")

    food_items = detect_food(detection_model, "hotdog.jpg")

    response = generate_response(language_model, tokenizer, food_items)

    print(response)


if __name__ == "__main__":
    main()