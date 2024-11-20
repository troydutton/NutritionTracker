from ultralytics import YOLO

from detection import get_food_items
from llm import generate_response, load_language_model


def main():
    # Load the model
    detection_model = YOLO("pretrained-weights")

    language_model, tokenizer = load_language_model("meta-llama/Llama-2-7b-chat-hf")

    food_items = get_food_items(detection_model, "image.jpg") 

    # TODO: Convert food_items into a prompt
    prompt = "Caesar Salad, Grilled Chicken, Hot Dog, Candy."

    response = generate_response(language_model, tokenizer, prompt)

    print(response)


if __name__ == "__main__":
    main()