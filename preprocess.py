
import json
import os

from tqdm import tqdm

TRAIN_ROOT = "data/raw_data/public_training_set_release_2.0"
VAL_ROOT = "data/raw_data/public_validation_set_2.0"
OUTPUT_ROOT = "data/processed_data"

with open(os.path.join(TRAIN_ROOT, "annotations.json"), 'r') as f:
    data = json.load(f)

# Map category id to class id
category_id_to_class_id = {category["id"]: i for i, category in enumerate(data["categories"])}

class_id_to_name = {category_id_to_class_id[category["id"]]: category["name_readable"] for category in data["categories"]}

def process_annotations(input_path: str, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # F
    for image in tqdm(data['images'], desc=f"Processing {input_path}", unit="image"):
        image_id = image['id']
        width, height = image["width"], image['height']
        file_name = os.path.splitext(image['file_name'])[0]
        
        with open(os.path.join(output_path, f"{file_name}.txt"), 'w') as f:
            annotations = [ann for ann in data["annotations"] if ann['image_id'] == image_id]
            
            for annotation in annotations:
                class_id = category_id_to_class_id[annotation['category_id']]
                x, y, w, h = annotation['bbox']

                # Convert to x_c, y_c, w, h
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w = w / width
                h = h / height
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

process_annotations(os.path.join(TRAIN_ROOT, "annotations.json"), os.path.join(OUTPUT_ROOT, "labels/train"))
process_annotations(os.path.join(VAL_ROOT, "annotations.json"), os.path.join(OUTPUT_ROOT, "labels/val"))

def process_images(input_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    # Move each image in the source directory
    for input_file_name in tqdm(os.listdir(input_path), desc=f"Processing {input_path}", unit="image"):
        # Skip non-jpg files
        if not input_file_name.endswith(".jpg"):
            continue

        input_file_path = os.path.join(input_path, input_file_name)
        output_file_path = os.path.join(output_path, input_file_name)

        # Move the image
        os.rename(input_file_path, output_file_path)

process_images(os.path.join(TRAIN_ROOT, "images"), os.path.join(OUTPUT_ROOT, "images/train"))
process_images(os.path.join(VAL_ROOT, "images"), os.path.join(OUTPUT_ROOT, "images/val"))

with open("configs/foodbenchmark.yaml", 'w') as f:
    f.write(f"""train: {OUTPUT_ROOT}/images/train
val: {OUTPUT_ROOT}/images/val
names: {OUTPUT_ROOT}/labels/classes.txt
""")
    
with open(os.path.join(OUTPUT_ROOT, "labels/classes.txt"), 'w') as f:
    for class_id, class_name in class_id_to_name.items():
        f.write(f"{class_name}\n")