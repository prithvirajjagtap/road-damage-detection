import os
import shutil
import random

BASE_DIR = r"C:\Users\aj595\OneDrive\Desktop\Road_Damage_Detection"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

IMAGE_SOURCE = os.path.join(DATASET_DIR, "data", "images")
LABEL_SOURCE = os.path.join(DATASET_DIR, "data", "labels-YOLO")

IMAGE_TRAIN = os.path.join(DATASET_DIR, "images", "train")
IMAGE_VAL = os.path.join(DATASET_DIR, "images", "val")

LABEL_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
LABEL_VAL = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(IMAGE_TRAIN, exist_ok=True)
os.makedirs(IMAGE_VAL, exist_ok=True)
os.makedirs(LABEL_TRAIN, exist_ok=True)
os.makedirs(LABEL_VAL, exist_ok=True)

images = [
    f for f in os.listdir(IMAGE_SOURCE)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.seed(42)
random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

def copy_files(image_list, image_destination, label_destination):
    for image in image_list:
        image_path = os.path.join(IMAGE_SOURCE, image)
        shutil.copy2(image_path, image_destination)

        label_name = os.path.splitext(image)[0] + ".txt"
        label_path = os.path.join(LABEL_SOURCE, label_name)

        if os.path.exists(label_path):
            shutil.copy2(label_path, label_destination)

copy_files(train_images, IMAGE_TRAIN, LABEL_TRAIN)
copy_files(val_images, IMAGE_VAL, LABEL_VAL)

print("Dataset preparation completed!")
print("Training images:", len(train_images))
print("Validation images:", len(val_images))