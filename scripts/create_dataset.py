import os
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from PIL import Image

DATASETS_DIR = Path("dataset")
TRAIN_IMAGES_DIR = DATASETS_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASETS_DIR / "labels" / "train"
TEST_IMAGES_DIR = DATASETS_DIR / "images" / "test"


def load_image_annotations(image_name: str) -> list[dict]:
    img_ext = image_name.split(".")[-1]
    label_path = TRAIN_LABELS_DIR / (image_name.replace(f".{img_ext}", ".txt"))
    if not label_path.exists():
        print(f"No annotations found for {image_name}.")
        return []

    with open(label_path, "r") as f:
        lines = f.readlines()

    annotations = []
    for j, line in enumerate(lines):
        if line.strip():  # Skip empty lines
            class_id, x_center, y_center, bbox_width, bbox_height = map(
                float, line.strip().split()
            )

            annotation = {
                "id": j,
                "label": int(class_id),
                "x_center": x_center,
                "y_center": y_center,
                "width": bbox_width,
                "height": bbox_height,
            }
            annotations.append(annotation)

    return annotations


def prepare_features():
    return Features(
        {
            "id": Value("int32"),
            "image_id": Value("string"),
            "image_name": Value("string"),
            "image": HFImage(),
            "width": Value("int32"),
            "height": Value("int32"),
            "annotations": Sequence(
                {
                    "label": Value("int32"),
                    "x_center": Value("float32"),
                    "y_center": Value("float32"),
                    "width": Value("float32"),
                    "height": Value("float32"),
                },
            ),
        }
    )


def load_data(path_to_split: Path) -> list[dict]:
    data_list = []

    for i, image_name in enumerate(os.listdir(path_to_split)):
        image_path = path_to_split / image_name
        image = Image.open(image_path)
        img_width, img_height = image.size

        data_item = {
            "id": i,
            "image_id": image_name.split(".")[0],
            "image": str(image_path),
            "image_name": image_name,
            "width": img_width,
            "height": img_height,
        }

        annotations = load_image_annotations(image_name)
        if not annotations:
            print(f"No annotations found for {image_name}.")
            continue

        data_item["annotations"] = annotations
        data_list.append(data_item)

    return data_list


def main():
    train_val_data = load_data(TRAIN_IMAGES_DIR)
    train_val_dataset = Dataset.from_list(train_val_data, features=prepare_features())

    split_dataset = train_val_dataset.train_test_split(test_size=0.2, seed=42)
    split_dataset["validation"] = split_dataset.pop("test")

    split_dataset.save_to_disk(DATASETS_DIR / "cocoa_coco_dataset")


if __name__ == "__main__":
    main()
