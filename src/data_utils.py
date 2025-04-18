"""
Dataset management for cocoa disease detection
"""

from functools import partial

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor


def format_image_annotations_as_coco(
    image_id: str, labels, bboxes, image_width, image_height
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        labels (List[int]): list of categories/class labels
        bboxes (List[Tuple[float]]): list of bounding boxes in YOLO format
            ([center_x, center_y, width, height] in normalized coordinates)
        image_width (int): width of the image in pixels
        image_height (int): height of the image in pixels

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for i, (label, bbox) in enumerate(zip(labels, bboxes)):
        # Convert from normalized YOLO format to absolute COCO format
        x_center_norm, y_center_norm, width_norm, height_norm = bbox

        # Denormalize to get pixel values
        width_px = width_norm * image_width
        height_px = height_norm * image_height
        x_center_px = x_center_norm * image_width
        y_center_px = y_center_norm * image_height

        # Convert center coordinates to top-left coordinates (COCO format)
        x_px = x_center_px - width_px / 2
        y_px = y_center_px - height_px / 2

        # Create annotation in COCO format with absolute pixel values
        annotation = {
            "id": i,
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [x_px, y_px, width_px, height_px],
            "area": width_px * height_px,
            "iscrowd": 0,
        }

        annotations.append(annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples,
    transform,
    image_processor,
):
    """Apply augmentations and format annotations in COCO format for object detection

    Args:
        examples: Dataset examples with images and annotations
        transform: Albumentations transform to apply
        image_processor: Hugging Face image processor

    Returns:
        Processed batch with pixel values and labels
    """
    # transform -> format as coco -> image processor
    images = []
    batch_annotations = []

    for image_id, image, annotations in zip(
        examples["id"], examples["image"], examples["annotations"]
    ):
        image = np.array(image.convert("RGB"))

        # Prepare bounding boxes in YOLO format
        bboxes = []
        labels = []
        for i in range(len(annotations["label"])):
            # Create a bbox in YOLO format [x_center, y_center, width, height]
            bbox = [
                annotations["x_center"][i],
                annotations["y_center"][i],
                annotations["width"][i],
                annotations["height"][i],
            ]
            bboxes.append(bbox)
            labels.append(annotations["label"][i])

        # Apply augmentations
        output = transform(image=image, bboxes=bboxes, label=labels)
        images.append(output["image"])

        transformed_height, transformed_width = output["image"].shape[:2]
        formatted_annotations = format_image_annotations_as_coco(
            image_id,
            output["label"],
            output["bboxes"],
            image_width=transformed_width,
            image_height=transformed_height,
        )
        batch_annotations.append(formatted_annotations)

    # Apply the image processor transformations
    result = image_processor(
        images=images, annotations=batch_annotations, return_tensors="pt"
    )

    # Help garbage collection
    del images, batch_annotations

    return result


class CocoaDatasetManager:
    """Class to manage the cocoa disease detection dataset."""

    def __init__(self, base_model: str, dataset_repo: str):
        self.base_model = base_model
        self.dataset_repo = dataset_repo
        self.image_processor = None
        self.train_dataset = None
        self.val_dataset = None
        self._initialize_datasets()

    def _initialize_datasets(self):
        """Initialize datasets and processors once."""
        print("Loading dataset and initializing image processor...")

        # Load dataset once
        dataset = load_dataset("julianz1/cocoa-disease-detection")

        # Initialize image processor once
        self.image_processor = AutoImageProcessor.from_pretrained(self.base_model)

        val_transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["label"],
                clip=True,
                min_area=1,
                min_width=1,
                min_height=1,
            ),
        )
        train_transform = val_transform

        # Create transform functions
        train_transform_batch = partial(
            augment_and_transform_batch,
            transform=train_transform,
            image_processor=self.image_processor,
        )

        val_transform_batch = partial(
            augment_and_transform_batch,
            transform=val_transform,
            image_processor=self.image_processor,
        )

        # type: ignore
        self.train_dataset = dataset["train"].with_transform(train_transform_batch)  # type: ignore
        # type: ignore
        self.val_dataset = dataset["validation"].with_transform(val_transform_batch)  # type: ignore

    def get_train_dataset(self):
        """Get the training dataset."""
        return self.train_dataset

    def get_val_dataset(self):
        """Get the validation dataset."""
        return self.val_dataset

    def collate_fn(self, batch):
        """Collate function for data loaders."""
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data
