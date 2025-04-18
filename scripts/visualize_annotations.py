import matplotlib.pyplot as plt
from PIL import ImageDraw


def visualize_annotations(image, annotations, id2label):
    """
    Visualize an image with its bounding boxes and labels.

    Args:
        image (PIL.Image): The image to visualize
        annotations (dict): Dictionary containing 'bbox' and 'category' lists
        id2label (dict): Mapping from category IDs to label names
    """
    # Create a copy of the image to draw on
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Draw each bounding box and label
    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        category = annotations["category"][i]

        # Draw rectangle
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline="red", width=2)

        # Add label text
        label = id2label[category]
        draw.text((x, y), label, fill="white")

    return image_with_boxes


def plot_images(dataset, indices, id2label):
    """
    Plot multiple images with their annotations.

    Args:
        dataset: The dataset containing images and annotations
        indices: List of indices to plot
        id2label: Mapping from category IDs to label names
    """
    n_images = len(indices)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    if n_images == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        sample = dataset[idx]
        image = sample["image"]
        annotations = sample["objects"]

        # Visualize annotations
        image_with_boxes = visualize_annotations(image, annotations, id2label)

        # Plot
        ax.imshow(image_with_boxes)
        ax.axis("off")
        ax.set_title(f"Image {idx}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from create_dataset import ID2LABEL
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir="dataset/cocoa_coco_dataset")

    # Plot some example images
    plot_images(dataset["train"], [0, 1, 2], ID2LABEL)
