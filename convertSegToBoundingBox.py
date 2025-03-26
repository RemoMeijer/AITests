import os
import json
import cv2
from glob import glob


def yolo_to_coco_bbox(segmentation, img_width, img_height):
    """Converts YOLO segmentation (normalized) to COCO bounding box format (absolute values)."""
    abs_coords = [(float(x) * img_width, float(y) * img_height) for x, y in segmentation]
    x_coords, y_coords = zip(*abs_coords)
    x_min, y_min, x_max, y_max = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    width, height = x_max - x_min, y_max - y_min
    return [x_min, y_min, width, height]  # COCO format (x, y, width, height)


def load_yolo_annotations(txt_file, img_width, img_height):
    """Parses YOLO segmentation annotations and converts them to COCO format."""
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0]) + 1  # Ensure category IDs start from 1
        segmentation = [(float(data[i]), float(data[i + 1])) for i in range(1, len(data), 2)]

        bbox = yolo_to_coco_bbox(segmentation, img_width, img_height)

        # Convert segmentation to COCO format (flatten list)
        abs_segmentation = [coord for xy in segmentation for coord in (xy[0] * img_width, xy[1] * img_height)]

        annotations.append({
            "id": None,  # Will be added later
            "image_id": None,  # Will be added later
            "category_id": class_id,
            "bbox": bbox,
            "segmentation": [abs_segmentation],  # Single list inside a list
            "area": bbox[2] * bbox[3],  # width * height
            "iscrowd": 0
        })
    return annotations


def process_dataset(yolo_annotations_path, images_path, output_json):
    """Processes YOLO segmentation annotations and saves them in COCO format."""
    dataset = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    annotation_id = 1

    for txt_file in glob(os.path.join(yolo_annotations_path, "*.txt")):
        image_file = os.path.join(images_path, os.path.basename(txt_file).replace(".txt", ".jpg"))
        if not os.path.exists(image_file):
            continue

        img = cv2.imread(image_file)
        img_height, img_width = img.shape[:2]

        dataset["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_file),
            "width": img_width,
            "height": img_height
        })

        annotations = load_yolo_annotations(txt_file, img_width, img_height)
        for ann in annotations:
            ann["id"] = annotation_id
            ann["image_id"] = image_id
            dataset["annotations"].append(ann)
            annotation_id += 1

        image_id += 1

    # Define categories
    dataset["categories"] = [
        {"id": 1, "name": "weed"},
        {"id": 2, "name": "crop"}
    ]

    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved COCO-style annotations to {output_json}")


# Example usage
yolo_annotations_path = "yoloFormats"
images_path = "/home/remo/Afstudeerproject/AgronomischePerformanceMeting/AnnotationAndTraining/Annotation/annotated_images"
output_json = "output_annotations.json"

process_dataset(yolo_annotations_path, images_path, output_json)
