from pathlib import Path

import cv2
from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset("VOC.yaml")
dataset_path = Path(dataset["path"])

image_path = sorted((dataset_path / "images" / "test2007").glob("*.jpg"))[0]
label_path = dataset_path / "labels" / "test2007" / f"{image_path.stem}.txt"

image = cv2.imread(str(image_path))
height, width = image.shape[:2]

with open(label_path) as file:
    for line in file:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        class_name = dataset["names"][int(class_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

output_path = Path("results/object_detection/annotations/voc_annotation_check.jpg")
cv2.imwrite(str(output_path), image)

print(f"Saved annotation preview to: {output_path}")
