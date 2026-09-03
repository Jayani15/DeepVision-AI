# YOLO Object Detection

## Overview
This module performs object detection using YOLOv8 Nano and transfer learning.

## Dataset
PASCAL VOC 2007 + 2012

- 20 object classes
- 16,551 training images
- 4,952 validation images
- YOLO bounding-box annotations

The dataset is not included in this repository. Ultralytics downloads and converts it automatically when `VOC.yaml` is used.

## Setup

```bash
pip install -r requirements.txt

```

## Training

The submitted Phase 1 model is already included in:

```text
models/yolov8n_voc_phase1_best.pt
```

Run this only to train a new model. It reuses the VOC dataset if already downloaded:

```bash
yolo detect train model=yolov8n.pt data=VOC.yaml epochs=20 imgsz=640 batch=4 device=mps workers=0
```

## Inference

```bash
python predict.py /path/to/image.jpg
```

Prediction images are saved in `results/predictions/`.

## Final Results

| Metric | Result |
|---|---:|
| Precision | ~0.77 |
| Recall | ~0.74 |
| mAP@50 | ~0.80 |
| mAP@50–95 | ~0.58 |

## Contributor

Nikshep
