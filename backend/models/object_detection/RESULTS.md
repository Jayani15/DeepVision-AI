# Object Detection — Phase 1 Results

## Dataset
- Dataset: PASCAL VOC 2007 + 2012
- Task: Object detection
- Classes: 20
- Training images: 16,551
- Validation images: 4,952
- Annotation format: YOLO bounding boxes

## Model
- Model: YOLOv8 Nano (`yolov8n.pt`)
- Method: Transfer learning
- Device: Apple M3 using MPS
- Image size: 640 × 640
- Batch size: 4
- Total training: 20 epochs

## Final Metrics
- Precision: approximately 0.77
- Recall: approximately 0.74
- mAP@50: approximately 0.80
- mAP@50–95: approximately 0.58

## Saved Model
`models/object_detection/yolov8n_voc_phase1_best.pt`

## Outputs
- Training plots: `runs/detect/voc_phase1_final/results.png`
- Prediction images: `results/object_detection/predictions/`

