from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "models" / "yolov8n_voc_phase1_best.pt"
OUTPUT_DIR = MODULE_DIR / "results" / "predictions"

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/object_detection/predict.py <image_path>")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        raise SystemExit(1)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        device="mps",
        verbose=False,
    )[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_prediction.jpg"
    cv2.imwrite(str(output_path), result.plot())

    print(f"Saved prediction to: {output_path}")

    if result.boxes is not None:
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            print(f"{class_name}: {confidence:.2%}")


if __name__ == "__main__":
    main()
