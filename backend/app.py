from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os

from models.autoencoder.model import DenoisingAutoencoder
from models.classification.models.cnn import CNNClassifier
from ultralytics import YOLO

app = FastAPI(title="DeepVision-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Autoencoder (Denoising) Setup
# ============================================================

AUTOENCODER_MODEL_PATH = os.path.join("models", "autoencoder", "checkpoints", "best_autoencoder.pth")

autoencoder_model = DenoisingAutoencoder().to(device)

if os.path.exists(AUTOENCODER_MODEL_PATH):
    autoencoder_model.load_state_dict(torch.load(AUTOENCODER_MODEL_PATH, map_location=device))
    autoencoder_model.eval()
    print("Successfully loaded autoencoder weights!")
else:
    print(f"Warning: Autoencoder checkpoint not found at {AUTOENCODER_MODEL_PATH}")

denoise_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def tensor_to_base64(tensor: torch.Tensor, target_size: tuple[int, int]) -> str:
    """
    Convert a model output tensor to a base64 PNG, resized back to
    the ORIGINAL uploaded image's dimensions. This keeps the noisy
    input and denoised output the same size/aspect ratio, so the
    frontend before/after slider lines up pixel-for-pixel instead
    of appearing to "zoom" between two differently-cropped images.
    """
    tensor = torch.clamp(tensor.squeeze(0).cpu(), 0.0, 1.0)
    image = transforms.ToPILImage()(tensor)
    image = image.resize(target_size, Image.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


# ============================================================
# CNN Classifier Setup
# ============================================================

CNN_MODEL_PATH = os.path.join("models", "classification", "models", "cnn_final.pth")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

cnn_model = CNNClassifier(dropout_rate=0.5).to(device)

if os.path.exists(CNN_MODEL_PATH):
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn_model.eval()
    print("Successfully loaded CNN classifier weights!")
else:
    print(f"Warning: CNN checkpoint not found at {CNN_MODEL_PATH}")

classify_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])


# ============================================================
# Object Detection (YOLOv8) Setup
# ============================================================

DETECTION_MODEL_PATH = os.path.join("models", "object_detection", "models", "yolov8n_voc_phase1_best.pt")

# YOLO's own device selector, kept separate from `device` above so we don't
# touch the existing autoencoder/classifier logic. Prefers Apple MPS (as used
# during training), falls back to CUDA, then CPU.
if torch.backends.mps.is_available():
    yolo_device = "mps"
elif torch.cuda.is_available():
    yolo_device = "cuda"
else:
    yolo_device = "cpu"

if os.path.exists(DETECTION_MODEL_PATH):
    yolo_model = YOLO(str(DETECTION_MODEL_PATH))
    print("Successfully loaded YOLOv8 object detection weights!")
else:
    yolo_model = None
    print(f"Warning: YOLO checkpoint not found at {DETECTION_MODEL_PATH}")


# ============================================================
# Routes
# ============================================================

@app.get("/")
def read_root():
    return {"message": "DeepVision-AI API is active!"}


@app.post("/api/denoise")
async def denoise_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Capture original size BEFORE resizing for the model,
        # so we can resize the output back to match afterward.
        original_size = pil_img.size  # (width, height)

        input_tensor = denoise_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = autoencoder_model(input_tensor)

        result_b64 = tensor_to_base64(output_tensor, original_size)
        return {"status": "success", "denoised_image": result_b64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = classify_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0)

        # Top-3 predictions, sorted by confidence descending
        top_probs, top_idxs = torch.topk(probabilities, k=3)

        predictions = [
            {
                "label": CLASS_NAMES[idx.item()],
                "confidence": round(prob.item() * 100, 1),
            }
            for prob, idx in zip(top_probs, top_idxs)
        ]

        return {"status": "success", "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="Object detection model not loaded")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = yolo_model.predict(
            source=pil_img,
            conf=0.25,
            device=yolo_device,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                confidence = float(box.conf[0])

                # Normalized (0-1) coordinates — independent of image size,
                # so the frontend can position boxes as simple percentages
                # regardless of the uploaded image's resolution/aspect ratio.
                x1, y1, x2, y2 = box.xyxyn[0].tolist()

                detections.append({
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        return {"status": "success", "detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)