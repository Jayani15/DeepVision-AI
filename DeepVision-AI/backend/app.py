from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import sys
import asyncio
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

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
# Neural Style Transfer Setup
# ============================================================
# Style transfer optimizes the output image from scratch on every request
# (no trained checkpoint), so it is run through the existing
# style_transfer.py script in a separate process. This keeps its VGG19 /
# optimizer memory isolated from the other models, and the script stays
# usable from the command line exactly as before.

STYLE_SCRIPT_PATH = (
    Path(__file__).resolve().parent / "models" / "style_transfer" / "style_transfer.py"
)
STYLE_STEPS = 300  # TESTING value (fast). Set back to 300 for full-quality results.
STYLE_SIZE = 384
STYLE_TIMEOUT_SECONDS = 900  # 15 min safety net for slow CPU-only machines

if STYLE_SCRIPT_PATH.exists():
    print("Style transfer script found!")
else:
    print(f"Warning: Style transfer script not found at {STYLE_SCRIPT_PATH}")


def _run_style_transfer(content_path: Path, style_path: Path, output_path: Path):
    """
    Blocking helper — always call via asyncio.to_thread.

    Streams the script's output line by line into THIS terminal so you can
    watch progress ("Step 25/300 ...") while the request is running.
    """
    command = [
        sys.executable,
        "-u",  # unbuffered, otherwise progress lines only show up at the end
        str(STYLE_SCRIPT_PATH),
        str(content_path),
        str(style_path),
        "--steps", str(STYLE_STEPS),
        "--size", str(STYLE_SIZE),
        "--output", str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge errors into the same stream
        text=True,
        bufsize=1,
    )

    timed_out = threading.Event()

    def _kill_on_timeout():
        timed_out.set()
        process.kill()

    watchdog = threading.Timer(STYLE_TIMEOUT_SECONDS, _kill_on_timeout)
    watchdog.start()

    started = time.time()
    lines = []

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            lines.append(line)
            print(f"[style-transfer] [{time.time() - started:6.1f}s] {line}", flush=True)
        process.wait()
    finally:
        watchdog.cancel()

    if timed_out.is_set():
        raise subprocess.TimeoutExpired(command, STYLE_TIMEOUT_SECONDS)

    return subprocess.CompletedProcess(
        command, process.returncode, stdout="\n".join(lines), stderr=""
    )


# ============================================================
# GAN (DCGAN Image Generator) Setup
# ============================================================
# The GAN package uses relative imports (`from .config import ...`), so it is
# imported as a package: backend/models/gan/. The import is wrapped so that a
# missing/broken GAN folder can never stop the other models from loading.

GAN_OUTPUT_SIZE = 128  # generator outputs 32x32; upscaled for display
GAN_LATENT_DIM = 100
gan_generator = None
gan_discriminator = None

try:
    from models.gan.generator import Generator as GANGenerator
    from models.gan.discriminator import Discriminator as GANDiscriminator
    from models.gan.config import (
        LATENT_DIM as GAN_LATENT_DIM,
        GENERATOR_CHECKPOINT as GAN_GENERATOR_CHECKPOINT,
        DISCRIMINATOR_CHECKPOINT as GAN_DISCRIMINATOR_CHECKPOINT,
    )

    if os.path.exists(GAN_GENERATOR_CHECKPOINT):
        try:
            _gan_g = GANGenerator().to(device)
            _gan_g_ckpt = torch.load(GAN_GENERATOR_CHECKPOINT, map_location=device)
            _gan_g.load_state_dict(_gan_g_ckpt["model_state_dict"])
            _gan_g.eval()
            gan_generator = _gan_g
            print("Successfully loaded GAN generator weights!")
        except Exception as e:
            print(f"Warning: failed to load GAN generator: {e}")
    else:
        print(f"Warning: GAN generator checkpoint not found at {GAN_GENERATOR_CHECKPOINT}")

    # The discriminator is optional: it is only used to give each generated
    # image a "realism" score (sigmoid of its output logit).
    if gan_generator is not None and os.path.exists(GAN_DISCRIMINATOR_CHECKPOINT):
        try:
            _gan_d = GANDiscriminator().to(device)
            _gan_d_ckpt = torch.load(GAN_DISCRIMINATOR_CHECKPOINT, map_location=device)
            _gan_d.load_state_dict(_gan_d_ckpt["model_state_dict"])
            _gan_d.eval()
            gan_discriminator = _gan_d
            print("Successfully loaded GAN discriminator weights!")
        except Exception as e:
            print(f"Warning: failed to load GAN discriminator (scores disabled): {e}")

except Exception as e:
    print(f"Warning: GAN package could not be imported: {e}")


def gan_tensor_to_data_url(img: torch.Tensor, size: int) -> str:
    """Generator output (3xHxW, range [-1, 1]) -> upscaled base64 PNG data URL."""
    img = ((img.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    pil_image = transforms.ToPILImage()(img).resize((size, size), Image.BICUBIC)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ============================================================
# Face Recognition (MTCNN + InceptionResnetV1) Setup
# ============================================================
# The face_recognition package uses relative imports (`from .face_embedder
# import ...`), so it is imported as a package: backend/models/face_recognition/.
# Like the GAN, the import is wrapped so a missing dependency (facenet-pytorch),
# missing gallery file, or broken folder can never stop the other models from
# loading.
#
# NOTE: requires models/face_recognition/gallery/gallery_embeddings.pt, which is
# created by running build_gallery.py (it is not part of the zip).

FACE_THRESHOLD = 0.50  # cosine similarity needed to count as a known identity
face_recognizer = None

try:
    from models.face_recognition.recognizer import FaceRecognizer

    face_recognizer = FaceRecognizer(
        threshold=FACE_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("Successfully loaded face recognition models + gallery!")
except FileNotFoundError as e:
    print(f"Warning: face recognition gallery missing: {e}")
except Exception as e:
    print(f"Warning: face recognition could not be loaded: {e}")


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


@app.post("/api/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
):
    if not STYLE_SCRIPT_PATH.exists():
        raise HTTPException(status_code=500, detail="Style transfer script not found")

    for name, upload in (("content_image", content_image), ("style_image", style_image)):
        if not (upload.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{name} must be an image file.")

    try:
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()

        # Fail fast with a clear message if either file isn't a decodable image.
        for name, data in (("content_image", content_bytes), ("style_image", style_bytes)):
            try:
                Image.open(io.BytesIO(data)).verify()
            except Exception:
                raise HTTPException(status_code=400, detail=f"{name} could not be read as an image.")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            content_path = temp_path / "content.jpg"
            style_path = temp_path / "style.jpg"
            output_path = temp_path / "stylized_output.jpg"

            content_path.write_bytes(content_bytes)
            style_path.write_bytes(style_bytes)

            # Runs in a worker thread so the (slow) optimization doesn't
            # block the other endpoints while it's running.
            print(f"[style-transfer] starting ({STYLE_STEPS} steps, size {STYLE_SIZE})...", flush=True)
            result = await asyncio.to_thread(
                _run_style_transfer, content_path, style_path, output_path
            )
            print(f"[style-transfer] finished, exit code {result.returncode}", flush=True)

            if result.returncode != 0 or not output_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Style transfer failed: {result.stderr or result.stdout}",
                )

            encoded = base64.b64encode(output_path.read_bytes()).decode("utf-8")

        return {
            "status": "success",
            "stylized_image": f"data:image/jpeg;base64,{encoded}",
        }

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Style transfer timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate")
def generate_gan_images(
    num_images: int = Query(8, ge=1, le=16),
    seed: Optional[int] = Query(None),
):
    if gan_generator is None:
        raise HTTPException(status_code=500, detail="GAN generator not loaded")

    try:
        # A fresh random seed per request; returned so a result can be reproduced.
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(num_images, GAN_LATENT_DIM, 1, 1, generator=rng).to(device)

        scores = None

        with torch.no_grad():
            fake_images = gan_generator(noise)

            if gan_discriminator is not None:
                probs = torch.sigmoid(gan_discriminator(fake_images)).cpu()
                scores = [round(float(p) * 100, 1) for p in probs]

        images = [gan_tensor_to_data_url(img, GAN_OUTPUT_SIZE) for img in fake_images]

        return {
            "status": "success",
            "images": images,
            "scores": scores,
            "seed": seed,
            "latent_dim": GAN_LATENT_DIM,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-recognize")
async def recognize_faces(file: UploadFile = File(...)):
    if face_recognizer is None:
        raise HTTPException(
            status_code=500,
            detail="Face recognition not loaded (check facenet-pytorch install and gallery_embeddings.pt)",
        )

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = pil_img.size

        # MTCNN + FaceNet are blocking; run in a worker thread.
        result = await asyncio.to_thread(face_recognizer.recognize, pil_img)

        faces = []
        for face in result["faces"]:
            # Pixel box -> normalized (0-1) box, so the frontend can place it
            # with simple percentages (same approach as /api/detect).
            box = face.get("box")
            norm_box = None
            if box is not None:
                x1, y1, x2, y2 = box
                norm_box = {
                    "x1": max(0.0, min(1.0, x1 / width)),
                    "y1": max(0.0, min(1.0, y1 / height)),
                    "x2": max(0.0, min(1.0, x2 / width)),
                    "y2": max(0.0, min(1.0, y2 / height)),
                }

            det_conf = face.get("detection_confidence")

            faces.append({
                "identity": face["identity"],
                "matched": bool(face["matched"]),
                "cosine_similarity": float(face["cosine_similarity"]),
                "similarity_percent": float(face["similarity_percent"]),
                "euclidean_distance": float(face["euclidean_distance"]),
                "threshold": float(face["threshold"]),
                "detection_confidence": (
                    round(float(det_conf) * 100, 1) if det_conf is not None else None
                ),
                "box": norm_box,
            })

        return {
            "status": "success",
            "face_count": result["face_count"],
            "faces": faces,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)