import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

APP_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = APP_DIR / "style_transfer.py"

app = FastAPI(title="Neural Style Transfer API")


@app.get("/")
def health_check():
    return {"status": "success", "message": "Style Transfer API is running"}


@app.post("/api/style-transfer")
async def create_style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
):
    valid_types = {"image/jpeg", "image/jpg", "image/png"}

    if content_image.content_type not in valid_types:
        raise HTTPException(400, "content_image must be a JPG or PNG image.")

    if style_image.content_type not in valid_types:
        raise HTTPException(400, "style_image must be a JPG or PNG image.")

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_path = Path(temp_directory)

        content_path = temp_path / "content.jpg"
        style_path = temp_path / "style.jpg"
        output_path = temp_path / "stylized_output.jpg"

        content_path.write_bytes(await content_image.read())
        style_path.write_bytes(await style_image.read())

        command = [
            sys.executable,
            str(SCRIPT_PATH),
            str(content_path),
            str(style_path),
            "--steps",
            "300",
            "--size",
            "384",
            "--output",
            str(output_path),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0 or not output_path.exists():
            raise HTTPException(
                500,
                f"Style transfer failed: {result.stderr or result.stdout}",
            )

        encoded_image = base64.b64encode(output_path.read_bytes()).decode("utf-8")

    return {
        "status": "success",
        "stylized_image": f"data:image/jpeg;base64,{encoded_image}",
    }

