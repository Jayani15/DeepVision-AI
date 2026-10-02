from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1

from .face_detector import FaceDetector


class FaceEmbedder:
    """
    Generates 512-dimensional face embeddings using
    InceptionResnetV1 pretrained on VGGFace2.
    """

    def __init__(self, device: Optional[str] = None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.detector = FaceDetector(device=self.device)

        self.model = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )

    def embed_faces(self, image: Image.Image):
        """
        Detect faces and generate one embedding per face.

        Returns:
            embeddings
            boxes
            probabilities
        """

        faces, boxes, probabilities = self.detector.extract_faces(
            image
        )

        if faces is None:
            return [], boxes, probabilities

        faces = faces.to(self.device)

        with torch.no_grad():
            embeddings = self.model(faces)

        # Normalize embeddings.
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return embeddings, boxes, probabilities

    def embed_single_face(self, image: Image.Image):
        """
        Generate an embedding for an image containing exactly one face.
        """

        embeddings, boxes, probabilities = self.embed_faces(image)

        if len(embeddings) == 0:
            raise ValueError("No face detected.")

        if len(embeddings) > 1:
            raise ValueError(
                "Multiple faces detected. "
                "Use an image containing exactly one face."
            )

        return embeddings[0]