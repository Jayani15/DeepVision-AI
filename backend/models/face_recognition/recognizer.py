from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .face_embedder import FaceEmbedder


BASE_DIR = Path(__file__).resolve().parent

GALLERY_FILE = (
    BASE_DIR
    / "gallery"
    / "gallery_embeddings.pt"
)


class FaceRecognizer:

    def __init__(
        self,
        threshold: float = 0.50,
        device: Optional[str] = None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.threshold = threshold

        if not GALLERY_FILE.exists():
            raise FileNotFoundError(
                "Gallery embeddings not found.\n"
                "Run build_gallery.py first."
            )

        gallery_data = torch.load(
            GALLERY_FILE,
            map_location=self.device,
            weights_only=False,
        )

        self.gallery = {
            name: embedding.to(self.device)
            for name, embedding
            in gallery_data["embeddings"].items()
        }

        self.embedder = FaceEmbedder(
            device=self.device
        )

    def compare_embedding(
        self,
        query_embedding: torch.Tensor,
    ) -> Dict:

        query_embedding = F.normalize(
            query_embedding,
            p=2,
            dim=0,
        )

        best_identity = None
        best_similarity = -1.0
        best_distance = float("inf")

        for identity, gallery_embedding in self.gallery.items():

            gallery_embedding = F.normalize(
                gallery_embedding,
                p=2,
                dim=0,
            )

            cosine_similarity = torch.dot(
                query_embedding,
                gallery_embedding,
            ).item()

            euclidean_distance = torch.norm(
                query_embedding
                - gallery_embedding,
                p=2,
            ).item()

            if cosine_similarity > best_similarity:

                best_similarity = cosine_similarity
                best_distance = euclidean_distance
                best_identity = identity

        matched = (
            best_similarity >= self.threshold
        )

        return {
            "identity": (
                best_identity
                if matched
                else "Unknown"
            ),
            "matched": matched,
            "cosine_similarity": round(
                best_similarity,
                4,
            ),
            "similarity_percent": round(
                best_similarity * 100,
                2,
            ),
            "euclidean_distance": round(
                best_distance,
                4,
            ),
            "threshold": self.threshold,
        }

    def recognize(
        self,
        image: Image.Image,
    ):

        embeddings, boxes, probabilities = (
            self.embedder.embed_faces(image)
        )

        if len(embeddings) == 0:
            return {
                "face_count": 0,
                "faces": [],
            }

        results = []

        for index, embedding in enumerate(
            embeddings
        ):

            result = self.compare_embedding(
                embedding
            )

            result["face_index"] = index

            if index < len(boxes):
                result["box"] = boxes[index]
            else:
                result["box"] = None

            if index < len(probabilities):
                result["detection_confidence"] = round(
                    probabilities[index],
                    4,
                )
            else:
                result["detection_confidence"] = None

            results.append(result)

        return {
            "face_count": len(results),
            "faces": results,
        }