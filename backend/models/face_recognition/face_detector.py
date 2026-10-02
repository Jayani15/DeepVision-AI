from typing import List, Dict, Optional

import torch
from PIL import Image
from facenet_pytorch import MTCNN


class FaceDetector:
    """
    Face detection and alignment using MTCNN.

    MTCNN detects faces and produces aligned 160x160 face crops
    suitable for the pretrained InceptionResnetV1 model.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device,
        )

    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Detect all faces in an image.

        Returns:
            List containing:
            - box
            - confidence
            - index
        """

        image = image.convert("RGB")

        boxes, probabilities = self.mtcnn.detect(image)

        results = []

        if boxes is None:
            return results

        for index, (box, probability) in enumerate(
            zip(boxes, probabilities)
        ):
            if probability is None:
                continue

            results.append(
                {
                    "index": index,
                    "box": [float(x) for x in box],
                    "confidence": float(probability),
                }
            )

        return results

    def extract_faces(self, image: Image.Image):
        """
        Detect and extract aligned face crops.

        Returns:
            faces: Tensor [N, 3, 160, 160]
            boxes: detected bounding boxes
            probabilities: detection confidence values
        """

        image = image.convert("RGB")

        faces = self.mtcnn(image)

        boxes, probabilities = self.mtcnn.detect(image)

        if faces is None:
            return None, [], []

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        boxes_list = []

        if boxes is not None:
            boxes_list = [
                [float(x) for x in box]
                for box in boxes
            ]

        probabilities_list = []

        if probabilities is not None:
            probabilities_list = [
                float(p)
                for p in probabilities
            ]

        return faces, boxes_list, probabilities_list