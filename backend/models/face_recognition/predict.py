import argparse
import json

from PIL import Image

from .recognizer import FaceRecognizer


def main():

    parser = argparse.ArgumentParser(
        description="DeepVision AI Face Recognition"
    )

    parser.add_argument(
        "image",
        help="Path to test image",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold",
    )

    args = parser.parse_args()

    image = Image.open(
        args.image
    ).convert("RGB")

    recognizer = FaceRecognizer(
        threshold=args.threshold
    )

    result = recognizer.recognize(
        image
    )

    print()
    print("=" * 60)
    print("FACE RECOGNITION RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()