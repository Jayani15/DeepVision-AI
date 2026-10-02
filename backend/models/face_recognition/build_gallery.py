import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .face_embedder import FaceEmbedder


BASE_DIR = Path(__file__).resolve().parent

SPLIT_FILE = BASE_DIR / "gallery" / "split.json"

OUTPUT_FILE = (
    BASE_DIR / "gallery" / "gallery_embeddings.pt"
)


def main():

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            "split.json not found. "
            "Run prepare_dataset.py first."
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print("Loading face embedding model...")

    embedder = FaceEmbedder(
        device=device
    )

    gallery = {}

    for identity, data in split["known"].items():

        print()
        print(f"Building gallery: {identity}")

        embeddings = []

        for relative_path in data["gallery"]:

            image_path = BASE_DIR / relative_path

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                embedding = (
                    embedder.embed_single_face(
                        image
                    )
                )

                embeddings.append(
                    embedding.cpu()
                )

                print(
                    f"  OK: {image_path.name}"
                )

            except Exception as e:

                print(
                    f"  SKIPPED: "
                    f"{image_path.name} "
                    f"-> {e}"
                )

        if not embeddings:
            print(
                f"WARNING: No valid images for "
                f"{identity}"
            )
            continue

        stacked = torch.stack(
            embeddings
        )

        prototype = stacked.mean(
            dim=0
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=0,
        )

        gallery[identity] = prototype

        print(
            f"  Created 512-D prototype "
            f"from {len(embeddings)} images."
        )

    if not gallery:
        raise RuntimeError(
            "Gallery is empty."
        )

    output = {
        "embeddings": gallery,
        "identities": list(gallery.keys()),
    }

    torch.save(
        output,
        OUTPUT_FILE,
    )

    print()
    print("=" * 60)
    print("GALLERY CREATED")
    print("=" * 60)
    print(f"Identities: {len(gallery)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()