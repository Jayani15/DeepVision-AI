import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from .recognizer import FaceRecognizer


BASE_DIR = Path(__file__).resolve().parent

SPLIT_FILE = BASE_DIR / "gallery" / "split.json"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        split = json.load(f)

    recognizer = FaceRecognizer(
        threshold=0.70
    )

    rows = []

    y_true = []
    y_pred = []

    similarities = []
    similarity_labels = []

    # -----------------------------------------------------
    # Known identities
    # -----------------------------------------------------

    for identity, data in split["known"].items():

        print()
        print(f"Testing known: {identity}")

        for relative_path in data["test"]:

            image_path = BASE_DIR / relative_path

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                result = recognizer.recognize(
                    image
                )

                if result["face_count"] == 0:
                    print(
                        f"  No face: "
                        f"{image_path.name}"
                    )
                    continue

                face_result = result["faces"][0]

                predicted = face_result[
                    "identity"
                ]

                similarity = face_result[
                    "cosine_similarity"
                ]

                y_true.append(identity)
                y_pred.append(predicted)

                similarities.append(
                    similarity
                )

                similarity_labels.append(
                    "known"
                )

                rows.append(
                    {
                        "true_identity": identity,
                        "predicted_identity": predicted,
                        "matched": face_result["matched"],
                        "similarity": similarity,
                        "distance": face_result[
                            "euclidean_distance"
                        ],
                        "image": str(image_path),
                    }
                )

                print(
                    f"  {image_path.name}: "
                    f"{predicted} "
                    f"({similarity:.4f})"
                )

            except Exception as e:

                print(
                    f"  ERROR: "
                    f"{image_path.name}: {e}"
                )

    # -----------------------------------------------------
    # Unknown identities
    # -----------------------------------------------------

    print()
    print("Testing unknown identities")

    unknown_correct = 0
    unknown_total = 0

    for identity, data in split["unknown"].items():

        for relative_path in data["test"]:

            image_path = BASE_DIR / relative_path

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                result = recognizer.recognize(
                    image
                )

                if result["face_count"] == 0:
                    continue

                face_result = result["faces"][0]

                predicted = face_result[
                    "identity"
                ]

                similarity = face_result[
                    "cosine_similarity"
                ]

                unknown_total += 1

                if predicted == "Unknown":
                    unknown_correct += 1

                similarities.append(
                    similarity
                )

                similarity_labels.append(
                    "unknown"
                )

                rows.append(
                    {
                        "true_identity": "Unknown",
                        "predicted_identity": predicted,
                        "matched": face_result["matched"],
                        "similarity": similarity,
                        "distance": face_result[
                            "euclidean_distance"
                        ],
                        "image": str(image_path),
                    }
                )

                print(
                    f"  {image_path.name}: "
                    f"{predicted} "
                    f"({similarity:.4f})"
                )

            except Exception as e:

                print(
                    f"  ERROR: "
                    f"{image_path.name}: {e}"
                )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    known_accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
        if y_true
        else 0.0
    )

    unknown_rejection_rate = (
        unknown_correct / unknown_total
        if unknown_total
        else 0.0
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "recognition_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "true_identity",
                "predicted_identity",
                "matched",
                "similarity",
                "distance",
                "image",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    if y_true:

        labels = sorted(
            set(y_true) | set(y_pred)
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        plt.figure(
            figsize=(10, 8)
        )

        plt.imshow(cm)

        plt.title(
            "Face Recognition Confusion Matrix"
        )

        plt.xlabel(
            "Predicted Identity"
        )

        plt.ylabel(
            "True Identity"
        )

        plt.xticks(
            range(len(labels)),
            labels,
            rotation=90,
        )

        plt.yticks(
            range(len(labels)),
            labels,
        )

        plt.colorbar()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            / "confusion_matrix.png",
            dpi=200,
        )

        plt.close()

    # -----------------------------------------------------
    # Similarity distribution
    # -----------------------------------------------------

    known_scores = [
        score
        for score, label
        in zip(
            similarities,
            similarity_labels,
        )
        if label == "known"
    ]

    unknown_scores = [
        score
        for score, label
        in zip(
            similarities,
            similarity_labels,
        )
        if label == "unknown"
    ]

    plt.figure(
        figsize=(10, 6)
    )

    if known_scores:
        plt.hist(
            known_scores,
            bins=20,
            alpha=0.7,
            label="Known",
        )

    if unknown_scores:
        plt.hist(
            unknown_scores,
            bins=20,
            alpha=0.7,
            label="Unknown",
        )

    plt.axvline(
        0.70,
        linestyle="--",
        label="Threshold = 0.70",
    )

    plt.xlabel(
        "Cosine Similarity"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Face Similarity Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "similarity_distribution.png",
        dpi=200,
    )

    plt.close()

    # -----------------------------------------------------
    # JSON results
    # -----------------------------------------------------

    results = {
        "threshold": 0.70,
        "known_accuracy": round(
            known_accuracy,
            4,
        ),
        "unknown_rejection_rate": round(
            unknown_rejection_rate,
            4,
        ),
        "known_samples": len(y_true),
        "unknown_samples": unknown_total,
    }

    json_path = (
        RESULTS_DIR
        / "evaluation_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Known recognition accuracy: "
        f"{known_accuracy * 100:.2f}%"
    )

    print(
        f"Unknown rejection rate: "
        f"{unknown_rejection_rate * 100:.2f}%"
    )

    print()
    print(
        f"Results saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()