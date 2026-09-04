import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from models.cnn import CNNClassifier


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64

MODEL_PATH = "models/cnn_sgd.pth"

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
    "truck"
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Load dataset
# ============================================================

_, _, test_loader = get_data_loaders(
    batch_size=BATCH_SIZE
)


# ============================================================
# Load model
# ============================================================

model = CNNClassifier().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")


# ============================================================
# Evaluation
# ============================================================

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# Overall results
# ============================================================

test_accuracy = 100 * correct / total

test_loss = (
    total_loss / len(test_loader)
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES
)

print(report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n==============================")
print("PER-CLASS ACCURACY")
print("==============================")


for i, class_name in enumerate(CLASS_NAMES):

    class_total = cm[i].sum()

    class_correct = cm[i][i]

    accuracy = (
        100 * class_correct / class_total
        if class_total > 0
        else 0
    )

    print(
        f"{class_name:12s}: "
        f"{accuracy:.2f}%"
    )

print(
    "\nConfusion matrix saved as confusion_matrix.png"
)