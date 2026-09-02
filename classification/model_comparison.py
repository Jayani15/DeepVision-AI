import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from models.mlp import MLP
from models.cnn import CNNClassifier


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64

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
# Test Data
# ============================================================

_, _, test_loader = get_data_loaders(
    batch_size=BATCH_SIZE
)


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = (
        100 * correct / total
    )

    return accuracy


# ============================================================
# Evaluate MLP
# ============================================================

mlp = MLP().to(device)

mlp.load_state_dict(
    torch.load(
        "models/mlp.pth",
        map_location=device
    )
)

mlp_accuracy = evaluate_model(mlp)

print(
    f"\nMLP Accuracy: "
    f"{mlp_accuracy:.2f}%"
)


# ============================================================
# Evaluate CNN Models
# ============================================================

cnn_models = {

    "CNN - SGD":
        "models/cnn_sgd.pth",

    "CNN - Momentum":
        "models/cnn_momentum.pth",

    "CNN - RMSProp":
        "models/cnn_rmsprop.pth"
}


results = {
    "MLP": mlp_accuracy
}


for model_name, model_path in cnn_models.items():

    try:

        model = CNNClassifier(
            dropout_rate=0.5
        ).to(device)

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device
            )
        )

        accuracy = evaluate_model(model)

        results[model_name] = accuracy

        print(
            f"{model_name} Accuracy: "
            f"{accuracy:.2f}%"
        )

    except FileNotFoundError:

        print(
            f"{model_name}: "
            f"Model file not found."
        )


# ============================================================
# Final Comparison
# ============================================================

print("\n===================================")
print("FINAL MODEL COMPARISON")
print("===================================")

for model_name, accuracy in results.items():

    print(
        f"{model_name:20s}: "
        f"{accuracy:.2f}%"
    )


# ============================================================
# Find Best Model
# ============================================================

best_model = max(
    results,
    key=results.get
)

best_accuracy = results[best_model]


print("\n===================================")
print("BEST MODEL")
print("===================================")

print(
    f"Model    : {best_model}"
)

print(
    f"Accuracy : {best_accuracy:.2f}%"
)

names = list(results.keys())
accuracies = list(results.values())


plt.figure(figsize=(9, 5))

plt.bar(
    names,
    accuracies
)

plt.ylabel(
    "Test Accuracy (%)"
)

plt.title(
    "MLP vs CNN and Optimizer Comparison"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nComparison graph saved as model_comparison.png"
)