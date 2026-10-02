import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from models.cnn import CNNClassifier


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Data
# ============================================================

train_loader, val_loader, _ = get_data_loaders(
    batch_size=BATCH_SIZE
)


# ============================================================
# Training Function
# ============================================================

def train_experiment(dropout_rate):

    if dropout_rate == 0:

        experiment_name = "Without Dropout"

    else:

        experiment_name = "With Dropout"


    print("\n" + "=" * 50)
    print(experiment_name)
    print("=" * 50)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CNNClassifier(
        dropout_rate=dropout_rate
    ).to(device)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )


    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(EPOCHS):


        # ====================================================
        # Training
        # ====================================================

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Loss
            loss = criterion(
                outputs,
                labels
            )


            # Clear gradients
            optimizer.zero_grad()


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            running_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        train_loss = (
            running_loss /
            len(train_loader)
        )


        train_accuracy = (
            100 * correct / total
        )


        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_loss_total = 0.0

        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                val_loss_total += loss.item()


                _, predicted = torch.max(
                    outputs,
                    1
                )


                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()


        val_loss = (
            val_loss_total /
            len(val_loader)
        )


        val_accuracy = (
            100 * val_correct /
            val_total
        )


        # ====================================================
        # Store results
        # ====================================================

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        val_accuracies.append(
            val_accuracy
        )


        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )


    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }


# ============================================================
# Run both experiments
# ============================================================

results_without_dropout = train_experiment(
    dropout_rate=0.0
)

results_with_dropout = train_experiment(
    dropout_rate=0.5
)


# ============================================================
# Accuracy Comparison
# ============================================================

epochs = range(
    1,
    EPOCHS + 1
)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    results_without_dropout["train_accuracy"],
    label="Train - No Dropout"
)

plt.plot(
    epochs,
    results_without_dropout["val_accuracy"],
    label="Validation - No Dropout"
)

plt.plot(
    epochs,
    results_with_dropout["train_accuracy"],
    label="Train - Dropout"
)

plt.plot(
    epochs,
    results_with_dropout["val_accuracy"],
    label="Validation - Dropout"
)


plt.xlabel("Epoch")

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Effect of Dropout on Generalization"
)

plt.legend()

plt.savefig(
    "dropout_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Loss Comparison
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    results_without_dropout["train_loss"],
    label="Train Loss - No Dropout"
)

plt.plot(
    epochs,
    results_without_dropout["val_loss"],
    label="Validation Loss - No Dropout"
)

plt.plot(
    epochs,
    results_with_dropout["train_loss"],
    label="Train Loss - Dropout"
)

plt.plot(
    epochs,
    results_with_dropout["val_loss"],
    label="Validation Loss - Dropout"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Effect of Dropout on Loss"
)

plt.legend()


plt.savefig(
    "dropout_loss_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nExperiment completed!")

print(
    "Saved: dropout_accuracy_comparison.png"
)

print(
    "Saved: dropout_loss_comparison.png"
)