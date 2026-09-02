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

train_loader, val_loader, test_loader = get_data_loaders(
    batch_size=BATCH_SIZE
)


# ============================================================
# Training Function
# ============================================================

def train_model(optimizer_name):

    print("\n" + "=" * 50)
    print("Optimizer:", optimizer_name)
    print("=" * 50)

    # --------------------------------------------------------
    # Create a NEW model for every optimizer experiment
    # --------------------------------------------------------

    model = CNNClassifier().to(device)

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Select Optimizer
    # --------------------------------------------------------

    if optimizer_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01
        )

    elif optimizer_name == "Momentum":

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    elif optimizer_name == "RMSProp":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=0.001
        )

    else:

        raise ValueError(
            "Unknown optimizer"
        )


    # --------------------------------------------------------
    # Store Training History
    # --------------------------------------------------------

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(EPOCHS):


        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0


        # ----------------------------------------------------
        # Mini-batch Loop
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )


            # Clear previous gradients
            optimizer.zero_grad()


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            # ------------------------------------------------
            # Training Statistics
            # ------------------------------------------------

            running_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


        # ----------------------------------------------------
        # Training Results
        # ----------------------------------------------------

        train_loss = (
            running_loss /
            len(train_loader)
        )


        train_accuracy = (
            100 * correct /
            total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0.0

        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)


                # Forward pass
                outputs = model(images)


                # Validation loss
                loss = criterion(
                    outputs,
                    labels
                )


                val_loss_total += loss.item()


                # Prediction
                _, predicted = torch.max(
                    outputs,
                    1
                )


                val_total += labels.size(0)


                val_correct += (
                    predicted == labels
                ).sum().item()


        # ----------------------------------------------------
        # Validation Results
        # ----------------------------------------------------

        val_loss = (
            val_loss_total /
            len(val_loader)
        )


        val_accuracy = (
            100 * val_correct /
            val_total
        )


        # ====================================================
        # Store Results
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


        # ====================================================
        # Print Results
        # ====================================================

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )


    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    model_path = (
        f"models/cnn_{optimizer_name.lower()}.pth"
    )


    torch.save(
        model.state_dict(),
        model_path
    )


    print(
        f"\nModel saved: {model_path}"
    )


    # ========================================================
    # Return Training History
    # ========================================================

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    }


# ============================================================
# Run All Optimizer Experiments
# ============================================================

results = {}


for optimizer_name in [
    "SGD",
    "Momentum",
    "RMSProp"
]:

    results[optimizer_name] = train_model(
        optimizer_name
    )


# ============================================================
# Plot Validation Accuracy
# ============================================================

plt.figure(figsize=(8, 5))


for optimizer_name, history in results.items():

    plt.plot(
        range(1, EPOCHS + 1),
        history["val_accuracy"],
        marker="o",
        label=optimizer_name
    )


plt.xlabel("Epoch")

plt.ylabel(
    "Validation Accuracy (%)"
)

plt.title(
    "Optimizer Comparison"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "optimizer_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nOptimizer comparison graph saved!"
)


# ============================================================
# Final Results
# ============================================================

print("\n" + "=" * 50)
print("FINAL OPTIMIZER RESULTS")
print("=" * 50)


for optimizer_name, history in results.items():

    final_train_accuracy = (
        history["train_accuracy"][-1]
    )

    final_val_accuracy = (
        history["val_accuracy"][-1]
    )

    print(
        f"{optimizer_name:10s} | "
        f"Train Accuracy: "
        f"{final_train_accuracy:.2f}% | "
        f"Validation Accuracy: "
        f"{final_val_accuracy:.2f}%"
    )