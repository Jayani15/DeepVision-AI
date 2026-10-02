import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import DenoisingAutoencoder


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 64

LEARNING_RATE = 0.001

NUM_EPOCHS = 20

NOISE_FACTOR = 0.2


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Create folders
# ==========================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================
# Dataset
# ==========================================

train_loader, val_loader, _ = get_dataloaders(
    data_dir="./",
    batch_size=BATCH_SIZE,
    noise_factor=NOISE_FACTOR
)


# ==========================================
# Model
# ==========================================

model = DenoisingAutoencoder().to(device)


# ==========================================
# Loss
# ==========================================

criterion = nn.MSELoss()


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# Store loss history
# ==========================================

train_losses = []

val_losses = []

best_val_loss = float("inf")


# ==========================================
# Training
# ==========================================

for epoch in range(NUM_EPOCHS):

    # --------------------------------------
    # TRAIN
    # --------------------------------------

    model.train()

    running_train_loss = 0.0

    for noisy_images, clean_images in train_loader:

        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed_images = model(noisy_images)

        # Reconstruction loss
        loss = criterion(
            reconstructed_images,
            clean_images
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_train_loss += (
            loss.item() * noisy_images.size(0)
        )

    epoch_train_loss = (
        running_train_loss /
        len(train_loader.dataset)
    )

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for noisy_images, clean_images in val_loader:

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            reconstructed_images = model(
                noisy_images
            )

            loss = criterion(
                reconstructed_images,
                clean_images
            )

            running_val_loss += (
                loss.item() *
                noisy_images.size(0)
            )

    epoch_val_loss = (
        running_val_loss /
        len(val_loader.dataset)
    )

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {epoch_train_loss:.6f} | "
        f"Val Loss: {epoch_val_loss:.6f}"
    )

    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_autoencoder.pth"
        )

        print(
            "  -> Best model saved!"
        )


# ==========================================
# Plot Loss Curve
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_EPOCHS + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, NUM_EPOCHS + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(
    "Denoising Autoencoder Training"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "results/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# Save loss values
# ==========================================

torch.save(
    {
        "train_losses": train_losses,
        "val_losses": val_losses
    },
    "results/training_history.pth"
)


print("\nTraining completed!")

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Best model saved at: "
    "checkpoints/best_autoencoder.pth"
)

print(
    "Loss graph saved at: "
    "results/loss_curve.png"
)