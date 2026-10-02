import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loaders
from models.cnn import CNNClassifier


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Data
# -----------------------------

train_loader, val_loader, test_loader = get_data_loaders(
    batch_size=64
)


# -----------------------------
# Model
# -----------------------------

model = CNNClassifier().to(device)


# -----------------------------
# Loss
# -----------------------------

criterion = nn.CrossEntropyLoss()


# -----------------------------
# Optimizer
# -----------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


# -----------------------------
# Training
# -----------------------------

epochs = 5


for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    "models/cnn_sgd.pth"
)

print("\nCNN model saved successfully!")