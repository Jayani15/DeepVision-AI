import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)


def get_dataloader():

    transform = transforms.Compose([
        transforms.ToTensor(),

        # Convert [0, 1] → [-1, 1]
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
    ])

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


if __name__ == "__main__":
    loader = get_dataloader()

    images, labels = next(iter(loader))

    print("Dataset loaded successfully!")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)