# Mini-batch gradient descent
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_data_loaders(batch_size=64):

    # Training transformation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD
        )
    ])

    # Validation/Test transformation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD
        )
    ])

    # Dataset with augmentation
    train_dataset_aug = datasets.CIFAR10(
        root="./classification/data",
        train=True,
        download=True,
        transform=train_transform
    )

    # Same images without augmentation
    train_dataset_eval = datasets.CIFAR10(
        root="./classification/data",
        train=True,
        download=False,
        transform=eval_transform
    )

    # Official test set
    test_dataset = datasets.CIFAR10(
        root="./classification/data",
        train=False,
        download=True,
        transform=eval_transform
    )

    # Fixed split
    train_size = 45000
    val_size = 5000

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(train_dataset_aug),
        generator=generator
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(
        train_dataset_aug,
        train_indices
    )

    val_dataset = Subset(
        train_dataset_eval,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader