import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. ADD GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(image, noise_factor=0.2):
    """
    Add Gaussian noise to an image.

    image:
        Tensor with pixel values between 0 and 1

    noise_factor:
        Controls how much noise is added
    """

    # Generate random Gaussian noise
    noise = torch.randn_like(image) * noise_factor

    # Add noise to the original image
    noisy_image = image + noise

    # Keep pixel values between 0 and 1
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image


# ============================================================
# 2. DENOISING CIFAR-10 DATASET
# ============================================================

class DenoisingCIFAR10(Dataset):

    def __init__(self, cifar_dataset, noise_factor=0.2):

        self.dataset = cifar_dataset
        self.noise_factor = noise_factor

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        # Get clean image and its label
        clean_image, _ = self.dataset[index]

        # Add artificial noise
        noisy_image = add_gaussian_noise(
            clean_image,
            self.noise_factor
        )

        # IMPORTANT:
        # Input  = noisy image
        # Target = clean image

        return noisy_image, clean_image


# ============================================================
# 3. CREATE DATA LOADERS
# ============================================================

def get_dataloaders(
    data_dir="./",
    batch_size=64,
    noise_factor=0.2
):

    # Convert PIL image → PyTorch Tensor
    # Pixel values become [0, 1]
    transform = transforms.ToTensor()


    # --------------------------------------------------------
    # LOAD CIFAR-10 TRAINING DATA
    # --------------------------------------------------------

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )


    # --------------------------------------------------------
    # SPLIT TRAINING DATA
    # --------------------------------------------------------

    train_size = 45000
    val_size = 5000

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )


    # --------------------------------------------------------
    # LOAD CIFAR-10 TEST DATA
    # --------------------------------------------------------

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )


    # --------------------------------------------------------
    # CONVERT TO DENOISING DATASETS
    # --------------------------------------------------------

    train_dataset = DenoisingCIFAR10(
        train_dataset,
        noise_factor=noise_factor
    )

    val_dataset = DenoisingCIFAR10(
        val_dataset,
        noise_factor=noise_factor
    )

    test_dataset = DenoisingCIFAR10(
        test_dataset,
        noise_factor=noise_factor
    )


    # --------------------------------------------------------
    # CREATE DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )


    return train_loader, val_loader, test_loader


# ============================================================
# 4. TEST DATASET
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    # Get one batch
    noisy_images, clean_images = next(
        iter(train_loader)
    )

    print("\n================================")
    print("CIFAR-10 DATASET TEST")
    print("================================")

    print(
        "Noisy batch shape :",
        noisy_images.shape
    )

    print(
        "Clean batch shape :",
        clean_images.shape
    )

    print(
        "Noisy image range:",
        noisy_images.min().item(),
        "to",
        noisy_images.max().item()
    )

    print(
        "Clean image range:",
        clean_images.min().item(),
        "to",
        clean_images.max().item()
    )

    print(
        "\nTraining samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(val_loader.dataset)
    )

    print(
        "Test samples:",
        len(test_loader.dataset)
    )

    print("\nDataset test completed successfully! ✓")