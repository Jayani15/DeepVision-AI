import matplotlib.pyplot as plt
import torch

from data_loader import get_data_loaders


classes = [
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


train_loader, _, _ = get_data_loaders(
    batch_size=16
)

images, labels = next(iter(train_loader))


# Undo normalization for visualization
mean = torch.tensor(
    (0.4914, 0.4822, 0.4465)
).view(3, 1, 1)

std = torch.tensor(
    (0.2470, 0.2435, 0.2616)
).view(3, 1, 1)


images = images * std + mean
images = torch.clamp(images, 0, 1)


plt.figure(figsize=(10, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    image = images[i].permute(1, 2, 0)

    plt.imshow(image)

    plt.title(classes[labels[i]])
    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "cifar10_samples.png",
    dpi=300,
    bbox_inches="tight"
)

print("Dataset visualization saved successfully!")