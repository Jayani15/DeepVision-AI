import torch
import torch.nn as nn

from .config import (
    IMAGE_CHANNELS,
    DISCRIMINATOR_FEATURES,
)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 32 × 32 → 16 × 16
            nn.Conv2d(
                IMAGE_CHANNELS,
                DISCRIMINATOR_FEATURES,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 × 16 → 8 × 8
            nn.Conv2d(
                DISCRIMINATOR_FEATURES,
                DISCRIMINATOR_FEATURES * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                DISCRIMINATOR_FEATURES * 2
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 × 8 → 4 × 4
            nn.Conv2d(
                DISCRIMINATOR_FEATURES * 2,
                DISCRIMINATOR_FEATURES * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                DISCRIMINATOR_FEATURES * 4
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 × 4 → 1 × 1
            nn.Conv2d(
                DISCRIMINATOR_FEATURES * 4,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.model(x).view(-1)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Discriminator().to(device)

    images = torch.randn(
        4,
        IMAGE_CHANNELS,
        32,
        32,
        device=device,
    )

    output = model(images)

    print("Discriminator test successful!")
    print("Input:", images.shape)
    print("Output:", output.shape)