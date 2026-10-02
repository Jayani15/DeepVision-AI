import torch
import torch.nn as nn

from .config import (
    LATENT_DIM,
    IMAGE_CHANNELS,
    GENERATOR_FEATURES,
)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 100 × 1 × 1
            nn.ConvTranspose2d(
                LATENT_DIM,
                GENERATOR_FEATURES * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(GENERATOR_FEATURES * 8),
            nn.ReLU(True),

            # 4 × 4
            nn.ConvTranspose2d(
                GENERATOR_FEATURES * 8,
                GENERATOR_FEATURES * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(GENERATOR_FEATURES * 4),
            nn.ReLU(True),

            # 8 × 8
            nn.ConvTranspose2d(
                GENERATOR_FEATURES * 4,
                GENERATOR_FEATURES * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(GENERATOR_FEATURES * 2),
            nn.ReLU(True),

            # 16 × 16
            nn.ConvTranspose2d(
                GENERATOR_FEATURES * 2,
                GENERATOR_FEATURES,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(GENERATOR_FEATURES),
            nn.ReLU(True),

            # 32 × 32
            nn.ConvTranspose2d(
                GENERATOR_FEATURES,
                IMAGE_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Generator().to(device)

    noise = torch.randn(
        4,
        LATENT_DIM,
        1,
        1,
        device=device,
    )

    output = model(noise)

    print("Generator test successful!")
    print("Input:", noise.shape)
    print("Output:", output.shape)