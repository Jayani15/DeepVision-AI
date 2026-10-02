import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # ==========================================
        # ENCODER
        # ==========================================

        self.encoder = nn.Sequential(

            # Input:
            # [B, 3, 32, 32]

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # [B, 32, 16, 16]

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # [B, 64, 8, 8]

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)

            # Latent representation:
            # [B, 128, 8, 8]
        )

        # ==========================================
        # DECODER
        # ==========================================

        self.decoder = nn.Sequential(

            # [B, 128, 8, 8]

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            # [B, 64, 16, 16]

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            # [B, 32, 32, 32]

            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=3,
                padding=1
            ),

            # Output pixel values between 0 and 1
            nn.Sigmoid()

            # Output:
            # [B, 3, 32, 32]
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# ==============================================
# Test architecture
# ==============================================

if __name__ == "__main__":

    model = DenoisingAutoencoder()

    test_input = torch.randn(
        4,
        3,
        32,
        32
    )

    output = model(test_input)

    latent = model.encoder(test_input)

    print(model)

    print("\nInput shape :", test_input.shape)
    print("Latent shape:", latent.shape)
    print("Output shape:", output.shape)

    assert output.shape == test_input.shape

    print("\nModel shape test PASSED!")