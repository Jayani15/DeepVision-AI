import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    DEVICE,
    LATENT_DIM,
    EPOCHS,
    LEARNING_RATE,
    BETA1,
    BETA2,
    GENERATOR_CHECKPOINT,
    DISCRIMINATOR_CHECKPOINT,
    TRAINING_HISTORY,
)

from .dataset import get_dataloader
from .generator import Generator
from .discriminator import Discriminator

from .utils import (
    save_generated_images,
    plot_losses,
    save_checkpoint,
)


def main():

    print("=" * 60)
    print("DeepVision AI - DCGAN Training")
    print("=" * 60)

    print("Device:", DEVICE)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataloader = get_dataloader()

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    # --------------------------------------------------
    # Fixed noise
    # --------------------------------------------------

    fixed_noise = torch.randn(
        64,
        LATENT_DIM,
        1,
        1,
        device=DEVICE,
    )

    generator_losses = []
    discriminator_losses = []

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        for real_images, _ in dataloader:

            real_images = real_images.to(DEVICE)

            batch_size = real_images.size(0)

            # ==========================================
            # Train Discriminator
            # ==========================================

            optimizer_d.zero_grad()

            # Real images
            real_labels = torch.ones(
                batch_size,
                device=DEVICE,
            )

            real_output = discriminator(
                real_images
            )

            real_loss = criterion(
                real_output,
                real_labels,
            )

            # Fake images
            noise = torch.randn(
                batch_size,
                LATENT_DIM,
                1,
                1,
                device=DEVICE,
            )

            fake_images = generator(noise)

            fake_labels = torch.zeros(
                batch_size,
                device=DEVICE,
            )

            fake_output = discriminator(
                fake_images.detach()
            )

            fake_loss = criterion(
                fake_output,
                fake_labels,
            )

            discriminator_loss = (
                real_loss + fake_loss
            ) / 2

            discriminator_loss.backward()

            optimizer_d.step()

            # ==========================================
            # Train Generator
            # ==========================================

            optimizer_g.zero_grad()

            noise = torch.randn(
                batch_size,
                LATENT_DIM,
                1,
                1,
                device=DEVICE,
            )

            generated_images = generator(noise)

            discriminator_output = discriminator(
                generated_images
            )

            # Generator wants discriminator
            # to classify fake images as real.
            generator_loss = criterion(
                discriminator_output,
                real_labels,
            )

            generator_loss.backward()

            optimizer_g.step()

            generator_loss_total += (
                generator_loss.item()
            )

            discriminator_loss_total += (
                discriminator_loss.item()
            )

        # --------------------------------------------------
        # Average losses
        # --------------------------------------------------

        avg_g_loss = (
            generator_loss_total /
            len(dataloader)
        )

        avg_d_loss = (
            discriminator_loss_total /
            len(dataloader)
        )

        generator_losses.append(avg_g_loss)
        discriminator_losses.append(avg_d_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Generator Loss: {avg_g_loss:.4f} "
            f"Discriminator Loss: {avg_d_loss:.4f}"
        )

        # --------------------------------------------------
        # Save generated samples
        # --------------------------------------------------

        with torch.no_grad():

            fake_samples = generator(
                fixed_noise
            ).detach().cpu()

        save_generated_images(
            fake_samples,
            f"epoch_{epoch + 1:03d}.png",
        )

        # --------------------------------------------------
        # Save checkpoints
        # --------------------------------------------------

        save_checkpoint(
            generator,
            optimizer_g,
            epoch + 1,
            GENERATOR_CHECKPOINT,
        )

        save_checkpoint(
            discriminator,
            optimizer_d,
            epoch + 1,
            DISCRIMINATOR_CHECKPOINT,
        )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    torch.save(
        {
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
        },
        TRAINING_HISTORY,
    )

    plot_losses(
        generator_losses,
        discriminator_losses,
    )

    print("\nTraining completed!")
    print(
        "Generator checkpoint:",
        GENERATOR_CHECKPOINT,
    )
    print(
        "Discriminator checkpoint:",
        DISCRIMINATOR_CHECKPOINT,
    )


if __name__ == "__main__":
    main()