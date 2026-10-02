from pathlib import Path

import torch
import matplotlib.pyplot as plt

from torchvision.utils import save_image

from .config import RESULTS_DIR


def save_generated_images(
    images,
    filename,
    normalize=True,
    nrow=8,
):
    path = RESULTS_DIR / filename

    save_image(
        images,
        path,
        normalize=normalize,
        nrow=nrow,
    )

    return path


def plot_losses(
    generator_losses,
    discriminator_losses,
):

    plt.figure(figsize=(10, 5))

    plt.plot(
        generator_losses,
        label="Generator Loss",
    )

    plt.plot(
        discriminator_losses,
        label="Discriminator Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")

    plt.legend()
    plt.grid(True)

    path = RESULTS_DIR / "loss_curve.png"

    plt.savefig(path)
    plt.close()

    return path


def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )