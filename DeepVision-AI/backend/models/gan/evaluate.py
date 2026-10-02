import torch

from .config import (
    DEVICE,
    LATENT_DIM,
    GENERATOR_CHECKPOINT,
    RESULTS_DIR,
)

from .generator import Generator

from torchvision.utils import save_image


def evaluate_generator(
    num_images=64,
):

    generator = Generator().to(DEVICE)

    checkpoint = torch.load(
        GENERATOR_CHECKPOINT,
        map_location=DEVICE,
    )

    generator.load_state_dict(
        checkpoint["model_state_dict"]
    )

    generator.eval()

    noise = torch.randn(
        num_images,
        LATENT_DIM,
        1,
        1,
        device=DEVICE,
    )

    with torch.no_grad():

        generated_images = generator(
            noise
        )

    output_path = (
        RESULTS_DIR /
        "evaluation_samples.png"
    )

    save_image(
        generated_images,
        output_path,
        normalize=True,
        nrow=8,
    )

    print("Evaluation completed.")
    print(
        "Generated samples:",
        output_path,
    )


if __name__ == "__main__":
    evaluate_generator()