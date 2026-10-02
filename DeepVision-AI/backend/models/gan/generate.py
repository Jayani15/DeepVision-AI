import torch

from .config import (
    DEVICE,
    LATENT_DIM,
    GENERATOR_CHECKPOINT,
    RESULTS_DIR,
)

from .generator import Generator

from torchvision.utils import save_image


def generate_images(
    num_images=16,
    filename="generated_images.png",
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

        images = generator(noise)

    output_path = RESULTS_DIR / filename

    save_image(
        images,
        output_path,
        normalize=True,
        nrow=4,
    )

    print(
        f"Generated {num_images} images."
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":

    generate_images(
        num_images=16
    )