import torch
from PIL import Image
from torchvision import transforms

from model import DenoisingAutoencoder
from dataset import add_gaussian_noise


MODEL_PATH = "checkpoints/best_autoencoder.pth"


class ImageDenoiser:

    def __init__(
        self,
        model_path=MODEL_PATH,
        noise_factor=0.2
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.noise_factor = noise_factor

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        self.model = DenoisingAutoencoder()

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])


        # ----------------------------------------------------
        # Image postprocessing
        # ----------------------------------------------------

        self.to_pil = transforms.ToPILImage()


    # ========================================================
    # DENOISE IMAGE
    # ========================================================

    def denoise_image(
        self,
        image,
        add_noise=False
    ):

        """
        Denoise an input image.

        Parameters
        ----------
        image : PIL.Image
            Input RGB image.

        add_noise : bool
            If True, artificial Gaussian noise is added
            before denoising.

            True  -> Demo/testing mode
            False -> Real application mode

        Returns
        -------
        PIL.Image
            Denoised image.
        """


        # ----------------------------------------------------
        # Ensure RGB
        # ----------------------------------------------------

        image = image.convert("RGB")


        # ----------------------------------------------------
        # Resize + convert to tensor
        # ----------------------------------------------------

        image_tensor = self.transform(
            image
        )


        # ----------------------------------------------------
        # Optional noise
        # ----------------------------------------------------

        if add_noise:

            image_tensor = add_gaussian_noise(
                image_tensor,
                self.noise_factor
            )


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        input_tensor = image_tensor.unsqueeze(
            0
        ).to(self.device)


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            output = self.model(
                input_tensor
            )


        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        output = output.squeeze(
            0
        ).cpu()


        # ----------------------------------------------------
        # Clamp pixel values
        # ----------------------------------------------------

        output = torch.clamp(
            output,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Convert tensor → PIL
        # ----------------------------------------------------

        output = self.to_pil(
            output
        )


        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    denoiser = ImageDenoiser()


    image = Image.open(
        "test_images/test_image.png"
    )


    # --------------------------------------------------------
    # DEMO MODE
    # Clean → Noise → Denoise
    # --------------------------------------------------------

    demo_result = denoiser.denoise_image(
        image,
        add_noise=True
    )

    demo_result.save(
        "results/denoiser_demo.png"
    )


    # --------------------------------------------------------
    # DIRECT MODE
    # Already noisy → Denoise
    # --------------------------------------------------------

    direct_result = denoiser.denoise_image(
        image,
        add_noise=False
    )

    direct_result.save(
        "results/denoiser_direct.png"
    )


    print(
        "Denoising tests completed successfully!"
    )

    print(
        "Demo result   : results/denoiser_demo.png"
    )

    print(
        "Direct result : results/denoiser_direct.png"
    )