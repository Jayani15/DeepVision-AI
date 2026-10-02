import math
import torch
import torch.nn.functional as F

from dataset import get_dataloaders
from model import DenoisingAutoencoder


# ============================================================
# CONFIGURATION
# ============================================================

NOISE_LEVELS = [0.1, 0.2, 0.3]

BATCH_SIZE = 64

MODEL_PATH = "checkpoints/best_autoencoder.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = DenoisingAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Trained model loaded successfully!")


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(mse):

    if mse == 0:
        return float("inf")

    return 10 * math.log10(
        1.0 / mse
    )


# ============================================================
# EXPERIMENT
# ============================================================

print("\n==============================================")
print("       NOISE LEVEL ROBUSTNESS EXPERIMENT")
print("==============================================")

print(
    "\nNoise Factor | Noisy MSE | Denoised MSE | "
    "Noisy PSNR | Denoised PSNR"
)

print("-" * 75)


results = []


for noise_factor in NOISE_LEVELS:

    # --------------------------------------------------------
    # Create test loader with current noise level
    # --------------------------------------------------------

    _, _, test_loader = get_dataloaders(
        data_dir="./",
        batch_size=BATCH_SIZE,
        noise_factor=noise_factor
    )


    total_noisy_mse = 0.0
    total_denoised_mse = 0.0

    total_noisy_psnr = 0.0
    total_denoised_psnr = 0.0

    total_images = 0


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    with torch.no_grad():

        for noisy_images, clean_images in test_loader:

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Autoencoder prediction
            outputs = model(
                noisy_images
            )


            batch_size = noisy_images.size(0)


            for i in range(batch_size):

                # ------------------------------------------
                # Noisy image MSE
                # ------------------------------------------

                noisy_mse = F.mse_loss(
                    noisy_images[i],
                    clean_images[i]
                ).item()


                # ------------------------------------------
                # Denoised image MSE
                # ------------------------------------------

                denoised_mse = F.mse_loss(
                    outputs[i],
                    clean_images[i]
                ).item()


                # ------------------------------------------
                # PSNR
                # ------------------------------------------

                noisy_psnr = calculate_psnr(
                    noisy_mse
                )

                denoised_psnr = calculate_psnr(
                    denoised_mse
                )


                total_noisy_mse += noisy_mse
                total_denoised_mse += denoised_mse

                total_noisy_psnr += noisy_psnr
                total_denoised_psnr += denoised_psnr

                total_images += 1


    # --------------------------------------------------------
    # Average results
    # --------------------------------------------------------

    avg_noisy_mse = (
        total_noisy_mse /
        total_images
    )

    avg_denoised_mse = (
        total_denoised_mse /
        total_images
    )

    avg_noisy_psnr = (
        total_noisy_psnr /
        total_images
    )

    avg_denoised_psnr = (
        total_denoised_psnr /
        total_images
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "noise_factor": noise_factor,
        "noisy_mse": avg_noisy_mse,
        "denoised_mse": avg_denoised_mse,
        "noisy_psnr": avg_noisy_psnr,
        "denoised_psnr": avg_denoised_psnr
    })


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"{noise_factor:^12.1f} | "
        f"{avg_noisy_mse:^10.6f} | "
        f"{avg_denoised_mse:^13.6f} | "
        f"{avg_noisy_psnr:^11.2f} | "
        f"{avg_denoised_psnr:^14.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/noise_level_results.txt",
    "w"
) as file:

    file.write(
        "NOISE LEVEL ROBUSTNESS EXPERIMENT\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        "Noise Factor | Noisy MSE | Denoised MSE | "
        "Noisy PSNR | Denoised PSNR\n"
    )

    for result in results:

        file.write(
            f"{result['noise_factor']:.1f} | "
            f"{result['noisy_mse']:.6f} | "
            f"{result['denoised_mse']:.6f} | "
            f"{result['noisy_psnr']:.2f} | "
            f"{result['denoised_psnr']:.2f}\n"
        )


print("\n==============================================")
print("Experiment completed!")
print(
    "Results saved to "
    "results/noise_level_results.txt"
)
print("==============================================")