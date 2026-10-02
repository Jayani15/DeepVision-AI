import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import DenoisingAutoencoder
from skimage.metrics import structural_similarity as ssim


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(prediction, target):

    mse = F.mse_loss(
        prediction,
        target
    ).item()

    if mse == 0:
        return float("inf")

    psnr = 10 * math.log10(
        1.0 / mse
    )

    return psnr


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(prediction, target):
    """
    Calculate average SSIM across a batch of RGB images.
    """

    prediction = prediction.detach().cpu()
    target = target.detach().cpu()

    total_ssim = 0.0

    for i in range(prediction.size(0)):

        # Convert:
        # [C, H, W]
        #
        # to:
        # [H, W, C]

        pred_image = prediction[i].permute(
            1, 2, 0
        ).numpy()

        target_image = target[i].permute(
            1, 2, 0
        ).numpy()

        score = ssim(
            target_image,
            pred_image,
            data_range=1.0,
            channel_axis=2
        )

        total_ssim += score

    return total_ssim / prediction.size(0)


# ============================================================
# DATASET
# ============================================================

_, _, test_loader = get_dataloaders(
    data_dir="./",
    batch_size=64,
    noise_factor=0.2
)


# ============================================================
# MODEL
# ============================================================

model = DenoisingAutoencoder()

model.load_state_dict(
    torch.load(
        "checkpoints/best_autoencoder.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# METRIC ACCUMULATORS
# ============================================================

total_output_mse = 0.0
total_noisy_mse = 0.0

total_output_psnr = 0.0
total_noisy_psnr = 0.0

total_output_ssim = 0.0
total_noisy_ssim = 0.0

total_images = 0


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for noisy_images, clean_images in test_loader:

        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        # ------------------------------------------
        # Autoencoder prediction
        # ------------------------------------------

        outputs = model(
            noisy_images
        )


        # ------------------------------------------
        # MSE
        # ------------------------------------------

        output_mse = F.mse_loss(
            outputs,
            clean_images,
            reduction="sum"
        ).item()

        noisy_mse = F.mse_loss(
            noisy_images,
            clean_images,
            reduction="sum"
        ).item()


        # ------------------------------------------
        # PSNR
        # ------------------------------------------

        batch_output_psnr = 0.0
        batch_noisy_psnr = 0.0

        batch_size = noisy_images.size(0)

        for i in range(batch_size):

            individual_output_mse = F.mse_loss(
                outputs[i],
                clean_images[i]
            ).item()

            individual_noisy_mse = F.mse_loss(
                noisy_images[i],
                clean_images[i]
            ).item()

            batch_output_psnr += (
                10 * math.log10(
                    1.0 / individual_output_mse
                )
            )

            batch_noisy_psnr += (
                10 * math.log10(
                    1.0 / individual_noisy_mse
                )
            )


        # ------------------------------------------
        # SSIM
        # ------------------------------------------

        batch_output_ssim = calculate_ssim(
            outputs,
            clean_images
        )

        batch_noisy_ssim = calculate_ssim(
            noisy_images,
            clean_images
        )


        # ------------------------------------------
        # Accumulate
        # ------------------------------------------

        total_output_mse += output_mse
        total_noisy_mse += noisy_mse

        total_output_psnr += batch_output_psnr
        total_noisy_psnr += batch_noisy_psnr

        total_output_ssim += (
            batch_output_ssim * batch_size
        )

        total_noisy_ssim += (
            batch_noisy_ssim * batch_size
        )

        total_images += batch_size


# ============================================================
# FINAL AVERAGES
# ============================================================

# MSE:
# Divide total squared error by number
# of all pixels in all images.

num_pixels_per_image = 3 * 32 * 32

average_output_mse = (
    total_output_mse /
    (total_images * num_pixels_per_image)
)

average_noisy_mse = (
    total_noisy_mse /
    (total_images * num_pixels_per_image)
)


# PSNR

average_output_psnr = (
    total_output_psnr /
    total_images
)

average_noisy_psnr = (
    total_noisy_psnr /
    total_images
)


# SSIM

average_output_ssim = (
    total_output_ssim /
    total_images
)

average_noisy_ssim = (
    total_noisy_ssim /
    total_images
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("      DENOISING AUTOENCODER RESULTS")
print("==========================================")

print("\nMSE")
print("------------------------------------------")

print(
    f"Noisy Image MSE     : "
    f"{average_noisy_mse:.6f}"
)

print(
    f"Denoised Image MSE  : "
    f"{average_output_mse:.6f}"
)


print("\nPSNR")
print("------------------------------------------")

print(
    f"Noisy Image PSNR    : "
    f"{average_noisy_psnr:.2f} dB"
)

print(
    f"Denoised Image PSNR : "
    f"{average_output_psnr:.2f} dB"
)


print("\nSSIM")
print("------------------------------------------")

print(
    f"Noisy Image SSIM    : "
    f"{average_noisy_ssim:.4f}"
)

print(
    f"Denoised Image SSIM : "
    f"{average_output_ssim:.4f}"
)


print("\n==========================================")


# ============================================================
# SAVE METRICS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    "results/metrics.txt",
    "w"
) as file:

    file.write(
        "DENOISING AUTOENCODER RESULTS\n"
    )

    file.write(
        "============================\n\n"
    )

    file.write(
        "MSE\n"
    )

    file.write(
        f"Noisy MSE: "
        f"{average_noisy_mse:.6f}\n"
    )

    file.write(
        f"Denoised MSE: "
        f"{average_output_mse:.6f}\n\n"
    )

    file.write(
        "PSNR\n"
    )

    file.write(
        f"Noisy PSNR: "
        f"{average_noisy_psnr:.2f} dB\n"
    )

    file.write(
        f"Denoised PSNR: "
        f"{average_output_psnr:.2f} dB\n\n"
    )

    file.write(
        "SSIM\n"
    )

    file.write(
        f"Noisy SSIM: "
        f"{average_noisy_ssim:.4f}\n"
    )

    file.write(
        f"Denoised SSIM: "
        f"{average_output_ssim:.4f}\n"
    )


# ============================================================
# VISUAL RESULTS
# ============================================================

noisy_images, clean_images = next(
    iter(test_loader)
)

noisy_images = noisy_images.to(device)

with torch.no_grad():

    outputs = model(
        noisy_images
    )


# Move to CPU

noisy_images = noisy_images.cpu()
clean_images = clean_images.cpu()
outputs = outputs.cpu()


# Number of examples

num_images = 6


fig, axes = plt.subplots(
    3,
    num_images,
    figsize=(12, 6)
)


for i in range(num_images):

    clean = clean_images[i].permute(
        1, 2, 0
    ).numpy()

    noisy = noisy_images[i].permute(
        1, 2, 0
    ).numpy()

    denoised = outputs[i].permute(
        1, 2, 0
    ).numpy()


    # Clean

    axes[0, i].imshow(clean)
    axes[0, i].axis("off")


    # Noisy

    axes[1, i].imshow(noisy)
    axes[1, i].axis("off")


    # Denoised

    axes[2, i].imshow(denoised)
    axes[2, i].axis("off")


# Row labels

axes[0, 0].set_ylabel(
    "Clean",
    fontsize=12
)

axes[1, 0].set_ylabel(
    "Noisy",
    fontsize=12
)

axes[2, 0].set_ylabel(
    "Denoised",
    fontsize=12
)


plt.tight_layout()


plt.savefig(
    "results/denoising_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print(
    "\nResults saved to results/"
)