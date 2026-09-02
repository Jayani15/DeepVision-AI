import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import DenoisingAutoencoder
from dataset import add_gaussian_noise


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "checkpoints/best_autoencoder.pth"

IMAGE_PATH = "test_images/test_image.png"

OUTPUT_PATH = "results/inference_result.png"

NOISE_FACTOR = 0.2


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD MODEL
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

print("Model loaded successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


print(
    "Original image size:",
    image.size
)


# ============================================================
# PREPROCESS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (32, 32)
    ),

    transforms.ToTensor()

])


clean_image = transform(
    image
)


# ============================================================
# ADD NOISE
# ============================================================

noisy_image = add_gaussian_noise(
    clean_image,
    noise_factor=NOISE_FACTOR
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

input_image = noisy_image.unsqueeze(
    0
).to(device)


# ============================================================
# MODEL INFERENCE
# ============================================================

with torch.no_grad():

    denoised_image = model(
        input_image
    )


# ============================================================
# REMOVE BATCH DIMENSION
# ============================================================

denoised_image = denoised_image.squeeze(
    0
).cpu()


# ============================================================
# VISUALIZATION
# ============================================================

clean_np = clean_image.permute(
    1, 2, 0
).numpy()

noisy_np = noisy_image.permute(
    1, 2, 0
).numpy()

denoised_np = denoised_image.permute(
    1, 2, 0
).numpy()


# ============================================================
# CREATE RESULT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)


# Clean

axes[0].imshow(clean_np)

axes[0].set_title(
    "Original / Clean"
)

axes[0].axis("off")


# Noisy

axes[1].imshow(noisy_np)

axes[1].set_title(
    "Noisy Input"
)

axes[1].axis("off")


# Denoised

axes[2].imshow(denoised_np)

axes[2].set_title(
    "Denoised Output"
)

axes[2].axis("off")


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print(
    "\nInference completed successfully!"
)

print(
    "Result saved to:",
    OUTPUT_PATH
)