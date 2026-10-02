from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

DATASET_NAME = "CIFAR10"

LATENT_DIM = 100

GENERATOR_FEATURES = 64
DISCRIMINATOR_FEATURES = 64

BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 0.0002

BETA1 = 0.5
BETA2 = 0.999

NUM_WORKERS = 0

NUM_GENERATED_IMAGES = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

GENERATOR_CHECKPOINT = (
    CHECKPOINT_DIR / "generator.pth"
)

DISCRIMINATOR_CHECKPOINT = (
    CHECKPOINT_DIR / "discriminator.pth"
)

TRAINING_HISTORY = (
    RESULTS_DIR / "training_history.pth"
)