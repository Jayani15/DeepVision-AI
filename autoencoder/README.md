# Convolutional Denoising Autoencoder

## Overview

This module implements a Convolutional Denoising Autoencoder for removing Gaussian noise from images.

The model is trained using the CIFAR-10 dataset and learns to reconstruct clean images from noisy inputs.

## Pipeline

```text
CIFAR-10 Image
      ↓
Gaussian Noise
      ↓
Noisy Image
      ↓
Convolutional Encoder
      ↓
Latent Representation
      ↓
Convolutional Decoder
      ↓
Denoised Image
```

## Dataset

CIFAR-10 is used for training, validation and testing.

- Image size: 32 × 32 pixels, RGB, 3 channels
- Dataset split:
Training: 45,000 images
Validation: 5,000 images
Testing: 10,000 images

The CIFAR-10 dataset is not included in this repository.

## Model

The model consists of:

### Encoder
- Convolutional layers
- ReLU activation
- Max pooling
- Feature extraction
- Latent representation
### Decoder
- Transposed convolutional layers
- ReLU activation
- Reconstruction layer
- Sigmoid output

The encoder compresses the noisy image into a latent representation, while the decoder reconstructs the clean image.

##Training

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 64
- Training epochs: 20
- Noise factor: 0.2
- Evaluation Metrics
The model is evaluated using:
Mean Squared Error (MSE)
Peak Signal-to-Noise Ratio (PSNR)
Structural Similarity Index (SSIM)
### Final Results
| Metric	| Noisy Input	| Denoised Output |
| -------- | -------- | -------- |
| MSE	| 0.033315 |	0.004302 |
| PSNR	| 14.79 dB |	23.90 dB |
| SSIM | 	0.4300 | 	0.7998 |

The denoising model significantly reduces reconstruction error while improving image quality and structural similarity.

### Noise Robustness Experiment

The trained model was evaluated with different Gaussian noise levels.

| Noise Factor | 	Noisy MSE |	Denoised MSE	Noisy | PSNR	| Denoised PSNR |
| -------- | -------- | -------- | -------- | -------- |
| 0.1 |	0.009303 |	0.003383 |	20.33 dB | 25.01 dB |
| 0.2	| 0.033290 |	0.004296 |	14.79 dB |	23.91 dB |
|0.3	| 0.064205| 	0.007456 |	11.94 dB |	21.40 dB |

The model improves PSNR across all tested noise levels, demonstrating robustness to different levels of Gaussian noise.

## Running the Project
#### 1. Install dependencies

```text

pip install torch torchvision matplotlib pillow scikit-image

```

#### 2. Prepare CIFAR-10

Download and extract the CIFAR-10 Python dataset.
Place the extracted folder as:
autoencoder/cifar-10-batches-py/

#### 3. Train
python train.py

#### 4. Evaluate
python evaluate.py

#### 5. Run inference
python inference.py

#### 6. Run noise robustness experiment
python noise_experiment.py

### Team Integration

The trained model can be integrated into the main project through the ImageDenoiser class.

Example:

```text

from PIL import Image
from denoiser import ImageDenoiser

denoiser = ImageDenoiser()

image = Image.open("input.png")

result = denoiser.denoise_image(image)

result.save("output.png")

```

#### Contributor
Jayani

#### Contribution:

- CIFAR-10 preprocessing
- Gaussian noise generation
- Convolutional denoising autoencoder
- Model training
- MSE, PSNR and SSIM evaluation
- Noise robustness analysis
- Inference pipeline
- Model integration interface