from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

LAYERS = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",
    "28": "conv5_1",
}

STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
CONTENT_LAYER = "conv4_2"


def load_image(path, max_size):
    image = Image.open(path).convert("RGB")
    scale = min(max_size / max(image.size), 1)
    size = (int(image.width * scale), int(image.height * scale))
    image = image.resize(size, Image.Resampling.LANCZOS)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ]
    )
    return transform(image).unsqueeze(0).to(DEVICE)


def save_image(tensor, output_path):
    image = tensor.detach().cpu().squeeze(0)
    image = (image * STD + MEAN).clamp(0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(image).save(output_path)


def gram_matrix(feature_map):
    batch, channels, height, width = feature_map.size()
    features = feature_map.view(batch, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)


def extract_features(image, model):
    features = {}

    for layer_number, layer in model._modules.items():
        image = layer(image)

        if layer_number in LAYERS:
            features[LAYERS[layer_number]] = image

    return features


def main():
    parser = ArgumentParser(description="Neural Style Transfer using VGG19")
    parser.add_argument("content_image")
    parser.add_argument("style_image")
    parser.add_argument("--output", default="results/stylized_output.jpg")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--style-weight", type=float, default=1_000_000)
    parser.add_argument("--content-weight", type=float, default=1.0)
    args = parser.parse_args()

    content = load_image(args.content_image, args.size)
    style = load_image(args.style_image, args.size)

    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        content_features = extract_features(content, model)
        style_features = extract_features(style, model)
        style_grams = {
            layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS
        }

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=0.02)

    print(f"Using device: {DEVICE}")
    print("Creating stylized image...")

    for step in range(1, args.steps + 1):
        generated_features = extract_features(generated, model)

        content_loss = torch.mean(
            (generated_features[CONTENT_LAYER] - content_features[CONTENT_LAYER]) ** 2
        )

        style_loss = sum(
            torch.mean((gram_matrix(generated_features[layer]) - style_grams[layer]) ** 2)
            for layer in STYLE_LAYERS
        )

        total_loss = (
            args.content_weight * content_loss
            + args.style_weight * style_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(-3, 3)

        if step == 1 or step % 25 == 0 or step == args.steps:
            print(
                f"Step {step}/{args.steps} | "
                f"content loss: {content_loss.item():.4f} | "
                f"style loss: {style_loss.item():.6f}"
            )

    output_path = Path(args.output)
    save_image(generated, output_path)
    print(f"Saved stylized image to: {output_path}")


if __name__ == "__main__":
    main()
