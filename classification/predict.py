import torch
from PIL import Image
from torchvision import transforms

from models.cnn import CNNClassifier


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/cnn_final.pth"


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Image Preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# Load Model
# ============================================================

model = CNNClassifier(
    dropout_rate=0.5
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


# ============================================================
# Prediction Function
# ============================================================

def predict_image(image_path):

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")


    # Resize to CIFAR-10 dimensions
    image = image.resize(
        (32, 32)
    )


    # Apply preprocessing
    image_tensor = transform(
        image
    )


    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predicted = torch.max(
            probabilities,
            1
        )


    predicted_class = CLASS_NAMES[
        predicted.item()
    ]


    confidence_value = (
        confidence.item() * 100
    )


    return (
        predicted_class,
        confidence_value
    )


# ============================================================
# Run from Command Line
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )


    prediction, confidence = predict_image(
        image_path
    )


    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    print(
        f"Class      : {prediction}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )