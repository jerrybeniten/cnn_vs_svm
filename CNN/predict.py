import torch
from PIL import Image
import torchvision.transforms as transforms

from model import CNN


def load_model(model_path="mnist_cnn.pth"):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # grayscale

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension

    return image


def predict(image_path):

    model = load_model()
    image = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if __name__ == "__main__":
    image_path = "six-3.png"
    prediction = predict(image_path)
    print("Predicted Digitx:", prediction)
