import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def get_svm_data():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Flatten images (28x28 → 784)
    X = dataset.data.numpy().reshape(-1, 784)
    y = dataset.targets.numpy()

    X_test = test_dataset.data.numpy().reshape(-1, 784)
    y_test = test_dataset.targets.numpy()

    # Create validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test, y_test
