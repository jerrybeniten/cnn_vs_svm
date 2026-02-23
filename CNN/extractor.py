import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


def main():

    # ==============================
    # Create directory if not exist
    # ==============================
    output_dir = "extractedImages"
    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST
    mnist_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True
    )

    # Select random sample
    random_index = random.randint(0, len(mnist_dataset) - 1)
    image, label = mnist_dataset[random_index]

    image_array = np.array(image)

    print("Random Index:", random_index)
    print("Label:", label)
    print("Original Size:", image.size)  # Should be (28, 28)

    # ===============================
    # 1️⃣ Save ORIGINAL 28×28 image
    # ===============================
    original_filename = os.path.join(
        output_dir,
        f"mnist_original_{random_index}.png"
    )
    image.save(original_filename)

    print("Saved original image as:", original_filename)

    # ======================================
    # 2️⃣ Create ENLARGED version with GRID
    # ======================================
    plt.figure(figsize=(6, 6))

    plt.imshow(image_array, cmap="gray", interpolation="nearest")

    plt.xticks(np.arange(-0.5, 28, 1))
    plt.yticks(np.arange(-0.5, 28, 1))

    plt.grid(True)
    plt.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False
    )

    plt.title(f"MNIST Digit (Label = {label})")

    enlarged_filename = os.path.join(
        output_dir,
        f"mnist_enlarged_with_grid_{random_index}.png"
    )

    plt.savefig(enlarged_filename, bbox_inches="tight")
    plt.close()

    print("Saved enlarged image with grid as:", enlarged_filename)


if __name__ == "__main__":
    main()