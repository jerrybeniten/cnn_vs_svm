import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, loader, criterion, device, detailed=False):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    if detailed:

        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_preds))

        cm = confusion_matrix(all_labels, all_preds)

        print("Confusion Matrix:\n")
        print(cm)

        # Plot Confusion Matrix
        plot_confusion_matrix(cm)

        # Save CSV (already implemented)
        report_dict = classification_report(
            all_labels,
            all_preds,
            output_dict=True
        )

        df = pd.DataFrame(report_dict).transpose()
        df = df.round(4)
        df.to_csv("cnn_classification_report.csv")

        print("\nCNN classification report saved to cnn_classification_report.csv")

    return avg_loss, accuracy


def plot_confusion_matrix(cm):

    plt.figure(figsize=(8, 6))

    # Grayscale heatmap
    plt.imshow(cm, cmap="gray")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    # Draw grid lines for all squares
    plt.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    plt.gca().set_xticks(np.arange(-.5, 10, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, 10, 1), minor=True)

    plt.colorbar()

    # 🔥 Always show numbers in every cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()
