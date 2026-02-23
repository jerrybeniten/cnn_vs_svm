import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_svm(model, scaler, X, y):

    X = scaler.transform(X)
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y, predictions))

    cm = confusion_matrix(y, predictions)

    print("Confusion Matrix:\n")
    print(cm)

    # Plot Confusion Matrix
    plot_confusion_matrix(cm)

    # Save classification report to CSV
    report_dict = classification_report(
        y,
        predictions,
        output_dict=True
    )

    df = pd.DataFrame(report_dict).transpose()
    df = df.round(4)
    df.to_csv("svm_classification_report.csv")

    print("\nSVM classification report saved to svm_classification_report.csv")

    return accuracy


def plot_confusion_matrix(cm):

    plt.figure(figsize=(8, 6))

    # Grayscale heatmap
    plt.imshow(cm, cmap="gray")
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    # Draw grid lines for each square
    plt.gca().set_xticks(np.arange(-.5, 10, 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, 10, 1), minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

    plt.colorbar()

    # Print values in all cells
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
