import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import CNN
from dataLoader import get_data_loaders
from train import train
from evaluate import evaluate
from utils import save_model


def main():

    torch.set_num_threads(os.cpu_count())
    device = torch.device("cpu")

    train_loader, val_loader, test_loader = get_data_loaders()

    learning_rates = [0.1, 0.01, 0.001]
    epochs = 3   # keep small for tuning speed

    best_val_acc = 0
    best_lr = None
    best_model_state = None

    print("\nStarting CNN Hyperparameter Tuning...\n")

    for lr in learning_rates:

        print(f"\nTesting Learning Rate: {lr}")

        model = CNN().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

        for epoch in range(epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
            detailed=False
        )

        print(f"Validation Accuracy for LR={lr}: {val_accuracy:.2f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_lr = lr
            best_model_state = model.state_dict()

    print("\nBest Learning Rate Found:", best_lr)
    print("Best Validation Accuracy:", best_val_acc)

    # 🔥 Load Best Model
    best_model = CNN().to(device)
    best_model.load_state_dict(best_model_state)

    # Final evaluation
    test_loss, test_accuracy = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        detailed=True
    )

    print("\nFinal Test Accuracy:", test_accuracy)

    save_model(best_model)


if __name__ == "__main__":
    main()