from svm_data_loader import get_svm_data
from svm_train import train_svm_with_tuning
from svm_evaluate import evaluate_svm


def main():

    X_train, X_val, y_train, y_val, X_test, y_test = get_svm_data()

    # Combine train + validation for final tuning
    import numpy as np
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))

    model, scaler = train_svm_with_tuning(X_combined, y_combined)

    print("\nFinal Test Performance")
    test_accuracy = evaluate_svm(model, scaler, X_test, y_test)

    print("\nFinal Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()