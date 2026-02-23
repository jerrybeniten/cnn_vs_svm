# svm_train.py

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_svm_with_tuning(X_train, y_train):

    # Subsample for faster tuning (very important)
    sample_size = 20000   # reduce from full 60k
    X_train = X_train[:sample_size]
    y_train = y_train[:sample_size]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Smaller hyperparameter grid (faster)
    param_grid = {
        "C": [1, 5],
        "gamma": ["scale"],
        "kernel": ["rbf"]
    }

    base_model = SVC()

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=2,               # 🔹 reduced from 3 to 2 folds
        scoring="accuracy",
        n_jobs=-1,
        verbose=3
    )

    print("\nRunning Fast Grid Search with 2-Fold Cross-Validation...\n")

    grid_search.fit(X_train, y_train)

    print("\nGrid Search Complete.")
    print("Best Parameters Found:")
    print(grid_search.best_params_)

    print("Best Cross-Validation Accuracy:",
          grid_search.best_score_)

    best_model = grid_search.best_estimator_

    return best_model, scaler