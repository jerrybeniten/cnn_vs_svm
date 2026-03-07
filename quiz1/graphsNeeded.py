import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================
# Load Data
# ===================================

train_df = pd.read_csv("housing_train.csv")
test_df = pd.read_csv("housing_test.csv")

X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]

X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# ===================================
# Train Models
# ===================================

# Base model
base_features = ["Bedrooms", "Bathrooms", "Floor Area", "Land Area"]

X_train_base = X_train[base_features]
X_test_base = X_test[base_features]

base_model = LinearRegression()
base_model.fit(X_train_base, y_train)

y_pred_base = base_model.predict(X_test_base)

# Improved model
improved_model = LinearRegression()
improved_model.fit(X_train, y_train)

y_pred_improved = improved_model.predict(X_test)

# ===================================
# Evaluation Metrics
# ===================================

mae_base = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)

mae_improved = mean_absolute_error(y_test, y_pred_improved)
rmse_improved = np.sqrt(mean_squared_error(y_test, y_pred_improved))
r2_improved = r2_score(y_test, y_pred_improved)

# ===================================
# Figure 1 — Price Distribution
# ===================================

plt.figure()

plt.hist(train_df["Price"], bins=30)

plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()

# ===================================
# Figure 2 — Correlation Heatmap
# ===================================

plt.figure(figsize=(10,8))

sns.heatmap(train_df.corr(), cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()

# ===================================
# Figure 3 — Regression Plot
# ===================================

plt.figure()

sns.regplot(x=train_df["Floor Area"], y=train_df["Price"])

plt.title("Floor Area vs Housing Price")

plt.show()

# ===================================
# Figure 5 — Model Performance Comparison
# ===================================

metrics = ["MAE", "RMSE", "R2"]

base_scores = [mae_base, rmse_base, r2_base]
improved_scores = [mae_improved, rmse_improved, r2_improved]

x = np.arange(len(metrics))

plt.figure()

plt.bar(x - 0.2, base_scores, 0.4, label="Base Model")
plt.bar(x + 0.2, improved_scores, 0.4, label="Improved Model")

plt.xticks(x, metrics)

plt.title("Model Performance Comparison")

plt.legend()

plt.show()

# ===================================
# Figure 6 — Actual vs Predicted
# ===================================

plt.figure()

plt.scatter(y_test, y_pred_improved)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Housing Prices")

plt.show()

# ===================================
# Figure 7 — Residual Plot
# ===================================

residuals = y_test - y_pred_improved

plt.figure()

plt.scatter(y_pred_improved, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()

# ===================================
# Figure 8 — Residual Histogram
# ===================================

plt.figure()

plt.hist(residuals, bins=30)

plt.title("Residual Distribution")

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.show()

# ===================================
# Figure 9 — Feature Importance
# ===================================

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": improved_model.coef_
})

coefficients = coefficients.sort_values(by="Coefficient")

plt.figure(figsize=(8,6))

plt.barh(coefficients["Feature"], coefficients["Coefficient"])

plt.title("Linear Regression Feature Importance")

plt.xlabel("Coefficient Value")

plt.show()