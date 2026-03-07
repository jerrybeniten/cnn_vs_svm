import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================
# 1. Load Housing Data
# ===================================

train_df = pd.read_csv("housing_train.csv")
test_df = pd.read_csv("housing_test.csv")

# ===================================
# 2. Separate Features and Target
# ===================================

X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]

X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# ===================================
# 3. Train Linear Regression Model
# ===================================

model = LinearRegression()
model.fit(X_train, y_train)

# ===================================
# 4. Generate Predictions
# ===================================

y_pred = model.predict(X_test)

# ===================================
# 5. Evaluation Metrics
# ===================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics")
print("------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# ===================================
# 6. Residual Analysis
# ===================================

residuals = y_test - y_pred

# Residual vs Predicted Plot
plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

# Residual Distribution
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Actual vs Predicted Plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Residual Summary
print("\nResidual Statistics")
print("-------------------")
print("Mean Residual:", np.mean(residuals))
print("Std Residual :", np.std(residuals))