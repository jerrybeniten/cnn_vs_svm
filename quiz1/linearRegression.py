import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===================================
# 1. Load Training and Testing Data
# ===================================

train_df = pd.read_csv("housing_train.csv")
test_df = pd.read_csv("housing_test.csv")

# Separate features and target

X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]

X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# ===================================
# 2. Train Linear Regression Model
# ===================================

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training complete.")

# ===================================
# 3. Make Predictions
# ===================================

y_pred = model.predict(X_test)

# ===================================
# 4. Evaluate Model Performance
# ===================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("RMSE:", rmse)
print("R² Score:", r2)

# ===================================
# 5. Interpret Model Coefficients
# ===================================

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients")
print(coefficients)

print("\nIntercept:", model.intercept_)

# ===================================
# 6. Assumption 1: Linearity
# ===================================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# ===================================
# 7. Assumption 2: Residual Analysis
# ===================================

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Homoscedasticity Check)")
plt.show()

# ===================================
# 8. Assumption 3: Normality of Residuals
# ===================================

plt.figure()
plt.hist(residuals, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

print("\nResidual Mean:", np.mean(residuals))
print("Residual Std:", np.std(residuals))