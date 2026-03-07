import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================
# 1. Load Housing Data
# ===================================

train_df = pd.read_csv("housing_train.csv")
test_df = pd.read_csv("housing_test.csv")

# Separate features and target
X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]

X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# ===================================
# 2. Base Linear Regression Model
# ===================================

base_features = [
    "Bedrooms",
    "Bathrooms",
    "Floor Area",
    "Land Area"
]

X_train_base = X_train[base_features]
X_test_base = X_test[base_features]

base_model = LinearRegression()
base_model.fit(X_train_base, y_train)

y_pred_base = base_model.predict(X_test_base)

# ===================================
# 3. Improved Linear Regression Model
# ===================================

improved_model = LinearRegression()
improved_model.fit(X_train, y_train)

y_pred_improved = improved_model.predict(X_test)

# ===================================
# 4. Evaluation Function
# ===================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n", name)
    print("-----------------------")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

# ===================================
# 5. Compare Models
# ===================================

evaluate_model("Base Linear Regression", y_test, y_pred_base)
evaluate_model("Improved Linear Regression", y_test, y_pred_improved)

# ===================================
# 6. Scatter Plot (Actual vs Predicted)
# ===================================

plt.figure()

plt.scatter(y_test, y_pred_improved)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Housing Prices")

plt.show()

# ===================================
# 7. Residual Plot
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
# 8. Correlation Heatmap
# ===================================

plt.figure(figsize=(10,8))

sns.heatmap(train_df.corr(), cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()

# ===================================
# 9. Regression Plot Example
# ===================================

plt.figure()

sns.regplot(x=train_df["Floor Area"], y=train_df["Price"])

plt.title("Floor Area vs Housing Price")

plt.show()