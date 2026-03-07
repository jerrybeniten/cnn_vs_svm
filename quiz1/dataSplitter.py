import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ===================================
# 1. Load Dataset
# ===================================

df = pd.read_csv("Housing_v2.csv")

print("Initial Shape:", df.shape)
print(df.info())

# ===================================
# 2. Basic EDA
# ===================================

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ===================================
# 3. Handle Missing Values
# ===================================

df = df.dropna(subset=["Price"])

numeric_columns = [
    "Bedrooms",
    "Bathrooms",
    "Floor Area",
    "Land Area",
    "Latitude",
    "Longitude"
]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

df["Location"] = df["Location"].fillna("Unknown")

# ===================================
# 4. Remove Location Column
# (City encoding no longer needed)
# ===================================

df = df.drop(columns=["Location"])

# ===================================
# 5. Handle Outliers (IQR Method)
# ===================================

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

print("Shape after removing outliers:", df.shape)

# ===================================
# 6. Feature Engineering
# ===================================

df["price_per_sqm"] = df["Price"] / df["Floor Area"]
df["total_rooms"] = df["Bedrooms"] + df["Bathrooms"]
df["floor_land_ratio"] = df["Floor Area"] / df["Land Area"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# ===================================
# 7. Distance Feature Engineering
# ===================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Anchor locations

MAKATI = (14.5547, 121.0244)
BGC = (14.5515, 121.0481)
ORTIGAS = (14.5869, 121.0610)
QUEZON_CITY = (14.6760, 121.0437)
ALABANG = (14.4181, 121.0383)
ANTIPOLO = (14.6255, 121.1245)

df["distance_to_makati_cbd"] = haversine(df["Latitude"], df["Longitude"], *MAKATI)
df["distance_to_bgc"] = haversine(df["Latitude"], df["Longitude"], *BGC)
df["distance_to_ortigas"] = haversine(df["Latitude"], df["Longitude"], *ORTIGAS)
df["distance_to_quezon_city"] = haversine(df["Latitude"], df["Longitude"], *QUEZON_CITY)
df["distance_to_alabang"] = haversine(df["Latitude"], df["Longitude"], *ALABANG)
df["distance_to_antipolo"] = haversine(df["Latitude"], df["Longitude"], *ANTIPOLO)

# Optional: remove raw coordinates

df = df.drop(columns=["Latitude", "Longitude"])

# ===================================
# 8. Feature Selection
# ===================================

X = df.drop(columns=["Price", "Description"])
y = df["Price"]

print("\nSelected Features:")
print(X.columns)

# ===================================
# 9. Dimensionality Reduction (Optional)
# ===================================

use_pca = False

if use_pca:
    pca = PCA(n_components=5)
    X = pca.fit_transform(X)

# ===================================
# 10. Train/Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# 11. Save to CSV
# ===================================

train_df = pd.DataFrame(X_train)
train_df["Price"] = y_train.values

test_df = pd.DataFrame(X_test)
test_df["Price"] = y_test.values

train_df.to_csv("housing_train.csv", index=False)
test_df.to_csv("housing_test.csv", index=False)

print("\nFinal Dataset Sizes")
print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("\nFiles Saved:")
print("housing_train.csv")
print("housing_test.csv")