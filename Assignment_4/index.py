# Step 2: Data Preparation (All-in-One Pipeline)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv('diabetic_data.csv')

# Optional: load ID mapping (not always required for modeling)
ids_map = pd.read_csv('IDS_mapping.csv')

# -----------------------------
# 2. Initial Cleaning
# -----------------------------
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop identifiers (no predictive value)
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
# Drop columns with too many missing values
threshold = 0.4  # 40% missing
df = df[df.columns[df.isnull().mean() < threshold]]

# Fill categorical missing values with 'Unknown'
for col in df.select_dtypes(include='object').columns:
    df[col].fillna('Unknown', inplace=True)

# Fill numerical missing values with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# -----------------------------
# 4. Feature Engineering
# -----------------------------

# Convert readmitted target into binary
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Length of stay proxy (time_in_hospital already exists)
df['long_stay'] = df['time_in_hospital'].apply(lambda x: 1 if x > 7 else 0)

# Total visits (prior healthcare usage)
df['total_visits'] = (
    df['number_outpatient'] +
    df['number_emergency'] +
    df['number_inpatient']
)

# Medication usage count (simple proxy)
med_cols = [col for col in df.columns if 'med' in col or 'insulin' in col]
df['medication_count'] = df[med_cols].apply(
    lambda row: sum(row != 'No'), axis=1
)

# -----------------------------
# 5. Encode Categorical Variables
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# 6. Split Features and Target
# -----------------------------
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # important for class imbalance
)

# -----------------------------
# 8. Output Summary
# -----------------------------
print("Data Preparation Complete")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")