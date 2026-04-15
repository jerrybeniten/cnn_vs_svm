import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# -----------------------------
# STEP 2: DATA PREPARATION
# -----------------------------

# Load dataset
df = pd.read_csv('diabetic_data.csv')

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop non-useful identifiers
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Drop columns with too many missing values (>40%)
threshold = 0.4
df = df[df.columns[df.isnull().mean() < threshold]]

# Fill missing values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('Unknown')

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# Feature Engineering
# -----------------------------

# Target variable (binary)
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Length of stay feature
df['long_stay'] = df['time_in_hospital'].apply(lambda x: 1 if x > 7 else 0)

# Total visits
df['total_visits'] = (
    df['number_outpatient'] +
    df['number_emergency'] +
    df['number_inpatient']
)

# Medication count (handle properly)
med_cols = [col for col in df.columns if 'med' in col or 'insulin' in col]

def count_meds(row):
    count = 0
    for col in med_cols:
        if row[col] not in ['No', 'Unknown', np.nan]:
            count += 1
    return count

df['medication_count'] = df.apply(count_meds, axis=1)

# -----------------------------
# Encode categorical variables
# -----------------------------
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Final safety cleanup
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = y.fillna(0)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data Preparation Complete")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# -----------------------------
# STEP 3: BASELINE MODEL
# -----------------------------

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nDecision Tree Baseline Performance:\n")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))