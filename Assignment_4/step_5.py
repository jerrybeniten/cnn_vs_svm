import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# -----------------------------
# STEP 2: DATA PREPARATION
# -----------------------------
df = pd.read_csv('diabetic_data.csv')

df.replace('?', np.nan, inplace=True)
df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

# Drop columns with too many missing values
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

# Target variable
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Length of stay
df['long_stay'] = df['time_in_hospital'].apply(lambda x: 1 if x > 7 else 0)

# Total visits
df['total_visits'] = (
    df['number_outpatient'] +
    df['number_emergency'] +
    df['number_inpatient']
)

# Medication count (FIXED)
med_cols = [col for col in df.columns if 'med' in col or 'insulin' in col]

def count_meds(row):
    return sum(
        1 for col in med_cols
        if pd.notna(row[col]) and row[col] not in ['No', 'Unknown']
    )

df['medication_count'] = df.apply(count_meds, axis=1)

# -----------------------------
# Encoding
# -----------------------------
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# Split Data
# -----------------------------
X = df.drop('readmitted', axis=1)
y = df['readmitted']

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = y.fillna(0)

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
# STEP 3: DECISION TREE
# -----------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print("\nDecision Tree Performance:\n")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_dt):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_dt):.4f}")

# -----------------------------
# STEP 4: RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100],
    'max_depth': [10, None],
    'min_samples_split': [2],
    'max_features': ['sqrt']
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Performance:\n")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# -----------------------------
# STEP 5: FEATURE IMPORTANCE
# -----------------------------
importances = best_rf.feature_importances_
features = X_train.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feat_imp.head(10))

plt.figure()
plt.barh(feat_imp['Feature'][:10][::-1], feat_imp['Importance'][:10][::-1])
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 5 (OPTIONAL): SHAP (FIXED + FAST)
# -----------------------------
try:
    import shap

    print("\nGenerating SHAP summary plot (FAST + FIXED)...")

    X_sample = X_test.sample(500, random_state=42)

    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_sample)
    else:
        shap.summary_plot(shap_values, X_sample)

except ImportError:
    print("\nSHAP not installed. Skipping SHAP.")
    print("Install using: pip install shap")