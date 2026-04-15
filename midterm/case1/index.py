# ============================================
# Case Study 1: FIXED VERSION (Your Dataset)
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv("Final_Marks_Data.csv")

print("Columns:", df.columns)

# ============================================
# 2. CREATE TARGET (CLASSIFICATION)
# ============================================

def categorize(score):
    if score >= 85:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 55:
        return 'C'
    else:
        return 'D'

df['grade_category'] = df['Final Exam Marks (out of 100)'].apply(categorize)

# ============================================
# 3. PREPROCESSING
# ============================================

# Drop unnecessary columns
df = df.drop(columns=['Student_ID', 'Final Exam Marks (out of 100)'])

target_column = 'grade_category'

# Encode target
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Normalize (VERY IMPORTANT for SVM & KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 5. MODELS
# ============================================

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf'),
    "Naive Bayes": GaussianNB()
}

results = {}

# ============================================
# 6. TRAIN & EVALUATE
# ============================================

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    print(classification_report(y_test, y_pred))

# ============================================
# 7. FINAL COMPARISON
# ============================================

results_df = pd.DataFrame(results).T

print("\n===== FINAL COMPARISON =====")
print(results_df.sort_values(by="F1 Score", ascending=False))

# ============================================
# 8. BEST MODEL
# ============================================

best_model = results_df["F1 Score"].idxmax()
print(f"\nBest Model Based on F1 Score: {best_model}")