"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin dataset,
evaluates each with Accuracy, AUC, Precision, Recall, F1, and MCC,
and saves:
  - model/*.joblib          (one trained model per algorithm)
  - model/scaler.joblib      (StandardScaler used for the scaled models)
  - test_data.csv            (held-out test split, used by the Streamlit app)
  - metrics_report.csv       (the comparison table required in the README)

Run once locally / on BITS Virtual Lab:  python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------
# Step 2: Train/test split (stratified to preserve class balance)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save the held-out test split as the "test_data.csv" required by the
# assignment (features + true label, so the Streamlit app can score it).
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# ---------------------------------------------------------------------
# Step 3: Scale features (Logistic Regression / kNN are scale-sensitive)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.joblib")

# ---------------------------------------------------------------------
# Step 4: Define models
#   - LR, kNN use scaled features
#   - Decision Tree, Naive Bayes, Random Forest use raw features
#     (tree/probability based models don't need scaling)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree":       (DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "kNN":                 (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes":         (GaussianNB(), False),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), False),
}

results = []

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)

    # Save each trained model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"model/{fname}.joblib")
    print(f"Trained & saved: {name} -> model/{fname}.joblib")

# ---------------------------------------------------------------------
# Step 5: Save comparison table (goes straight into your README table)
# ---------------------------------------------------------------------
report_df = pd.DataFrame(results)
report_df.to_csv("metrics_report.csv", index=False)

print("\n=== Model Comparison Table ===")
print(report_df.to_string(index=False))
print("\nSaved: test_data.csv, metrics_report.csv, model/*.joblib")
