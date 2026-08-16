"""
app.py
------
Streamlit demo app for the Breast Cancer classification models.

Features (per assignment requirements):
  a. Dataset upload option (CSV)         -> file_uploader below
  b. Model selection dropdown            -> st.selectbox below
  c. Display of evaluation metrics       -> metric cards + table
  d. Confusion matrix / classification report -> heatmap + text report

Expected CSV format: the same columns as test_data.csv, i.e. the 30
breast-cancer feature columns plus a "target" column (0/1).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Model Explorer", layout="wide")

MODEL_FILES = {
    "Logistic Regression": ("model/logistic_regression.joblib", True),
    "Decision Tree": ("model/decision_tree.joblib", False),
    "kNN": ("model/knn.joblib", True),
    "Naive Bayes": ("model/naive_bayes.joblib", False),
    "Random Forest (Ensemble)": ("model/random_forest_ensemble.joblib", False),
}


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.joblib")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🔬 Classification Model Explorer")
st.caption("BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning Assignment 2")

st.markdown("""
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — 30 numeric features,
binary target (`0` = malignant, `1` = benign).
""")

# --- a. Dataset upload -------------------------------------------------
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded — using the bundled test_data.csv as a demo.")
    df = pd.read_csv("test_data.csv")

if "target" not in df.columns:
    st.error("Uploaded CSV must include a 'target' column with the true labels.")
    st.stop()

X = df.drop(columns=["target"])
y_true = df["target"]

st.write(f"Loaded **{X.shape[0]}** rows and **{X.shape[1]}** feature columns.")
with st.expander("Preview data"):
    st.dataframe(df.head())

# --- b. Model selection dropdown ---------------------------------------
model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
model_path, needs_scaling = MODEL_FILES[model_name]
model = load_model(model_path)

if needs_scaling:
    scaler = load_scaler()
    X_for_model = scaler.transform(X)
else:
    X_for_model = X

y_pred = model.predict(X_for_model)
y_proba = model.predict_proba(X_for_model)[:, 1]

# --- c. Evaluation metrics ----------------------------------------------
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

st.subheader(f"Results — {model_name}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{acc:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{prec:.4f}")
c4.metric("Recall", f"{rec:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

# --- d. Confusion matrix + classification report -------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_b:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, target_names=["Malignant", "Benign"])
    st.text(report)

st.markdown("---")
st.subheader("Compare all models on this data")

comparison_rows = []
for name, (path, scale_flag) in MODEL_FILES.items():
    m = load_model(path)
    Xin = load_scaler().transform(X) if scale_flag else X
    pred = m.predict(Xin)
    proba = m.predict_proba(Xin)[:, 1]
    comparison_rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, pred), 4),
        "AUC": round(roc_auc_score(y_true, proba), 4),
        "Precision": round(precision_score(y_true, pred), 4),
        "Recall": round(recall_score(y_true, pred), 4),
        "F1": round(f1_score(y_true, pred), 4),
        "MCC": round(matthews_corrcoef(y_true, pred), 4),
    })

st.dataframe(pd.DataFrame(comparison_rows).set_index("Model"), use_container_width=True)
