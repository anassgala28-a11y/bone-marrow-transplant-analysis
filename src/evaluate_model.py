import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import preprocess_pipeline


# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path: str):
    """Load a saved model from a .pkl file."""
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


# ─────────────────────────────────────────────
# 2. CLASSIFICATION REPORT
# ─────────────────────────────────────────────
def print_classification_report(model, X_test, y_test):
    """Print full classification report."""
    y_pred = model.predict(X_test)
    print("\n📋 CLASSIFICATION REPORT")
    print("─" * 50)
    print(classification_report(y_test, y_pred,
          target_names=["Not Survived", "Survived"]))


# ─────────────────────────────────────────────
# 3. CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, output_dir: str = "outputs"):
    """Plot and save the confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Not Survived", "Survived"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"💾 Confusion matrix saved → {path}")
    return path


# ─────────────────────────────────────────────
# 4. ROC CURVE
# ─────────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test, output_dir: str = "outputs"):
    """Plot and save the ROC curve."""
    os.makedirs(output_dir, exist_ok=True)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"💾 ROC curve saved → {path}")
    return path


# ─────────────────────────────────────────────
# 5. FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────
def evaluate_pipeline(model_path: str, data_path: str, output_dir: str = "outputs"):
    """Run full evaluation: report + plots."""
    # Load data
    print("🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    # Load model
    model = load_model(model_path)

    # Evaluation
    print_classification_report(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test, output_dir)
    plot_roc_curve(model, X_test, y_test, output_dir)

    print(f"\n✅ Evaluation complete! All plots saved in '{output_dir}/'")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/xgboost.pkl"
    data_path  = sys.argv[2] if len(sys.argv) > 2 else "data/bone-marrow.arff"

    evaluate_pipeline(model_path, data_path)


