"""
evaluate.py
===========
Evaluate the trained model on the held-out test split produced by train.py and
write a confusion matrix, ROC curve, and a full metrics report to results/.

Run AFTER train.py:
    python evaluate.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

warnings.filterwarnings("ignore")
RESULTS_DIR = "results"


def _load():
    payload = joblib.load("model.pkl")
    with open(os.path.join(RESULTS_DIR, "split.json")) as f:
        split = json.load(f)
    X = np.load(os.path.join(RESULTS_DIR, "X.npy"))
    y = np.array(split["labels"])
    te = np.array(split["test_idx"])
    return payload, X, y, te


def main():
    if not (os.path.exists("model.pkl") and
            os.path.exists(os.path.join(RESULTS_DIR, "X.npy"))):
        raise SystemExit("Run train.py first (need model.pkl and results/X.npy).")

    payload, X, y, te = _load()
    model = payload["model"]
    Xte, yte = X[te], y[te]

    proba = model.predict_proba(Xte)[:, 1]
    thr = payload.get("threshold", 0.5)
    pred = (proba >= thr).astype(int)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    cm = confusion_matrix(yte, pred)

    print("=" * 56)
    print(f" Held-out evaluation  ({payload['model_name']})")
    print("=" * 56)
    print(f" Test images : {len(yte)}  ({(yte==0).sum()} real / {(yte==1).sum()} screen)")
    print(f" Accuracy    : {acc:.4f}")
    print(f" Precision   : {prec:.4f}")
    print(f" Recall      : {rec:.4f}")
    print(f" F1 score    : {f1:.4f}")
    print(f" ROC-AUC     : {auc:.4f}")
    print(f"\n Confusion matrix (rows=true, cols=pred):")
    print(f"               pred_real  pred_screen")
    print(f"   true_real    {cm[0,0]:7d}    {cm[0,1]:9d}")
    print(f"   true_screen  {cm[1,0]:7d}    {cm[1,1]:9d}")
    print("\n" + classification_report(yte, pred,
          target_names=["real", "screen"], zero_division=0))

    # ---- Confusion-matrix plot --------------------------------------------- #
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["real", "screen"])
    ax.set_yticks([0, 1], ["real", "screen"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix  (acc={acc:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)

    # ---- ROC curve --------------------------------------------------------- #
    if len(np.unique(yte)) > 1:
        fpr, tpr, _ = roc_curve(yte, proba)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve"); ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=120)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
               "roc_auc": auc, "confusion_matrix": cm.tolist(),
               "n_test": int(len(yte))}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Saved confusion_matrix.png, roc_curve.png, metrics.json to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
