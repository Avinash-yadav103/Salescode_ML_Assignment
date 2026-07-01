"""
train.py
========
Compare screen-recapture detectors and ship the most accurate one.

Representations (all scored on the SAME group cross-validation):
    Hand                 42 hand-crafted physics features   (cheap baseline)
    MobileNetV3-Small    ImageNet embedding  @320  (576-d)
    EfficientNet-B0      ImageNet embedding  @384  (1280-d)
    ConvNeXt-Tiny        ImageNet embedding  @256  (768-d)   <- usually wins

Each representation is paired with two classifier heads (logistic regression and
an RBF SVM) on top of the frozen features. Accuracy is measured by GROUP 5-FOLD
CROSS-VALIDATION: every one of the ~120 images is held out exactly once and
groups never straddle folds, so the number reflects unseen photos rather than a
single lucky split -- essential on a small dataset.

Frozen embeddings (not fine-tuning) are used on purpose: with only ~120 images,
fine-tuning the backbone overfits and scored WORSE in testing, while a strong
frozen backbone + a small head generalises. Augmentation is likewise omitted --
it helps the hand baseline but slightly hurt the embeddings.

The best (representation, head) by cross-validated accuracy is retrained on all
data, probability-calibrated, and saved to model.pkl with the recipe predict.py
needs to rebuild the representation.

Run:
    python train.py                 # uses ./dataset
    python train.py --folds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings

import numpy as np
import joblib
from joblib import Parallel, delayed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from features import extract_features, FEATURE_NAMES
from embeddings import embed_images, EMB_DIMS, BACKBONE_INPUT

warnings.filterwarnings("ignore")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic")
RESULTS_DIR = "results"

# Representation -> (backbone key or None, uses hand features?).
REPS = {
    "Hand":              (None, True),
    "MobileNetV3-Small": ("mobilenet", False),
    "EfficientNet-B0":   ("efficientnet", False),
    "ConvNeXt-Tiny":     ("convnext", False),
}

# Rough on-device cost (backbone weights + measured CPU embed/extract latency).
REP_COST = {
    "Hand":              {"size_mb": 0.02, "latency_ms": 120},
    "MobileNetV3-Small": {"size_mb": 9.8,  "latency_ms": 55},
    "EfficientNet-B0":   {"size_mb": 20.5, "latency_ms": 70},
    "ConvNeXt-Tiny":     {"size_mb": 110.0, "latency_ms": 75},
}


 
# Data loading
 
def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXT)]


def _group_id(path):
    """Group paired scenes so they never straddle the split (no scene leakage)."""
    base = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(c for c in base if c.isdigit())
    return digits if digits else base


def load_dataset(data_dir):
    real = _list_images(os.path.join(data_dir, "real"))
    screen = _list_images(os.path.join(data_dir, "screen"))
    if not real or not screen:
        raise SystemExit(
            f"Need images in {data_dir}/real and {data_dir}/screen. "
            f"Found {len(real)} real, {len(screen)} screen.")
    paths = real + screen
    labels = np.array([0] * len(real) + [1] * len(screen))
    groups = np.array([_group_id(p) for p in paths])
    print(f"Loaded {len(real)} real + {len(screen)} screen = {len(paths)} images")
    return paths, labels, groups


 
# Representation building (hand features + CNN embeddings), disk-cached
 
def _sig(paths):
    h = hashlib.md5()
    for p in paths:
        st = os.stat(p)
        h.update(p.encode()); h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:12]


def _cached(name, sig, build):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}_{sig}.npy")
    if os.path.exists(path):
        return np.load(path)
    arr = build()
    np.save(path, arr)
    return arr


def build_representations(paths):
    sig = _sig(paths)
    print("Building representations (cached in results/)...")
    t0 = time.perf_counter()
    mats = {}
    mats["Hand"] = _cached("hand", sig, lambda: np.vstack(
        Parallel(n_jobs=-1, batch_size=8)(
            delayed(extract_features)(p) for p in paths)).astype(np.float32))
    for name, (bk, _) in REPS.items():
        if bk is None:
            continue
        mats[name] = _cached(
            f"emb_{bk}", sig,
            lambda bk=bk: embed_images(paths, backbone=bk,
                                       input_size=BACKBONE_INPUT[bk],
                                       progress=True))
    print(f"  " + " | ".join(f"{n} {m.shape[1]}d" for n, m in mats.items())
          + f"  ({time.perf_counter() - t0:.1f}s)")
    return mats


 
# Heads + cross-validation
 
def make_heads():
    return {
        "LogReg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")),
        "SVM": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced")),
    }


def _score(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-clf.decision_function(X)))


def cross_validate(X, y, groups, head_ctor, n_folds):
    """Out-of-fold predictions -- every image scored by a model that never saw
    it (nor any image sharing its group)."""
    n = len(y)
    pred = np.zeros(n, dtype=int)
    score = np.zeros(n, dtype=float)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        clf = head_ctor()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        score[te] = _score(clf, X[te])
    return pred, score


def metrics_from_oof(y, pred, score):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else float("nan"),
    }


def print_table(rows):
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
            "latency_ms", "size_mb"]
    head = f"{'model':28s} " + " ".join(f"{c:>10s}" for c in cols[1:])
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['model']:28s} "
        for c in cols[1:]:
            v = r.get(c, float("nan"))
            line += f"{v:10.3f} " if isinstance(v, (int, float)) else f"{str(v):>10s} "
        print(line)
    print()


 
# Main
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="model.pkl")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    paths, y, groups = load_dataset(args.data)
    mats = build_representations(paths)
    heads = make_heads()

    print(f"\nGroup {args.folds}-fold cross-validation:")
    rows, oof_store = [], {}
    for rep_name in REPS:
        for head_name, ctor in heads.items():
            pred, sc = cross_validate(mats[rep_name], y, groups, ctor, args.folds)
            m = metrics_from_oof(y, pred, sc)
            m["model"] = f"{rep_name} + {head_name}"
            m.update(REP_COST[rep_name])
            rows.append(m)
            oof_store[m["model"]] = (pred, sc)
            print(f"  {m['model']:28s} acc={m['accuracy']:.3f} "
                  f"auc={m['roc_auc']:.3f} f1={m['f1']:.3f}")
    print_table(rows)

    # Select most accurate (tie -> higher AUC, then smaller backbone).
    rows.sort(key=lambda r: (-r["accuracy"], -r["roc_auc"], r["size_mb"]))
    best = rows[0]
    best_name = best["model"]
    rep_name, head_name = best_name.split(" + ")
    backbone, use_hand = REPS[rep_name]
    print(f"Selected: {best_name}  "
          f"(acc={best['accuracy']:.3f}, auc={best['roc_auc']:.3f})")

    # Honest out-of-fold predictions of the winner (for evaluate.py).
    pred, sc = oof_store[best_name]
    with open(os.path.join(RESULTS_DIR, "oof.json"), "w") as f:
        json.dump({"model": best_name, "y_true": y.tolist(),
                   "y_pred": pred.tolist(), "y_score": sc.tolist(),
                   "paths": paths}, f)

    # Ship: fit the head on ALL data and wrap in a probability calibrator. The
    # honest accuracy already comes from the group OOF above; the calibrator only
    # shapes the sigmoid, so a plain stratified CV is sufficient here (there are
    # no augmented/duplicate rows that could leak across its folds).
    X = mats[rep_name]
    base = heads[head_name]()
    final = CalibratedClassifierCV(base, method="sigmoid",
                                   cv=min(args.folds, int(np.bincount(y).min())))
    final.fit(X, y)

    payload = {
        "model": final,
        "rep": rep_name,
        "backbone": backbone,
        "use_hand": use_hand,
        "input_size": BACKBONE_INPUT.get(backbone) if backbone else None,
        "feature_names": FEATURE_NAMES,
        "model_name": best_name,
        "threshold": 0.5,
        "cv_accuracy": float(best["accuracy"]),
        "cv_roc_auc": float(best["roc_auc"]),
        "trained_on": {"real": int((y == 0).sum()), "screen": int((y == 1).sum())},
        "version": 3,
    }
    joblib.dump(payload, args.out, compress=3)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) "
          f"[rep={rep_name}, backbone={backbone}, hand={use_hand}]")

    with open(os.path.join(RESULTS_DIR, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"Wrote comparison + oof predictions to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
