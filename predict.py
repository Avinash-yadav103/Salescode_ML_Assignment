"""
predict.py
==========
One-line predictor for screen-recapture (fake photo) detection.

    python predict.py some_image.jpg   ->   0.93

Prints ONE number in [0, 1]:
    0 = real photo,   1 = photo of a screen (recapture / fraud).

Primary path  : the calibrated tree model trained by train.py (model.pkl).
Fallback path : if model.pkl is missing/unloadable, a physically-grounded
                frequency+glare heuristic that needs no training, so the script
                always returns a sensible score and never crashes.

Output is ONLY the number (everything else goes to stderr), so this is safe to
pipe / capture from another process.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


# --------------------------------------------------------------------------- #
# Heuristic fallback (no model required)
# --------------------------------------------------------------------------- #
def _heuristic_score(feat: dict) -> float:
    """Map the most reliable physical cues to a fraud probability with a hand-
    tuned logistic. Used only when no trained model is present, so it just needs
    to be directionally right. Coefficients are centred on the class midpoints
    observed in the data; the dominant terms are the FFT moire-peak features,
    which are the physically meaningful, transferable cues (a higher / sharper
    off-axis spectral peak == screen)."""
    import math
    z = (2.0 * (feat["fft_peak_max"] - 2.8)        # moire peak height (real~2.1 / screen~3.5)
         + 0.6 * (feat["fft_peak_ratio"] - 12.5)   # peak sharpness   (real~8.8 / screen~16)
         + 18.0 * (feat["glare_fraction"] - 0.005) # glossy-panel glare
         + 12.0 * (0.89 - feat["lbp_uniform_ratio"]))  # sub-pixel micro-texture
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# --------------------------------------------------------------------------- #
# Main predictor
# --------------------------------------------------------------------------- #
def _build_vector(payload, image_path):
    """Reconstruct the exact representation the shipped model was trained on:
    hand-crafted features, a CNN embedding, or both concatenated (hand first)."""
    from features import extract_features, FEATURE_NAMES, _feature_dict
    import numpy as np

    parts = []
    if payload.get("use_hand", True):
        names = payload.get("feature_names", FEATURE_NAMES)
        d = _feature_dict(image_path)
        parts.append(np.array([d.get(n, 0.0) for n in names], dtype=np.float32))
    backbone = payload.get("backbone")
    if backbone:
        from embeddings import embed_image
        emb = embed_image(image_path, backbone=backbone,
                          input_size=payload.get("input_size"))
        parts.append(emb.astype(np.float32))
    x = np.concatenate(parts).reshape(1, -1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def predict(image_path: str) -> float:
    from features import _feature_dict
    import numpy as np

    # Trained-model path.
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            payload = joblib.load(MODEL_PATH)
            model = payload["model"]
            x = _build_vector(payload, image_path)
            return float(model.predict_proba(x)[0, 1])
        except Exception as e:                       # never fail the interface
            print(f"[predict] model path failed ({e}); using heuristic",
                  file=sys.stderr)

    # Heuristic fallback (needs no model, no torch).
    return float(_heuristic_score(_feature_dict(image_path)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python predict.py image.jpg", file=sys.stderr)
        sys.exit(1)
    score = predict(sys.argv[1])
    # ONLY the number on stdout.
    print(round(score, 4))
