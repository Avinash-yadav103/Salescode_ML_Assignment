# Run Guide — Spot the Fake Photo

Real-photo vs. photo-of-a-screen (recapture) detector.
Output is a single score in `[0, 1]`: **0 = real photo, 1 = photo of a screen**.

> Platform note: commands below use **Windows PowerShell**. On macOS/Linux just swap
> the backslashes (`\`) in paths for forward slashes (`/`).

---

## 0. Setup (one time)

```powershell
python -m pip install -r requirements.txt
```

The core dependencies (`numpy`, `scipy`, `scikit-image`, `scikit-learn`, `joblib`,
`pillow`) are all `predict.py` / `train.py` / `evaluate.py` need. The training extras
(`matplotlib`, `xgboost`, `lightgbm`) and demo deps (`streamlit`, `flask`) are optional —
`xgboost`/`lightgbm` are skipped automatically if not installed.

---

## 1. Predict on one image (core deliverable)

```powershell
python predict.py "dataset\real\IMG_20260629_215106_528.jpg"
```

Prints **only** the number (e.g. `0.1186`) on stdout, so it is safe to pipe/capture.
If `model.pkl` is missing it falls back to a no-training heuristic instead of crashing.

---

## 2. (Optional) Generate the synthetic bootstrap dataset

Only needed to (re)create the physically-motivated synthetic data — e.g. to restore the
original shipped `model.pkl`.

```powershell
python gen_synthetic.py
```

---

## 3. Train

Reads `dataset\real` + `dataset\screen`, compares models (RandomForest,
HistGradientBoosting, LogisticRegression, and XGBoost/LightGBM if installed), calibrates,
picks the best within the size/latency budget, and writes `model.pkl` plus tables/plots
to `results\`.

```powershell
python train.py
# explicit form:
python train.py --data dataset --out model.pkl --test-size 0.25
```

---

## 4. Evaluate

Scores the trained model on the held-out split and writes a confusion matrix, ROC curve,
and `metrics.json` to `results\`. **Run after `train.py`** (needs `model.pkl` and
`results\X.npy`).

```powershell
python evaluate.py
```

---

## 5. Benchmark (latency / cost / size)

```powershell
python benchmark.py
# or:
python benchmark.py --data dataset --runs 100
```

Reports feature-extraction time, inference time, end-to-end latency per image, peak RAM,
`model.pkl` size, and a cloud cost estimate at scale.

---

## 6. Demo UIs (optional)

Both wrap the same `predict()` — pick one.

```powershell
# Flask: lightweight web page + JSON API at POST /predict
python app.py            # then open http://127.0.0.1:5000

# Streamlit: polished camera/upload demo
streamlit run streamlit_app.py
```

- **Flask** (`app.py`) — also exposes a real JSON API (`{"score": 0.93, "label": "screen", "ms": 70}`); use it for integration/backends.
- **Streamlit** (`streamlit_app.py`) — friendliest click-through demo for a human reviewer.

---

## Typical full workflow

```powershell
python train.py
python evaluate.py
python benchmark.py
python app.py
```

---

## Dataset layout

```
dataset\
  real\     # genuine photos        -> label 0
  screen\   # photos of a screen    -> label 1
```

Drop your own `.jpg/.jpeg/.png/.bmp/.webp/.heic` files into these two folders and re-run
`python train.py` — no code changes needed. More **distinct** real scenes (aim for ~100+
per class across varied devices/screens/angles/lighting) is the single biggest lever on
accuracy.
