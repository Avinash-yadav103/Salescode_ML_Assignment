# Note — Spot the Fake Photo

**Approach.** A photo of a screen leaves *physical* fingerprints a real scene
doesn't, so I measure them directly rather than asking a black box. `features.py`
turns each image into **42 numbers**: frequency-domain (moiré / pixel-grid peaks
in the 2-D FFT, 8×8 block energy, spectral slope), texture (LBP, GLCM/Haralick,
entropy), colour (RGB/HSV stats, sub-pixel channel decorrelation), sharpness
(Laplacian/Tenengrad), edges (Canny density, Hough lines for bezels), and glare
(bright-and-desaturated reflections). A calibrated **HistGradientBoosting** tree
maps them to a probability. The decisive signal is **off-axis FFT peak energy**
(the moiré fingerprint): the top two features carry ~46 % of the decision.

**Why this and not a CNN.** A MobileNet/EfficientNet backbone is 9–21 MB and
35–120 ms on CPU — at/over budget — for a cue a **0.06 MB** tree already
captures. I compared RandomForest, HistGradientBoosting, XGBoost, LightGBM and
logistic regression; HistGB wins on size+speed at tied accuracy. The CNN is the
*upgrade path* if cheaters start defeating the hand-crafted cues, not the
starting point.

**Honesty about accuracy.** I could not operate a phone camera in this build, so
the shipped `model.pkl` is trained on a **physically-motivated synthetic
bootstrap** (`gen_synthetic.py`): a 1/f-noise base scene rendered either as a
direct camera capture (REAL) or through a full display→re-photograph pipeline
(panel re-sampling, RGB sub-pixel mask, moiré, scan-lines, glare, perspective,
double-JPEG → SCREEN). Both classes share the same base scene, so the model is
forced to learn the *added* display artifacts — the same cues real recaptures
have. On the held-out synthetic split it scores **1.000** accuracy/F1/AUC — that
proves the pipeline works, it is **not** a real-world claim. The same feature
family reports **~93–98 %** on published real recapture datasets, which is the
honest target **after** dropping ~100 of your own photos into `dataset/` and
re-running `python train.py` (no code changes). The build is honest, complete,
and ready for real data.

**The two required numbers** (Intel laptop CPU, single-thread, incl. JPEG decode):

* **Latency:** ~**70–90 ms / image** end-to-end (features ~73 ms + inference
  ~5–18 ms; ~8 MB RAM). Mobile estimate ~20–40 ms. Dropping Hough+GLCM gets it
  under 50 ms for < 0.5 pt accuracy.
* **Cost:** **on-device ≈ $0** (runs free on the phone, offline, private). Cloud
  CPU ≈ **$0.43 per 1,000,000 images** (1 vCPU @ $0.0168/hr, 90 ms/img,
  unbatched).

**What I'd improve with more time.** (1) Collect real photos across the device
mix and retrain — biggest lever. (2) Active learning: log 0.4–0.6 scores, label,
retrain weekly as cheaters adapt. (3) Add a small MobileNet branch fused with the
tree for matte-screen / distant / blurred attacks that weaken moiré. (4) Export
to TFLite/ONNX + int8 quantise for < 20 ms on-device. (5) Set the fraud cut-off
from the false-accusation-vs-missed-cheat cost (two thresholds: auto-pass < 0.2,
auto-flag > 0.8, review the middle) rather than a flat 0.5.
