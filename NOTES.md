# Note — Spot the Fake Photo

**Approach.** A photo of a screen carries physical fingerprints a real scene
doesn't — moiré from the camera grid beating the pixel grid, an RGB sub-pixel
stripe, panel glare, double-JPEG banding, shifted colour. The shipped detector
runs the image through a **frozen, ImageNet-pretrained ConvNeXt-Tiny** backbone at
256×256, global-average-pools it to a 768-d embedding, and classifies with a
**standardised, calibrated logistic-regression head**. The backbone's general
edge/texture/frequency detectors capture those recapture cues far more completely
than a fixed hand-coded rule set.

**Evaluation (now).** Group 5-fold cross-validation on **57 real + 66 screen**
real phone photos — every image held out once, related shots kept in the same
fold (no scene leakage): **95.1 % accuracy, 0.968 ROC-AUC, 0.97 precision,
0.94 recall** (6 errors / 123: 2 false flags, 4 missed screens).

**Model choices that mattered.**
* **Backbone:** ConvNeXt-Tiny cross-validated highest — 95.1 %, vs 89.4 %
  EfficientNet-B0, 87.0 % MobileNetV3-Small, 83.7 % hand-features baseline.
* **Resolution 256 px:** recapture cues are high-frequency, so a bit above the
  usual 224 keeps the moiré alive; 256 beat 224/320/384 in the sweep.
* **Frozen, not fine-tuned:** end-to-end fine-tuning on only ~120 images *overfit*
  and scored several points **worse** in CV. A strong frozen backbone + a small
  head generalises and needs no GPU.
* **Linear head:** with a strong embedding the backbone does the work, so a
  logistic head both wins and calibrates cleanly; an RBF-SVM head was tried and
  lost.

**Hand-crafted baseline (`features.py`).** 42 interpretable physics features (FFT
moiré peaks, 8×8-block energy, LBP/GLCM texture, HSV colour, glare, Hough bezels).
Kept as a transparent baseline (0.84 CV) and as the **no-torch heuristic fallback**
in `predict.py`, so the interface never crashes without the CNN.

**The two required numbers** (Intel laptop CPU, single-thread, incl. JPEG decode):
* **Latency:** ~**82 ms / image** end-to-end (ConvNeXt forward ~75 ms +
  head ~1 ms). Backbone weights ~110 MB, a few hundred MB RAM. This heavier
  footprint is the deliberate cost of moving from ~84 % to 95 % accuracy.
* **Cost:** **on-device ≈ $0** (offline, private). Cloud CPU ≈ **$0.38 per
  1,000,000 images** (1 vCPU @ $0.0168/hr, 82 ms/img, unbatched).

**What I'd improve with more time.** (1) Collect more real photos across the
device mix — biggest lever, and the point at which fine-tuning overtakes frozen.
(2) Active learning: log 0.4–0.6 scores, label, retrain as cheaters adapt.
(3) Export ConvNeXt to TFLite/ONNX + int8 (~7 MB, tens of ms) for on-device.
(4) Fuse CNN + hand features for matte-screen / distant / blurred attacks.
(5) Set the fraud cut-off from the false-accusation-vs-missed-cheat cost
(auto-pass < 0.2, auto-flag > 0.8, review the middle) rather than a flat 0.5.
</content>
