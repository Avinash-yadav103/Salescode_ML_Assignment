"""
benchmark.py
============
Measure the two numbers the brief asks for - latency and (implicitly) cost - plus
RAM and model size, on real images.

    python benchmark.py                 # benchmarks on ./dataset
    python benchmark.py --data dataset --runs 100

Reports:
    * feature-extraction time / image
    * model inference time / image
    * end-to-end time / image  (what predict.py actually costs)
    * peak RAM during inference
    * model.pkl size on disk
    * a cloud cost estimate at scale
"""

from __future__ import annotations

import os
# Pin BLAS / OpenMP to a single thread BEFORE importing numpy. On a small 42-d
# feature vector, multi-threading only adds thread-pool spin-up jitter and
# oversubscription when extraction + inference run back-to-back; single-thread
# gives the stable, representative per-image latency a phone would see.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
import tracemalloc
import platform
import warnings

import numpy as np
import joblib

from features import extract_features, FEATURE_NAMES, _feature_dict

warnings.filterwarnings("ignore")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _gather(data_dir, limit):
    paths = []
    for sub in ("real", "screen"):
        d = os.path.join(data_dir, sub)
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith(IMG_EXT)]
    np.random.shuffle(paths)
    return paths[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    paths = _gather(args.data, args.runs)
    if not paths:
        raise SystemExit(f"No images under {args.data}/real|screen")

    payload = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
    model = payload["model"] if payload else None
    names = payload["feature_names"] if payload else FEATURE_NAMES

    # Warm-up (first call pays import / JIT costs we don't want to time).
    _ = _feature_dict(paths[0])

    feat_times, infer_times, e2e_times = [], [], []
    tracemalloc.start()
    for p in paths:
        t0 = time.perf_counter()
        d = _feature_dict(p)
        x = np.array([[d.get(n, 0.0) for n in names]], dtype=np.float32)
        x = np.nan_to_num(x)
        t1 = time.perf_counter()
        if model is not None:
            _ = model.predict_proba(x)[0, 1]
        t2 = time.perf_counter()
        feat_times.append((t1 - t0) * 1000)
        infer_times.append((t2 - t1) * 1000)
        e2e_times.append((t2 - t0) * 1000)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def stats(a):
        a = np.array(a)
        return a.mean(), np.median(a), np.percentile(a, 95)

    fm, fmd, f95 = stats(feat_times)
    im, imd, i95 = stats(infer_times)
    em, emd, e95 = stats(e2e_times)
    size_mb = os.path.getsize("model.pkl") / 1e6 if payload else 0.0

    print("=" * 60)
    print(" LATENCY BENCHMARK")
    print("=" * 60)
    print(f" Device          : {platform.processor() or platform.machine()}")
    print(f" Python          : {platform.python_version()}  ({platform.system()})")
    print(f" Images timed    : {len(paths)}")
    print("-" * 60)
    print(f" {'stage':22s} {'mean':>8s} {'median':>8s} {'p95':>8s}   (ms/img)")
    print(f" {'feature extraction':22s} {fm:8.1f} {fmd:8.1f} {f95:8.1f}")
    print(f" {'model inference':22s} {im:8.2f} {imd:8.2f} {i95:8.2f}")
    print(f" {'END-TO-END':22s} {em:8.1f} {emd:8.1f} {e95:8.1f}")
    print("-" * 60)
    print(f" Model size       : {size_mb:.2f} MB on disk")
    print(f" Peak RAM (infer) : {peak / 1e6:.1f} MB")
    print("=" * 60)

    # ---- Cost at scale ----------------------------------------------------- #
    # Assume a small cloud CPU box (e.g. AWS t4g.small ~ $0.0168/hr, 1 vCPU).
    # One image = end-to-end seconds of CPU. Two regimes: on-device vs cloud.
    sec = em / 1000.0
    box_per_hr = 0.0168
    imgs_per_hr = 3600.0 / sec
    cost_per_img = box_per_hr / imgs_per_hr
    print("\n COST ESTIMATE (assumptions in README)")
    print(f"  on-device   : ~$0.00  (runs free on the user's phone)")
    print(f"  cloud CPU    : ${cost_per_img*1000:.4f} / 1,000 images")
    print(f"               ${cost_per_img*1e6:8.2f} / 1,000,000 images")
    print(f"  (assumes 1 vCPU @ ${box_per_hr}/hr, {sec*1000:.0f} ms/img, no batching)")


if __name__ == "__main__":
    main()
