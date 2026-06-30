"""
streamlit_app.py
================
Tiny live demo for the recapture detector.

    streamlit run streamlit_app.py

Upload an image OR use your camera; the app shows the picture, the fraud score,
and a confidence read-out. Pure CPU, runs offline once deps are installed.
"""

import time
import numpy as np
import streamlit as st
from PIL import Image

from predict import predict, MODEL_PATH
import os

st.set_page_config(page_title="Spot the Fake Photo", page_icon="🕵️", layout="centered")
st.title("🕵️ Spot the Fake Photo")
st.caption("Real photo vs. photo-of-a-screen (recapture) detector. "
           "0 = real, 1 = screen.")

if not os.path.exists(MODEL_PATH):
    st.warning("model.pkl not found - running on the no-training heuristic "
               "fallback. Run `python train.py` for the full model.")

tab_cam, tab_up = st.tabs(["📷 Camera", "📁 Upload"])
img = None
with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        img = Image.open(shot).convert("RGB")
with tab_up:
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up:
        img = Image.open(up).convert("RGB")

if img is not None:
    # predict() takes a path; save to a temp file so the same code path runs.
    tmp = "._demo_tmp.jpg"
    img.save(tmp, quality=95)
    t0 = time.perf_counter()
    score = predict(tmp)
    dt = (time.perf_counter() - t0) * 1000
    os.remove(tmp)

    st.image(img, caption="input", use_container_width=True)
    label = "PHOTO OF A SCREEN (likely fraud)" if score >= 0.5 else "REAL photo"
    conf = score if score >= 0.5 else 1 - score
    (st.error if score >= 0.5 else st.success)(
        f"**{label}**  —  score = {score:.3f}  ({conf*100:.0f}% confidence)")
    st.progress(float(score))
    st.caption(f"Inference: {dt:.0f} ms (includes image decode).")
