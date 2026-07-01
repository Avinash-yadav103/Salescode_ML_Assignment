"""
embeddings.py
=============
Deep CNN embeddings for screen-recapture detection.

We use ImageNet-pretrained convolutional backbones (MobileNetV3-Small and
EfficientNet-B0) purely as *frozen feature extractors*: run the image through the
convolutional trunk, global-average-pool the final feature map, and take the
resulting vector as a rich, learned image descriptor. A small classifier head
(fit in train.py) then separates REAL from SCREEN.

WHY this beats the 42 hand-crafted features here.
The hand features encode a *fixed* physics hypothesis (moire peaks, glare, ...).
A pretrained backbone brings millions of general visual primitives -- edges,
textures, colour-gradient and frequency detectors -- learned from ImageNet, so it
also picks up recapture cues we never hand-coded (panel texture, banding, subtle
tone/gamut shifts). On this dataset that lifts accuracy well past the linear
hand-feature baseline. The cost is size/latency (a backbone is 6-21 MB and
tens of ms on CPU) -- an accuracy-over-budget trade the project now chooses.

The backbones are downloaded once (torchvision caches them in ~/.cache/torch) and
run on CPU with grads disabled. Embeddings are deterministic, so train.py caches
them to results/.

    from embeddings import embed_images, EMB_DIMS
    E = embed_images(list_of_rgb_uint8_or_paths, backbone="efficientnet")
"""

from __future__ import annotations

import os

# On Windows, PyTorch ships its own Intel OpenMP (libiomp5md.dll) which collides
# with the one already pulled in by numpy/scipy MKL, aborting the process. Allow
# the duplicate (safe for our inference-only, single-process use) and keep thread
# counts modest so back-to-back embeds don't oversubscribe the CPU. Must be set
# before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from PIL import Image

# Backbone registry: name -> (ctor attr, weights enum attr, dim, default input).
# The default input size is the one that cross-validated best for THIS task (see
# NOTES): recapture cues are high-frequency, so the sweet spot is above the usual
# 224 but not so large that the fixed-scale moire pattern is lost. ConvNeXt-Tiny
# @256 is the shipped choice (best accuracy here).
BACKBONES = {
    "mobilenet":    ("mobilenet_v3_small", "MobileNet_V3_Small_Weights", 576,  320),
    "efficientnet": ("efficientnet_b0",    "EfficientNet_B0_Weights",    1280, 384),
    "convnext":     ("convnext_tiny",      "ConvNeXt_Tiny_Weights",      768,  256),
}
EMB_DIMS = {k: v[2] for k, v in BACKBONES.items()}
BACKBONE_INPUT = {k: v[3] for k, v in BACKBONES.items()}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Default backbone input. Recapture cues (moire / sub-pixel grid) are HIGH
# frequency, so feeding the conv trunk a larger image than the usual 224 keeps
# those fine patterns alive and measurably helps here. Backbones are fully
# convolutional up to the global pool, so any square size works.
_INPUT = 320

# Cache of loaded (model, ) keyed by backbone name so repeated calls are cheap.
_MODELS: dict = {}


def _get_model(backbone: str):
    """Lazily build a frozen, eval-mode feature trunk for `backbone`."""
    if backbone in _MODELS:
        return _MODELS[backbone]
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; "
                         f"choose from {list(BACKBONES)}")
    import torch
    import torchvision.models as tvm

    ctor_name, weights_name, _, _ = BACKBONES[backbone]
    weights = getattr(tvm, weights_name).DEFAULT
    net = getattr(tvm, ctor_name)(weights=weights)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    # `features` is the conv trunk for both MobileNetV3 and EfficientNet.
    trunk = net.features
    _MODELS[backbone] = (torch, trunk)
    return _MODELS[backbone]


def _load_rgb(x, input_size: int) -> np.ndarray:
    """Accept a path or an RGB uint8 array; return an SxS float[0,1] array."""
    if isinstance(x, np.ndarray):
        img = Image.fromarray(x).convert("RGB")
    else:
        img = Image.open(x)
        try:                      # cheap libjpeg down-scale during decode
            img.draft("RGB", (input_size * 2, input_size * 2))
        except Exception:
            pass
        img = img.convert("RGB")
    img = img.resize((input_size, input_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _forward(torch, trunk, arr):
    """arr: (b,H,W,3) float[0,1] -> (b,D) float32 pooled embedding."""
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    t = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())
    feat = trunk(t)
    feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
    return feat.flatten(1).cpu().numpy().astype(np.float32)


def embed_images(images, backbone: str = "convnext", batch_size: int = 16,
                 input_size: int = None, tta: bool = False,
                 progress: bool = False) -> np.ndarray:
    """Return an (N, EMB_DIMS[backbone]) float32 array of L2-normalised
    embeddings. `input_size` defaults to the backbone's tuned resolution. If
    `tta`, average the embedding over the image and its horizontal mirror (a
    cheap, label-preserving test-time augmentation)."""
    if input_size is None:
        input_size = BACKBONE_INPUT[backbone]
    torch, trunk = _get_model(backbone)
    out = []
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = images[i:i + batch_size]
            arr = np.stack([_load_rgb(x, input_size) for x in chunk])
            feat = _forward(torch, trunk, arr)
            if tta:
                feat = feat + _forward(torch, trunk, arr[:, :, ::-1, :].copy())
                feat *= 0.5
            out.append(feat)
            if progress:
                print(f"  embedded {min(i + batch_size, n)}/{n}", end="\r")
    if progress:
        print()
    E = np.vstack(out)
    # L2-normalise: makes a linear / SVM head scale-stable and is standard for
    # frozen-embedding classifiers.
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / np.maximum(norm, 1e-9)).astype(np.float32)


def embed_image(image, backbone: str = "convnext", input_size: int = None,
                tta: bool = False) -> np.ndarray:
    """Single-image convenience wrapper -> 1-D embedding."""
    return embed_images([image], backbone=backbone, input_size=input_size,
                        tta=tta)[0]


if __name__ == "__main__":
    import sys
    bk = sys.argv[2] if len(sys.argv) > 2 else "efficientnet"
    e = embed_image(sys.argv[1], backbone=bk)
    print(f"{bk}: dim={e.shape[0]}  |e|={np.linalg.norm(e):.3f}  "
          f"mean={e.mean():.4f}")
