"""
gen_synthetic.py
================
Bootstrap dataset generator.

WHY THIS EXISTS (read me - honesty matters):
The real deliverable for this take-home is the *approach* and the code. The
ideal training set is ~50 real phone photos + ~50 phone-photos-of-a-screen.
This automated build cannot physically hold a phone, so to ship a `model.pkl`
that runs out-of-the-box and to prove the whole pipeline end-to-end, we generate
a *physically motivated* synthetic dataset here.

The trick that makes it useful rather than a toy:

* The base scene is **1/f ("pink") noise** plus a few structures. Natural images
  have an ~1/f power spectrum, so the REAL class gets a realistic natural
  spectrum.
* The SCREEN class takes that same base and pushes it through a faithful
  *display + re-photograph* pipeline: panel re-sampling, an RGB sub-pixel stripe
  mask, moire from a slightly-rotated grid beating against the sampling lattice,
  scan-lines, a glare blob, a small perspective warp, and a second JPEG pass.

Because both classes share the same base scene, the classifier is forced to
learn the *added* display fingerprints (grid peaks, glare, colour shift) - which
are exactly the cues that exist in genuine recaptures. That is what gives the
synthetic-trained model a fighting chance of transferring to real photos.

>>> Replace dataset/real and dataset/screen with your OWN photos and re-run
    train.py for the best, honest accuracy. <<<

Usage:
    python gen_synthetic.py --n 200 --out dataset
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from PIL import Image
from scipy import ndimage

RNG = np.random.default_rng(20240629)


# --------------------------------------------------------------------------- #
# Base scene: 1/f noise (natural image statistics) + occasional structure
# --------------------------------------------------------------------------- #
def pink_noise(h, w, beta=1.0):
    """Coloured noise with a 1/f**beta power spectrum (natural-image-like)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1e-6
    spectrum = 1.0 / (f ** beta)
    phase = RNG.uniform(0, 2 * np.pi, (h, w))
    field = np.fft.ifft2(spectrum * np.exp(1j * phase)).real
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return field


def base_scene(h=720, w=960):
    beta = RNG.uniform(0.8, 1.6)
    ch = [pink_noise(h, w, beta + RNG.uniform(-0.2, 0.2)) for _ in range(3)]
    img = np.stack(ch, axis=-1)
    # global colour cast + contrast (camera white balance variety)
    img = img * RNG.uniform(0.7, 1.0, 3) + RNG.uniform(0.0, 0.2, 3)
    # occasional smooth blobs / structures so it's not pure noise
    if RNG.random() < 0.6:
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(RNG.integers(1, 4)):
            cy, cx = RNG.uniform(0, h), RNG.uniform(0, w)
            r = RNG.uniform(80, 300)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
            img += blob[..., None] * RNG.uniform(-0.4, 0.4, 3)
    img = ndimage.gaussian_filter(img, sigma=(RNG.uniform(0.5, 2.0), RNG.uniform(0.5, 2.0), 0))
    return np.clip(img, 0, 1)


# --------------------------------------------------------------------------- #
# Camera pipeline shared by both classes (noise / blur / vignette / jpeg)
# --------------------------------------------------------------------------- #
def vignette(img):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    v = 1 - RNG.uniform(0.0, 0.35) * d ** 2
    return img * v[..., None]


def camera_capture(img):
    """Mild, realistic in-camera processing for a *direct* photo of a scene."""
    img = vignette(img)
    img = ndimage.gaussian_filter(img, sigma=(RNG.uniform(0.3, 1.0),
                                              RNG.uniform(0.3, 1.0), 0))
    img = img + RNG.normal(0, RNG.uniform(0.005, 0.02), img.shape)  # sensor noise
    img = np.clip(img, 0, 1)
    return _jpeg(img, RNG.integers(75, 96))


# --------------------------------------------------------------------------- #
# Display + re-photograph pipeline (the SCREEN class)
# --------------------------------------------------------------------------- #
def subpixel_mask(h, w, period):
    """RGB vertical-stripe sub-pixel mask, the dominant colour fingerprint of a
    photographed LCD/OLED panel."""
    cols = (np.arange(w) // max(1, period // 3)) % 3
    mask = np.ones((h, w, 3), np.float32) * 0.75
    for c in range(3):
        mask[:, cols == c, c] = 1.0
    return mask


def moire(img, period):
    """Beat the panel sampling grid against a slightly rotated re-sampling grid,
    producing the characteristic moire interference."""
    angle = RNG.uniform(-3.5, 3.5)
    rot = ndimage.rotate(img, angle, reshape=False, order=1, mode="reflect")
    h, w = rot.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    grid = (0.5 + 0.5 * np.cos(2 * np.pi * xx / period)) * \
           (0.5 + 0.5 * np.cos(2 * np.pi * yy / period))
    rot = rot * (1 - 0.18 * grid[..., None])
    return ndimage.rotate(rot, -angle, reshape=False, order=1, mode="reflect")


def glare(img):
    """Broad blown-out, desaturated reflection blob from the glossy panel."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = RNG.uniform(0.2, 0.8) * h, RNG.uniform(0.2, 0.8) * w
    r = RNG.uniform(120, 320)
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    strength = RNG.uniform(0.2, 0.6)
    return np.clip(img + strength * g[..., None], 0, 1)


def perspective(img):
    """Small keystone warp - the screen is rarely shot perfectly head-on."""
    h, w = img.shape[:2]
    sh = RNG.uniform(-0.04, 0.04)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xx + sh * (yy - h / 2)
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = ndimage.map_coordinates(img[..., c], [yy, xs], order=1, mode="reflect")
    return out


def screen_recapture(img):
    """Full display -> re-photograph chain applied to a base scene."""
    h, w = img.shape[:2]
    # 1) panel re-sampling: downscale to a panel resolution and back up.
    panel_h = RNG.integers(220, 420)
    panel_w = int(panel_h * w / h)
    small = np.asarray(Image.fromarray((img * 255).astype(np.uint8))
                       .resize((panel_w, panel_h), Image.BILINEAR), np.float32) / 255
    img = np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                     .resize((w, h), Image.NEAREST), np.float32) / 255
    period = RNG.integers(4, 9)            # pixel pitch as seen by the camera
    # 2) sub-pixel RGB stripe mask
    img = img * subpixel_mask(h, w, period)
    # 3) moire interference
    img = moire(img, RNG.uniform(period * 1.5, period * 3.0))
    # 4) faint scan-lines (common on OLED / refresh interaction)
    if RNG.random() < 0.6:
        yy = np.arange(h)[:, None, None]
        img = img * (1 - 0.06 * (0.5 + 0.5 * np.cos(2 * np.pi * yy / RNG.uniform(2, 4))))
    # 5) slight colour-gamut shift (panel can't reproduce full gamut)
    img = np.clip(img * RNG.uniform(0.9, 1.05, 3) + RNG.uniform(-0.03, 0.03, 3), 0, 1)
    # 6) glare + perspective + camera capture on top
    if RNG.random() < 0.7:
        img = glare(img)
    img = perspective(img)
    img = ndimage.gaussian_filter(img, sigma=(RNG.uniform(0.3, 0.9),
                                              RNG.uniform(0.3, 0.9), 0))
    img = img + RNG.normal(0, RNG.uniform(0.005, 0.02), img.shape)
    img = np.clip(img, 0, 1)
    # 7) second JPEG pass (display screenshot was already JPEG, now re-photographed)
    return _jpeg(img, RNG.integers(70, 92))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jpeg(img, quality):
    import io
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255


def _save(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="images per class")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    real_dir = os.path.join(args.out, "real")
    screen_dir = os.path.join(args.out, "screen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(screen_dir, exist_ok=True)

    for i in range(args.n):
        h = int(RNG.integers(600, 820))
        w = int(RNG.integers(800, 1100))
        scene = base_scene(h, w)
        _save(camera_capture(scene.copy()), os.path.join(real_dir, f"real_{i:04d}.jpg"))
        _save(screen_recapture(scene.copy()), os.path.join(screen_dir, f"screen_{i:04d}.jpg"))
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{args.n} pairs")

    print(f"Done. {args.n} real + {args.n} screen images in '{args.out}/'")


if __name__ == "__main__":
    main()
