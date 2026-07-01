"""
augment.py
==========
Label-preserving image augmentation for screen-recapture detection.

WHY augment here.
The real dataset is small (~120 photos). A small tabular model over 42 features
generalises much better when each *training* image is shown under several
plausible capture conditions. Augmentation multiplies effective training size
and, more importantly, teaches the classifier to be *invariant* to nuisances
(lighting, angle, re-compression, sensor noise) while still keying on the
physical screen fingerprints (moire, sub-pixel stripe, glare).

THE HARD CONSTRAINT: every transform must be *label preserving*.
  * A REAL photo must stay REAL  -> never inject moire / sub-pixel stripes /
    panel glare (that would turn it into a fake SCREEN and flip the label).
  * A SCREEN photo must stay a SCREEN -> never blur so hard that the moire /
    grid cue is destroyed.
So we only apply nuisances that a camera legitimately adds to *both* classes:
horizontal flip, small rotation, brightness / gamma / contrast jitter, mild
zoom-crop, light sensor noise, and JPEG re-compression. These change how the
scene was captured, not *what class* it is.

USAGE (see train.py): augment ONLY the training split, and give every variant
the same group id as its parent so no augmented copy of a test image can leak
into training.

    from augment import augment_image
    variants = augment_image(rgb_uint8, rng, n=4)   # list of HxWx3 uint8
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage


# --------------------------------------------------------------------------- #
# Individual transforms  (each takes/returns an RGB uint8 array)
# --------------------------------------------------------------------------- #
def _hflip(rgb, rng):
    return rgb[:, ::-1].copy()


def _rotate_small(rgb, rng):
    """Small in-plane rotation (screens are rarely shot perfectly upright).
    reflect padding avoids black corners that would fake a dark bezel."""
    angle = rng.uniform(-6.0, 6.0)
    return ndimage.rotate(rgb, angle, reshape=False, order=1, mode="reflect")


def _brightness_gamma(rgb, rng):
    """Exposure + gamma variation (daylight vs dim room)."""
    x = rgb.astype(np.float32) / 255.0
    gamma = rng.uniform(0.75, 1.35)
    gain = rng.uniform(0.85, 1.15)
    x = gain * np.power(np.clip(x, 0, 1), gamma)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _contrast(rgb, rng):
    """Contrast stretch / squeeze around the image mean."""
    x = rgb.astype(np.float32)
    m = x.mean()
    f = rng.uniform(0.8, 1.2)
    return np.clip((x - m) * f + m, 0, 255).astype(np.uint8)


def _zoom_crop(rgb, rng):
    """Random central-ish crop then resize back = mild zoom / framing change.
    Kept mild so the frequency-domain scale of moire is not distorted much."""
    h, w = rgb.shape[:2]
    scale = rng.uniform(0.82, 0.98)
    ch, cw = int(h * scale), int(w * scale)
    top = rng.integers(0, h - ch + 1)
    left = rng.integers(0, w - cw + 1)
    crop = rgb[top:top + ch, left:left + cw]
    return np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR),
                      dtype=np.uint8)


def _sensor_noise(rgb, rng):
    """A little extra shot noise (different camera / higher ISO)."""
    sigma = rng.uniform(2.0, 7.0)
    x = rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def _jpeg(rgb, rng):
    """Re-encode at a random quality. Both real and recaptured photos are
    delivered as JPEG, so this is a genuinely class-neutral nuisance and it
    also makes the model robust to the messaging-app recompression seen in the
    dataset (WhatsApp images)."""
    q = int(rng.integers(60, 93))
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


# Pool of nuisance transforms (geometry handled separately so a variant can be
# e.g. "flip + jitter + jpeg" and still look like one plausible capture).
_PHOTOMETRIC = [_brightness_gamma, _contrast, _sensor_noise, _jpeg]
_GEOMETRIC = [_rotate_small, _zoom_crop]


def augment_once(rgb, rng):
    """Produce ONE augmented variant by composing a small random subset of
    label-preserving transforms."""
    out = rgb
    if rng.random() < 0.5:
        out = _hflip(out, rng)
    # 0-1 geometric nuisance
    if rng.random() < 0.6:
        out = rng.choice(_GEOMETRIC)(out, rng)
    # 1-2 photometric nuisances
    k = int(rng.integers(1, 3))
    for fn in rng.choice(_PHOTOMETRIC, size=k, replace=False):
        out = fn(out, rng)
    # A JPEG pass at the end is realistic (final delivery) roughly half the time.
    if rng.random() < 0.5:
        out = _jpeg(out, rng)
    return np.ascontiguousarray(out)


def augment_image(rgb, rng, n=4):
    """Return `n` label-preserving augmented variants of an RGB uint8 image."""
    return [augment_once(rgb, rng) for _ in range(n)]


if __name__ == "__main__":
    import sys
    from features import load_image
    rng = np.random.default_rng(0)
    img = load_image(sys.argv[1])
    for i, v in enumerate(augment_image(img, rng, n=4)):
        Image.fromarray(v).save(f"aug_{i}.jpg", quality=92)
        print(f"aug_{i}.jpg  shape={v.shape}")
