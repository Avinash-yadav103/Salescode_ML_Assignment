"""
features.py
===========
Hand-crafted feature extraction for screen-recapture (re-photograph) detection.

The whole detector rests on one idea: a photo of a *screen* carries physical
fingerprints that a photo of a *real scene* does not. We turn those fingerprints
into ~40 numbers a tiny classifier can separate.

The strongest signals, in rough order of usefulness:

1. Moire / pixel-grid aliasing  -> sharp off-axis peaks in the 2-D FFT.
   The camera sensor grid beats against the display's pixel grid, producing
   periodic structure no natural scene has.
2. JPEG / display 8x8 block periodicity -> energy spikes at f = fs/8, fs/4 ...
3. Sub-pixel RGB stripe texture -> channel-dependent high-frequency content.
4. Glare / specular reflection off the glossy panel -> broad blown-out regions.
5. Slightly-off colour & reduced gamut -> HSV / saturation statistics.
6. Straight bezel / grid lines -> Hough line count.

Everything here is pure NumPy / SciPy / scikit-image so it runs anywhere with no
OpenCV and no GPU. Extraction of one image is a few milliseconds.

`extract_features(path_or_array)` -> 1-D np.float32 vector aligned with
`FEATURE_NAMES`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.feature import canny, local_binary_pattern, graycomatrix, graycoprops
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import shannon_entropy

# All features are computed on a fixed-size square so that frequency-domain
# measurements are comparable across images of different resolutions. 256 keeps
# the moire / pixel-grid structure well resolved while staying fast on a CPU.
WORK_SIZE = 256


def _rgb2hsv(x: np.ndarray) -> np.ndarray:
    """Fast vectorised RGB->HSV for an [0,1] float array (skimage's version is
    the single biggest cost otherwise). Returns H,S,V in [0,1]."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    diff = mx - mn
    v = mx
    s = np.where(mx > 1e-9, diff / (mx + 1e-9), 0.0)
    h = np.zeros_like(mx)
    mask = diff > 1e-9
    # piecewise hue
    rc = np.where(mask, (mx - r) / (diff + 1e-9), 0.0)
    gc = np.where(mask, (mx - g) / (diff + 1e-9), 0.0)
    bc = np.where(mask, (mx - b) / (diff + 1e-9), 0.0)
    h = np.where(mx == r, bc - gc, h)
    h = np.where(mx == g, 2.0 + rc - bc, h)
    h = np.where(mx == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


# --------------------------------------------------------------------------- #
# Image loading / normalisation
# --------------------------------------------------------------------------- #
def load_image(path_or_array) -> np.ndarray:
    """Return an RGB uint8 array, resized so the short side is WORK_SIZE and
    centre-cropped to WORK_SIZE x WORK_SIZE. Standardising the size keeps the
    FFT-based features (which are scale sensitive) consistent."""
    if isinstance(path_or_array, np.ndarray):
        img = Image.fromarray(path_or_array).convert("RGB")
    else:
        img = Image.open(path_or_array).convert("RGB")

    w, h = img.size
    scale = WORK_SIZE / min(w, h)
    img = img.resize((max(WORK_SIZE, round(w * scale)),
                      max(WORK_SIZE, round(h * scale))), Image.BILINEAR)

    w, h = img.size
    left, top = (w - WORK_SIZE) // 2, (h - WORK_SIZE) // 2
    img = img.crop((left, top, left + WORK_SIZE, top + WORK_SIZE))
    return np.asarray(img, dtype=np.uint8)


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma in [0, 1]."""
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                      dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# 1. Frequency-domain features  (the moire / grid fingerprint)
# --------------------------------------------------------------------------- #
def _radial_profile(mag: np.ndarray) -> np.ndarray:
    """Azimuthally-averaged magnitude spectrum -> 1-D profile vs. radius."""
    cy, cx = np.array(mag.shape) // 2
    y, x = np.indices(mag.shape)
    r = np.hypot(x - cx, y - cy).astype(np.int32)
    nbins = r.max() + 1
    tbin = np.bincount(r.ravel(), mag.ravel(), minlength=nbins)
    nr = np.bincount(r.ravel(), minlength=nbins)
    return tbin / np.maximum(nr, 1)


def frequency_features(gray: np.ndarray) -> dict:
    # Windowed FFT (Hann window kills spectral leakage from image borders that
    # would otherwise masquerade as periodic structure).
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    mag = np.abs(f)
    logmag = np.log1p(mag)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(mag.shape)
    radius = np.hypot(x - cx, y - cy)
    rmax = radius.max()

    total = mag.sum() + 1e-9
    high = mag[radius > 0.5 * rmax].sum()
    mid = mag[(radius > 0.25 * rmax) & (radius <= 0.5 * rmax)].sum()
    feats = {
        "fft_high_freq_ratio": high / total,
        "fft_mid_freq_ratio": mid / total,
    }

    # Radial profile shape: natural images fall off ~1/f (smooth). Screens add
    # bumps. We fit the log-log slope and measure how bumpy the tail is.
    prof = _radial_profile(mag)
    rr = np.arange(1, len(prof))
    lp = np.log(prof[1:] + 1e-9)
    slope = np.polyfit(np.log(rr), lp, 1)[0]
    feats["fft_radial_slope"] = slope
    tail = prof[len(prof) // 4:]
    feats["fft_radial_tail_std"] = float(np.std(np.log(tail + 1e-9)))

    # --- Moire peak detection -------------------------------------------------
    # Suppress DC, the central low-freq disk, and the axis cross (which carry
    # ordinary image edges), then look for sharp isolated peaks = aliasing.
    masked = logmag.copy()
    central = radius < 0.10 * rmax
    masked[central] = 0
    masked[cy - 1:cy + 2, :] = 0
    masked[:, cx - 1:cx + 2] = 0

    local_mean = ndimage.uniform_filter(masked, size=9)
    peak_strength = masked - local_mean
    pk = peak_strength[radius > 0.10 * rmax]
    feats["fft_peak_max"] = float(pk.max())
    feats["fft_peak_ratio"] = float(pk.max() / (pk.std() + 1e-9))
    # Number of strong, isolated peaks (a few = grid aliasing; many/none = scene).
    thresh = pk.mean() + 4 * pk.std()
    feats["fft_peak_count"] = float((peak_strength > thresh).sum())

    # --- 8x8 block / display-grid periodicity --------------------------------
    # Average the spectrum along rows & cols; energy at normalised freq 1/8, 1/4
    # signals JPEG blocks or a display sampling grid.
    col_spec = mag.mean(axis=0)
    row_spec = mag.mean(axis=1)

    def _grid_energy(spec):
        n = len(spec)
        c = n // 2
        half = spec[c:]
        half = half / (half.sum() + 1e-9)
        idx = [int(round(n * frac)) - c for frac in (0.125, 0.25, 0.375)]
        return float(sum(half[i] for i in idx if 0 <= i < len(half)))

    feats["fft_grid_energy"] = 0.5 * (_grid_energy(col_spec) + _grid_energy(row_spec))

    # Periodicity of the 1-D spectra via peak count (screens -> regular comb).
    cs = col_spec[len(col_spec) // 2:]
    rs = row_spec[len(row_spec) // 2:]
    cs = (cs - cs.min()) / (np.ptp(cs) + 1e-9)
    rs = (rs - rs.min()) / (np.ptp(rs) + 1e-9)
    pc, _ = find_peaks(cs, height=0.15, distance=4)
    pr, _ = find_peaks(rs, height=0.15, distance=4)
    feats["fft_spectral_peaks"] = float(len(pc) + len(pr))
    return feats


# --------------------------------------------------------------------------- #
# 2. Texture features
# --------------------------------------------------------------------------- #
def texture_features(gray: np.ndarray) -> dict:
    g = (gray * 255).astype(np.uint8)

    # Local Binary Pattern (uniform) histogram -> entropy + uniform-pattern share.
    lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
    lbp_entropy = -np.sum(hist * np.log2(hist + 1e-9))

    # Gray-Level Co-occurrence Matrix (Haralick) averaged over 4 directions.
    # 16 grey levels keeps the matrix small/fast without losing the contrast and
    # homogeneity contrast that separates smooth display gradients from texture.
    glcm = graycomatrix((g // 16), distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=16, symmetric=True, normed=True)
    return {
        "lbp_entropy": float(lbp_entropy),
        "lbp_uniform_ratio": float(hist[:9].sum()),
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "shannon_entropy": float(shannon_entropy(g)),
    }


# --------------------------------------------------------------------------- #
# 3. Colour / image statistics
# --------------------------------------------------------------------------- #
def color_features(rgb: np.ndarray) -> dict:
    # Colour statistics don't need full resolution: a 2x stride is plenty and
    # roughly quarters the cost of the HSV conversion.
    x = rgb[::2, ::2].astype(np.float32) / 255.0
    hsv = _rgb2hsv(x)
    feats = {}
    for i, c in enumerate("rgb"):
        feats[f"{c}_mean"] = float(x[..., i].mean())
        feats[f"{c}_std"] = float(x[..., i].std())
    for i, c in enumerate(["hue", "sat", "val"]):
        feats[f"{c}_mean"] = float(hsv[..., i].mean())
        feats[f"{c}_std"] = float(hsv[..., i].std())

    # Saturation histogram peakiness: screens often clip / shift saturation.
    sat_hist, _ = np.histogram(hsv[..., 1], bins=16, range=(0, 1), density=True)
    feats["sat_hist_peak"] = float(sat_hist.max() / (sat_hist.mean() + 1e-9))
    # Channel correlation (subsampled): subpixel structure decorrelates RGB at
    # high frequency. np.errstate guards flat images where std == 0.
    r = x[..., 0].ravel()[::4]
    g = x[..., 1].ravel()[::4]
    if r.std() < 1e-6 or g.std() < 1e-6:
        feats["rg_corr"] = 1.0
    else:
        feats["rg_corr"] = float(np.corrcoef(r, g)[0, 1])
    return feats


# --------------------------------------------------------------------------- #
# 4. Sharpness
# --------------------------------------------------------------------------- #
def sharpness_features(gray: np.ndarray) -> dict:
    lap = ndimage.laplace(gray)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return {
        "laplacian_var": float(lap.var()),
        "tenengrad": float((gx ** 2 + gy ** 2).mean()),
    }


# --------------------------------------------------------------------------- #
# 5. Edge / line features
# --------------------------------------------------------------------------- #
def edge_features(gray: np.ndarray) -> dict:
    edges = canny(gray, sigma=1.5)
    feats = {"canny_density": float(edges.mean())}

    # Hough straight-line count: bezels / window frames / on-screen UI add long
    # straight lines that natural scenes rarely have in such regularity. Run on a
    # half-resolution edge map -- line detection is robust to it and it halves cost.
    try:
        small = edges[::2, ::2]
        h, theta, d = hough_line(small)
        accum, _, _ = hough_line_peaks(h, theta, d, num_peaks=20,
                                       threshold=0.4 * h.max())
        feats["hough_line_count"] = float(len(accum))
    except Exception:
        feats["hough_line_count"] = 0.0
    return feats


# --------------------------------------------------------------------------- #
# 6. Reflection / glare features
# --------------------------------------------------------------------------- #
def reflection_features(rgb: np.ndarray) -> dict:
    x = rgb.astype(np.float32) / 255.0
    val = x.max(axis=2)            # HSV value
    sat = (val - x.min(axis=2)) / (val + 1e-9)
    # Glare = bright AND desaturated (specular highlight from a glossy panel).
    glare = (val > 0.92) & (sat < 0.20)
    feats = {
        "glare_fraction": float(glare.mean()),
        "bright_fraction": float((val > 0.95).mean()),
    }
    # Count of distinct specular blobs.
    labelled, n = ndimage.label(glare)
    feats["specular_blobs"] = float(min(n, 50))
    return feats


# --------------------------------------------------------------------------- #
# 7. Dynamic range / contrast / brightness
# --------------------------------------------------------------------------- #
def range_features(gray: np.ndarray) -> dict:
    p1, p50, p99 = np.percentile(gray, [1, 50, 99])
    return {
        "dynamic_range": float(p99 - p1),
        "contrast_std": float(gray.std()),
        "brightness_mean": float(gray.mean()),
        "brightness_median": float(p50),
        "brightness_p99": float(p99),
    }


# --------------------------------------------------------------------------- #
# Master extractor
# --------------------------------------------------------------------------- #
def _feature_dict(path_or_array) -> dict:
    rgb = load_image(path_or_array)
    gray = _to_gray(rgb)
    feats = {}
    feats.update(frequency_features(gray))
    feats.update(texture_features(gray))
    feats.update(color_features(rgb))
    feats.update(sharpness_features(gray))
    feats.update(edge_features(gray))
    feats.update(reflection_features(rgb))
    feats.update(range_features(gray))
    return feats


# Stable, ordered feature names (derived once from a blank image at import time).
FEATURE_NAMES = list(_feature_dict(np.zeros((WORK_SIZE, WORK_SIZE, 3),
                                            dtype=np.uint8)).keys())


def extract_features(path_or_array) -> np.ndarray:
    """Return the feature vector for one image, aligned with FEATURE_NAMES."""
    d = _feature_dict(path_or_array)
    vec = np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)
    # Frequency / log features can occasionally produce non-finite values on
    # degenerate (flat) images; keep the classifier safe.
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    import sys
    v = extract_features(sys.argv[1])
    for name, val in zip(FEATURE_NAMES, v):
        print(f"{name:24s} {val:12.5f}")
    print(f"\n{len(FEATURE_NAMES)} features")
