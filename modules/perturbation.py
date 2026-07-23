"""On-the-fly image perturbations for metamorphic stability testing, plus a
model-free "proxy detector" (Otsu threshold) used to probe stability when
the actual segmentation model (e.g. Cellpose) cannot be re-invoked inside
this app.

Metamorphic testing idea: a good segmentation pipeline should give
(approximately) the same result on a perturbed image as a correspondingly
transformed version of the result on the original image. Since this app
only *consumes* precomputed masks rather than running a live model, we
either (a) use a real predicted mask on the perturbed image if the user
supplies one, or (b) fall back to a deterministic Otsu-threshold "detector"
as a consistent stand-in, so the app remains fully self-contained.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.filters import threshold_otsu


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: Optional[int] = None) -> np.ndarray:
    """Add zero-mean Gaussian noise with std ``sigma`` (in the image's own
    intensity units) -- a standard perturbation for testing robustness to
    sensor/read noise.
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0, sigma, size=image.shape)
    return np.clip(noisy, 0, None)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale intensity around the image mean by ``factor`` (e.g. 1.15 =
    +15% contrast, 0.85 = -15%), simulating illumination/exposure drift.
    """
    image = image.astype(np.float64)
    mean = image.mean()
    return np.clip(mean + (image - mean) * factor, 0, None)


def rotate90(array: np.ndarray, k: int) -> np.ndarray:
    """Rotate by 90*k degrees (k = 1, 2, 3). Lossless and label-preserving
    (no interpolation), unlike an arbitrary-angle rotation."""
    return np.rot90(array, k=k)


def flip(array: np.ndarray, axis: int) -> np.ndarray:
    """Mirror the array along ``axis`` (0 = vertical flip, 1 = horizontal flip)."""
    return np.flip(array, axis=axis)


def otsu_proxy_binary_mask(image: np.ndarray) -> np.ndarray:
    """A minimal, deterministic "detector" (Otsu threshold) used only as a
    stand-in when the real segmentation model can't be re-run on a
    perturbed image inside this app. It probes whether a perturbation
    destroys the recoverability of the foreground signal, without
    requiring a live model.
    """
    image = image.astype(np.float64)
    if image.max() <= image.min():
        return np.zeros_like(image, dtype=bool)
    threshold = threshold_otsu(image)
    return image > threshold


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Binary Jaccard similarity (IoU) between two (label or boolean) masks."""
    a, b = mask_a > 0, mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)
