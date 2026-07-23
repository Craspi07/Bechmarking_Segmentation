"""Unsupervised, intrinsic (no ground truth needed) per-object quality
metrics computed directly from raw intensity + mask boundaries: edge
sharpness, intra-object homogeneity, and signal-to-surround contrast,
combined into a single "Composite Segmentation Confidence Score" per
object.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from . import snr_utils


def compute_boundary_sharpness(mask: np.ndarray, raw_image: np.ndarray) -> pd.DataFrame:
    """Mean Sobel gradient magnitude of the raw image along each object's
    boundary pixels. A confident, intensity-supported segmentation
    boundary tends to sit on a real edge (high gradient); a boundary that
    doesn't align with any intensity edge (low gradient) suggests the
    mask's shape isn't well supported by the data.
    """
    gradient = sobel(raw_image.astype(np.float64))
    boundaries = find_boundaries(mask, mode="inner")

    records = []
    for prop in regionprops(mask.astype(np.int32)):
        r0, c0, r1, c1 = prop.bbox
        local_boundary = boundaries[r0:r1, c0:c1] & (mask[r0:r1, c0:c1] == prop.label)
        boundary_grad = gradient[r0:r1, c0:c1][local_boundary]
        records.append(
            {
                "label": prop.label,
                "boundary_gradient_mean": float(boundary_grad.mean()) if boundary_grad.size else np.nan,
                "boundary_gradient_std": float(boundary_grad.std()) if boundary_grad.size else np.nan,
                "n_boundary_pixels": int(boundary_grad.size),
            }
        )
    return pd.DataFrame(records)


def compute_homogeneity(mask: np.ndarray, raw_image: np.ndarray) -> pd.DataFrame:
    """Intra-object intensity homogeneity: coefficient of variation (CV)
    of raw pixel intensities within each object. A cleanly segmented
    dot/cell tends to have fairly uniform internal signal; a high CV can
    indicate the mask spans two merged objects or leaks into background.
    """
    records = []
    for prop in regionprops(mask.astype(np.int32), intensity_image=raw_image):
        pixels = prop.image_intensity[prop.image]
        mean_i = float(pixels.mean())
        std_i = float(pixels.std())
        cv = std_i / mean_i * 100 if mean_i != 0 else np.nan
        records.append({"label": prop.label, "mean_intensity": mean_i, "std_intensity": std_i, "cv_pct": cv})
    return pd.DataFrame(records)


def _minmax_norm(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1]; a flat/degenerate series maps to a
    neutral 0.5 everywhere rather than raising a division error."""
    filled = series.fillna(series.median())
    lo, hi = filled.min(), filled.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(filled), 0.5), index=filled.index)
    return ((filled - lo) / (hi - lo)).clip(0, 1)


def compute_composite_confidence(
    mask: np.ndarray,
    raw_image: np.ndarray,
    ring_width: int = 6,
    ring_gap: int = 2,
    sensitivity_k: float = 2.0,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Combine boundary sharpness, intra-object homogeneity, and
    signal-to-surround contrast (SBR) into a single 0-1 "Composite
    Segmentation Confidence Score" per object, via min-max normalization
    of each metric across the population followed by a weighted average.

    Metrics are all oriented so that HIGHER = more confident:
    * sharpness: higher boundary gradient -> higher confidence.
    * homogeneity: lower CV -> higher confidence (so we use ``1 - norm(CV)``).
    * contrast: higher SBR -> higher confidence.
    """
    weights = weights or {"sharpness": 1.0, "homogeneity": 1.0, "contrast": 1.0}

    sharpness_df = compute_boundary_sharpness(mask, raw_image)
    homogeneity_df = compute_homogeneity(mask, raw_image)
    snr_df = snr_utils.compute_sbr_snr(mask, raw_image, ring_width, ring_gap, sensitivity_k)

    if sharpness_df.empty:
        return pd.DataFrame(
            columns=["label", "boundary_gradient_mean", "cv_pct", "SBR", "SNR", "confidence_score"]
        )

    df = sharpness_df.merge(homogeneity_df, on="label").merge(
        snr_df[["label", "SBR", "SNR", "detection_confidence"]], on="label"
    )

    sharp_n = _minmax_norm(df["boundary_gradient_mean"])
    homog_n = 1 - _minmax_norm(df["cv_pct"])
    contrast_n = _minmax_norm(df["SBR"])

    total_weight = sum(weights.values()) or 1.0
    df["confidence_score"] = (
        weights["sharpness"] * sharp_n + weights["homogeneity"] * homog_n + weights["contrast"] * contrast_n
    ) / total_weight

    return df


def flag_low_confidence(df: pd.DataFrame, percentile: float = 10.0) -> pd.DataFrame:
    """Flag objects whose confidence score falls at or below the given
    population percentile as "suspicious"."""
    out = df.copy()
    if out.empty:
        out["suspicious"] = pd.Series(dtype=bool)
        return out
    cutoff = np.percentile(out["confidence_score"], percentile)
    out["suspicious"] = out["confidence_score"] <= cutoff
    return out
