"""Morphological & intensity feature extraction for segmented instances,
plus heuristic outlier flagging (candidate merged objects / split
fragments) based on the population statistics of the mask itself -- no
manual ground truth required.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

# `regionprops_table` property names (skimage >= 0.19 naming).
_PROPS_NO_INTENSITY = (
    "label",
    "area",
    "equivalent_diameter_area",
    "eccentricity",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "solidity",
    "centroid",
)
_PROPS_WITH_INTENSITY = _PROPS_NO_INTENSITY + ("intensity_mean", "intensity_max")

_RENAME = {
    "equivalent_diameter_area": "equivalent_diameter",
    "axis_major_length": "major_axis_length",
    "axis_minor_length": "minor_axis_length",
    "intensity_mean": "mean_intensity",
    "intensity_max": "max_intensity",
}


def compute_region_props(mask: np.ndarray, intensity_image: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Compute per-object morphology (and, if an intensity image is given,
    intensity) features with ``skimage.measure.regionprops_table``.

    Circularity (``4*pi*Area / Perimeter^2``, 1.0 = a perfect circle) is
    derived manually since it is not a built-in regionprops property.
    """
    columns = list(_PROPS_WITH_INTENSITY if intensity_image is not None else _PROPS_NO_INTENSITY)
    if mask.max() == 0:
        return pd.DataFrame(columns=[_RENAME.get(c, c) for c in columns] + ["circularity"])

    table = regionprops_table(
        mask.astype(np.int32),
        intensity_image=intensity_image,
        properties=tuple(columns),
    )
    df = pd.DataFrame(table).rename(columns=_RENAME)

    # Perimeter is 0 for degenerate (1-2 pixel) objects; guard the division.
    with np.errstate(divide="ignore", invalid="ignore"):
        circularity = 4 * np.pi * df["area"] / df["perimeter"] ** 2
    df["circularity"] = circularity.replace([np.inf, -np.inf], np.nan).clip(upper=1.0)
    return df


def _robust_zscore(series: pd.Series) -> pd.Series:
    """Median / MAD-based z-score. Unlike a mean/std z-score, this is
    resistant to being skewed by the very outliers it is meant to detect.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - median) / mad


def detect_outliers(
    df: pd.DataFrame,
    area_z_thresh: float = 2.5,
    circularity_thresh: float = 0.6,
    small_area_percentile: float = 5.0,
) -> pd.DataFrame:
    """Flag objects that are likely segmentation artifacts, using only the
    statistics of the segmented population itself:

    * **Potential merge** -- abnormally large area *and* low circularity:
      the classic signature of two touching cells segmented as one blob.
    * **Potential fragment** -- area near the bottom of the population's
      size distribution, suggesting a real object was over-segmented
      (split) or that noise was picked up as a spurious tiny object.
    """
    out = df.copy()
    if out.empty:
        out["area_zscore"] = pd.Series(dtype=float)
        out["flag"] = pd.Series(dtype=object)
        return out

    out["area_zscore"] = _robust_zscore(out["area"])
    small_area_cutoff = np.percentile(out["area"], small_area_percentile)

    is_merge = (out["area_zscore"] > area_z_thresh) & (out["circularity"] < circularity_thresh)
    is_fragment = out["area"] <= small_area_cutoff

    out["flag"] = np.where(is_merge, "Potential merge", np.where(is_fragment, "Potential fragment", "OK"))
    return out
