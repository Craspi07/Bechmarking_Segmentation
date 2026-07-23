"""Aggregate summary statistics and condition-vs-condition comparisons for
replicate / control-vs-treatment consistency checks.

With only a single image per condition, classical replicate statistics
aren't directly available. We generate spatial *pseudo-replicates* by
tiling each image into an N x N grid and counting objects per tile -- this
lets us estimate within-image variability (CV) and run a real significance
test between two conditions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import regionprops


def summarize_counts(mask: np.ndarray, pixel_size_um: float = 1.0) -> Dict[str, float]:
    """Whole-image summary: total object count and density per 100 um^2."""
    labels = np.unique(mask)
    count = int(np.count_nonzero(labels))
    area_px = mask.shape[0] * mask.shape[1]
    area_um2 = area_px * (pixel_size_um ** 2)
    density_per_100um2 = (count / area_um2) * 100 if area_um2 > 0 else np.nan
    return {
        "total_count": count,
        "image_area_um2": area_um2,
        "density_per_100um2": density_per_100um2,
    }


def grid_region_counts(mask: np.ndarray, grid_size: int = 4) -> pd.DataFrame:
    """Split the mask into a ``grid_size x grid_size`` set of tiles and
    count objects whose centroid falls in each tile. These tile counts
    serve as spatial pseudo-replicates for estimating within-image
    variability (CV) and for significance testing when only a single image
    is available per condition.
    """
    h, w = mask.shape
    row_edges = np.linspace(0, h, grid_size + 1)
    col_edges = np.linspace(0, w, grid_size + 1)

    counts = np.zeros((grid_size, grid_size), dtype=int)
    for prop in regionprops(mask.astype(np.int32)):
        cr, cc = prop.centroid
        ri = min(int(np.searchsorted(row_edges, cr, side="right") - 1), grid_size - 1)
        ci = min(int(np.searchsorted(col_edges, cc, side="right") - 1), grid_size - 1)
        counts[ri, ci] += 1

    tiles = [
        {"row": r, "col": c, "count": int(counts[r, c])}
        for r in range(grid_size)
        for c in range(grid_size)
    ]
    return pd.DataFrame(tiles)


def coefficient_of_variation(values: np.ndarray) -> float:
    """CV expressed as a percentage; NaN for a zero mean (undefined)."""
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    return float(values.std() / mean * 100) if mean != 0 else np.nan


def compare_conditions(counts_a: np.ndarray, counts_b: np.ndarray) -> Dict[str, float]:
    """Non-parametric (Mann-Whitney U) and parametric (Welch's t-test)
    comparison of tile-level counts between two conditions.
    """
    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)
    result = {
        "mean_a": float(counts_a.mean()) if counts_a.size else np.nan,
        "mean_b": float(counts_b.mean()) if counts_b.size else np.nan,
        "cv_a": coefficient_of_variation(counts_a),
        "cv_b": coefficient_of_variation(counts_b),
    }
    if len(counts_a) >= 2 and len(counts_b) >= 2 and (counts_a.std() > 0 or counts_b.std() > 0):
        u_stat, u_p = stats.mannwhitneyu(counts_a, counts_b, alternative="two-sided")
        t_stat, t_p = stats.ttest_ind(counts_a, counts_b, equal_var=False)
        result.update({"mannwhitney_p": float(u_p), "welch_t_p": float(t_p)})
    else:
        result.update({"mannwhitney_p": np.nan, "welch_t_p": np.nan})
    return result
