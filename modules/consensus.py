"""Consensus / pseudo-ground-truth generation from multiple segmentation
masks (pixel-wise majority vote, or a simplified STAPLE-style EM label
fusion), plus classical Otsu + Watershed fallback baselines for use as
extra "voters" when the user hasn't supplied at least two masks.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.segmentation import watershed


def otsu_watershed_baseline(raw_image: np.ndarray, min_distance: int = 5) -> Dict[str, np.ndarray]:
    """Generate two classic, model-free instance masks from a raw
    intensity image, used as extra consensus "voters" when fewer than two
    real segmentation masks are available.
    """
    smoothed = ndi.gaussian_filter(raw_image.astype(np.float64), sigma=1.0)
    if smoothed.max() <= smoothed.min():
        empty = np.zeros(raw_image.shape, dtype=np.int32)
        return {"Otsu (auto)": empty, "Watershed (auto)": empty.copy()}

    threshold = threshold_otsu(smoothed)
    binary = smoothed > threshold
    otsu_mask = sk_label(binary).astype(np.int32)

    # Watershed splits touching blobs using the distance transform's local
    # maxima as seed markers -- a standard classical instance-segmentation
    # baseline for round, mostly-convex objects (cells/dots).
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    markers = np.zeros_like(binary, dtype=np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    watershed_mask = watershed(-distance, markers, mask=binary).astype(np.int32)

    return {"Otsu (auto)": otsu_mask, "Watershed (auto)": watershed_mask}


def majority_vote_consensus(masks: List[np.ndarray], min_votes_fraction: float = 0.5) -> np.ndarray:
    """Pixel-wise majority vote across N binarized masks.

    A pixel is foreground in the consensus if at least
    ``min_votes_fraction`` of the input masks call it foreground. The
    resulting binary consensus is relabeled into instances via connected
    components.
    """
    vote_sum = np.zeros(masks[0].shape, dtype=np.float64)
    for m in masks:
        vote_sum += (m > 0).astype(np.float64)
    consensus_binary = vote_sum >= (min_votes_fraction * len(masks))
    return sk_label(consensus_binary).astype(np.int32)


def staple_consensus(masks: List[np.ndarray], max_iter: int = 25, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified STAPLE (Simultaneous Truth And Performance Level
    Estimation, Warfield et al. 2004) binary label fusion via
    Expectation-Maximization.

    Each rater ``r`` is characterized by a sensitivity ``p_r`` and
    specificity ``q_r``, estimated jointly with the hidden true-label
    probability map ``W`` (pixel-independent; no spatial MRF prior, as in
    the base STAPLE formulation). Returns ``(consensus_instance_mask,
    true_label_probability_map)``.
    """
    binaries = [(m > 0).astype(np.float64) for m in masks]
    n_raters = len(binaries)
    stacked = np.stack(binaries, axis=0)  # shape (n_raters, H, W)

    # Initialize the hidden true-label estimate as the simple mean vote.
    w = stacked.mean(axis=0)
    sens = np.full(n_raters, 0.9)
    spec = np.full(n_raters, 0.9)

    for _ in range(max_iter):
        w_prev = w

        # M-step: re-estimate each rater's sensitivity/specificity given W.
        denom_pos = w.sum()
        denom_neg = (1 - w).sum()
        for r in range(n_raters):
            d_r = stacked[r]
            if denom_pos > 0:
                sens[r] = np.clip((d_r * w).sum() / denom_pos, 1e-3, 1 - 1e-3)
            if denom_neg > 0:
                spec[r] = np.clip(((1 - d_r) * (1 - w)).sum() / denom_neg, 1e-3, 1 - 1e-3)

        # E-step: update the hidden true-label probability given rater performance.
        log_a = np.zeros(w.shape)
        log_b = np.zeros(w.shape)
        for r in range(n_raters):
            d_r = stacked[r]
            log_a += d_r * np.log(sens[r]) + (1 - d_r) * np.log(1 - sens[r])
            log_b += d_r * np.log(1 - spec[r]) + (1 - d_r) * np.log(spec[r])

        prior = 0.5  # uninformative prior on true foreground probability
        a = prior * np.exp(log_a)
        b = (1 - prior) * np.exp(log_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where((a + b) > 0, a / (a + b), 0.0)

        if np.abs(w - w_prev).mean() < tol:
            break

    consensus_binary = w >= 0.5
    return sk_label(consensus_binary).astype(np.int32), w


def binary_overlap_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, float]:
    """Whole-image binary (foreground vs. background) IoU and Dice,
    independent of how each mask numbers its instances.
    """
    a, b = mask_a > 0, mask_b > 0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    area_sum = int(a.sum() + b.sum())
    iou = intersection / union if union > 0 else np.nan
    dice = 2 * intersection / area_sum if area_sum > 0 else np.nan
    return {"IoU": float(iou), "Dice": float(dice), "intersection_px": intersection, "union_px": union}
