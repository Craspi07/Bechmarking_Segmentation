"""Reverse Classification Accuracy (RCA) proxy.

We train a lightweight classifier to separate the primary mask's
foreground objects from randomly sampled background patches, using simple
intensity/shape/gradient/local-background features. High cross-validated
separability (ROC-AUC / PR-AUC close to 1) means the segmentation's
foreground calls are consistently, cleanly distinguishable from
background -- a statistical proxy for segmentation quality that requires
no manual ground truth, since the "labels" (foreground vs. background)
come directly from the mask being evaluated.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.filters import sobel
from skimage.measure import regionprops_table
from skimage.morphology import dilation
from skimage.morphology import disk as morph_disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import morphology, quality_metrics, snr_utils

FEATURE_COLUMNS: List[str] = [
    "mean_intensity",
    "std_intensity",
    "area",
    "equivalent_diameter",
    "circularity",
    "eccentricity",
    "gradient_mean",
    "local_background_mean",
    "contrast",
]


def extract_positive_features(
    mask: np.ndarray, raw_image: np.ndarray, ring_width: int = 6, ring_gap: int = 2
) -> pd.DataFrame:
    """Feature vector for every real detected object (the "foreground"
    / positive class), built from the mask's true shape via regionprops."""
    props_df = morphology.compute_region_props(mask, raw_image)
    if props_df.empty:
        return pd.DataFrame(columns=["label", *FEATURE_COLUMNS, "class"])

    # `morphology.compute_region_props` gives shape features + mean_intensity,
    # but not std_intensity -- pull that from the homogeneity helper instead
    # of recomputing regionprops a third time.
    homogeneity_df = quality_metrics.compute_homogeneity(mask, raw_image)[["label", "std_intensity"]]

    gradient = sobel(raw_image.astype(np.float64))
    grad_table = regionprops_table(mask.astype(np.int32), intensity_image=gradient, properties=("label", "intensity_mean"))
    grad_df = pd.DataFrame(grad_table).rename(columns={"intensity_mean": "gradient_mean"})

    snr_df = snr_utils.compute_sbr_snr(mask, raw_image, ring_width, ring_gap, sensitivity_k=2.0)

    merged = (
        props_df.merge(homogeneity_df, on="label")
        .merge(grad_df, on="label")
        .merge(snr_df[["label", "mean_background", "SBR"]], on="label")
    )
    merged = merged.rename(columns={"mean_background": "local_background_mean", "SBR": "contrast"})
    merged["class"] = 1
    return merged[["label", *FEATURE_COLUMNS, "class"]]


def sample_negative_patches(
    mask: np.ndarray,
    raw_image: np.ndarray,
    positive_radii: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    max_overlap_frac: float = 0.1,
    ring_width: int = 6,
    ring_gap: int = 2,
    max_attempts: int = 4000,
) -> pd.DataFrame:
    """Randomly sample circular background patches (comparable in size to
    the real objects) to serve as the "background" / negative class.
    Rejection-sampled so each patch mostly avoids real foreground pixels.
    """
    gradient = sobel(raw_image.astype(np.float64))
    h, w = mask.shape
    records = []
    attempts = 0
    radii_pool = positive_radii[np.isfinite(positive_radii) & (positive_radii > 0)]
    if radii_pool.size == 0:
        radii_pool = np.array([5.0])

    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        radius = max(float(rng.choice(radii_pool)), 2.0)
        if h <= 2 * radius or w <= 2 * radius:
            radius = max(min(h, w) / 4, 1.0)
        row = rng.uniform(radius, max(h - radius, radius))
        col = rng.uniform(radius, max(w - radius, radius))

        rr, cc = disk((row, col), radius, shape=mask.shape)
        if rr.size < 4:
            continue
        if np.mean(mask[rr, cc] > 0) > max_overlap_frac:
            continue

        region_mask = np.zeros(mask.shape, dtype=bool)
        region_mask[rr, cc] = True
        ring_inner = dilation(region_mask, morph_disk(int(radius) + ring_gap)) if ring_gap > 0 else region_mask
        ring_outer = dilation(region_mask, morph_disk(int(radius) + ring_gap + ring_width))
        ring = ring_outer & ~ring_inner & (mask == 0)

        bg_pixels = raw_image[ring]
        local_bg = float(bg_pixels.mean()) if bg_pixels.size >= 5 else float(raw_image.mean())
        pixels = raw_image[region_mask]
        mean_i = float(pixels.mean())

        records.append(
            {
                "label": -(len(records) + 1),
                "mean_intensity": mean_i,
                "std_intensity": float(pixels.std()),
                "area": float(region_mask.sum()),
                "equivalent_diameter": float(2 * radius),
                "circularity": 1.0,  # by construction (a disk)
                "eccentricity": 0.0,  # by construction
                "gradient_mean": float(gradient[region_mask].mean()),
                "local_background_mean": local_bg,
                "contrast": mean_i / local_bg if local_bg else np.nan,
                "class": 0,
            }
        )

    return pd.DataFrame(records)


def build_training_set(
    mask: np.ndarray,
    raw_image: np.ndarray,
    ring_width: int = 6,
    ring_gap: int = 2,
    negative_ratio: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Assemble the pseudo-labeled foreground/background dataset used to
    train the RCA proxy classifier."""
    positives = extract_positive_features(mask, raw_image, ring_width, ring_gap)
    rng = np.random.default_rng(seed)
    n_negative = max(int(round(len(positives) * negative_ratio)), 1)
    radii = (positives["equivalent_diameter"] / 2).to_numpy()
    negatives = sample_negative_patches(mask, raw_image, radii, n_negative, rng, ring_width=ring_width, ring_gap=ring_gap)
    data = pd.concat([positives, negatives], ignore_index=True)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(lambda s: s.fillna(s.median()))
    return data


def run_rca_cross_validation(
    data: pd.DataFrame,
    classifier_name: str = "Random Forest",
    n_splits: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Cross-validated ROC-AUC / PR-AUC for the foreground-vs-background
    classifier, plus feature importances from a full-data fit.
    """
    X = data[FEATURE_COLUMNS].to_numpy()
    y = data["class"].to_numpy()

    if classifier_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed)

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    precision, recall, _ = precision_recall_curve(y, probs)

    pipeline.fit(X, y)
    fitted_clf = pipeline.named_steps["clf"]
    if hasattr(fitted_clf, "feature_importances_"):
        importances = fitted_clf.feature_importances_
    else:
        importances = np.abs(fitted_clf.coef_[0])
    feature_importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "feature_importance": feature_importance,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
    }
