"""Signal-to-background / signal-to-noise validation.

For every segmented object we estimate the *local* background from a ring
of pixels dilated outward from the object footprint (excluding a small gap
and any neighboring objects' pixels), then derive per-object SBR/SNR and a
"detection confidence" that ties directly into the sidebar sensitivity
parameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.morphology import dilation, disk


def compute_sbr_snr(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    ring_width: int = 5,
    ring_gap: int = 2,
    sensitivity_k: float = 2.0,
) -> pd.DataFrame:
    """Per-object Signal-to-Background Ratio (SBR) and Signal-to-Noise
    Ratio (SNR), computed from a local background ring.

    * ``SBR`` = mean(signal) / mean(local background)
    * ``SNR`` = (mean(signal) - mean(local background)) / std(local background)
    * ``detection_confidence`` = fraction of the object's own pixels whose
      intensity exceeds ``mean(background) + sensitivity_k * std(background)``
      -- a proxy for how confidently each object stands out from its local
      background at the chosen sensitivity.

    Working on a small padded bounding-box crop per object (rather than the
    whole image) keeps this fast even for images with thousands of objects.
    """
    records = []
    props = regionprops(mask.astype(np.int32))
    pad = ring_width + ring_gap + 1

    for prop in props:
        r0, c0, r1, c1 = prop.bbox
        r0p, c0p = max(r0 - pad, 0), max(c0 - pad, 0)
        r1p, c1p = min(r1 + pad, mask.shape[0]), min(c1 + pad, mask.shape[1])

        local_mask = mask[r0p:r1p, c0p:c1p]
        local_intensity = intensity_image[r0p:r1p, c0p:c1p]
        object_mask = local_mask == prop.label

        # Ring = dilate(object, gap+width) minus dilate(object, gap) minus
        # any pixels belonging to a *different* object.
        gap_zone = dilation(object_mask, disk(ring_gap)) if ring_gap > 0 else object_mask
        ring_outer = dilation(object_mask, disk(ring_gap + ring_width))
        ring = ring_outer & ~gap_zone & (local_mask == 0)

        signal_pixels = local_intensity[object_mask]
        bg_pixels = local_intensity[ring]

        mean_signal = float(signal_pixels.mean())

        if bg_pixels.size >= 5:
            mean_bg = float(bg_pixels.mean())
            std_bg = float(bg_pixels.std())
        else:
            mean_bg, std_bg = np.nan, np.nan

        sbr = mean_signal / mean_bg if (not np.isnan(mean_bg) and mean_bg != 0) else np.nan
        snr = (mean_signal - mean_bg) / std_bg if (not np.isnan(std_bg) and std_bg > 0) else np.nan

        if not np.isnan(mean_bg) and not np.isnan(std_bg):
            threshold = mean_bg + sensitivity_k * std_bg
            detection_confidence = float(np.mean(signal_pixels > threshold))
        else:
            threshold, detection_confidence = np.nan, np.nan

        records.append(
            {
                "label": prop.label,
                "area": prop.area,
                "mean_signal": mean_signal,
                "mean_background": mean_bg,
                "std_background": std_bg,
                "SBR": sbr,
                "SNR": snr,
                "local_threshold": threshold,
                "detection_confidence": detection_confidence,
                "n_background_pixels": int(bg_pixels.size),
            }
        )

    return pd.DataFrame.from_records(records)
