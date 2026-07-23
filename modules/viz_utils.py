"""Helpers for tracing a specific labeled object back to its location in
the original image (used by the per-object inspectors in Tabs 2 and 3).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi


def object_bbox(mask: np.ndarray, label: int, pad: int = 15) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded ``(row0, col0, row1, col1)`` bounding box for
    ``label`` in ``mask``, or ``None`` if the label isn't present.

    Uses ``scipy.ndimage.find_objects``, which locates every label's
    bounding box in a single pass -- much cheaper than iterating all of
    ``regionprops`` just to look up one object on a mask with many
    instances.
    """
    label = int(label)
    if label <= 0:
        return None
    slices = ndi.find_objects(mask, max_label=label)
    if label > len(slices) or slices[label - 1] is None:
        return None
    row_slice, col_slice = slices[label - 1]
    h, w = mask.shape
    r0 = max(row_slice.start - pad, 0)
    c0 = max(col_slice.start - pad, 0)
    r1 = min(row_slice.stop + pad, h)
    c1 = min(col_slice.stop + pad, w)
    return (r0, c0, r1, c1)
