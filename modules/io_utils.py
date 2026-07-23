"""I/O utilities for loading raw fluorescence images and segmentation masks
uploaded through Streamlit (TIFF or NumPy ``.npy``), plus small helpers for
display normalization and collapsing multi-dimensional arrays to 2D.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import tifffile


def load_array_from_upload(uploaded_file) -> np.ndarray:
    """Load a numpy array from a Streamlit ``UploadedFile``.

    Supports multi-page/multi-channel TIFF (``.tif``/``.tiff``) and raw
    NumPy arrays (``.npy``). Singleton dimensions (e.g. a stray channel
    axis of length 1) are squeezed out so downstream shape checks behave
    intuitively.
    """
    name = uploaded_file.name.lower()
    buffer = io.BytesIO(uploaded_file.getvalue())

    if name.endswith((".tif", ".tiff")):
        array = tifffile.imread(buffer)
    elif name.endswith(".npy"):
        array = np.load(buffer, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    return np.squeeze(np.asarray(array))


def guess_axis_role(shape: Tuple[int, ...]) -> str:
    """Heuristic describing whether a >2D array looks like a channel-last
    stack (small trailing axis) or a generic stack (e.g. z-planes), used
    only to pre-select a sensible default reduction method in the UI.
    """
    if len(shape) == 2:
        return "2d"
    if len(shape) == 3 and shape[-1] in (2, 3, 4):
        return "channel_last"
    return "stack"


def reduce_to_2d(array: np.ndarray, method: str, index: int = 0) -> np.ndarray:
    """Collapse a >2D array (multi-channel image or z-stack) to a single 2D
    plane so it can be treated as a standard grayscale/intensity image.

    Parameters
    ----------
    array : the N-D array as loaded from disk (already squeezed).
    method : one of {"channel_last_index", "first_axis_index",
        "max_projection", "mean_projection"}.
    index : plane/channel index used by the ``*_index`` methods.
    """
    if array.ndim == 2:
        return array
    if method == "channel_last_index":
        return array[..., index]
    if method == "first_axis_index":
        return array[index, ...]
    if method == "max_projection":
        # Common for z-stacks: a maximum-intensity projection preserves
        # bright foreground puncta/cells that might sit in different planes.
        return array.max(axis=0)
    if method == "mean_projection":
        return array.mean(axis=0)
    raise ValueError(f"Unknown reduction method: {method}")


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Rescale an intensity image to uint8 [0, 255] using a robust
    (0.5-99.5 percentile) contrast stretch, for st.image/plotly display.
    Falls back to min/max, then to an all-zero image, on degenerate input.
    """
    image = image.astype(np.float64)
    lo, hi = np.percentile(image, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((image - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def array_to_npy_bytes(array: np.ndarray) -> bytes:
    """Serialize a numpy array to ``.npy`` bytes for st.download_button."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()
