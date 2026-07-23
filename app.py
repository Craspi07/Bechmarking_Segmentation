"""
Segmentation QC Benchmark
=========================

A Streamlit app for benchmarking cell/particle instance-segmentation
quality WITHOUT a full manual ground-truth dataset. It combines five
complementary, no-ground-truth-required validation strategies:

1. Crop & Count Micro-Validation   -- spot-check a small, fully-verifiable patch.
2. Morphological & Intensity QC    -- flag statistical outliers (merges/fragments).
3. SNR & Contrast Validation       -- relate detection confidence to local SNR.
4. Synthetic Dot Benchmark         -- exact-ground-truth stress test.
5. Replicate Consistency           -- compare two conditions/replicates.

Input: a raw fluorescence image (TIFF) and a predicted instance
segmentation mask (TIFF or .npy, 0 = background, unique int > 0 = one
instance each).
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tifffile
from skimage.color import label2rgb
from skimage.measure import regionprops

from modules import io_utils, morphology, snr_utils, stats_utils, synthetic, viz_utils

st.set_page_config(page_title="Segmentation QC Benchmark", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar -- data input & global parameters (shared by every tab)
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 Segmentation QC Benchmark")
st.sidebar.markdown(
    "Upload a raw image and its predicted instance mask to quality-check "
    "segmentation without a full manual ground truth."
)

st.sidebar.header("1. Data input")
raw_file = st.sidebar.file_uploader("Raw fluorescence image (TIFF)", type=["tif", "tiff"], key="raw_upload")
mask_file = st.sidebar.file_uploader(
    "Predicted segmentation mask (TIFF or .npy)", type=["tif", "tiff", "npy"], key="mask_upload"
)


def _load_and_reduce(uploaded_file, label_prefix: str, is_label_mask: bool = False) -> np.ndarray:
    """Load an uploaded file and, if it has more than 2 dimensions, let the
    user pick how to collapse it to a single 2D plane (multi-channel
    intensity images vs. z-stack label masks need different defaults).
    """
    array = io_utils.load_array_from_upload(uploaded_file)
    if array.ndim == 2:
        return array

    role = io_utils.guess_axis_role(array.shape)
    st.sidebar.caption(f"{label_prefix}: shape {array.shape} detected -- select how to flatten to 2D.")

    if is_label_mask:
        # Label masks should never be projected (that would corrupt integer
        # instance IDs) -- only a single plane/channel selection makes sense.
        method = "channel_last_index" if role == "channel_last" else "first_axis_index"
        axis_len = array.shape[-1] if method == "channel_last_index" else array.shape[0]
        index = st.sidebar.number_input(
            f"{label_prefix} plane/channel index", 0, max(axis_len - 1, 0), 0, key=f"{label_prefix}_index"
        )
        return io_utils.reduce_to_2d(array, method, int(index))

    options = ["max_projection", "mean_projection", "channel_last_index", "first_axis_index"]
    default_idx = 2 if role == "channel_last" else 0
    method = st.sidebar.selectbox(f"{label_prefix} reduction", options, index=default_idx, key=f"{label_prefix}_method")
    index = 0
    if method in ("channel_last_index", "first_axis_index"):
        axis_len = array.shape[-1] if method == "channel_last_index" else array.shape[0]
        index = st.sidebar.number_input(f"{label_prefix} index", 0, max(axis_len - 1, 0), 0, key=f"{label_prefix}_idx2")
    return io_utils.reduce_to_2d(array, method, int(index))


def _tiff_bytes(array: np.ndarray) -> bytes:
    """Serialize a numpy array to TIFF bytes for st.download_button."""
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, array)
    return buffer.getvalue()


def render_flag_overlay(raw_img_local, mask_img_local, flagged_df_local):
    """Whole-image overlay showing every detected object at once, colored
    by its outlier flag -- the "big picture" view of where problems cluster,
    complementing the single-object inspector below it.
    """
    if raw_img_local is not None:
        base = io_utils.normalize_for_display(raw_img_local)
    else:
        base = np.where(mask_img_local > 0, 255, 0).astype(np.uint8)
    rgb = np.repeat(base[:, :, None], 3, axis=2).astype(np.float64)

    color_map = {"OK": (0, 200, 0), "Potential fragment": (255, 165, 0), "Potential merge": (255, 0, 0)}
    alpha = 0.45
    for flag_name, color in color_map.items():
        labels_for_flag = flagged_df_local.loc[flagged_df_local["flag"] == flag_name, "label"].astype(int).to_numpy()
        if labels_for_flag.size == 0:
            continue
        flag_mask = np.isin(mask_img_local, labels_for_flag)
        for c in range(3):
            rgb[:, :, c] = np.where(flag_mask, rgb[:, :, c] * (1 - alpha) + color[c] * alpha, rgb[:, :, c])

    fig = px.imshow(rgb.astype(np.uint8), title="All objects, colored by outlier flag")
    st.plotly_chart(fig, width="stretch")
    st.caption("🟢 OK &nbsp;&nbsp; 🟠 Potential fragment &nbsp;&nbsp; 🔴 Potential merge", unsafe_allow_html=True)


def render_snr_overlay(raw_img_local, mask_img_local, snr_df_local):
    """Whole-image overview marking every object's centroid, colored by its
    SNR value -- lets you spot at a glance where low-confidence detections
    cluster before drilling into any single object.
    """
    centroids = {p.label: p.centroid for p in regionprops(mask_img_local.astype(np.int32))}
    labels = [l for l in snr_df_local["label"] if l in centroids]
    if not labels:
        return
    xs = [centroids[l][1] for l in labels]
    ys = [centroids[l][0] for l in labels]
    snr_values = snr_df_local.set_index("label").loc[labels, "SNR"]

    fig = px.imshow(io_utils.normalize_for_display(raw_img_local), color_continuous_scale="gray", title="All objects, colored by SNR")
    fig.update_coloraxes(showscale=False)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=10,
                color=snr_values,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="SNR"),
                line=dict(width=1, color="black"),
            ),
            text=[f"label {l}, SNR={s:.2f}" for l, s in zip(labels, snr_values)],
            name="objects",
        )
    )
    st.plotly_chart(fig, width="stretch")


def render_object_inspector(raw_img_local, mask_img_local, df, key_prefix, default_label=None):
    """Let the user pick an object label from ``df`` and see exactly where
    it sits in the full image, plus a zoomed, mask-overlaid crop -- so a
    flagged/suspicious object can be visually judged against the raw data
    instead of trusting the numbers alone.
    """
    if df.empty or "label" not in df.columns:
        st.info("No objects available to inspect.")
        return

    labels_sorted = sorted(int(l) for l in df["label"].tolist())
    default_index = labels_sorted.index(int(default_label)) if default_label in labels_sorted else 0
    selected_label = st.selectbox(
        "Select object label to inspect", labels_sorted, index=default_index, key=f"{key_prefix}_inspect_label"
    )

    bbox = viz_utils.object_bbox(mask_img_local, selected_label, pad=20)
    if bbox is None:
        st.warning("Selected object was not found in the mask.")
        return
    r0, c0, r1, c1 = bbox
    mask_crop = mask_img_local[r0:r1, c0:c1]
    highlight = (mask_crop == selected_label).astype(np.int32)

    if raw_img_local is not None:
        full_display = io_utils.normalize_for_display(raw_img_local)
        crop_display = io_utils.normalize_for_display(raw_img_local[r0:r1, c0:c1])
        full_title = f"Full raw image -- object {selected_label} location"
    else:
        # No raw image available (mask-only mode): fall back to a binary
        # foreground map so the object can still be located spatially.
        full_display = np.where(mask_img_local > 0, 255, 0).astype(np.uint8)
        crop_display = np.where(mask_crop > 0, 255, 0).astype(np.uint8)
        full_title = f"Full mask footprint -- object {selected_label} location"

    full_fig = px.imshow(full_display, color_continuous_scale="gray", title=full_title)
    full_fig.add_shape(type="rect", x0=c0, y0=r0, x1=c1, y1=r1, line=dict(color="red", width=2))
    st.plotly_chart(full_fig, width="stretch")

    overlay = label2rgb(highlight, image=crop_display, bg_label=0, alpha=0.5, colors=["red"])
    crop_col1, crop_col2 = st.columns(2)
    with crop_col1:
        st.plotly_chart(px.imshow(crop_display, color_continuous_scale="gray", title="Zoomed crop"), width="stretch")
    with crop_col2:
        st.plotly_chart(px.imshow(overlay, title=f"Object {selected_label} highlighted"), width="stretch")

    st.dataframe(df[df["label"] == selected_label], width="stretch")


raw_img, mask_img = None, None
if raw_file is not None:
    raw_img = _load_and_reduce(raw_file, "Raw image").astype(np.float64)
if mask_file is not None:
    mask_img = _load_and_reduce(mask_file, "Mask", is_label_mask=True).astype(np.int32)

st.sidebar.header("2. Global parameters")
pixel_size_um = st.sidebar.number_input(
    "Pixel size (microns / pixel)", min_value=0.0001, value=0.1075, step=0.0005, format="%.4f"
)
sensitivity_k = st.sidebar.slider(
    "Detection sensitivity k (x background sigma)",
    0.5,
    6.0,
    2.0,
    0.1,
    help="An object pixel counts as 'confidently detected' if its intensity exceeds "
    "local_background_mean + k * local_background_std. Used in Tab 3.",
)
ring_width = st.sidebar.slider("Background ring width (px)", 1, 40, 6, help="Used in Tab 3's SNR/SBR estimate.")
ring_gap = st.sidebar.slider("Ring gap from object edge (px)", 0, 15, 2, help="Used in Tab 3's SNR/SBR estimate.")

st.sidebar.subheader("Outlier flagging (Tab 2)")
area_z_thresh = st.sidebar.slider("Area robust z-score threshold", 1.0, 6.0, 2.5, 0.1)
circularity_thresh = st.sidebar.slider("Circularity threshold (merge flag)", 0.0, 1.0, 0.6, 0.05)
small_area_pct = st.sidebar.slider("Small-object percentile (fragment flag)", 1, 25, 5)

if mask_img is not None and raw_img is not None and mask_img.shape != raw_img.shape:
    st.sidebar.error(f"Shape mismatch: raw {raw_img.shape} vs mask {mask_img.shape}. Downstream tabs may fail.")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1 · Crop & Count",
        "2 · Morphology & Intensity",
        "3 · SNR & Contrast",
        "4 · Synthetic Benchmark",
        "5 · Replicate Consistency",
    ]
)

# ===========================================================================
# TAB 1 -- Crop & Count Micro-Validation
# ===========================================================================
with tab1:
    st.header("Crop & Count Micro-Validation")
    st.markdown(
        "Spot-check segmentation quality on a small patch that is small enough to fully "
        "verify by eye, without hand-annotating the whole image."
    )

    if raw_img is None or mask_img is None:
        st.info("Upload both a raw image and a mask in the sidebar to use this tab.")
    else:
        h, w = raw_img.shape
        patch_mode = st.radio("Patch selection", ["Manual bounding box", "Random N x N patch"], horizontal=True)

        if patch_mode == "Manual bounding box":
            c1, c2 = st.columns(2)
            with c1:
                row0, row1 = st.slider("Row range", 0, h, (0, min(150, h)))
            with c2:
                col0, col1 = st.slider("Column range", 0, w, (0, min(150, w)))
        else:
            patch_n = st.number_input("Patch size N (px)", 20, int(min(h, w)), min(100, int(min(h, w))), 10)
            if st.button("Draw new random patch") or "patch_coords" not in st.session_state:
                rng = np.random.default_rng()
                r0 = int(rng.integers(0, max(h - patch_n, 1)))
                c0 = int(rng.integers(0, max(w - patch_n, 1)))
                st.session_state["patch_coords"] = (r0, r0 + patch_n, c0, c0 + patch_n)
            row0, row1, col0, col1 = st.session_state["patch_coords"]

        row0, row1 = sorted((row0, row1))
        col0, col1 = sorted((col0, col1))

        if row1 - row0 < 2 or col1 - col0 < 2:
            st.warning("Patch too small; widen the selection.")
        else:
            raw_crop = raw_img[row0:row1, col0:col1]
            mask_crop = mask_img[row0:row1, col0:col1]
            crop_labels = np.unique(mask_crop)
            predicted_count = int(np.count_nonzero(crop_labels))

            disp_raw = io_utils.normalize_for_display(raw_crop)
            overlay = label2rgb(mask_crop, image=disp_raw, bg_label=0, alpha=0.4)

            cc1, cc2 = st.columns(2)
            with cc1:
                fig_raw = px.imshow(disp_raw, color_continuous_scale="gray", title="Raw crop")
                st.plotly_chart(fig_raw, width="stretch")
            with cc2:
                fig_overlay = px.imshow(overlay, title=f"Mask overlay ({predicted_count} objects)")
                st.plotly_chart(fig_overlay, width="stretch")

            st.metric("Objects detected by mask in this patch", predicted_count)

            verify_mode = st.radio(
                "Verification mode", ["Quick manual count", "Detailed point verification"], horizontal=True
            )

            if verify_mode == "Quick manual count":
                manual_count = st.number_input(
                    "Manual reference count (count dots/cells yourself in the raw crop)", 0, 100000, predicted_count
                )
                tp = min(predicted_count, manual_count)
                fp = max(predicted_count - manual_count, 0)
                fn = max(manual_count - predicted_count, 0)
            else:
                st.caption(
                    "Click a marker below to flag it as a **false positive** (wrong/spurious detection). "
                    "Then enter how many real objects you can see with no marker nearby (**false negatives**)."
                )
                centroids = regionprops(mask_crop.astype(np.int32))
                scatter_fig = px.imshow(disp_raw, color_continuous_scale="gray")
                scatter_fig.add_trace(
                    go.Scatter(
                        x=[p.centroid[1] for p in centroids],
                        y=[p.centroid[0] for p in centroids],
                        mode="markers",
                        marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=2)),
                        text=[f"label {p.label}" for p in centroids],
                        name="detected objects",
                    )
                )
                scatter_fig.update_layout(title="Click a marker to flag a false positive")
                selection = st.plotly_chart(
                    scatter_fig, width="stretch", on_select="rerun", selection_mode="points", key="tab1_click"
                )
                flagged = len(selection["selection"]["points"]) if selection and selection.get("selection") else 0
                st.write(f"Flagged as false positive: **{flagged}**")
                fn = st.number_input("Missed objects (visible in raw image, no marker nearby)", 0, 100000, 0)
                fp = flagged
                tp = max(predicted_count - fp, 0)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Precision", f"{precision:.2f}")
            m2.metric("Recall", f"{recall:.2f}")
            m3.metric("F1-score", f"{f1:.2f}")

# ===========================================================================
# TAB 2 -- Morphological & Intensity Consistency Analysis
# ===========================================================================
with tab2:
    st.header("Morphological & Intensity Consistency Analysis")

    if mask_img is None:
        st.info("Upload a segmentation mask in the sidebar to use this tab.")
    else:
        props_df = morphology.compute_region_props(mask_img, raw_img)
        if props_df.empty:
            st.warning("No labeled objects found in the mask.")
        else:
            flagged_df = morphology.detect_outliers(props_df, area_z_thresh, circularity_thresh, small_area_pct)

            st.subheader("Per-object feature table")
            st.dataframe(flagged_df, width="stretch")

            st.subheader("Distributions")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(px.histogram(flagged_df, x="area", nbins=40, title="Area distribution"), width="stretch")
                st.plotly_chart(
                    px.box(flagged_df, y="circularity", points="all", title="Circularity distribution"),
                    width="stretch",
                )
            with col2:
                if "mean_intensity" in flagged_df:
                    st.plotly_chart(
                        px.histogram(flagged_df, x="mean_intensity", nbins=40, title="Mean intensity distribution"),
                        width="stretch",
                    )
                st.plotly_chart(
                    px.box(flagged_df, y="equivalent_diameter", points="all", title="Equivalent diameter distribution"),
                    width="stretch",
                )

            if "mean_intensity" in flagged_df:
                st.subheader("Area vs. Mean Intensity")
                fig = px.scatter(
                    flagged_df,
                    x="area",
                    y="mean_intensity",
                    color="flag",
                    hover_data=["label", "circularity"],
                    title="Area vs. Mean Intensity (colored by outlier flag)",
                )
                st.plotly_chart(fig, width="stretch")

            st.subheader("Automated outlier detection")

            with st.expander("What do 'Potential merge' and 'Potential fragment' mean?"):
                st.markdown(
                    f"""
**Potential merge** -- the object's area is a statistical outlier (robust z-score above the sidebar's
*Area robust z-score threshold*, currently **{area_z_thresh:.1f}**) **and** its circularity is below the
*Circularity threshold* (currently **{circularity_thresh:.2f}**). This is the classic signature of two
touching/overlapping cells segmented as a single blob: real cells tend to be fairly round, so an object
that is both unusually large **and** irregularly shaped is suspicious.

**Potential fragment** -- the object's area falls at or below the sidebar's *Small-object percentile*
cutoff (currently the smallest **{small_area_pct}%** of detected objects by area). This flag does **not**
automatically mean the object is wrong -- it can indicate any of:
1. **Over-segmentation / splitting** -- a single real object was incorrectly cut into multiple smaller pieces.
2. **Spurious detection** -- noise, debris, or an imaging artifact was picked up as if it were a real object.
3. **A genuinely small, valid object** in a population that has a wide natural size range.

Use the object inspector below to zoom into a flagged object on the raw image and judge which case applies --
statistics alone can't tell these three apart.
                    """
                )

            n_merge = int((flagged_df["flag"] == "Potential merge").sum())
            n_fragment = int((flagged_df["flag"] == "Potential fragment").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Total objects", len(flagged_df))
            m2.metric("Potential merges", n_merge)
            m3.metric("Potential fragments", n_fragment)

            suspicious = flagged_df[flagged_df["flag"] != "OK"].sort_values("flag")
            st.dataframe(suspicious, width="stretch")
            if not suspicious.empty:
                st.download_button(
                    "Download suspicious objects (CSV)",
                    suspicious.to_csv(index=False),
                    "suspicious_objects.csv",
                    "text/csv",
                )

            st.subheader("Full image overview")
            render_flag_overlay(raw_img, mask_img, flagged_df)

            st.subheader("Trace an object back to the original image")
            default_lbl = int(suspicious["label"].iloc[0]) if not suspicious.empty else int(flagged_df["label"].iloc[0])
            render_object_inspector(raw_img, mask_img, flagged_df, "tab2", default_label=default_lbl)

# ===========================================================================
# TAB 3 -- Signal-to-Noise & Contrast Ratio Validation
# ===========================================================================
with tab3:
    st.header("Signal-to-Noise & Contrast Ratio Validation")

    if mask_img is None or raw_img is None:
        st.info("Upload both a raw image and a mask in the sidebar to use this tab.")
    else:
        snr_df = snr_utils.compute_sbr_snr(mask_img, raw_img, ring_width, ring_gap, sensitivity_k)
        if snr_df.empty:
            st.warning("No labeled objects found in the mask.")
        else:
            st.subheader("Per-object SBR / SNR table")
            st.dataframe(snr_df, width="stretch")

            st.subheader("Full image overview")
            render_snr_overlay(raw_img, mask_img, snr_df)

            st.subheader("Trace an object back to the original image")
            valid_snr = snr_df.dropna(subset=["SNR"])
            default_lbl = (
                int(valid_snr.sort_values("SNR").iloc[0]["label"]) if not valid_snr.empty else int(snr_df["label"].iloc[0])
            )
            render_object_inspector(raw_img, mask_img, snr_df, "tab3", default_label=default_lbl)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    px.histogram(snr_df, x="SBR", nbins=40, title="Signal-to-Background Ratio (SBR) distribution"),
                    width="stretch",
                )
            with col2:
                st.plotly_chart(
                    px.histogram(snr_df, x="SNR", nbins=40, title="Signal-to-Noise Ratio (SNR) distribution"),
                    width="stretch",
                )

            st.subheader("Detection confidence & area stability vs. local SNR")
            st.caption(
                "Objects are grouped into SNR bins. For each bin we report the mean detection confidence "
                f"(fraction of object pixels above background_mean + {sensitivity_k:.1f} x background_std) and "
                "the coefficient of variation of object area, as a proxy for segmentation stability at low SNR."
            )
            valid = snr_df.dropna(subset=["SNR"]).copy()
            if len(valid) >= 4:
                n_bins = min(6, valid["SNR"].nunique())
                valid["SNR_bin"] = pd.qcut(valid["SNR"], q=n_bins, duplicates="drop")
                binned = (
                    valid.groupby("SNR_bin", observed=True)
                    .agg(
                        mean_detection_confidence=("detection_confidence", "mean"),
                        mean_area=("area", "mean"),
                        area_cv_pct=("area", lambda s: s.std() / s.mean() * 100 if s.mean() else np.nan),
                        n_objects=("label", "count"),
                    )
                    .reset_index()
                )
                binned["SNR_bin"] = binned["SNR_bin"].astype(str)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=binned["SNR_bin"],
                        y=binned["mean_detection_confidence"],
                        name="Mean detection confidence",
                        mode="lines+markers",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=binned["SNR_bin"],
                        y=binned["area_cv_pct"] / 100,
                        name="Area CV (fraction)",
                        mode="lines+markers",
                    )
                )
                fig.update_layout(
                    title="Detection confidence & area stability across SNR bins",
                    xaxis_title="SNR bin (low -> high)",
                    yaxis_title="Fraction",
                )
                st.plotly_chart(fig, width="stretch")
                st.dataframe(binned, width="stretch")
            else:
                st.info("Need at least 4 objects with a valid local background estimate to bin by SNR.")

# ===========================================================================
# TAB 4 -- Synthetic Dot Simulation Benchmark
# ===========================================================================
with tab4:
    st.header("Synthetic Dot Simulation Benchmark")
    st.markdown(
        "Generate a synthetic fluorescence image with an **exact, known ground truth** to benchmark "
        "detection/segmentation performance where no real annotated data exists."
    )

    st.subheader("Generator parameters")
    g1, g2, g3 = st.columns(3)
    with g1:
        grid_h = st.number_input("Image height (px)", 64, 2048, 512, 32)
        grid_w = st.number_input("Image width (px)", 64, 2048, 512, 32)
        n_dots = st.number_input("Number of dots", 1, 5000, 150, 5)
    with g2:
        radius_mean = st.number_input("Dot radius mean (px)", 1.0, 50.0, 5.0, 0.5)
        radius_std = st.number_input("Dot radius std (px)", 0.0, 20.0, 1.5, 0.1)
        blur_sigma = st.number_input("PSF Gaussian blur sigma (px)", 0.0, 10.0, 1.2, 0.1)
    with g3:
        poisson_scale = st.number_input("Poisson noise scale (higher = less shot noise)", 0.01, 100.0, 1.0, 0.1)
        gaussian_noise_sigma = st.number_input("Gaussian read-noise sigma", 0.0, 100.0, 5.0, 0.5)
        seed = st.number_input("Random seed", 0, 999999, 42, 1)

    if st.button("Generate synthetic dataset"):
        synth_raw, synth_gt = synthetic.generate_synthetic_dots(
            shape=(int(grid_h), int(grid_w)),
            n_dots=int(n_dots),
            radius_mean=radius_mean,
            radius_std=radius_std,
            blur_sigma=blur_sigma,
            poisson_scale=poisson_scale,
            gaussian_noise_sigma=gaussian_noise_sigma,
            seed=int(seed),
        )
        st.session_state["synth_raw"] = synth_raw
        st.session_state["synth_gt"] = synth_gt

    if "synth_raw" in st.session_state:
        synth_raw = st.session_state["synth_raw"]
        synth_gt = st.session_state["synth_gt"]
        n_gt = int(synth_gt.max())

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.imshow(io_utils.normalize_for_display(synth_raw), color_continuous_scale="gray", title="Synthetic raw image"),
                width="stretch",
            )
        with c2:
            gt_overlay = label2rgb(synth_gt, image=io_utils.normalize_for_display(synth_raw), bg_label=0, alpha=0.4)
            st.plotly_chart(px.imshow(gt_overlay, title=f"Ground truth ({n_gt} objects)"), width="stretch")

        st.caption("Download the synthetic data to run through an external pipeline, or segment it in-app below.")
        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            st.download_button(
                "Raw image (.npy)", io_utils.array_to_npy_bytes(synth_raw), "synthetic_raw.npy", key="dl_raw_npy"
            )
        with dl2:
            st.download_button(
                "Raw image (.tif)", _tiff_bytes(synth_raw.astype(np.float32)), "synthetic_raw.tif",
                "image/tiff", key="dl_raw_tif",
            )
        with dl3:
            st.download_button(
                "Ground truth (.npy)", io_utils.array_to_npy_bytes(synth_gt), "synthetic_ground_truth.npy", key="dl_gt_npy"
            )
        with dl4:
            st.download_button(
                "Ground truth (.tif)", _tiff_bytes(synth_gt.astype(np.int32)), "synthetic_ground_truth.tif",
                "image/tiff", key="dl_gt_tif",
            )

        st.subheader("Evaluate a predicted mask against this ground truth")
        eval_source = st.radio(
            "Prediction source", ["Upload a predicted mask", "Run Cellpose in-app"], horizontal=True, key="synth_eval_source"
        )

        pred_mask_for_eval = None

        if eval_source == "Upload a predicted mask":
            st.caption("Run your segmentation pipeline on the downloaded raw image, then upload its predicted mask here.")
            pred_file = st.file_uploader(
                "Predicted mask for the synthetic image (TIFF or .npy)", type=["tif", "tiff", "npy"], key="synth_pred_upload"
            )
            if pred_file is not None:
                pred_mask = io_utils.load_array_from_upload(pred_file)
                if pred_mask.ndim != 2:
                    pred_mask = io_utils.reduce_to_2d(pred_mask, "first_axis_index", 0)
                if pred_mask.shape != synth_gt.shape:
                    st.error(f"Predicted mask shape {pred_mask.shape} does not match synthetic image shape {synth_gt.shape}.")
                else:
                    pred_mask_for_eval = pred_mask.astype(np.int32)

        else:
            st.caption(
                "Run a Cellpose model directly on the synthetic image -- either a built-in pretrained model or your "
                "own trained model file -- then immediately QC-check the result against the exact synthetic ground truth."
            )
            cp1, cp2 = st.columns(2)
            with cp1:
                model_source = st.radio(
                    "Model source", ["Built-in pretrained model", "Upload a custom-trained model"], key="cp_model_source"
                )
                if model_source == "Built-in pretrained model":
                    builtin_model_name = st.selectbox(
                        "Pretrained model name",
                        ["cpsam_v2", "cyto3", "cyto2", "cyto", "nuclei", "cpdino"],
                        help="Exact names available depend on your installed Cellpose version. "
                        "'cpsam_v2' is the current default (Cellpose-SAM) model as of Cellpose >= 4.",
                    )
                    custom_model_file = None
                else:
                    builtin_model_name = None
                    custom_model_file = st.file_uploader("Cellpose model file (trained checkpoint)", key="cp_model_file")
            with cp2:
                diameter_px = st.number_input("Cell/dot diameter (px, 0 = auto-estimate)", 0.0, 500.0, 0.0, 1.0)
                flow_threshold = st.slider("Flow threshold", 0.0, 3.0, 0.4, 0.05)
                cellprob_threshold = st.slider("Cell probability threshold", -6.0, 6.0, 0.0, 0.5)
                use_gpu = st.checkbox("Use GPU if available", value=False)

            if st.button("Run Cellpose segmentation on synthetic image"):
                try:
                    from cellpose import models as cellpose_models
                except ImportError:
                    st.error(
                        "Cellpose is not installed in this environment. Install it with `pip install cellpose` "
                        "(already listed in requirements.txt) to use in-app inference."
                    )
                else:
                    if model_source == "Upload a custom-trained model" and custom_model_file is None:
                        st.warning("Please upload a model file first.")
                    else:
                        with st.spinner("Running Cellpose inference -- this can take a while on CPU..."):
                            try:
                                if model_source == "Upload a custom-trained model":
                                    tmp_dir = tempfile.mkdtemp(prefix="cellpose_model_")
                                    tmp_path = os.path.join(tmp_dir, custom_model_file.name)
                                    with open(tmp_path, "wb") as fh:
                                        fh.write(custom_model_file.getvalue())
                                    model = cellpose_models.CellposeModel(gpu=use_gpu, pretrained_model=tmp_path)
                                else:
                                    model = cellpose_models.CellposeModel(gpu=use_gpu, model_type=builtin_model_name)

                                diam = None if diameter_px == 0 else float(diameter_px)
                                masks_out, _flows_out, _styles_out = model.eval(
                                    synth_raw,
                                    diameter=diam,
                                    flow_threshold=flow_threshold,
                                    cellprob_threshold=cellprob_threshold,
                                )
                                cellpose_mask = np.asarray(masks_out).astype(np.int32)
                                st.session_state["cellpose_pred_mask"] = cellpose_mask
                                st.session_state["cellpose_pred_shape"] = cellpose_mask.shape
                                st.success(f"Cellpose detected {int(cellpose_mask.max())} objects.")
                            except Exception as exc:  # noqa: BLE001 -- surface any model/runtime error to the user
                                st.error(f"Cellpose run failed: {exc}")

            cached_pred = st.session_state.get("cellpose_pred_mask")
            if cached_pred is not None and cached_pred.shape == synth_gt.shape:
                pred_mask_for_eval = cached_pred
                cp_overlay = label2rgb(
                    pred_mask_for_eval, image=io_utils.normalize_for_display(synth_raw), bg_label=0, alpha=0.4
                )
                st.plotly_chart(
                    px.imshow(cp_overlay, title=f"Cellpose prediction ({int(pred_mask_for_eval.max())} objects)"),
                    width="stretch",
                )
            elif cached_pred is not None:
                st.warning(
                    f"Cached Cellpose prediction shape {cached_pred.shape} no longer matches the current synthetic "
                    f"image shape {synth_gt.shape} -- re-run Cellpose above."
                )

        if pred_mask_for_eval is not None:
            eval_df = synthetic.evaluate_against_ground_truth(synth_gt, pred_mask_for_eval)
            st.dataframe(eval_df, width="stretch")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eval_df["iou_threshold"], y=eval_df["AP"], name="AP", mode="lines+markers"))
            fig.add_trace(
                go.Scatter(x=eval_df["iou_threshold"], y=eval_df["precision"], name="Precision", mode="lines+markers")
            )
            fig.add_trace(
                go.Scatter(
                    x=eval_df["iou_threshold"], y=eval_df["recall"], name="Recall (detection rate)", mode="lines+markers"
                )
            )
            fig.update_layout(
                title="AP / Precision / Recall vs. IoU threshold", xaxis_title="IoU threshold", yaxis_title="Score"
            )
            st.plotly_chart(fig, width="stretch")

            at_50 = eval_df[eval_df["iou_threshold"] == 0.5].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AP @ IoU 0.5", f"{at_50['AP']:.3f}")
            m2.metric("Precision @ 0.5", f"{at_50['precision']:.3f}")
            m3.metric("Recall @ 0.5", f"{at_50['recall']:.3f}")
            m4.metric("Mean AP (0.5-0.95)", f"{eval_df['AP'].mean():.3f}")

# ===========================================================================
# TAB 5 -- Downstream Statistical & Replicate Consistency
# ===========================================================================
with tab5:
    st.header("Downstream Statistical & Replicate Consistency")
    st.markdown(
        "Compare the primary image pair (Condition A, loaded in the sidebar) against a second uploaded pair "
        "(Condition B) -- e.g. control vs. treatment, or technical replicates."
    )

    if mask_img is None:
        st.info("Upload a mask (Condition A) in the sidebar to use this tab.")
    else:
        st.subheader("Condition B upload")
        raw_file_b = st.file_uploader("Raw image B (TIFF)", type=["tif", "tiff"], key="raw_b_upload")
        mask_file_b = st.file_uploader("Mask B (TIFF or .npy)", type=["tif", "tiff", "npy"], key="mask_b_upload")

        grid_size = st.slider(
            "Spatial grid size for pseudo-replicate tiles (N x N)",
            2,
            8,
            4,
            help="Each image is split into an N x N grid; per-tile object counts serve as spatial pseudo-replicates "
            "for variability/statistical testing when only one image per condition is available.",
        )

        if mask_file_b is None:
            st.info("Upload Condition B's mask to run the comparison.")
        else:
            mask_b = io_utils.load_array_from_upload(mask_file_b)
            if mask_b.ndim != 2:
                mask_b = io_utils.reduce_to_2d(mask_b, "first_axis_index", 0)
            mask_b = mask_b.astype(np.int32)

            if raw_file_b is not None:
                raw_b = io_utils.load_array_from_upload(raw_file_b)
                if raw_b.ndim != 2:
                    raw_b = io_utils.reduce_to_2d(raw_b, "max_projection")

            summary_a = stats_utils.summarize_counts(mask_img, pixel_size_um)
            summary_b = stats_utils.summarize_counts(mask_b, pixel_size_um)

            tiles_a = stats_utils.grid_region_counts(mask_img, grid_size)
            tiles_b = stats_utils.grid_region_counts(mask_b, grid_size)

            comparison = stats_utils.compare_conditions(tiles_a["count"].values, tiles_b["count"].values)

            st.subheader("Summary statistics")
            summary_table = pd.DataFrame(
                [
                    {
                        "Condition": "A (sidebar)",
                        **summary_a,
                        "tile_CV_pct": stats_utils.coefficient_of_variation(tiles_a["count"].values),
                    },
                    {
                        "Condition": "B (uploaded)",
                        **summary_b,
                        "tile_CV_pct": stats_utils.coefficient_of_variation(tiles_b["count"].values),
                    },
                ]
            )
            st.dataframe(summary_table, width="stretch")

            fig_bar = px.bar(
                summary_table,
                x="Condition",
                y="total_count",
                error_y=[tiles_a["count"].std(), tiles_b["count"].std()],
                title="Total object count by condition (error bars = tile-level std. dev.)",
            )
            st.plotly_chart(fig_bar, width="stretch")

            combined_tiles = pd.concat([tiles_a.assign(condition="A"), tiles_b.assign(condition="B")])
            fig_box = px.box(
                combined_tiles,
                x="condition",
                y="count",
                points="all",
                title="Per-tile object count distribution (spatial pseudo-replicates)",
            )
            st.plotly_chart(fig_box, width="stretch")

            st.subheader("Significance testing")
            st.caption(
                "Comparing per-tile counts between conditions (Mann-Whitney U is preferred for small/non-normal "
                "count data; Welch's t-test shown for reference)."
            )
            m1, m2 = st.columns(2)
            mwu_p, tt_p = comparison["mannwhitney_p"], comparison["welch_t_p"]
            m1.metric("Mann-Whitney U p-value", f"{mwu_p:.4f}" if not np.isnan(mwu_p) else "N/A")
            m2.metric("Welch's t-test p-value", f"{tt_p:.4f}" if not np.isnan(tt_p) else "N/A")

            alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)
            if not np.isnan(mwu_p):
                verdict = "statistically significant" if mwu_p < alpha else "not statistically significant"
                st.write(
                    f"At alpha = {alpha}, the difference in object density between conditions is "
                    f"**{verdict}** (Mann-Whitney p = {mwu_p:.4f})."
                )
            else:
                st.write("Not enough variability across tiles to run a significance test (need >= 2 tiles per side with non-zero spread).")
