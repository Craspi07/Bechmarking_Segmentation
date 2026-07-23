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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from skimage.color import label2rgb
from skimage.measure import regionprops

from modules import io_utils, morphology, snr_utils, stats_utils, synthetic

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
            st.caption(
                "'Potential merge' = unusually large area + low circularity (touching cells segmented as one). "
                "'Potential fragment' = area near the bottom of the population's size distribution (over-segmentation "
                "or spurious tiny detections)."
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

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download synthetic raw image (.npy)", io_utils.array_to_npy_bytes(synth_raw), "synthetic_raw.npy"
            )
        with dl2:
            st.download_button(
                "Download ground-truth mask (.npy)", io_utils.array_to_npy_bytes(synth_gt), "synthetic_ground_truth.npy"
            )

        st.subheader("Evaluate a predicted mask against this ground truth")
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
                eval_df = synthetic.evaluate_against_ground_truth(synth_gt, pred_mask.astype(np.int32))
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
